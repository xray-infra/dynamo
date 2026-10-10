//! Same-process native HTTP maintenance. A reversible admission seal is not
//! shutdown, and HTTP quiescence is not evidence of physical worker cleanup.
use std::{
    collections::BTreeMap,
    convert::Infallible,
    fs,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt},
        },
    },
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use anyhow::{Result, bail};
use axum::{
    Router,
    body::{Body, Bytes, HttpBody},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
};
use hyper::{
    body::{Frame, Incoming},
    service::{Service, service_fn},
};
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, UnixListener, UnixStream},
    task::{JoinHandle, JoinSet},
};
use tokio_util::sync::CancellationToken;

use crate::{
    http::service::service_v2::{ServiceStage, State},
    native_cleanup::{CleanupState, NativeCleanupRegistry},
};

struct ConnectionState {
    active: u64,
    closing: bool,
    close: CancellationToken,
}

/// Private ingress decision consumed inside the existing public auth layer.
/// It is never deserialized from a user header or body.
#[derive(Clone, Copy)]
pub(crate) struct NativeHttpAdmission(pub bool);

pub struct NativeManagement {
    state: Arc<State>,
    cleanup: Arc<NativeCleanupRegistry>,
    connections: Mutex<BTreeMap<u64, ConnectionState>>,
    body_changed: tokio::sync::Notify,
    next_connection: AtomicU64,
}

#[derive(Debug, Serialize)]
pub struct NativeManagementStatus {
    pub service_stage: String,
    pub sealed: bool,
    pub connections: usize,
    pub active_http_bodies: u64,
    pub closing_connections: usize,
    pub inflight_inference: u64,
    pub physical_running: usize,
    pub physical_blocked: usize,
    pub blocked_cleanup: BTreeMap<String, String>,
}

impl NativeManagement {
    pub fn new(state: Arc<State>, cleanup: Arc<NativeCleanupRegistry>) -> Self {
        Self {
            state,
            cleanup,
            connections: Mutex::new(BTreeMap::new()),
            body_changed: tokio::sync::Notify::new(),
            next_connection: AtomicU64::new(1),
        }
    }

    pub fn is_for_state(&self, state: &Arc<State>) -> bool {
        Arc::ptr_eq(&self.state, state)
    }

    pub fn seal(&self) -> NativeManagementStatus {
        // The same lock serializes the gate and per-connection request entry.
        let connections = self.connections.lock().unwrap();
        self.state.seal();
        drop(connections);
        self.status()
    }

    pub fn resume(&self) -> Result<NativeManagementStatus> {
        let connections = self.connections.lock().unwrap();
        if !self.state.resume() {
            bail!("terminal service draining/stopping cannot be resumed");
        }
        drop(connections);
        Ok(self.status())
    }

    pub fn status(&self) -> NativeManagementStatus {
        let connections = self.connections.lock().unwrap();
        let physical = self.cleanup.states();
        NativeManagementStatus {
            service_stage: self.state.service_stage().to_string(),
            sealed: self.state.is_sealed(),
            connections: connections.len(),
            active_http_bodies: connections
                .values()
                .map(|connection| connection.active)
                .sum(),
            closing_connections: connections
                .values()
                .filter(|connection| connection.closing)
                .count(),
            inflight_inference: self.state.inflight_count(),
            physical_running: physical
                .values()
                .filter(|state| matches!(state, CleanupState::Running))
                .count(),
            physical_blocked: physical
                .values()
                .filter(|state| matches!(state, CleanupState::Blocked(_)))
                .count(),
            blocked_cleanup: physical
                .iter()
                .filter_map(|(id, state)| match state {
                    CleanupState::Blocked(error) => Some((id.clone(), error.to_string())),
                    CleanupState::Running => None,
                })
                .collect(),
        }
    }

    /// Wait for tracked HTTP calls and bodies, independently of native physical
    /// release. Seal/terminal-drain first to prevent new generation admission.
    pub async fn wait_http_bodies_zero(&self) {
        loop {
            let notified = self.body_changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self
                .connections
                .lock()
                .unwrap()
                .values()
                .all(|connection| connection.active == 0)
            {
                return;
            }
            notified.await;
        }
    }

    /// Mark truly idle registered HTTP connections for protocol-level graceful
    /// shutdown. No task abort, socket shutdown, or inference-count heuristic.
    pub fn close_idle(&self) -> usize {
        self.close_connections(false)
    }

    fn close_connections(&self, including_active: bool) -> usize {
        let mut connections = self.connections.lock().unwrap();
        let mut closing = 0;
        for connection in connections.values_mut() {
            if !connection.closing && (including_active || connection.active == 0) {
                connection.closing = true;
                connection.close.cancel();
                closing += 1;
            }
        }
        closing
    }

    fn register(self: &Arc<Self>) -> (ConnectionPermit, CancellationToken) {
        let id = self.next_connection.fetch_add(1, Ordering::Relaxed);
        let close = CancellationToken::new();
        self.connections.lock().unwrap().insert(
            id,
            ConnectionState {
                active: 0,
                closing: false,
                close: close.clone(),
            },
        );
        (
            ConnectionPermit {
                management: self.clone(),
                id,
            },
            close,
        )
    }

    fn enter(self: &Arc<Self>, id: u64, system: bool) -> Option<(RequestPermit, bool)> {
        let mut connections = self.connections.lock().unwrap();
        let connection = connections.get_mut(&id)?;
        let admitted = !connection.closing
            && self.state.service_stage() == ServiceStage::Ready
            && (system || !self.state.is_sealed());
        connection.active += 1;
        Some((
            RequestPermit {
                management: self.clone(),
                id,
            },
            admitted,
        ))
    }

    /// Bind before starting public HTTP so permission/bind failures fail startup.
    /// One bounded JSON command per same-user UNIX connection; no public route.
    pub fn spawn_unix_control(
        self: &Arc<Self>,
        path: &Path,
        cancel: CancellationToken,
    ) -> Result<JoinHandle<Result<()>>> {
        let (listener, socket) = bind_private_socket(path)?;
        let management = self.clone();
        Ok(tokio::spawn(async move {
            let _socket = socket;
            let mut clients = JoinSet::new();
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    finished = clients.join_next(), if !clients.is_empty() => {
                        if let Some(Err(error)) = finished { tracing::warn!(%error, "private management client failed"); }
                    },
                    accepted = listener.accept(), if clients.len() < 32 => {
                        let (stream, _) = accepted?;
                        let management = management.clone();
                        clients.spawn(async move { management_client(stream, management).await });
                    }
                }
            }
            drop(listener);
            while clients.join_next().await.is_some() {}
            Ok(())
        }))
    }
}

struct ConnectionPermit {
    management: Arc<NativeManagement>,
    id: u64,
}
impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.management.connections.lock().unwrap().remove(&self.id);
        self.management.body_changed.notify_waiters();
    }
}
struct RequestPermit {
    management: Arc<NativeManagement>,
    id: u64,
}
impl Drop for RequestPermit {
    fn drop(&mut self) {
        if let Some(connection) = self
            .management
            .connections
            .lock()
            .unwrap()
            .get_mut(&self.id)
        {
            connection.active -= 1;
        }
        self.management.body_changed.notify_waiters();
    }
}

struct TrackedBody {
    body: Pin<Box<Body>>,
    permit: Option<RequestPermit>,
}
impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let frame = self.body.as_mut().poll_frame(context);
        if matches!(&frame, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            self.permit.take();
        }
        frame
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}

fn unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, axum::Json(serde_json::json!({"error":{"type":"server_error","message":"Native frontend is sealed or draining","code":503,"param":null}}))).into_response()
}

/// Native plaintext serving only. Keep accepting after reversible sealing;
/// idle-close targets individual connections, never the listener/runtime.
pub async fn serve_native(
    listener: TcpListener,
    router: Router,
    state: Arc<State>,
    cancel: CancellationToken,
    management: Arc<NativeManagement>,
) -> Result<()> {
    if !management.is_for_state(&state) {
        bail!("management belongs to another service");
    }
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            completed = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = completed { tracing::warn!(%error, "native HTTP connection task failed"); }
            },
            accepted = listener.accept() => {
                let (stream, _) = match accepted {
                    Ok(accepted) => accepted,
                    Err(error) => {
                        tracing::warn!(%error, "native HTTP accept failed");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };
                let _ = stream.set_nodelay(true);
                let (connection_permit, close) = management.register();
                let id = connection_permit.id;
                let management = management.clone();
                let service = TowerToHyperService::new(router.clone().into_service::<Incoming>());
                connections.spawn(async move {
                    let _connection_permit = connection_permit;
                    let service = service_fn(move |mut request: Request<Incoming>| {
                        let system = matches!(request.uri().path(), "/health" | "/live" | "/metrics");
                        let permit = management.enter(id, system);
                        let service = service.clone();
                        async move {
                            let Some((permit, admitted)) = permit else { return Ok::<_, Infallible>(unavailable()); };
                            request.extensions_mut().insert(NativeHttpAdmission(admitted));
                            let response = service.call(request).await?;
                            let (parts, body) = response.into_parts();
                            Ok(Response::from_parts(parts, Body::new(TrackedBody { body: Box::pin(body), permit: Some(permit) })))
                        }
                    });
                    let builder = Builder::new(TokioExecutor::new());
                    let connection = builder.serve_connection_with_upgrades(TokioIo::new(stream), service);
                    tokio::pin!(connection);
                    tokio::select! {
                        outcome = &mut connection => { if let Err(error) = outcome { tracing::debug!(%error, "native HTTP connection ended"); } },
                        _ = close.cancelled() => {
                            connection.as_mut().graceful_shutdown();
                            if let Err(error) = connection.await { tracing::debug!(%error, "native HTTP connection drained"); }
                        }
                    }
                });
            }
        }
    }
    drop(listener);
    state.start_draining();
    management.seal();
    management.close_connections(true);
    // Active HTTP bodies have no arbitrary deadline. Physical registry wait is
    // owned by the caller after this route/body lifetime drain completes.
    while let Some(result) = connections.join_next().await {
        result?;
    }
    Ok(())
}

struct PrivateSocket {
    path: PathBuf,
    device: u64,
    inode: u64,
}
impl Drop for PrivateSocket {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.dev() == self.device && metadata.ino() == self.inode)
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn private_socket_connection_refused(path: &Path) -> Result<bool> {
    let bytes = path.as_os_str().as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if bytes.contains(&0) || bytes.len() >= address.sun_path.len() {
        bail!("management socket path is not a valid Unix address");
    }
    address.sun_family = libc::AF_UNIX as _;
    for (target, byte) in address.sun_path.iter_mut().zip(bytes) {
        *target = *byte as _;
    }
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let socket = unsafe { OwnedFd::from_raw_fd(fd) };
    for (get, set, flag) in [
        (libc::F_GETFL, libc::F_SETFL, libc::O_NONBLOCK),
        (libc::F_GETFD, libc::F_SETFD, libc::FD_CLOEXEC),
    ] {
        let flags = unsafe { libc::fcntl(socket.as_raw_fd(), get) };
        if flags < 0 || unsafe { libc::fcntl(socket.as_raw_fd(), set, flags | flag) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    let length = std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1;
    let connected = unsafe {
        libc::connect(
            socket.as_raw_fd(),
            std::ptr::from_ref(&address).cast(),
            length as libc::socklen_t,
        )
    };
    // A full live-listener backlog returns EAGAIN with this nonblocking FD.
    // Only actual ECONNREFUSED permits removing an owned stale socket.
    Ok(connected < 0
        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECONNREFUSED))
}

fn bind_private_socket(path: &Path) -> Result<(UnixListener, PrivateSocket)> {
    if !path.is_absolute() {
        bail!("management socket path must be absolute");
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("management socket needs a private parent"))?;
    if !parent.exists() {
        fs::DirBuilder::new().mode(0o700).create(parent)?;
    }
    let directory = fs::symlink_metadata(parent)?;
    let uid = unsafe { libc::geteuid() };
    if !directory.is_dir()
        || directory.file_type().is_symlink()
        || directory.uid() != uid
        || directory.mode() & 0o777 != 0o700
    {
        bail!("management socket parent must be an owned nonsymlink directory with mode 0700");
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if !metadata.file_type().is_socket()
                || metadata.uid() != uid
                || metadata.mode() & 0o777 != 0o600
                || !private_socket_connection_refused(path)?
            {
                bail!("management socket path already exists and is not an owned refused socket");
            }
            let current = fs::symlink_metadata(path)?;
            let current_parent = fs::symlink_metadata(parent)?;
            if current.dev() != metadata.dev()
                || current.ino() != metadata.ino()
                || current.uid() != metadata.uid()
                || current.gid() != metadata.gid()
                || current.mode() != metadata.mode()
                || !current.file_type().is_socket()
                || current_parent.dev() != directory.dev()
                || current_parent.ino() != directory.ino()
                || current_parent.uid() != directory.uid()
                || current_parent.gid() != directory.gid()
                || current_parent.mode() != directory.mode()
                || !current_parent.is_dir()
            {
                bail!("management socket identity changed during refused connection probe");
            }
            fs::remove_file(path)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(path)?;
    let metadata = fs::symlink_metadata(path)?;
    let socket = PrivateSocket {
        path: path.to_owned(),
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    // Trusted parent prevents other users from connecting before chmod.
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok((listener, socket))
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Command {
    Seal,
    Resume,
    Status,
    IdleClose,
}

// A strict object envelope is necessary: serde's internally tagged unit
// variants consume the tag and can ignore remaining object fields.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlCommand {
    command: Command,
}

async fn management_client(
    mut stream: UnixStream,
    management: Arc<NativeManagement>,
) -> Result<()> {
    if stream.peer_cred()?.uid() != unsafe { libc::geteuid() } {
        bail!("management peer UID rejected");
    }
    let mut request = Vec::new();
    let read = {
        let mut reader = BufReader::new((&mut stream).take(4097));
        tokio::time::timeout(
            Duration::from_secs(5),
            reader.read_until(b'\n', &mut request),
        )
        .await??
    };
    let result = if read == 0 || request.len() > 4096 || !request.ends_with(b"\n") {
        serde_json::json!({"ok":false,"error":"one newline-terminated JSON command of at most 4096 bytes is required"})
    } else {
        match serde_json::from_slice::<ControlCommand>(&request).map(|request| request.command) {
            Ok(Command::Seal) => serde_json::json!({"ok":true,"status":management.seal()}),
            Ok(Command::Resume) => match management.resume() {
                Ok(status) => serde_json::json!({"ok":true,"status":status}),
                Err(error) => serde_json::json!({"ok":false,"error":error.to_string()}),
            },
            Ok(Command::Status) => serde_json::json!({"ok":true,"status":management.status()}),
            Ok(Command::IdleClose) => {
                serde_json::json!({"ok":true,"closing_requested":management.close_idle(),"status":management.status()})
            }
            Err(_) => serde_json::json!({"ok":false,"error":"invalid management command"}),
        }
    };
    let mut response = serde_json::to_vec(&result)?;
    response.push(b'\n');
    tokio::time::timeout(Duration::from_secs(5), stream.write_all(&response)).await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use futures::StreamExt;

    #[tokio::test]
    async fn native_management_stale_socket_recovery_closed_owned_socket() {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join("control.sock");
        let previous = std::os::unix::net::UnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        // Closing the actual listener without unlink leaves the same filesystem
        // residue as an abrupt process exit; no cleanup receipt is simulated.
        drop(previous);
        assert!(private_socket_connection_refused(&path).unwrap());
        let (listener, owner) = bind_private_socket(&path).unwrap();
        assert_eq!(fs::symlink_metadata(&path).unwrap().mode() & 0o777, 0o600);
        let client = tokio::time::timeout(Duration::from_secs(2), UnixStream::connect(&path))
            .await.unwrap().unwrap();
        let (accepted, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
            .await.unwrap().unwrap();
        drop(client);
        drop(accepted);
        drop(listener);
        drop(owner);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn native_management_stale_socket_recovery_live_and_full_backlog_are_bounded() {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join("control.sock");
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let original = fs::symlink_metadata(&path).unwrap();
        assert!(!private_socket_connection_refused(&path).unwrap());
        assert!(bind_private_socket(&path).is_err());
        // Empty the successful probe connections before the deliberate full
        // backlog case. This is the real Unix listener, not an error fixture.
        listener.set_nonblocking(true).unwrap();
        let drain_began = std::time::Instant::now();
        let mut drained = 0;
        loop {
            assert!(drain_began.elapsed() < Duration::from_secs(1));
            match listener.accept() {
                Ok((stream, _)) => {
                    drop(stream);
                    drained += 1;
                    assert!(drained <= 2);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("actual probe queue drain failed: {error}"),
            }
        }
        assert_eq!(drained, 2, "both live-socket probes must reach the listener");
        assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 0) }, 0);
        let queued = std::os::unix::net::UnixStream::connect(&path).unwrap();
        let began = std::time::Instant::now();
        assert!(!private_socket_connection_refused(&path).unwrap());
        assert!(bind_private_socket(&path).is_err());
        assert!(began.elapsed() < Duration::from_secs(1));
        let current = fs::symlink_metadata(&path).unwrap();
        assert_eq!((current.dev(), current.ino(), current.uid(), current.mode()),
                   (original.dev(), original.ino(), original.uid(), original.mode()));
        drop(queued);
        drop(listener);
    }

    #[tokio::test]
    async fn native_management_stale_socket_recovery_preserves_invalid_paths_and_foreign_parent() {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join("control.sock");
        fs::write(&path, b"preserve regular file").unwrap();
        assert!(bind_private_socket(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"preserve regular file");
        fs::remove_file(&path).unwrap();
        let target = directory.path().join("target");
        fs::write(&target, b"preserve symlink target").unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(bind_private_socket(&path).is_err());
        assert_eq!(fs::read_link(&path).unwrap(), target);
        assert_eq!(fs::read(&target).unwrap(), b"preserve symlink target");
        fs::remove_file(&path).unwrap();
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        drop(listener);
        let original = fs::symlink_metadata(&path).unwrap();
        assert!(bind_private_socket(&path).is_err());
        assert_eq!(fs::symlink_metadata(&path).unwrap().ino(), original.ino());
        assert_eq!(fs::symlink_metadata(&path).unwrap().mode() & 0o777, 0o644);
        // The normal CPU SDK runs as1001. Use a real foreign-owned parent
        // without creating or changing any file in that parent.
        let foreign = Path::new("/");
        if fs::symlink_metadata(foreign).unwrap().uid() != unsafe { libc::geteuid() } {
            let rejected = foreign.join(format!("qwen-never-create-{}.sock", uuid::Uuid::new_v4()));
            assert!(bind_private_socket(&rejected).is_err());
            assert!(!rejected.exists());
        }
    }

    fn management() -> Arc<NativeManagement> {
        let service = crate::http::service::service_v2::HttpService::builder()
            .build()
            .unwrap();
        Arc::new(NativeManagement::new(
            service.state_clone(),
            Arc::new(NativeCleanupRegistry::new()),
        ))
    }

    async fn wait_active(management: &NativeManagement, expected: u64) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while management.status().active_http_bodies != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn native_management_idle_close_preserves_active_body_and_resumes_same_listener() {
        let management = management();
        let (sender, receiver) = tokio::sync::mpsc::channel::<Bytes>(4);
        let receiver = Arc::new(Mutex::new(Some(receiver)));
        let router = Router::new()
            .route("/idle", get(|| async { "idle" }))
            .route(
                "/slow",
                get({
                    let receiver = receiver.clone();
                    move || {
                        let receiver = receiver.lock().unwrap().take().unwrap();
                        async move {
                            Body::from_stream(
                                tokio_stream::wrappers::ReceiverStream::new(receiver)
                                    .map(Ok::<_, Infallible>),
                            )
                        }
                    }
                }),
            )
            .layer(axum::middleware::from_fn(
                |request: axum::extract::Request, next: axum::middleware::Next| async move {
                    if request
                        .extensions()
                        .get::<NativeHttpAdmission>()
                        .is_some_and(|admission| !admission.0)
                    {
                        unavailable()
                    } else {
                        next.run(request).await
                    }
                },
            ));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        let server = tokio::spawn(serve_native(
            listener,
            router,
            management.state.clone(),
            cancel.clone(),
            management.clone(),
        ));

        // A real HTTP/1 keepalive socket, independent of client pool retries.
        let mut idle = tokio::net::TcpStream::connect(address).await.unwrap();
        idle.write_all(b"GET /idle HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let mut chunk = [0; 1024];
                let length = idle.read(&mut chunk).await.unwrap();
                assert_ne!(length, 0);
                bytes.extend_from_slice(&chunk[..length]);
                if bytes.ends_with(b"\r\n\r\nidle") {
                    break;
                }
            }
        })
        .await
        .unwrap();
        wait_active(&management, 0).await;

        let client = reqwest::Client::builder().http1_only().build().unwrap();
        sender.send(Bytes::from_static(b"A")).await.unwrap();
        let mut active = client
            .get(format!("http://{address}/slow"))
            .send()
            .await
            .unwrap();
        assert_eq!(active.chunk().await.unwrap().unwrap(), "A");
        wait_active(&management, 1).await;
        assert_eq!(management.close_idle(), 1);
        let mut byte = [0];
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), idle.read(&mut byte))
                .await
                .unwrap()
                .unwrap(),
            0
        );
        assert_eq!(management.status().active_http_bodies, 1);

        management.seal();
        assert!(!management.state.is_ready());
        let rejected = client
            .get(format!("http://{address}/idle"))
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
        drop(rejected);
        management.resume().unwrap();
        assert!(management.state.is_ready());
        assert_eq!(
            client
                .get(format!("http://{address}/idle"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
            "idle"
        );

        // A body admitted before seal completes unchanged after seal/resume.
        sender.send(Bytes::from_static(b"B")).await.unwrap();
        drop(sender);
        assert_eq!(active.chunk().await.unwrap().unwrap(), "B");
        assert!(active.chunk().await.unwrap().is_none());
        wait_active(&management, 0).await;
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(management.status().connections, 0);
        assert!(management.resume().is_err());
    }

    async fn command(path: &Path, command: &str) -> serde_json::Value {
        let mut socket = UnixStream::connect(path).await.unwrap();
        socket.write_all(command.as_bytes()).await.unwrap();
        let mut response = String::new();
        BufReader::new(socket)
            .read_line(&mut response)
            .await
            .unwrap();
        serde_json::from_str(&response).unwrap()
    }

    #[tokio::test]
    async fn native_management_private_unix_permissions_and_strict_commands() {
        let management = management();
        let directory = tempfile::tempdir().unwrap();
        let private = directory.path().join("management");
        let path = private.join("control.sock");
        let cancel = CancellationToken::new();
        let task = management
            .spawn_unix_control(&path, cancel.clone())
            .unwrap();
        assert_eq!(
            fs::symlink_metadata(&private).unwrap().mode() & 0o777,
            0o700
        );
        assert_eq!(fs::symlink_metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(
            command(&path, "{\"command\":\"seal\"}\n").await["status"]["sealed"]
                .as_bool()
                .unwrap()
        );
        assert!(
            !command(&path, "{\"command\":\"resume\"}\n").await["status"]["sealed"]
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            command(&path, "{\"command\":\"status\",\"unexpected\":true}\n").await["ok"],
            false
        );
        management.state.start_draining();
        assert_eq!(
            command(&path, "{\"command\":\"resume\"}\n").await["ok"],
            false
        );
        cancel.cancel();
        task.await.unwrap().unwrap();
        assert!(!path.exists());

        fs::set_permissions(&private, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            management
                .spawn_unix_control(&path, CancellationToken::new())
                .is_err()
        );
        fs::set_permissions(&private, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(&path, b"must preserve").unwrap();
        assert!(
            management
                .spawn_unix_control(&path, CancellationToken::new())
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"must preserve");
    }
}

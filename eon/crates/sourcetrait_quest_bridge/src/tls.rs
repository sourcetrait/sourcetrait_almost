//! Mutual TLS from the installed material: both ends, one leaf.
use crate::*;

/// The name a loopback peer's certificate is checked against.
pub const LOOPBACK_NAME: &str = "localhost";

/// The four installed artifacts, as the profile that owns them names.
pub fn material(secret_data: &Path, profile: &str) -> srcert::CertFiles {
    srcert::CertFiles::new(&srcert::secret_certs_dir(secret_data, profile), profile)
}

/// A client that PRESENTS our leaf and trusts only our authority.
///
/// Client authentication is the point rather than an option: the leaf
/// carries both usages precisely so it can be shown at this end.
pub fn client_config(files: &srcert::CertFiles) -> BridgeResult<rustls::ClientConfig> {
    let roots = authority_store(&files.authority_public)?;
    let (chain, key) = leaf(files)?;
    match builder()?
        .with_root_certificates(roots)
        .with_client_auth_cert(chain, key)
    {
        Ok(config) => Ok(config),
        Err(error) => snafu::whatever!("the client configuration was refused: {error}"),
    }
}

/// A server that REQUIRES a client certificate from our authority.
pub fn server_config(files: &srcert::CertFiles) -> BridgeResult<rustls::ServerConfig> {
    let roots = authority_store(&files.authority_public)?;
    let verifier = match rustls::server::WebPkiClientVerifier::builder_with_provider(
        std::sync::Arc::new(roots),
        provider(),
    )
    .build()
    {
        Ok(verifier) => verifier,
        Err(error) => snafu::whatever!("the client verifier was refused: {error}"),
    };
    let (chain, key) = leaf(files)?;
    match server_builder()?
        .with_client_cert_verifier(verifier)
        .with_single_cert(chain, key)
    {
        Ok(config) => Ok(config),
        Err(error) => snafu::whatever!("the server configuration was refused: {error}"),
    }
}

/// The name a client checks the server's certificate against.
pub fn loopback_server_name() -> BridgeResult<r::tls::ServerName<'static>> {
    match r::tls::ServerName::try_from(LOOPBACK_NAME) {
        Ok(name) => Ok(name),
        Err(error) => snafu::whatever!("{LOOPBACK_NAME:?} is not a server name: {error}"),
    }
}

/// Ring, named rather than taken from process-global state.
fn provider() -> std::sync::Arc<rustls::crypto::CryptoProvider> {
    std::sync::Arc::new(rustls::crypto::ring::default_provider())
}

/// A builder on the named provider, so nothing depends on install order.
fn builder() -> BridgeResult<rustls::ConfigBuilder<rustls::ClientConfig, rustls::WantsVerifier>> {
    match rustls::ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
    {
        Ok(builder) => Ok(builder),
        Err(error) => snafu::whatever!("the protocol versions were refused: {error}"),
    }
}

/// The server half of the same builder; the two types do not unify.
fn server_builder()
-> BridgeResult<rustls::ConfigBuilder<rustls::ServerConfig, rustls::WantsVerifier>> {
    match rustls::ServerConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
    {
        Ok(builder) => Ok(builder),
        Err(error) => snafu::whatever!("the protocol versions were refused: {error}"),
    }
}

/// Our authority, and ONLY our authority. No platform roots.
fn authority_store(authority: &Path) -> BridgeResult<rustls::RootCertStore> {
    let certificate = match r::tls::CertificateDer::from_pem_file(authority) {
        Ok(certificate) => certificate,
        Err(error) => {
            snafu::whatever!("reading {}: {error}", authority.display())
        }
    };
    let mut roots = rustls::RootCertStore::empty();
    if let Err(error) = roots.add(certificate) {
        snafu::whatever!("{} is not a usable authority: {error}", authority.display());
    }
    Ok(roots)
}

/// The leaf this end presents, and the key proving it is ours.
fn leaf(
    files: &srcert::CertFiles,
) -> BridgeResult<(Vec<r::tls::CertificateDer<'static>>, r::tls::PrivateKeyDer<'static>)> {
    let certificate = match r::tls::CertificateDer::from_pem_file(&files.entity_public) {
        Ok(certificate) => certificate,
        Err(error) => {
            snafu::whatever!("reading {}: {error}", files.entity_public.display())
        }
    };
    let key = match r::tls::PrivateKeyDer::from_pem_file(&files.entity_private) {
        Ok(key) => key,
        Err(error) => {
            snafu::whatever!("reading {}: {error}", files.entity_private.display())
        }
    };
    Ok((vec![certificate], key))
}

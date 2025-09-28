# TLS Implementation

> **Relevant source files**
> * [src/src/tls-gnu.c](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c)
> * [src/src/tls-openssl.c](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c)

## Purpose and Scope

This document covers Exim's Transport Layer Security (TLS) implementation, which provides encrypted communication for both SMTP client and server operations. Exim supports two TLS backends: OpenSSL and GnuTLS, with a unified interface that abstracts the underlying cryptographic library differences.

For general transport configuration, see [Transport Mechanisms](/Exim/exim/6-transport-mechanisms). For security-related ACL processing, see [Access Control Lists (ACLs)](/Exim/exim/2.4-access-control-lists-(acls)).

## Architecture Overview

Exim's TLS implementation uses a dual-backend architecture that supports both OpenSSL and GnuTLS libraries through compile-time selection. The implementation provides comprehensive TLS features including SNI, OCSP, ALPN, session resumption, and DANE verification.

### TLS Backend Selection

```mermaid
flowchart TD

COMPILE["Compile-time Selection"]
OPENSSL["OpenSSL Backend<br>tls-openssl.c"]
GNUTLS["GnuTLS Backend<br>tls-gnu.c"]
COMMON["Common Interface<br>tls.c"]
SMTP_IN["smtp_in.c<br>Server Operations"]
SMTP_OUT["SMTP Client<br>Transport Layer"]

COMPILE --> OPENSSL
COMPILE --> GNUTLS
OPENSSL --> COMMON
GNUTLS --> COMMON
COMMON --> SMTP_IN
COMMON --> SMTP_OUT
```

**TLS Backend Architecture**

Sources: [src/src/tls-openssl.c L1-L50](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1-L50)

 [src/src/tls-gnu.c L1-L50](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L1-L50)

### State Management Structure

The TLS implementation centers around main state structures that manage session context, credentials, and configuration for both client and server operations.

```mermaid
flowchart TD

OSSL_STATE["exim_openssl_state_st"]
OSSL_CTX["SSL_CTX *lib_ctx"]
OSSL_SSL["SSL *lib_ssl"]
OSSL_CREDS["Certificate/Key Storage"]
GNU_STATE["exim_gnutls_state_st"]
GNU_SESSION["gnutls_session_t session"]
GNU_CREDS["gnutls_certificate_credentials_t x509_cred"]
GNU_PRIORITY["gnutls_priority_t pri_cache"]
TLS_SUPPORT["tls_support *tlsp"]
HOST_INFO["host_item *host"]
VERIFY_REQ["verify_requirement"]
EVENT_ACTION["event_action"]

OSSL_STATE --> TLS_SUPPORT
GNU_STATE --> TLS_SUPPORT

subgraph subGraph2 ["Common Elements"]
    TLS_SUPPORT
    HOST_INFO
    VERIFY_REQ
    EVENT_ACTION
    TLS_SUPPORT --> HOST_INFO
    TLS_SUPPORT --> VERIFY_REQ
    TLS_SUPPORT --> EVENT_ACTION
end

subgraph subGraph1 ["GnuTLS Implementation"]
    GNU_STATE
    GNU_SESSION
    GNU_CREDS
    GNU_PRIORITY
    GNU_STATE --> GNU_SESSION
    GNU_STATE --> GNU_CREDS
    GNU_STATE --> GNU_PRIORITY
end

subgraph subGraph0 ["OpenSSL Implementation"]
    OSSL_STATE
    OSSL_CTX
    OSSL_SSL
    OSSL_CREDS
    OSSL_STATE --> OSSL_CTX
    OSSL_STATE --> OSSL_SSL
    OSSL_STATE --> OSSL_CREDS
end
```

**TLS State Management Structure**

Sources: [src/src/tls-openssl.c L402-L441](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L402-L441)

 [src/src/tls-gnu.c L200-L256](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L200-L256)

## Initialization and Credential Management

### Server Initialization Flow

TLS server initialization involves credential preloading, certificate setup, and callback registration for various TLS extensions.

```mermaid
sequenceDiagram
  participant Daemon Process
  participant tls_server_creds_init()
  participant Certificate Loading
  participant OCSP Setup
  participant Callback Registration

  Daemon Process->>tls_server_creds_init(): Initialize server credentials
  tls_server_creds_init()->>Certificate Loading: tls_add_certfile()
  Certificate Loading->>tls_server_creds_init(): Certificate loaded
  tls_server_creds_init()->>OCSP Setup: Setup OCSP stapling
  OCSP Setup->>tls_server_creds_init(): OCSP configured
  tls_server_creds_init()->>Callback Registration: Register SNI callback
  Callback Registration->>tls_server_creds_init(): tls_servername_cb registered
  tls_server_creds_init()->>Callback Registration: Register ALPN callback
  Callback Registration->>tls_server_creds_init(): tls_server_alpn_cb registered
  tls_server_creds_init()->>Daemon Process: Server ready
```

**Server Initialization Sequence**

Sources: [src/src/tls-openssl.c L1742-L1861](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1742-L1861)

 [src/src/tls-gnu.c L1593-L1708](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L1593-L1708)

### Client Initialization Flow

Client initialization supports credential preloading and transport-specific configuration, enabling efficient connection establishment.

```mermaid
flowchart TD

CLIENT_INIT["tls_client_creds_init()"]
TRANSPORT["Transport Options"]
PRELOAD["Credential Preloading"]
CERT_LOAD["Certificate Loading<br>tls_add_certfile()"]
CA_BUNDLE["CA Bundle Loading<br>creds_load_cabundle()"]
CRL_LOAD["CRL Loading<br>creds_load_crl()"]
WATCH["File Watching<br>tls_set_watch()"]

CLIENT_INIT --> TRANSPORT
TRANSPORT --> PRELOAD
PRELOAD --> CERT_LOAD
PRELOAD --> CA_BUNDLE
PRELOAD --> CRL_LOAD
CERT_LOAD --> WATCH
CA_BUNDLE --> WATCH
CRL_LOAD --> WATCH
```

**Client Credential Initialization**

Sources: [src/src/tls-openssl.c L1871-L1941](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1871-L1941)

 [src/src/tls-gnu.c L1717-L1810](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L1717-L1810)

## TLS Extensions and Features

### Server Name Indication (SNI)

SNI allows multiple SSL certificates on a single IP address by enabling certificate selection based on the requested hostname.

```mermaid
flowchart TD

CLIENT_HELLO["Client Hello<br>with SNI Extension"]
SNI_CB["SNI Callback<br>tls_servername_cb()"]
CERT_SELECT["Certificate Selection"]
NEW_CTX["New SSL Context<br>server_sni"]
EXPAND["Variable Expansion<br>tls_sni, tls_in_sni"]
REEXPAND["Re-expand Files<br>tls_expand_session_files()"]

CLIENT_HELLO --> SNI_CB
SNI_CB --> CERT_SELECT
CERT_SELECT --> NEW_CTX
SNI_CB --> EXPAND
EXPAND --> REEXPAND
REEXPAND --> NEW_CTX
```

**SNI Processing Flow**

Sources: [src/src/tls-openssl.c L2216-L2302](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L2216-L2302)

 [src/src/tls-gnu.c L2799-L2854](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2799-L2854)

### OCSP Stapling

Online Certificate Status Protocol (OCSP) stapling allows the server to provide certificate revocation status during the TLS handshake.

```mermaid
flowchart TD

OCSP_FILE["OCSP Response File"]
OCSP_LOAD["ocsp_load_response()"]
OCSP_CB["tls_server_stapling_cb()"]
CLIENT_CB["tls_client_stapling_cb()"]
VERIFY_STORE["X509_STORE verification"]
OCSP_BASIC["OCSP_basic_verify()"]
HANDSHAKE["TLS Handshake"]

OCSP_CB --> HANDSHAKE
HANDSHAKE --> CLIENT_CB

subgraph subGraph1 ["Client OCSP Verification"]
    CLIENT_CB
    VERIFY_STORE
    OCSP_BASIC
    CLIENT_CB --> VERIFY_STORE
    VERIFY_STORE --> OCSP_BASIC
end

subgraph subGraph0 ["Server OCSP Setup"]
    OCSP_FILE
    OCSP_LOAD
    OCSP_CB
    OCSP_FILE --> OCSP_LOAD
    OCSP_LOAD --> OCSP_CB
end
```

**OCSP Stapling Implementation**

Sources: [src/src/tls-openssl.c L2388-L2467](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L2388-L2467)

 [src/src/tls-openssl.c L2488-L2795](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L2488-L2795)

### Application Layer Protocol Negotiation (ALPN)

ALPN enables protocol negotiation during the TLS handshake, commonly used for HTTP/2 selection.

```mermaid
sequenceDiagram
  participant TLS Client
  participant TLS Server
  participant tls_server_alpn_cb()
  participant tls_alpn Configuration

  TLS Client->>TLS Server: Client Hello + ALPN Extension
  TLS Server->>tls_server_alpn_cb(): Process ALPN protocols
  tls_server_alpn_cb()->>tls_alpn Configuration: Check configured protocols
  tls_alpn Configuration->>tls_server_alpn_cb(): Return matches
  tls_server_alpn_cb()->>TLS Server: Select protocol or reject
  TLS Server->>TLS Client: Server Hello + Selected Protocol
```

**ALPN Protocol Negotiation**

Sources: [src/src/tls-openssl.c L2316-L2370](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L2316-L2370)

 [src/src/tls-gnu.c L3054-L3076](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L3054-L3076)

## Certificate Verification

### Verification Process

The certificate verification process supports multiple verification modes and integrates with DANE for DNS-based certificate validation.

```mermaid
flowchart TD

PEER_CERT["Peer Certificate"]
VERIFY_MODE["Verification Mode"]
NONE["VERIFY_NONE"]
OPTIONAL["VERIFY_OPTIONAL"]
REQUIRED["VERIFY_REQUIRED"]
DANE["VERIFY_DANE"]
BASIC_VERIFY["Basic Certificate Verification"]
HOSTNAME_CHECK["Hostname Verification"]
DANE_VERIFY["DANE Verification"]
EVENT_HOOK["Event Hook Processing"]

VERIFY_MODE --> NONE
VERIFY_MODE --> OPTIONAL
VERIFY_MODE --> REQUIRED
VERIFY_MODE --> DANE
PEER_CERT --> BASIC_VERIFY
BASIC_VERIFY --> HOSTNAME_CHECK
HOSTNAME_CHECK --> DANE_VERIFY
DANE_VERIFY --> EVENT_HOOK
REQUIRED --> BASIC_VERIFY
DANE --> DANE_VERIFY
OPTIONAL --> BASIC_VERIFY
```

**Certificate Verification Flow**

Sources: [src/src/tls-openssl.c L1096-L1233](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1096-L1233)

 [src/src/tls-gnu.c L2534-L2755](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2534-L2755)

### DANE Support

DNS-based Authentication of Named Entities (DANE) provides certificate validation using DNS TLSA records.

```mermaid
flowchart TD

TLSA_RECORDS["DNS TLSA Records"]
DANE_STATE["dane_state_t"]
DANE_QUERY["dane_query_t"]
CERT_CHAIN["Certificate Chain"]
DANE_VERIFY["dane_verify_crt_raw()"]
DANE_TA["DANE_TA (Trust Anchor)"]
DANE_EE["DANE_EE (End Entity)"]
VERIFICATION_RESULT["Verification Result"]

DANE_VERIFY --> DANE_TA
DANE_VERIFY --> DANE_EE
DANE_TA --> VERIFICATION_RESULT
DANE_EE --> VERIFICATION_RESULT

subgraph subGraph1 ["Usage Types"]
    DANE_TA
    DANE_EE
end

subgraph subGraph0 ["DANE Verification Process"]
    TLSA_RECORDS
    DANE_STATE
    DANE_QUERY
    CERT_CHAIN
    DANE_VERIFY
    TLSA_RECORDS --> DANE_STATE
    DANE_STATE --> DANE_QUERY
    CERT_CHAIN --> DANE_VERIFY
    DANE_QUERY --> DANE_VERIFY
end
```

**DANE Verification Architecture**

Sources: [src/src/tls-openssl.c L1241-L1281](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1241-L1281)

 [src/src/tls-gnu.c L2556-L2677](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2556-L2677)

## Session Management and Resumption

### Session Resumption

TLS session resumption reduces handshake overhead by reusing previous session parameters.

```mermaid
flowchart TD

TICKET_KEY["Session Ticket Key<br>exim_stek"]
TICKET_CB["ticket_key_callback()"]
KEY_ROTATION["Key Rotation<br>tk_init()"]
CLIENT_TICKET["Client Presents Ticket"]
SERVER_VALIDATE["Server Validates Ticket"]
SESSION_RESTORE["Session Restoration"]
NEW_SESSION["New Session Creation"]

TICKET_CB --> SERVER_VALIDATE

subgraph subGraph1 ["Resumption Flow"]
    CLIENT_TICKET
    SERVER_VALIDATE
    SESSION_RESTORE
    NEW_SESSION
    CLIENT_TICKET --> SERVER_VALIDATE
    SERVER_VALIDATE --> SESSION_RESTORE
    SERVER_VALIDATE --> NEW_SESSION
end

subgraph subGraph0 ["Session Ticket Management"]
    TICKET_KEY
    TICKET_CB
    KEY_ROTATION
    TICKET_KEY --> TICKET_CB
    KEY_ROTATION --> TICKET_KEY
end
```

**Session Resumption Mechanism**

Sources: [src/src/tls-openssl.c L2117-L2179](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L2117-L2179)

 [src/src/tls-gnu.c L2956-L3013](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2956-L3013)

## Configuration and Options

### TLS Options Processing

The TLS implementation processes various configuration options for ciphers, protocols, and security features.

```mermaid
flowchart TD

SSL_OPTIONS["SSL_OP_* flags"]
OPTION_PARSER["tls_openssl_options_parse()"]
INIT_OPTIONS["init_options global"]
PRIORITY_STRING["Priority String"]
PRIORITY_CACHE["gnutls_priority_t"]
CIPHER_LOAD["creds_load_pristring()"]
TLS_REQUIRE_CIPHERS["tls_require_ciphers"]
TLS_DHPARAM["tls_dhparam"]
TLS_ECCURVE["tls_eccurve"]

TLS_REQUIRE_CIPHERS --> SSL_OPTIONS
TLS_REQUIRE_CIPHERS --> PRIORITY_STRING
TLS_DHPARAM --> INIT_OPTIONS
TLS_ECCURVE --> INIT_OPTIONS

subgraph subGraph2 ["Common Configuration"]
    TLS_REQUIRE_CIPHERS
    TLS_DHPARAM
    TLS_ECCURVE
end

subgraph subGraph1 ["GnuTLS Priority"]
    PRIORITY_STRING
    PRIORITY_CACHE
    CIPHER_LOAD
    PRIORITY_STRING --> CIPHER_LOAD
    CIPHER_LOAD --> PRIORITY_CACHE
end

subgraph subGraph0 ["OpenSSL Options"]
    SSL_OPTIONS
    OPTION_PARSER
    INIT_OPTIONS
    SSL_OPTIONS --> OPTION_PARSER
    OPTION_PARSER --> INIT_OPTIONS
end
```

**TLS Configuration Processing**

Sources: [src/src/tls-openssl.c L175-L294](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L175-L294)

 [src/src/tls-gnu.c L2578-L2590](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2578-L2590)

## Error Handling and Debugging

### Error Processing

Both TLS backends provide comprehensive error handling with detailed error reporting and debugging capabilities.

```mermaid
flowchart TD

TLS_ERROR["tls_error()"]
TLS_ERROR_GNU["tls_error_gnu()"]
TLS_ERROR_SYS["tls_error_sys()"]
ERROR_CONTEXT["Error Context"]
ERROR_MSG["Error Message"]
HOST_INFO["Host Information"]
ERROR_STR["Output Error String"]
OPENSSL_ERR["ERR_error_string_n()"]
GNUTLS_ERR["gnutls_strerror()"]
SYSTEM_ERR["strerror()"]

TLS_ERROR --> ERROR_CONTEXT
TLS_ERROR --> ERROR_MSG
TLS_ERROR --> HOST_INFO
TLS_ERROR --> ERROR_STR
TLS_ERROR_GNU --> GNUTLS_ERR
TLS_ERROR_SYS --> SYSTEM_ERR
ERROR_MSG --> OPENSSL_ERR
```

**Error Handling Framework**

Sources: [src/src/tls-openssl.c L500-L513](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L500-L513)

 [src/src/tls-gnu.c L396-L425](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L396-L425)

## Integration Points

The TLS implementation integrates with multiple Exim subsystems including SMTP processing, transport management, and access control systems.

```mermaid
flowchart TD

ACL_VERIFY["Certificate Verification"]
TLS_VARS["TLS Variables"]
EVENT_ACTIONS["Event Actions"]
TRANSPORT_OPTS["Transport Options"]
PRELOAD_CREDS["Preloaded Credentials"]
CLIENT_CERTS["Client Certificates"]
SMTP_IN["smtp_in.c"]
TLS_ACTIVE["tls_active"]
TLS_GETC["tls_getc()"]
TLS_WRITE["tls_write()"]
SMTP_OUT["SMTP Transport"]

subgraph subGraph2 ["ACL Integration"]
    ACL_VERIFY
    TLS_VARS
    EVENT_ACTIONS
    ACL_VERIFY --> TLS_VARS
    TLS_VARS --> EVENT_ACTIONS
end

subgraph subGraph1 ["Transport Integration"]
    TRANSPORT_OPTS
    PRELOAD_CREDS
    CLIENT_CERTS
    TRANSPORT_OPTS --> PRELOAD_CREDS
    PRELOAD_CREDS --> CLIENT_CERTS
end

subgraph subGraph0 ["SMTP Integration"]
    SMTP_IN
    TLS_ACTIVE
    TLS_GETC
    TLS_WRITE
    SMTP_OUT
    TLS_ACTIVE --> SMTP_IN
    TLS_GETC --> SMTP_IN
    TLS_WRITE --> SMTP_OUT
end
```

**TLS System Integration**

Sources: [src/src/tls-openssl.c L3049-L3054](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L3049-L3054)

 [src/src/tls-gnu.c L2185-L2301](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L2185-L2301)
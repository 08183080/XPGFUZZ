# Security Features

> **Relevant source files**
> * [src/src/ip.c](https://github.com/Exim/exim/blob/29568b25/src/src/ip.c)
> * [src/src/malware.c](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c)
> * [src/src/mime.c](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c)
> * [src/src/mime.h](https://github.com/Exim/exim/blob/29568b25/src/src/mime.h)
> * [src/src/regex.c](https://github.com/Exim/exim/blob/29568b25/src/src/regex.c)
> * [src/src/smtp_out.c](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_out.c)
> * [src/src/spam.c](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c)
> * [src/src/spool_mbox.c](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c)
> * [src/src/tls-gnu.c](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c)
> * [src/src/tls-openssl.c](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c)
> * [test/aux-fixed/4502.msg2.txt](https://github.com/Exim/exim/blob/29568b25/test/aux-fixed/4502.msg2.txt)
> * [test/aux-fixed/4502.msg3.txt](https://github.com/Exim/exim/blob/29568b25/test/aux-fixed/4502.msg3.txt)
> * [test/confs/4000](https://github.com/Exim/exim/blob/29568b25/test/confs/4000)
> * [test/confs/4500](https://github.com/Exim/exim/blob/29568b25/test/confs/4500)
> * [test/log/4000](https://github.com/Exim/exim/blob/29568b25/test/log/4000)
> * [test/log/4500](https://github.com/Exim/exim/blob/29568b25/test/log/4500)
> * [test/log/4501](https://github.com/Exim/exim/blob/29568b25/test/log/4501)
> * [test/log/4502](https://github.com/Exim/exim/blob/29568b25/test/log/4502)
> * [test/log/4506](https://github.com/Exim/exim/blob/29568b25/test/log/4506)
> * [test/mail/4000.userx](https://github.com/Exim/exim/blob/29568b25/test/mail/4000.userx)
> * [test/rejectlog/4000](https://github.com/Exim/exim/blob/29568b25/test/rejectlog/4000)
> * [test/scripts/4000-scanning/4000](https://github.com/Exim/exim/blob/29568b25/test/scripts/4000-scanning/4000)
> * [test/scripts/4500-DKIM/4500](https://github.com/Exim/exim/blob/29568b25/test/scripts/4500-DKIM/4500)
> * [test/scripts/4500-DKIM/4506](https://github.com/Exim/exim/blob/29568b25/test/scripts/4500-DKIM/4506)
> * [test/stderr/4507](https://github.com/Exim/exim/blob/29568b25/test/stderr/4507)
> * [test/stdout/4000](https://github.com/Exim/exim/blob/29568b25/test/stdout/4000)

This document covers Exim's comprehensive security mechanisms including encryption, authentication, and content filtering. These features protect mail transmission through TLS/SSL, verify message authenticity via DKIM, and filter malicious content through integrated scanning systems.

For information about Access Control Lists (ACLs) that enforce security policies, see [Access Control Lists (ACLs)](/Exim/exim/2.4-access-control-lists-(acls)). For details on routing and transport security, see [Transport and Delivery](/Exim/exim/2.3-transport-and-delivery).

## TLS/SSL Implementation

Exim provides robust TLS encryption through a pluggable backend architecture supporting both OpenSSL and GnuTLS libraries. The implementation handles certificate verification, session management, and various security protocols.

### TLS Backend Architecture

```mermaid
flowchart TD

TLS_MAIN["tls.c<br>Main TLS Interface"]
OPENSSL["tls-openssl.c<br>OpenSSL Backend"]
GNUTLS["tls-gnu.c<br>GnuTLS Backend"]
INIT["tls_init()"]
HANDSHAKE["tls_server_start()<br>tls_client_start()"]
VERIFY["verify_callback()"]
CERTS["setup_certs()"]
OCSP["OCSP Validation"]
SNI["Server Name Indication"]
RESUME["Session Resumption"]
ALPN["Application Layer Protocol Negotiation"]

TLS_MAIN --> OPENSSL
TLS_MAIN --> GNUTLS
OPENSSL --> INIT
OPENSSL --> HANDSHAKE
OPENSSL --> VERIFY
OPENSSL --> CERTS
GNUTLS --> INIT
GNUTLS --> HANDSHAKE
GNUTLS --> VERIFY
GNUTLS --> CERTS
VERIFY --> OCSP
HANDSHAKE --> SNI
HANDSHAKE --> RESUME
HANDSHAKE --> ALPN

subgraph subGraph3 ["Security Features"]
    OCSP
    SNI
    RESUME
    ALPN
end

subgraph subGraph2 ["Core Functions"]
    INIT
    HANDSHAKE
    VERIFY
    CERTS
end

subgraph subGraph1 ["Backend Implementations"]
    OPENSSL
    GNUTLS
end

subgraph subGraph0 ["TLS Interface Layer"]
    TLS_MAIN
end
```

The TLS implementation uses compile-time selection between backends through conditional compilation. Key structures include `exim_openssl_state_st` for OpenSSL and `exim_gnutls_state_st` for GnuTLS, maintaining connection state and configuration.

Sources: [src/src/tls-openssl.c L1-L2000](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1-L2000)

 [src/src/tls-gnu.c L1-L2000](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L1-L2000)

### Certificate Verification and OCSP

```mermaid
flowchart TD

CERT_INPUT["Certificate Chain"]
VERIFY_FUNC["verify_callback()"]
HOSTNAME_CHECK["Hostname Verification"]
CHAIN_VALID["Chain Validation"]
OCSP_REQ["OCSP Request"]
OCSP_RESP["OCSP Response"]
OCSP_VERIFY["OCSP Verification"]
CERT_VERIFIED["Certificate Verified"]
PEER_CERT["Peer Certificate Stored"]
TLS_VARS["TLS Variables Set"]

VERIFY_FUNC --> OCSP_REQ
CHAIN_VALID --> CERT_VERIFIED
HOSTNAME_CHECK --> CERT_VERIFIED
OCSP_VERIFY --> CERT_VERIFIED

subgraph Results ["Results"]
    CERT_VERIFIED
    PEER_CERT
    TLS_VARS
    CERT_VERIFIED --> PEER_CERT
    CERT_VERIFIED --> TLS_VARS
end

subgraph subGraph1 ["OCSP Processing"]
    OCSP_REQ
    OCSP_RESP
    OCSP_VERIFY
    OCSP_REQ --> OCSP_RESP
    OCSP_RESP --> OCSP_VERIFY
end

subgraph subGraph0 ["Certificate Chain Validation"]
    CERT_INPUT
    VERIFY_FUNC
    HOSTNAME_CHECK
    CHAIN_VALID
    CERT_INPUT --> VERIFY_FUNC
    VERIFY_FUNC --> HOSTNAME_CHECK
    VERIFY_FUNC --> CHAIN_VALID
end
```

Certificate verification includes hostname checking via `X509_check_host()` for OpenSSL or custom verification for GnuTLS. OCSP validation provides real-time certificate status checking with configurable response handling.

Sources: [src/src/tls-openssl.c L1096-L1400](https://github.com/Exim/exim/blob/29568b25/src/src/tls-openssl.c#L1096-L1400)

 [src/src/tls-gnu.c L1500-L1800](https://github.com/Exim/exim/blob/29568b25/src/src/tls-gnu.c#L1500-L1800)

## DKIM Verification

Exim implements comprehensive DKIM (DomainKeys Identified Mail) signature verification to authenticate message origin and detect tampering. The DKIM subsystem operates during message reception and integrates with the ACL system.

### DKIM Verification Flow

```mermaid
flowchart TD

MSG_RECV["Message Reception"]
DKIM_HEADERS["Extract DKIM Headers"]
SIG_PARSE["Parse DKIM-Signature"]
DNS_LOOKUP["DNS TXT Lookup"]
PUBKEY_PARSE["Parse Public Key"]
KEY_IMPORT["Import RSA/Ed25519 Key"]
HASH_CALC["Calculate Body Hash"]
HASH_COMPARE["Compare Hashes"]
SIG_VERIFY["Verify Signature"]
DKIM_PASS["dkim=pass"]
DKIM_FAIL["dkim=fail"]
DKIM_VARS["Set dkim_* Variables"]
AUTH_RESULTS["Authentication-Results Header"]

SIG_PARSE --> DNS_LOOKUP
SIG_PARSE --> HASH_CALC
KEY_IMPORT --> SIG_VERIFY
HASH_COMPARE --> DKIM_PASS
HASH_COMPARE --> DKIM_FAIL
SIG_VERIFY --> DKIM_PASS
SIG_VERIFY --> DKIM_FAIL

subgraph Results ["Results"]
    DKIM_PASS
    DKIM_FAIL
    DKIM_VARS
    AUTH_RESULTS
    DKIM_PASS --> DKIM_VARS
    DKIM_FAIL --> DKIM_VARS
    DKIM_VARS --> AUTH_RESULTS
end

subgraph Verification ["Verification"]
    HASH_CALC
    HASH_COMPARE
    SIG_VERIFY
    HASH_CALC --> HASH_COMPARE
end

subgraph subGraph1 ["DNS Operations"]
    DNS_LOOKUP
    PUBKEY_PARSE
    KEY_IMPORT
    DNS_LOOKUP --> PUBKEY_PARSE
    PUBKEY_PARSE --> KEY_IMPORT
end

subgraph subGraph0 ["Message Processing"]
    MSG_RECV
    DKIM_HEADERS
    SIG_PARSE
    MSG_RECV --> DKIM_HEADERS
    DKIM_HEADERS --> SIG_PARSE
end
```

DKIM verification supports multiple algorithms (rsa-sha1, rsa-sha256, ed25519-sha256) and canonicalization methods (simple, relaxed). The process includes DNS key retrieval, signature validation, and body hash verification.

Sources: Test files [test/log/4502 L1-L28](https://github.com/Exim/exim/blob/29568b25/test/log/4502#L1-L28)

 [test/scripts/4500-DKIM/4500 L1-L100](https://github.com/Exim/exim/blob/29568b25/test/scripts/4500-DKIM/4500#L1-L100)

 [test/confs/4500 L1-L50](https://github.com/Exim/exim/blob/29568b25/test/confs/4500#L1-L50)

## Content Scanning

Exim provides integrated content scanning through external malware scanners and spam detection systems. The scanning architecture supports multiple backends with failover capabilities.

### Malware Scanning Architecture

```mermaid
flowchart TD

CLAMD["M_CLAMD<br>ClamAV Daemon"]
FPROT["M_FPROTD<br>F-Prot Daemon"]
DRWEB["M_DRWEB<br>Dr.Web"]
AVAST["M_AVAST<br>Avast"]
CMDLINE["M_CMDL<br>Command Line"]
SOCK["M_SOCK<br>Generic Socket"]
TCP["MC_TCP<br>TCP Socket"]
UNIX["MC_UNIX<br>Unix Socket"]
STREAM["MC_STRM<br>Stream Socket"]
MALWARE_FUNC["malware()"]
MALWARE_INTERNAL["malware_internal()"]
SPOOL_MBOX["spool_mbox()"]
MBOX_FILE["MBOX Spool File"]
SCANNER_COMM["Scanner Communication"]
RESULT_PARSE["Result Parsing"]

SPOOL_MBOX --> MBOX_FILE
SCANNER_COMM --> TCP
SCANNER_COMM --> UNIX
SCANNER_COMM --> STREAM
TCP --> CLAMD
TCP --> FPROT
UNIX --> DRWEB
STREAM --> AVAST

subgraph subGraph3 ["Message Processing"]
    MBOX_FILE
    SCANNER_COMM
    RESULT_PARSE
    MBOX_FILE --> SCANNER_COMM
    SCANNER_COMM --> RESULT_PARSE
end

subgraph subGraph2 ["Core Functions"]
    MALWARE_FUNC
    MALWARE_INTERNAL
    SPOOL_MBOX
    MALWARE_FUNC --> MALWARE_INTERNAL
    MALWARE_INTERNAL --> SPOOL_MBOX
end

subgraph subGraph1 ["Connection Types"]
    TCP
    UNIX
    STREAM
end

subgraph subGraph0 ["Scanner Types"]
    CLAMD
    FPROT
    DRWEB
    AVAST
    CMDLINE
    SOCK
end
```

The malware scanning system creates MBOX-format spool files for scanner consumption and supports various communication protocols. Scanner selection includes priority and weight-based load balancing.

Sources: [src/src/malware.c L57-L102](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L57-L102)

 [src/src/malware.c L573-L700](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L573-L700)

 [src/src/spool_mbox.c L32-L180](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c#L32-L180)

### Spam Detection Integration

```mermaid
flowchart TD

SPAM_FUNC["spam()"]
SPAMD_CONN["SpamAssassin/Rspamd Connection"]
SERVER_SELECT["spamd_get_server()"]
SPAMD_SERVERS["spamd_address_container[]"]
WEIGHT_CALC["Weight-based Selection"]
FAILOVER["Automatic Failover"]
SPAMC_PROTO["SPAMC Protocol"]
RSPAMD_PROTO["Rspamd Protocol"]
RESULT_PARSE["Score/Action Parsing"]
SPAM_SCORE["$spam_score"]
SPAM_ACTION["$spam_action"]
SPAM_REPORT["$spam_report"]

SERVER_SELECT --> SPAMD_SERVERS
SPAMD_CONN --> SPAMC_PROTO
SPAMD_CONN --> RSPAMD_PROTO
RESULT_PARSE --> SPAM_SCORE
RESULT_PARSE --> SPAM_ACTION
RESULT_PARSE --> SPAM_REPORT

subgraph Variables ["Variables"]
    SPAM_SCORE
    SPAM_ACTION
    SPAM_REPORT
end

subgraph Communication ["Communication"]
    SPAMC_PROTO
    RSPAMD_PROTO
    RESULT_PARSE
    SPAMC_PROTO --> RESULT_PARSE
    RSPAMD_PROTO --> RESULT_PARSE
end

subgraph subGraph1 ["Server Management"]
    SPAMD_SERVERS
    WEIGHT_CALC
    FAILOVER
    SPAMD_SERVERS --> WEIGHT_CALC
    WEIGHT_CALC --> FAILOVER
end

subgraph subGraph0 ["Spam Detection"]
    SPAM_FUNC
    SPAMD_CONN
    SERVER_SELECT
    SPAM_FUNC --> SERVER_SELECT
    SERVER_SELECT --> SPAMD_CONN
end
```

Spam detection supports both SpamAssassin and Rspamd with different protocol implementations. The system includes server prioritization, retry logic, and comprehensive result reporting.

Sources: [src/src/spam.c L177-L400](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L177-L400)

 [src/src/spam.c L32-L134](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L32-L134)

## MIME Processing and Security

Exim's MIME processing provides content analysis, decoding, and security validation for multipart messages. The system integrates with content scanning and supports various encoding schemes.

### MIME Processing Pipeline

```mermaid
flowchart TD

MIME_STREAM["mime_stream"]
HEADER_PARSE["mime_get_header()"]
BOUNDARY_DETECT["Boundary Detection"]
PART_EXTRACT["Part Extraction"]
QP_DECODE["mime_decode_qp()<br>Quoted-Printable"]
B64_DECODE["mime_decode_base64()<br>Base64"]
DECODE_ASIS["mime_decode_asis()<br>No Encoding"]
MIME_VARS["MIME Variables"]
CONTENT_TYPE["Content-Type Analysis"]
FILENAME_EXTRACT["Filename Extraction"]
SIZE_CALC["Content Size"]
ANOMALY_DETECT["mime_set_anomaly()"]
BROKEN_QP["Broken QP Detection"]
BROKEN_B64["Broken Base64 Detection"]

PART_EXTRACT --> QP_DECODE
PART_EXTRACT --> B64_DECODE
PART_EXTRACT --> DECODE_ASIS
QP_DECODE --> CONTENT_TYPE
B64_DECODE --> CONTENT_TYPE
DECODE_ASIS --> CONTENT_TYPE
QP_DECODE --> ANOMALY_DETECT
B64_DECODE --> ANOMALY_DETECT

subgraph subGraph3 ["Security Features"]
    ANOMALY_DETECT
    BROKEN_QP
    BROKEN_B64
    ANOMALY_DETECT --> BROKEN_QP
    ANOMALY_DETECT --> BROKEN_B64
end

subgraph subGraph2 ["Content Analysis"]
    MIME_VARS
    CONTENT_TYPE
    FILENAME_EXTRACT
    SIZE_CALC
    CONTENT_TYPE --> MIME_VARS
    CONTENT_TYPE --> FILENAME_EXTRACT
    CONTENT_TYPE --> SIZE_CALC
end

subgraph subGraph1 ["Content Decoding"]
    QP_DECODE
    B64_DECODE
    DECODE_ASIS
end

subgraph subGraph0 ["MIME Parsing"]
    MIME_STREAM
    HEADER_PARSE
    BOUNDARY_DETECT
    PART_EXTRACT
    MIME_STREAM --> HEADER_PARSE
    HEADER_PARSE --> BOUNDARY_DETECT
    BOUNDARY_DETECT --> PART_EXTRACT
end
```

MIME processing includes robust error handling for malformed content and provides detailed metadata through expansion variables. The system supports nested multipart structures and various encoding validation.

Sources: [src/src/mime.c L220-L301](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L220-L301)

 [src/src/mime.c L48-L105](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L48-L105)

 [src/src/mime.c L304-L400](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L304-L400)

## Network Security Features

Exim implements various network-level security features including connection management, IP filtering, and protocol security enhancements.

### Connection Security Architecture

```mermaid
flowchart TD

IP_SOCKET["ip_socket()"]
IP_BIND["ip_bind()"]
IP_CONNECT["ip_connect()"]
SMTP_CONNECT["smtp_connect()"]
TFO_SUPPORT["TCP Fast Open"]
DSCP_MARKING["DSCP Traffic Marking"]
PIPELINING["SMTP Pipelining"]
CHUNKING["BDAT/CHUNKING"]
EARLY_DATA["TLS Early Data"]
INTERFACE_BIND["Interface Binding"]
PORT_SELECTION["Port Selection"]
TIMEOUT_MGMT["Timeout Management"]
SECURITY_CONTROLS["SECURITY_CONTROLS"]

IP_SOCKET --> SMTP_CONNECT
IP_BIND --> INTERFACE_BIND
IP_CONNECT --> TFO_SUPPORT
SMTP_CONNECT --> PIPELINING
SMTP_CONNECT --> CHUNKING
TFO_SUPPORT --> EARLY_DATA
INTERFACE_BIND --> SECURITY_CONTROLS
PORT_SELECTION --> SECURITY_CONTROLS
TIMEOUT_MGMT --> SECURITY_CONTROLS

subgraph subGraph3 ["Security Controls"]
    INTERFACE_BIND
    PORT_SELECTION
    TIMEOUT_MGMT
end

subgraph subGraph2 ["Protocol Features"]
    PIPELINING
    CHUNKING
    EARLY_DATA
end

subgraph subGraph1 ["SMTP Security"]
    SMTP_CONNECT
    TFO_SUPPORT
    DSCP_MARKING
    SMTP_CONNECT --> DSCP_MARKING
end

subgraph subGraph0 ["Network Layer"]
    IP_SOCKET
    IP_BIND
    IP_CONNECT
end
```

Network security includes source IP binding, DSCP traffic classification, and TCP Fast Open support for improved performance with security considerations.

Sources: [src/src/smtp_out.c L276-L334](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_out.c#L276-L334)

 [src/src/ip.c L193-L400](https://github.com/Exim/exim/blob/29568b25/src/src/ip.c#L193-L400)

 [src/src/smtp_out.c L349-L530](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_out.c#L349-L530)

### Content Filtering Integration

| Scanner Type | Protocol | Configuration | Features |
| --- | --- | --- | --- |
| `M_CLAMD` | TCP/Unix Socket | `av_scanner = clamd:/tmp/clamd` | Multiple servers, retry logic |
| `M_FPROTD` | TCP | `av_scanner = f-protd:localhost:10200` | HTTP-based protocol |
| `M_DRWEB` | Unix Socket | `av_scanner = drweb:/var/run/drweb.sock` | Binary protocol |
| `M_AVAST` | Stream Socket | `av_scanner = avast:/var/run/avast/scan.sock` | Stream-based scanning |
| `M_SOCK` | Generic Socket | `av_scanner = sock:/tmp/scanner.sock` | Custom protocol support |

The content filtering system provides unified access to multiple scanner backends through a common interface while preserving scanner-specific optimizations and features.

Sources: [src/src/malware.c L57-L102](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L57-L102)
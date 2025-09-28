# Security Features

> **Relevant source files**
> * [FAQ](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ)
> * [README.TLS](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.TLS)
> * [README.Virtual-Users](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users)
> * [src/tls.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.c)
> * [src/tls.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.h)

This document covers Pure-FTPd's comprehensive security architecture, including encryption, authentication mechanisms, access controls, and process isolation features. Pure-FTPd implements multiple layers of security to protect FTP communications and prevent unauthorized access.

For information about specific authentication backends, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management). For TLS configuration details, see [TLS/SSL Encryption](/jedisct1/pure-ftpd/3.1-tlsssl-encryption).

## Overview of Security Architecture

Pure-FTPd employs a multi-layered security approach combining transport encryption, authentication verification, access control, and process isolation:

```mermaid
flowchart TD

TLS["TLS/SSL Encryption<br>tls.c, tls.h"]
SNI["SNI Support<br>ssl_servername_cb()"]
CERT["Certificate Management<br>tls_load_cert_file()"]
HASH["Password Hashing<br>argon2, scrypt, bcrypt"]
VUSER["Virtual Users<br>pureftpd.pdb"]
MULTI["Multi-Backend Auth<br>MySQL, LDAP, PAM"]
IP["IP Restrictions<br>allow/deny lists"]
TIME["Time Restrictions<br>hhmm-hhmm ranges"]
QUOTA["Quota Enforcement<br>file/size limits"]
CHROOT["Chroot Jailing<br>-A, -a options"]
PRIVSEP["Privilege Separation<br>privsep.c"]
LIMITS["Connection Limits<br>per-IP, per-user"]
CLIENT["FTP Client"]

CLIENT --> TLS
TLS --> HASH
HASH --> IP
TIME --> CHROOT

subgraph subGraph3 ["Process Security"]
    CHROOT
    PRIVSEP
    LIMITS
    CHROOT --> PRIVSEP
end

subgraph subGraph2 ["Access Control"]
    IP
    TIME
    QUOTA
    IP --> TIME
end

subgraph subGraph1 ["Authentication Security"]
    HASH
    VUSER
    MULTI
    HASH --> VUSER
    VUSER --> MULTI
end

subgraph subGraph0 ["Transport Security"]
    TLS
    SNI
    CERT
    TLS --> CERT
    TLS --> SNI
end
```

Sources: [src/tls.c L1-L608](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.c#L1-L608)

 [src/tls.h L1-L53](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.h#L1-L53)

 [README.TLS L1-L313](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.TLS#L1-L313)

 [README.Virtual-Users L1-L318](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L1-L318)

## TLS/SSL Encryption System

Pure-FTPd provides comprehensive TLS support for encrypting both control and data connections:

```mermaid
flowchart TD

CIPHER["SSL cipher configuration"]
DHPARAM["tls_init_dhparams()"]
ECDH["tls_init_ecdh_curve()"]
INIT["tls_init_library()"]
CTX["tls_create_new_context()"]
OPTS["tls_init_options()"]
NEWSESS["tls_init_new_session()"]
DATASESS["tls_init_data_session()"]
CLOSE["tls_close_session()"]
LOAD["tls_load_cert_file()"]
VERIFY["tls_init_client_cert_verification()"]
SNI_CB["ssl_servername_cb()"]
tls_cnx["tls_cnx SSL connection"]
tls_data_cnx["tls_data_cnx SSL connection"]
EXTCERT["External Certificate Handler"]

OPTS --> LOAD
NEWSESS --> tls_cnx
DATASESS --> tls_data_cnx
SNI_CB --> EXTCERT

subgraph subGraph2 ["Certificate Handling"]
    LOAD
    VERIFY
    SNI_CB
    LOAD --> VERIFY
end

subgraph subGraph1 ["Session Management"]
    NEWSESS
    DATASESS
    CLOSE
end

subgraph subGraph0 ["TLS Initialization"]
    INIT
    CTX
    OPTS
    INIT --> CTX
    CTX --> OPTS
end

subgraph subGraph3 ["Security Configuration"]
    CIPHER
    DHPARAM
    ECDH
end
```

### TLS Security Features

| Feature | Implementation | Configuration |
| --- | --- | --- |
| **Protocol Support** | TLS 1.2, TLS 1.3 | `--tls=0/1/2/3` |
| **Cipher Selection** | Server preference enforced | `--tlsciphersuite` |
| **Certificate Verification** | Client cert validation | `-C` prefix in cipher suite |
| **SNI Support** | Dynamic certificate selection | `ssl_servername_cb()` |
| **Session Caching** | SSL session reuse | `tls_init_cache()` |
| **Perfect Forward Secrecy** | ECDHE/DHE key exchange | `tls_init_ecdh_curve()` |

### TLS Security Configuration

The TLS implementation enforces strong security defaults:

* **Disabled Protocols**: SSLv2, SSLv3, TLS 1.0, TLS 1.1 are disabled by default
* **Minimum Cipher Strength**: 128-bit encryption minimum (`MINIMAL_CIPHER_STRENGTH_BITS`)
* **Certificate Depth**: Maximum 6 certificates in chain (`MAX_CERTIFICATE_DEPTH`)
* **Session Security**: No session resumption on renegotiation

Sources: [src/tls.c L328-L370](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.c#L328-L370)

 [src/tls.c L437-L460](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.c#L437-L460)

 [src/tls.h L47-L49](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.h#L47-L49)

 [README.TLS L28-L131](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.TLS#L28-L131)

## Authentication Security Framework

Pure-FTPd implements secure authentication with multiple backend support and strong password protection:

```mermaid
flowchart TD

HASH_ALG["Hash Algorithm Priority<br>1. argon2<br>2. scrypt<br>3. bcrypt<br>4. SHA-512<br>5. MD5"]
MEMORY["Memory-hard Functions<br>64MB per login attempt"]
CPU_COST["CPU Cost<br>100% CPU core usage"]
PASSWD_FILE["pureftpd.passwd<br>Text format"]
PDB_FILE["pureftpd.pdb<br>Indexed binary format"]
PURE_PW["pure-pw command<br>User management"]
PUREDB["PureDB Backend<br>-lpuredb:path"]
MYSQL["MySQL Backend<br>-lmysql:config"]
LDAP["LDAP Backend<br>-lldap:config"]
PAM["PAM Backend<br>-lpam"]
AUTH_RESULT["Authentication Result"]

HASH_ALG --> PASSWD_FILE
PDB_FILE --> PUREDB
PUREDB --> AUTH_RESULT
MYSQL --> AUTH_RESULT
LDAP --> AUTH_RESULT
PAM --> AUTH_RESULT

subgraph subGraph2 ["Authentication Backends"]
    PUREDB
    MYSQL
    LDAP
    PAM
end

subgraph subGraph1 ["Virtual User System"]
    PASSWD_FILE
    PDB_FILE
    PURE_PW
    PASSWD_FILE --> PDB_FILE
    PURE_PW --> PASSWD_FILE
end

subgraph subGraph0 ["Password Security"]
    HASH_ALG
    MEMORY
    CPU_COST
end
```

### Password Security Implementation

Virtual users employ state-of-the-art password hashing:

| Hash Function | Security Level | Memory Usage | CPU Cost |
| --- | --- | --- | --- |
| **Argon2** | Highest (recommended) | Up to 64MB | High |
| **scrypt** | High (recommended) | Up to 64MB | High |
| **bcrypt** | Good | Low | Medium |
| **SHA-512** | Legacy (deprecated) | Low | Low |
| **MD5** | Weak (deprecated) | Low | Low |

The password hashing priority ensures the strongest available algorithm is used based on system capabilities and libsodium availability.

Sources: [README.Virtual-Users L113-L124](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L113-L124)

 [README.Virtual-Users L59-L126](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L59-L126)

## Access Control and Restrictions

Pure-FTPd provides granular access control mechanisms:

```mermaid
flowchart TD

ALLOW_CLIENT["Allowed Client IPs<br>-r ip/mask"]
DENY_CLIENT["Denied Client IPs<br>-R ip/mask"]
ALLOW_LOCAL["Allowed Local IPs<br>-i ip/mask"]
DENY_LOCAL["Denied Local IPs<br>-I ip/mask"]
TIME_RANGE["Time Restrictions<br>-z hhmm-hhmm"]
ACTIVE_SESSION["Active Session<br>Can continue past time limit"]
MAX_CLIENTS["Max Clients per IP<br>--maxclientsperip"]
MAX_SESSIONS["Max Concurrent Sessions<br>-y option per user"]
GLOBAL_LIMIT["Global Connection Limit<br>-c option"]
FILE_QUOTA["File Count Quota<br>-n max files"]
SIZE_QUOTA["Size Quota<br>-N max MB"]
BANDWIDTH["Bandwidth Limits<br>-t/-T up/down"]
CLIENT_CONNECT["Client Connection"]
ACCESS_GRANTED["Access Granted"]

CLIENT_CONNECT --> ALLOW_CLIENT
DENY_CLIENT --> TIME_RANGE
TIME_RANGE --> MAX_CLIENTS
MAX_SESSIONS --> FILE_QUOTA
BANDWIDTH --> ACCESS_GRANTED

subgraph subGraph3 ["Resource Quotas"]
    FILE_QUOTA
    SIZE_QUOTA
    BANDWIDTH
    FILE_QUOTA --> SIZE_QUOTA
    SIZE_QUOTA --> BANDWIDTH
end

subgraph subGraph2 ["Connection Limits"]
    MAX_CLIENTS
    MAX_SESSIONS
    GLOBAL_LIMIT
    MAX_CLIENTS --> MAX_SESSIONS
end

subgraph subGraph1 ["Time-based Restrictions"]
    TIME_RANGE
    ACTIVE_SESSION
end

subgraph subGraph0 ["IP-based Restrictions"]
    ALLOW_CLIENT
    DENY_CLIENT
    ALLOW_LOCAL
    DENY_LOCAL
    ALLOW_CLIENT --> DENY_CLIENT
end
```

### Access Control Configuration

| Control Type | Virtual User Field | Command Line Option | Description |
| --- | --- | --- | --- |
| **Client IP Allow** | Field 15 | `-r ip/mask` | Allowed source IPs |
| **Client IP Deny** | Field 16 | `-R ip/mask` | Denied source IPs |
| **Local IP Allow** | Field 13 | `-i ip/mask` | Allowed local IPs |
| **Local IP Deny** | Field 14 | `-I ip/mask` | Denied local IPs |
| **Time Restrictions** | Field 17 | `-z hhmm-hhmm` | Allowed time range |
| **Session Limit** | Field 10 | `-y count` | Max concurrent sessions |

Sources: [README.Virtual-Users L44-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L44-L50)

 [README.Virtual-Users L90-L103](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L90-L103)

## Process Security and Isolation

Pure-FTPd implements multiple process security mechanisms:

```mermaid
flowchart TD

CHROOT_ALL["Chroot Everyone<br>-A option"]
CHROOT_EXCEPT["Chroot Except Group<br>-a gid option"]
VIRTUAL_CHROOT["Virtual Chroot<br>--with-virtualchroot"]
DOT_SLASH["Per-user Chroot<br>/./ in home path"]
ROOT_START["Start as Root<br>Bind privileged ports"]
DROP_PRIVS["Drop Privileges<br>Switch to user context"]
MIN_UID["Minimum UID<br>-u option"]
TRUST_CHECK["Trust Verification<br>Shell in /etc/shells"]
DOT_FILES_WRITE["Dot Files Write Protection<br>-x option"]
DOT_FILES_READ["Dot Files Read Protection<br>-X option"]
UMASK["File Permissions<br>-m umask"]
CUSTOMER_PROOF["Customer Proof Mode<br>--customerproof"]

ROOT_START --> CHROOT_ALL
CHROOT_ALL --> DROP_PRIVS
TRUST_CHECK --> DOT_FILES_WRITE

subgraph subGraph2 ["File System Security"]
    DOT_FILES_WRITE
    DOT_FILES_READ
    UMASK
    CUSTOMER_PROOF
    DOT_FILES_WRITE --> DOT_FILES_READ
    DOT_FILES_READ --> CUSTOMER_PROOF
end

subgraph subGraph1 ["Privilege Management"]
    ROOT_START
    DROP_PRIVS
    MIN_UID
    TRUST_CHECK
    DROP_PRIVS --> MIN_UID
    MIN_UID --> TRUST_CHECK
end

subgraph subGraph0 ["Chroot Implementation"]
    CHROOT_ALL
    CHROOT_EXCEPT
    VIRTUAL_CHROOT
    DOT_SLASH
end
```

### Chroot Security Models

Pure-FTPd offers two chroot implementations:

| Model | Symlink Behavior | Use Case | Compilation Flag |
| --- | --- | --- | --- |
| **Traditional Chroot** | Restricted to jail | High security | Default |
| **Virtual Chroot** | Can follow outside links | Shared directories | `--with-virtualchroot` |

### Security Enforcement Options

| Security Feature | Option | Description |
| --- | --- | --- |
| **Dot File Write Protection** | `-x` | Prevent modification of hidden files |
| **Dot File Read Protection** | `-X` | Prevent access to hidden files |
| **Customer Proof Mode** | `--customerproof` | Enhanced security for hosting |
| **Minimum UID** | `-u uid` | Reject logins below UID threshold |
| **Broken Client Support** | `-b` | Compatibility with security implications |

Sources: [FAQ L40-L51](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L40-L51)

 [FAQ L99-L113](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L99-L113)

 [FAQ L644-L669](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L644-L669)

 [README.Virtual-Users L82-L84](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Virtual-Users#L82-L84)

## Connection Security Features

Pure-FTPd implements comprehensive connection security:

```mermaid
flowchart TD

DNS_CHECK["DNS Resolution<br>-H to disable"]
REVERSE_DNS["Reverse DNS Lookup<br>Client verification"]
HOST_VALIDATION["Hostname Validation<br>Certificate matching"]
IDLE_TIMEOUT["Idle Timeout<br>Connection cleanup"]
MAX_LOGIN_ATTEMPTS["Login Attempt Limits<br>Brute force protection"]
SESSION_TRACKING["Session Tracking<br>pure-ftpwho monitoring"]
PASSIVE_PORTS["Passive Port Range<br>-p low:high"]
FIREWALL_COMPAT["Firewall Compatibility<br>Fixed port ranges"]
DATA_ENCRYPTION["Data Channel TLS<br>Encrypted transfers"]
CLIENT_CONN["Client Connection"]

CLIENT_CONN --> DNS_CHECK
REVERSE_DNS --> MAX_LOGIN_ATTEMPTS
IDLE_TIMEOUT --> PASSIVE_PORTS

subgraph subGraph2 ["Data Transfer Security"]
    PASSIVE_PORTS
    FIREWALL_COMPAT
    DATA_ENCRYPTION
    PASSIVE_PORTS --> DATA_ENCRYPTION
end

subgraph subGraph1 ["Session Management"]
    IDLE_TIMEOUT
    MAX_LOGIN_ATTEMPTS
    SESSION_TRACKING
    MAX_LOGIN_ATTEMPTS --> IDLE_TIMEOUT
end

subgraph subGraph0 ["Connection Validation"]
    DNS_CHECK
    REVERSE_DNS
    HOST_VALIDATION
    DNS_CHECK --> REVERSE_DNS
end
```

### Security Monitoring and Logging

| Feature | Implementation | Security Benefit |
| --- | --- | --- |
| **Session Monitoring** | `pure-ftpwho` | Real-time connection tracking |
| **Alternative Logging** | `altlog.c` | Comprehensive audit trails |
| **Syslog Integration** | Facility-based logging | Centralized security monitoring |
| **Upload Notifications** | `pure-uploadscript` | File transfer alerting |

Sources: [FAQ L164-L195](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L164-L195)

 [FAQ L630-L641](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L630-L641)

 [FAQ L832-L847](https://github.com/jedisct1/pure-ftpd/blob/3818577a/FAQ#L832-L847)

 [src/tls.c L518-L561](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/tls.c#L518-L561)
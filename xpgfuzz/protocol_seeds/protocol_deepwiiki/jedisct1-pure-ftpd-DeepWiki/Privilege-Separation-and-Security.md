# Privilege Separation and Security

> **Relevant source files**
> * [src/altlog.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c)
> * [src/log_puredb.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c)
> * [src/privsep.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c)
> * [src/privsep_p.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep_p.h)
> * [src/pure-pw.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c)
> * [src/pure-quotacheck.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c)
> * [src/upload-pipe.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c)

This document covers Pure-FTPd's privilege separation architecture and security mechanisms. It focuses on process isolation, privilege dropping, secure inter-process communication, and other security features that protect the FTP server from privilege escalation and unauthorized access.

For information about TLS/SSL encryption, see [TLS/SSL Encryption](/jedisct1/pure-ftpd/3.1-tlsssl-encryption). For authentication methods and user management, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management).

## Privilege Separation Architecture

Pure-FTPd implements a privilege separation model where the main FTP server process runs with minimal privileges, while a separate privileged helper process handles operations that require elevated permissions.

### Process Structure

```mermaid
flowchart TD

BIND_PORT["Bind Reserved Ports"]
MAIN["pure-ftpd Main Process<br>(Unprivileged)"]
PRIV["pure-ftpd (PRIV)<br>(Privileged Helper)"]
CMD_PROC["Command Processing"]
FILE_OPS["File Operations"]
AUTH["Authentication"]
FTPWHO_RM["Remove FTP Who Entries"]
PRIV_FILES["Privileged File Access"]

subgraph subGraph2 ["Privilege Separation Model"]
    MAIN
    PRIV
    MAIN --> PRIV
    MAIN --> CMD_PROC
    MAIN --> FILE_OPS
    MAIN --> AUTH
    PRIV --> BIND_PORT
    PRIV --> FTPWHO_RM
    PRIV --> PRIV_FILES

subgraph subGraph1 ["Privileged Operations"]
    BIND_PORT
    FTPWHO_RM
    PRIV_FILES
end

subgraph subGraph0 ["Main Process Operations"]
    CMD_PROC
    FILE_OPS
    AUTH
end
end
```

Sources: [src/privsep.c L335-L364](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L335-L364)

The privilege separation is initialized through `privsep_init()`, which creates a Unix domain socket pair and forks into two processes:

* **Main Process**: Handles FTP protocol commands, file transfers, and user interactions
* **Privileged Helper**: Performs operations requiring root privileges like binding to reserved ports

### Initialization Sequence

```mermaid
sequenceDiagram
  participant Main Process
  participant privsep_init()
  participant Privileged Process

  Main Process->>privsep_init(): Initialize privilege separation
  privsep_init()->>privsep_init(): Create Unix socket pair
  privsep_init()->>privsep_init(): fork()
  loop [Child Process (Privileged)]
    privsep_init()->>Privileged Process: Set process name "pure-ftpd (PRIV)"
    Privileged Process->>Privileged Process: Close unnecessary descriptors
    Privileged Process->>Privileged Process: Initialize privsep user account
    Privileged Process->>Privileged Process: Drop to privsep_uid
    Privileged Process->>Privileged Process: Enter main loop
    privsep_init()-->>Main Process: Return success
    Main Process->>Main Process: Continue with normal operation
  end
```

Sources: [src/privsep.c L335-L364](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L335-L364)

 [src/privsep.c L268-L278](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L268-L278)

## Inter-Process Communication Protocol

The two processes communicate through a well-defined protocol using Unix domain sockets with support for file descriptor passing.

### Command Structure

The communication protocol uses structured messages defined in `PrivSepQuery` and `PrivSepAnswer` unions:

| Command | Purpose | Privilege Required |
| --- | --- | --- |
| `PRIVSEPCMD_BINDRESPORT` | Bind to reserved port | Root |
| `PRIVSEPCMD_REMOVEFTPWHOENTRY` | Remove FTP who entry | File system access |
| `PRIVSEPCMD_ANSWER_FD` | File descriptor response | - |

Sources: [src/privsep_p.h L22-L30](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep_p.h#L22-L30)

 [src/privsep_p.h L48-L54](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep_p.h#L48-L54)

### File Descriptor Passing

```mermaid
flowchart TD

REQ["Request Reserved Port"]
SEND["privsep_sendcmd()"]
RECV_FD["privsep_recvfd()"]
BIND["Bind to Port"]
SEND_FD["privsep_sendfd()"]

REQ --> SEND
SEND --> BIND
SEND_FD --> RECV_FD

subgraph subGraph2 ["Privileged Process"]
    BIND
    SEND_FD
    BIND --> SEND_FD
end

subgraph subGraph1 ["Socket Communication"]
    SEND
    RECV_FD
end

subgraph subGraph0 ["Main Process"]
    REQ
end
```

Sources: [src/privsep.c L41-L88](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L41-L88)

 [src/privsep.c L90-L140](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L90-L140)

The `privsep_sendfd()` and `privsep_recvfd()` functions use `SCM_RIGHTS` control messages to safely pass file descriptors between processes, enabling the privileged process to create sockets and pass them to the unprivileged main process.

## Security Mechanisms

### User Account Isolation

Pure-FTPd creates a dedicated system user account for privilege separation:

```mermaid
flowchart TD

USERS["Preferred Users<br>PRIVSEP_USER<br>pure-ftpd"]
LOOKUP["getpwnam() lookup"]
ACCOUNT["Selected Account"]
SETGROUPS["setgroups()"]
SETGID["setgid()"]
SETEUID["seteuid()"]

ACCOUNT --> SETGROUPS

subgraph subGraph1 ["Privilege Dropping"]
    SETGROUPS
    SETGID
    SETEUID
    SETGROUPS --> SETGID
    SETGID --> SETEUID
end

subgraph subGraph0 ["User Account Resolution"]
    USERS
    LOOKUP
    ACCOUNT
    USERS --> LOOKUP
    LOOKUP --> ACCOUNT
end
```

Sources: [src/privsep.c L308-L333](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L308-L333)

 [src/privsep_p.h L18-L20](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep_p.h#L18-L20)

### Process State Management

The privileged process alternates between privileged and unprivileged states:

* `privsep_priv_user()`: Elevates to root for privileged operations
* `privsep_unpriv_user()`: Drops back to `privsep_uid` for safety

Sources: [src/privsep.c L142-L154](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L142-L154)

### Resource Cleanup

The privileged process closes unnecessary file descriptors to reduce attack surface:

```
// Closes upload pipes, log files, and standard streams
privsep_privpart_closejunk()
```

Sources: [src/privsep.c L280-L306](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L280-L306)

## Authentication Security

### Password Hashing

Pure-FTPd supports modern password hashing algorithms through libsodium and fallback methods:

```mermaid
flowchart TD

LIBSODIUM["libsodium<br>crypto_pwhash_str()"]
SCRYPT["scryptsalsa208sha256"]
BCRYPT["bcrypt ($2a$)"]
SHA512["SHA-512 ($6$)"]
MD5["MD5 ($1$)"]
TIMING["Timing Attack Protection"]
MEMORY["Memory Cost Control"]
CONCURRENT["Concurrent Login Limits"]

LIBSODIUM --> TIMING
LIBSODIUM --> MEMORY
LIBSODIUM --> CONCURRENT

subgraph subGraph1 ["Security Features"]
    TIMING
    MEMORY
    CONCURRENT
end

subgraph subGraph0 ["Password Hashing Priority"]
    LIBSODIUM
    SCRYPT
    BCRYPT
    SHA512
    MD5
    LIBSODIUM --> SCRYPT
    SCRYPT --> BCRYPT
    BCRYPT --> SHA512
    SHA512 --> MD5
end
```

Sources: [src/pure-pw.c L269-L374](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L269-L374)

The `best_crypt()` function implements adaptive password hashing with:

* Time-based calibration to maintain consistent authentication time
* Memory usage limits based on concurrent user limits
* Protection against timing attacks

### Authentication Flow Security

```mermaid
sequenceDiagram
  participant FTP Client
  participant Main Process
  participant PureDB Backend
  participant Password Verification

  FTP Client->>Main Process: USER/PASS commands
  Main Process->>PureDB Backend: pw_puredb_check()
  PureDB Backend->>PureDB Backend: puredb_find_s()
  PureDB Backend->>PureDB Backend: Parse user record
  PureDB Backend->>Password Verification: Verify password hash
  loop [Modern Hash
    Password Verification->>Password Verification: crypto_pwhash_str_verify()
    Password Verification->>Password Verification: crypt() + timing-safe compare
  end
  Password Verification-->>Main Process: Authentication result
  Main Process-->>FTP Client: Login response
```

Sources: [src/log_puredb.c L369-L400](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L369-L400)

 [src/log_puredb.c L215-L367](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L215-L367)

## File System Security

### Quota Enforcement

The quota checking system implements multiple security measures:

```mermaid
flowchart TD

INODE_TRACK["Inode/Device Tracking"]
LOOP_DETECT["Loop Detection"]
RACE_PREVENT["Race Condition Prevention"]
ROOT_CHECK["Root Privilege Check"]
PATH_VALIDATION["Path Validation"]
OWNERSHIP["File Ownership Verification"]
PERMISSIONS["Permission Enforcement"]
CHROOT["chroot() to user directory"]
UID_DROP["Drop to user UID/GID"]
TRAVERSE["Safe directory traversal"]

ROOT_CHECK --> CHROOT
PATH_VALIDATION --> UID_DROP
OWNERSHIP --> TRAVERSE
PERMISSIONS --> TRAVERSE

subgraph Operations ["Operations"]
    CHROOT
    UID_DROP
    TRAVERSE
end

subgraph subGraph0 ["Security Checks"]
    ROOT_CHECK
    PATH_VALIDATION
    OWNERSHIP
    PERMISSIONS
end

subgraph subGraph1 ["Traversal Protection"]
    INODE_TRACK
    LOOP_DETECT
    RACE_PREVENT
    INODE_TRACK --> LOOP_DETECT
    LOOP_DETECT --> RACE_PREVENT
end
```

Sources: [src/pure-quotacheck.c L68-L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L68-L166)

 [src/pure-quotacheck.c L212-L219](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L212-L219)

### Access Control

IP-based access control is implemented with CIDR support:

```mermaid
flowchart TD

CLIENT_IP["Client IP"]
MATCH["access_ip_match()"]
LOCAL_IP["Local IP"]
ALLOW_PATTERNS["Allow Patterns"]
DENY_PATTERNS["Deny Patterns"]
DECISION["Access Decision"]

subgraph subGraph0 ["IP Access Control"]
    CLIENT_IP
    MATCH
    LOCAL_IP
    ALLOW_PATTERNS
    DENY_PATTERNS
    DECISION
    CLIENT_IP --> MATCH
    LOCAL_IP --> MATCH
    ALLOW_PATTERNS --> MATCH
    DENY_PATTERNS --> MATCH
    MATCH --> DECISION
end
```

Sources: [src/log_puredb.c L74-L156](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L74-L156)

 [src/log_puredb.c L160-L185](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L160-L185)

## Upload Processing Security

The upload script system implements secure file handling:

```mermaid
flowchart TD

LOCK_FILE["Upload Lock File"]
PIPE_FILE["Upload Pipe (FIFO)"]
OWNERSHIP["File Ownership Check"]
PERMISSIONS["Permission Verification (0600)"]
STAT_CHECK["fstat() validation"]
LSTAT_CHECK["lstat() validation"]
OWNER_MATCH["Owner matching"]
MODE_CHECK["Mode verification"]

LOCK_FILE --> STAT_CHECK
PIPE_FILE --> STAT_CHECK

subgraph subGraph1 ["Security Validation"]
    STAT_CHECK
    LSTAT_CHECK
    OWNER_MATCH
    MODE_CHECK
    STAT_CHECK --> LSTAT_CHECK
    LSTAT_CHECK --> OWNER_MATCH
    OWNER_MATCH --> MODE_CHECK
end

subgraph subGraph0 ["Upload Pipe Security"]
    LOCK_FILE
    PIPE_FILE
    OWNERSHIP
    PERMISSIONS
end
```

Sources: [src/upload-pipe.c L14-L91](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c#L14-L91)

The upload pipe system prevents:

* Race conditions through file locking
* Unauthorized access through ownership verification
* Symlink attacks through `lstat()` validation
* Permission escalation through mode checking

## Logging Security

Alternative logging formats include security considerations:

### Secure Log Writing

```mermaid
flowchart TD

LOCK["File Locking"]
ATOMIC["Atomic Writes"]
ENCODING["Output Encoding"]
RACE["Race Conditions"]
INJECTION["Log Injection"]
CORRUPTION["Data Corruption"]

LOCK --> RACE
ENCODING --> INJECTION
ATOMIC --> CORRUPTION

subgraph subGraph1 ["Protection Against"]
    RACE
    INJECTION
    CORRUPTION
end

subgraph subGraph0 ["Log Security"]
    LOCK
    ATOMIC
    ENCODING
    LOCK --> ATOMIC
    ATOMIC --> ENCODING
end
```

Sources: [src/altlog.c L20-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L20-L50)

 [src/altlog.c L102-L152](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L102-L152)

The logging system uses:

* File locking to prevent concurrent write issues
* URL encoding to prevent log injection attacks
* Atomic write operations for data integrity

This comprehensive security model ensures that Pure-FTPd operates with minimal privileges while maintaining the ability to perform necessary privileged operations through a controlled, auditable interface.
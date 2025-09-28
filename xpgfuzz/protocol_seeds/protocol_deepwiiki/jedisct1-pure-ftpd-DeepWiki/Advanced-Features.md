# Advanced Features

> **Relevant source files**
> * [src/altlog.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c)
> * [src/ftpwho-read.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpwho-read.c)
> * [src/log_puredb.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c)
> * [src/mysnprintf.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/mysnprintf.c)
> * [src/privsep.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c)
> * [src/privsep_p.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep_p.h)
> * [src/pure-authd.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-authd.c)
> * [src/pure-certd.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-certd.c)
> * [src/pure-ftpwho.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-ftpwho.c)
> * [src/pure-pw.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c)
> * [src/pure-quotacheck.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c)
> * [src/pure-uploadscript.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-uploadscript.c)
> * [src/upload-pipe.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c)

This section covers Pure-FTPd's sophisticated features that extend beyond basic FTP operations, including automated post-upload processing, advanced user management, resource monitoring, and security enhancements. These features enable enterprise-grade deployments with fine-grained control over user access, resource usage, and operational monitoring.

For basic server configuration, see [Runtime Configuration](/jedisct1/pure-ftpd/5.2-runtime-configuration). For core authentication methods, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management). For privilege separation details, see [Privilege Separation and Security](/jedisct1/pure-ftpd/3.2-privilege-separation-and-security).

## Architecture Overview

Pure-FTPd's advanced features are built around a modular architecture that separates concerns while maintaining tight integration with the core server process. The system uses helper daemons, inter-process communication mechanisms, and pluggable processing pipelines to deliver enterprise functionality.

```mermaid
flowchart TD

FTPD["pure-ftpd main process<br>ftpd.c"]
GLOBALS["Global State<br>globals.h"]
UPLOADPIPE["Upload Notification<br>upload-pipe.c"]
UPLOADSCRIPT["pure-uploadscript daemon<br>pure-uploadscript.c"]
USERSCRIPT["User Script<br>External Process"]
PUREPW["pure-pw tool<br>pure-pw.c"]
AUTHD["pure-authd daemon<br>pure-authd.c"]
EXTAUTH["External Auth Script"]
QUOTACHECK["pure-quotacheck<br>pure-quotacheck.c"]
QUOTAFILE["Quota Files<br>.ftpquota"]
FTPWHO["pure-ftpwho<br>pure-ftpwho.c"]
FTPWHOREAD["Session Reader<br>ftpwho-read.c"]
ALTLOG["Alternative Logging<br>altlog.c"]
SCOREBOARD["Scoreboard Files<br>/var/run/pure-ftpd/"]
PRIVSEP["Privilege Separation<br>privsep.c"]
CERTD["Certificate Daemon<br>pure-certd.c"]

FTPD --> UPLOADPIPE
FTPD --> AUTHD
PUREPW --> FTPD
FTPD --> QUOTACHECK
FTPD --> FTPWHOREAD
FTPD --> ALTLOG
FTPD --> PRIVSEP
FTPD --> CERTD

subgraph subGraph5 ["Security Infrastructure"]
    PRIVSEP
    CERTD
end

subgraph subGraph4 ["Monitoring & Logging"]
    FTPWHO
    FTPWHOREAD
    ALTLOG
    SCOREBOARD
    FTPWHO --> SCOREBOARD
    FTPWHOREAD --> SCOREBOARD
end

subgraph subGraph3 ["Resource Management"]
    QUOTACHECK
    QUOTAFILE
    QUOTACHECK --> QUOTAFILE
end

subgraph subGraph2 ["Advanced Authentication"]
    PUREPW
    AUTHD
    EXTAUTH
    AUTHD --> EXTAUTH
end

subgraph subGraph1 ["Upload Processing Pipeline"]
    UPLOADPIPE
    UPLOADSCRIPT
    USERSCRIPT
    UPLOADPIPE --> UPLOADSCRIPT
    UPLOADSCRIPT --> USERSCRIPT
end

subgraph subGraph0 ["Core FTP Server Process"]
    FTPD
    GLOBALS
    FTPD --> GLOBALS
end
```

Sources: [src/pure-uploadscript.c L1-L502](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-uploadscript.c#L1-L502)

 [src/pure-pw.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L1-L50)

 [src/pure-quotacheck.c L1-L30](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L1-L30)

 [src/pure-ftpwho.c L1-L30](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-ftpwho.c#L1-L30)

 [src/pure-authd.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-authd.c#L1-L50)

 [src/privsep.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L1-L50)

 [src/altlog.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L1-L50)

## Upload Processing Pipeline

The upload processing system enables automated post-upload actions through a secure pipeline that notifies external scripts when files are uploaded. This system uses named pipes for communication and maintains security through privilege separation.

```mermaid
flowchart TD

CLIENT["FTP Client"]
STOR["dostor/doappe<br>File Upload"]
PIPE["upload_pipe_push<br>upload-pipe.c:95"]
FIFO["Named FIFO<br>/var/run/pure-ftpd/pipe"]
DAEMON["pure-uploadscript<br>Daemon Process"]
SCRIPT["User Script<br>External Process"]
LOCK["File Lock<br>UPLOAD_PIPE_LOCK"]
PRIVDROP["Privilege Drop<br>changeuidgid:287"]
ENV["Environment Setup<br>fillenv:355"]

PIPE --> FIFO
DAEMON --> PRIVDROP
ENV --> SCRIPT
PIPE --> LOCK

subgraph subGraph2 ["Security Context"]
    LOCK
    PRIVDROP
    ENV
    PRIVDROP --> ENV
end

subgraph subGraph1 ["Notification Pipeline"]
    FIFO
    DAEMON
    SCRIPT
    FIFO --> DAEMON
end

subgraph subGraph0 ["Upload Flow"]
    CLIENT
    STOR
    PIPE
    CLIENT --> STOR
    STOR --> PIPE
end
```

The upload notification process begins when a file upload completes. The main server process calls `upload_pipe_push()` which writes a structured message containing the username and file path to a named FIFO. The `pure-uploadscript` daemon reads from this FIFO and executes user-defined scripts with environment variables containing upload metadata.

**Key Implementation Details:**

* **Secure Communication**: Uses named pipes with strict permission checking [src/upload-pipe.c L47-L84](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c#L47-L84)
* **Message Format**: Binary protocol with username and file path [src/upload-pipe.c L95-L139](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c#L95-L139)
* **Process Isolation**: Scripts run in separate processes with dropped privileges [src/pure-uploadscript.c L393-L411](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-uploadscript.c#L393-L411)
* **Virtual Host Support**: File paths include virtual host prefixes [src/pure-uploadscript.c L128-L145](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-uploadscript.c#L128-L145)

Sources: [src/upload-pipe.c L14-L155](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c#L14-L155)

 [src/pure-uploadscript.c L75-L500](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-uploadscript.c#L75-L500)

## Advanced Password Hashing and User Management

Pure-FTPd implements sophisticated password hashing using modern cryptographic functions, with automatic parameter tuning based on system capabilities and security requirements.

```mermaid
flowchart TD

PUREPW["pure-pw useradd<br>Command Line"]
GETPWD["do_get_passwd<br>pure-pw.c:761"]
BESTCRYPT["best_crypt<br>pure-pw.c:269"]
LIBSODIUM["libsodium Detection<br>crypto_pwhash_str"]
BCRYPT["bcrypt Fallback<br>$2a$ format"]
SHA512["SHA-512 Fallback<br>$6$ format"]
MD5["MD5 Fallback<br>$1$ format"]
CONCURRENT["max_concurrent_logins<br>AUTH_CORES:72"]
MEMORY["max_auth_memory<br>DEFAULT_TOTAL_AUTH_MEMORY:52"]
TIMING["auth_time_ms<br>DEFAULT_AUTH_TIME_MS:63"]
PROBE["Timing Probe<br>gettimeofday:297"]
PWLINE["PWInfo Structure<br>add_new_pw_line:645"]
PUREDB["PureDB Database<br>mkdb command"]

BESTCRYPT --> LIBSODIUM
BESTCRYPT --> CONCURRENT
BESTCRYPT --> MEMORY
BESTCRYPT --> TIMING
BESTCRYPT --> PROBE
BESTCRYPT --> PWLINE

subgraph subGraph3 ["Database Storage"]
    PWLINE
    PUREDB
    PWLINE --> PUREDB
end

subgraph subGraph2 ["Parameter Tuning"]
    CONCURRENT
    MEMORY
    TIMING
    PROBE
end

subgraph subGraph1 ["Cryptographic Selection"]
    LIBSODIUM
    BCRYPT
    SHA512
    MD5
    LIBSODIUM --> BCRYPT
    BCRYPT --> SHA512
    SHA512 --> MD5
end

subgraph subGraph0 ["Password Creation Flow"]
    PUREPW
    GETPWD
    BESTCRYPT
    PUREPW --> GETPWD
    GETPWD --> BESTCRYPT
end
```

The password hashing system automatically selects the strongest available cryptographic function and tunes parameters based on system performance and security requirements. The `best_crypt()` function [src/pure-pw.c L269-L374](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L269-L374)

 implements a timing-based approach to determine optimal parameters for password hashing operations.

**Adaptive Security Features:**

* **Performance Tuning**: Automatically adjusts operations count based on timing measurements [src/pure-pw.c L297-L318](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L297-L318)
* **Memory Management**: Divides available memory across concurrent authentication attempts [src/pure-pw.c L284-L296](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L284-L296)
* **Fallback Chain**: Gracefully degrades from libsodium to traditional crypt() functions [src/pure-pw.c L327-L373](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L327-L373)
* **Constant-Time Operations**: Uses timing-safe comparison functions for security [src/log_puredb.c L244](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L244-L244)

Sources: [src/pure-pw.c L269-L374](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L269-L374)

 [src/pure-pw.c L761-L809](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-pw.c#L761-L809)

 [src/log_puredb.c L215-L400](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L215-L400)

## Resource Management and Monitoring

The quota system provides real-time disk usage tracking and enforcement through a combination of background scanning and cached quota files.

```mermaid
flowchart TD

QUOTACHECK["pure-quotacheck<br>main:272"]
TRAVERSE["traverse<br>pure-quotacheck.c:68"]
NODES["Node Tracking<br>Loop Prevention:31"]
QUOTAFILE[".ftpquota File<br>QUOTA_FILE"]
WRITEQUOTA["writequota<br>pure-quotacheck.c:221"]
FILELOCK["File Locking<br>F_WRLCK:250"]
QUOTACHECK_CALL["Quota Checks<br>Server Integration"]
LIMITS["User Limits<br>Files & Size"]
DENIAL["Upload Denial<br>Quota Exceeded"]

TRAVERSE --> WRITEQUOTA
QUOTAFILE --> QUOTACHECK_CALL

subgraph subGraph2 ["Runtime Integration"]
    QUOTACHECK_CALL
    LIMITS
    DENIAL
    QUOTACHECK_CALL --> LIMITS
    LIMITS --> DENIAL
end

subgraph subGraph1 ["Quota Enforcement"]
    QUOTAFILE
    WRITEQUOTA
    FILELOCK
    WRITEQUOTA --> FILELOCK
    FILELOCK --> QUOTAFILE
end

subgraph subGraph0 ["Quota Calculation"]
    QUOTACHECK
    TRAVERSE
    NODES
    QUOTACHECK --> TRAVERSE
    TRAVERSE --> NODES
end
```

The quota system uses a recursive directory traversal with cycle detection to calculate total disk usage. The `traverse()` function [src/pure-quotacheck.c L68-L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L68-L166)

 maintains a list of visited inodes to prevent infinite loops in case of symbolic links or bind mounts.

Sources: [src/pure-quotacheck.c L68-L270](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L68-L270)

 [src/pure-quotacheck.c L221-L270](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L221-L270)

## Session Monitoring Infrastructure

Pure-FTPd provides comprehensive session monitoring through a scoreboard system that tracks active connections and enables real-time status reporting.

```mermaid
flowchart TD

FTPWHO_UPDATE["ftpwho_update<br>Session State"]
SCOREBOARD_FILES["Scoreboard Files<br>SCOREBOARD_PATH"]
FTPWHO_ENTRY["FTPWhoEntry<br>Data Structure"]
FTPWHO_CMD["pure-ftpwho<br>Status Display"]
OUTPUT_FORMAT["Output Formats<br>Text/HTML/XML"]
FTPWHO_READ["ftpwho_read_count<br>User Counting"]
TEXT_OUTPUT["text_output_line<br>pure-ftpwho.c:124"]
HTML_OUTPUT["html_output_line<br>pure-ftpwho.c:338"]
XML_OUTPUT["xml_output_line<br>pure-ftpwho.c:414"]
CHECKPROC["checkproc<br>Process Validation:102"]
CLEANUP["scoreboard_cleanup<br>Dead Process Removal:18"]

FTPWHO_READ --> SCOREBOARD_FILES
OUTPUT_FORMAT --> TEXT_OUTPUT
OUTPUT_FORMAT --> HTML_OUTPUT
OUTPUT_FORMAT --> XML_OUTPUT
FTPWHO_READ --> CHECKPROC
CLEANUP --> SCOREBOARD_FILES

subgraph subGraph3 ["Process Management"]
    CHECKPROC
    CLEANUP
    CHECKPROC --> CLEANUP
end

subgraph subGraph2 ["Display Functions"]
    TEXT_OUTPUT
    HTML_OUTPUT
    XML_OUTPUT
end

subgraph subGraph1 ["Monitoring Tools"]
    FTPWHO_CMD
    OUTPUT_FORMAT
    FTPWHO_READ
    FTPWHO_CMD --> FTPWHO_READ
    FTPWHO_CMD --> OUTPUT_FORMAT
end

subgraph subGraph0 ["Session Tracking"]
    FTPWHO_UPDATE
    SCOREBOARD_FILES
    FTPWHO_ENTRY
    FTPWHO_UPDATE --> SCOREBOARD_FILES
    SCOREBOARD_FILES --> FTPWHO_ENTRY
end
```

The monitoring system maintains individual files for each active session in the scoreboard directory. The `pure-ftpwho` utility reads these files to provide real-time status information with multiple output formats suitable for different use cases.

Sources: [src/pure-ftpwho.c L102-L202](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-ftpwho.c#L102-L202)

 [src/ftpwho-read.c L13-L86](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpwho-read.c#L13-L86)

 [src/pure-ftpwho.c L338-L393](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-ftpwho.c#L338-L393)

## Alternative Logging System

Pure-FTPd supports multiple logging formats through a pluggable logging architecture that can generate logs compatible with various analysis tools.

```mermaid
flowchart TD

CLF["Common Log Format<br>altlog_writexfer_clf:156"]
XFERLOG["WuFTPd Format<br>altlog_writexfer_xferlog:236"]
W3C["W3C Extended Format<br>altlog_writexfer_w3c:314"]
STATS["Statistics Format<br>altlog_writexfer_stats:54"]
URLENCODE["urlencode<br>altlog.c:102"]
ALTLOG_WRITE["altlog_write<br>File Locking:20"]
FORMAT_SELECT["altlog_writexfer<br>Format Selection:412"]
TRANSFER_COMPLETE["Transfer Completion"]
ALTLOG_FD["altlog_fd<br>File Descriptor"]
SAFE_WRITE["safe_write<br>Thread-Safe I/O"]

TRANSFER_COMPLETE --> FORMAT_SELECT
FORMAT_SELECT --> CLF
FORMAT_SELECT --> XFERLOG
FORMAT_SELECT --> W3C
FORMAT_SELECT --> STATS
CLF --> URLENCODE
XFERLOG --> URLENCODE
W3C --> URLENCODE
CLF --> ALTLOG_WRITE
XFERLOG --> ALTLOG_WRITE
W3C --> ALTLOG_WRITE
STATS --> ALTLOG_WRITE
ALTLOG_WRITE --> ALTLOG_FD

subgraph subGraph2 ["Integration Points"]
    TRANSFER_COMPLETE
    ALTLOG_FD
    SAFE_WRITE
    ALTLOG_FD --> SAFE_WRITE
end

subgraph subGraph1 ["Processing Functions"]
    URLENCODE
    ALTLOG_WRITE
    FORMAT_SELECT
end

subgraph subGraph0 ["Log Formats"]
    CLF
    XFERLOG
    W3C
    STATS
end
```

The alternative logging system uses a dispatch mechanism to select the appropriate logging format and applies URL encoding for file names when necessary. Each format has specific timestamp, field ordering, and escaping requirements.

Sources: [src/altlog.c L54-L430](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L54-L430)

 [src/altlog.c L156-L232](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L156-L232)

 [src/altlog.c L236-L312](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L236-L312)

 [src/altlog.c L314-L366](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L314-L366)

## Security Infrastructure Integration

The advanced features integrate with Pure-FTPd's security infrastructure through privilege separation and secure inter-process communication mechanisms.

```mermaid
flowchart TD

PRIVSEP_INIT["privsep_init<br>privsep.c:335"]
SOCKETPAIR["socketpair<br>Communication Channel"]
PRIV_PROCESS["Privileged Process<br>privsep_privpart_main:268"]
UNPRIV_PROCESS["Unprivileged Process<br>Main FTP Server"]
SENDCMD["privsep_sendcmd<br>privsep.c:15"]
RECVCMD["privsep_recvcmd<br>privsep.c:28"]
SENDFD["privsep_sendfd<br>File Descriptor Passing:41"]
RECVFD["privsep_recvfd<br>privsep.c:90"]
BINDRESPORT["Privileged Port Binding<br>privsep_bindresport:234"]
FTPWHO_REMOVE["Scoreboard Cleanup<br>privsep_removeftpwhoentry:172"]

UNPRIV_PROCESS --> SENDCMD
RECVCMD --> PRIV_PROCESS
PRIV_PROCESS --> SENDFD
RECVFD --> UNPRIV_PROCESS
PRIV_PROCESS --> BINDRESPORT
PRIV_PROCESS --> FTPWHO_REMOVE

subgraph subGraph2 ["Protected Operations"]
    BINDRESPORT
    FTPWHO_REMOVE
end

subgraph subGraph1 ["IPC Operations"]
    SENDCMD
    RECVCMD
    SENDFD
    RECVFD
    SENDCMD --> RECVCMD
    SENDFD --> RECVFD
end

subgraph subGraph0 ["Privilege Separation"]
    PRIVSEP_INIT
    SOCKETPAIR
    PRIV_PROCESS
    UNPRIV_PROCESS
    PRIVSEP_INIT --> SOCKETPAIR
    SOCKETPAIR --> PRIV_PROCESS
    SOCKETPAIR --> UNPRIV_PROCESS
end
```

The privilege separation system isolates security-sensitive operations in a separate process that retains root privileges while the main FTP server runs with reduced privileges. Communication occurs through a Unix domain socket with structured message passing.

Sources: [src/privsep.c L335-L364](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L335-L364)

 [src/privsep.c L15-L140](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L15-L140)

 [src/privsep.c L187-L245](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L187-L245)

## Integration and Data Flow

The advanced features work together through well-defined interfaces and shared data structures, enabling complex workflows while maintaining system stability and security.

| Feature | Integration Point | Data Exchange | Security Model |
| --- | --- | --- | --- |
| Upload Scripts | `upload_pipe_push()` | Binary protocol over FIFO | Privilege dropping, chroot |
| Authentication | `pw_puredb_check()` | Database queries | Timing-safe comparisons |
| Quotas | File system traversal | `.ftpquota` files | User isolation |
| Monitoring | Scoreboard files | Structured binary data | Process validation |
| Logging | Transfer completion hooks | Formatted text output | File locking |
| Privilege Separation | Socket IPC | Command/response protocol | Process isolation |

This architecture enables administrators to deploy sophisticated FTP services with automated processing, comprehensive monitoring, and enterprise-grade security while maintaining the simplicity and reliability that Pure-FTPd is known for.

Sources: [src/upload-pipe.c L95-L139](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/upload-pipe.c#L95-L139)

 [src/log_puredb.c L369-L400](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/log_puredb.c#L369-L400)

 [src/pure-quotacheck.c L221-L270](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-quotacheck.c#L221-L270)

 [src/pure-ftpwho.c L33-L86](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/pure-ftpwho.c#L33-L86)

 [src/altlog.c L412-L430](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/altlog.c#L412-L430)

 [src/privsep.c L234-L245](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/privsep.c#L234-L245)
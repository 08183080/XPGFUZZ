# Architecture Overview

> **Relevant source files**
> * [src/ftp_parser.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c)
> * [src/ftpd.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c)
> * [src/ftpd.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h)
> * [src/ftpd_p.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h)
> * [src/globals.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h)
> * [src/main.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c)

This document provides a high-level view of Pure-FTPd's system architecture, focusing on the core server components, their interactions, and the main execution flow. It covers the primary modules, data structures, and control flow that form the foundation of the FTP server.

For detailed information about specific subsystems, see [Core Server Components](/jedisct1/pure-ftpd/2-core-server-components), [Security Features](/jedisct1/pure-ftpd/3-security-features), [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management), and [Configuration and Administration](/jedisct1/pure-ftpd/5-configuration-and-administration).

## System Overview

Pure-FTPd is implemented as a modular FTP server with a clear separation between core protocol handling, authentication, security, and file operations. The architecture is designed around a main server process that handles FTP protocol commands while delegating specific responsibilities to specialized subsystems.

```mermaid
flowchart TD

MAIN["main<br>main.c"]
PUREFTPD_START["pureftpd_start<br>ftpd.c"]
PARSER["parser<br>ftp_parser.c"]
SFGETS["sfgets<br>ftp_parser.c"]
COMMAND_HANDLERS["Command Handlers<br>douser, dopass, doretr, etc."]
GLOBALS["Global State<br>globals.h"]
CLIENT_IO["Client I/O<br>client_printf, doreply"]
STATE_TRACKING["State Tracking<br>loggedin, chrooted, etc."]
TLS_SUPPORT["TLS Support<br>tls_cnx, data_protection_level"]
PRIVSEP["Privilege Separation<br>privsep"]
SIGNAL_HANDLING["Signal Handling<br>sigalarm, sigterm"]

PUREFTPD_START --> PARSER
PARSER --> GLOBALS
COMMAND_HANDLERS --> CLIENT_IO
PARSER --> TLS_SUPPORT
PUREFTPD_START --> PRIVSEP
PUREFTPD_START --> SIGNAL_HANDLING

subgraph subGraph3 ["Security Layer"]
    TLS_SUPPORT
    PRIVSEP
    SIGNAL_HANDLING
end

subgraph subGraph2 ["Session Management"]
    GLOBALS
    CLIENT_IO
    STATE_TRACKING
    GLOBALS --> STATE_TRACKING
end

subgraph subGraph1 ["Core Protocol Engine"]
    PARSER
    SFGETS
    COMMAND_HANDLERS
    PARSER --> SFGETS
    PARSER --> COMMAND_HANDLERS
end

subgraph subGraph0 ["Application Entry"]
    MAIN
    PUREFTPD_START
    MAIN --> PUREFTPD_START
end
```

**Sources:** [src/main.c L1-L8](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c#L1-L8)

 [src/ftpd.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L1-L50)

 [src/ftp_parser.c L224-L824](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L224-L824)

 [src/globals.h L1-L201](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L1-L201)

## Core Server Process

The main server process is centered around the `pureftpd_start` function in `ftpd.c`, which initializes the server environment and enters the main command processing loop via the `parser()` function in `ftp_parser.c`.

```mermaid
flowchart TD

INIT_TZ["init_tz<br>ftpd.c"]
CLEARARGS["clearargs<br>ftpd.c"]
SIGNAL_SETUP["Signal Setup<br>sigalarm, sigterm_client"]
PARSER_LOOP["parser<br>ftp_parser.c"]
SFGETS_READ["sfgets<br>ftp_parser.c"]
CMD_DISPATCH["Command Dispatch<br>user, pass, retr, stor, etc."]
REPLY_SYSTEM["doreply<br>ftpd.c"]
GLOBAL_VARS["Global Variables<br>clientfd, loggedin, account"]
REPLY_QUEUE["Reply Queue<br>firstreply, lastreply"]
TRANSFER_STATE["Transfer State<br>xferfd, datafd, restartat"]

INIT_TZ --> PARSER_LOOP
CLEARARGS --> PARSER_LOOP
SIGNAL_SETUP --> PARSER_LOOP
CMD_DISPATCH --> GLOBAL_VARS
REPLY_SYSTEM --> REPLY_QUEUE

subgraph subGraph2 ["State Management"]
    GLOBAL_VARS
    REPLY_QUEUE
    TRANSFER_STATE
    GLOBAL_VARS --> TRANSFER_STATE
end

subgraph subGraph1 ["Main Loop"]
    PARSER_LOOP
    SFGETS_READ
    CMD_DISPATCH
    REPLY_SYSTEM
    PARSER_LOOP --> SFGETS_READ
    SFGETS_READ --> CMD_DISPATCH
    CMD_DISPATCH --> REPLY_SYSTEM
end

subgraph subGraph0 ["Server Initialization"]
    INIT_TZ
    CLEARARGS
    SIGNAL_SETUP
end
```

**Sources:** [src/ftpd.c L165-L187](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L165-L187)

 [src/ftpd.c L435-L507](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L435-L507)

 [src/ftp_parser.c L224-L824](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L224-L824)

 [src/ftp_parser.c L70-L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L70-L166)

## Command Processing Architecture

The FTP command processing follows a well-defined pipeline from network input to response generation, with comprehensive command parsing and validation.

| Component | Function | File | Purpose |
| --- | --- | --- | --- |
| Input Reader | `sfgets()` | ftp_parser.c | Reads and buffers FTP commands from client |
| Command Parser | `parser()` | ftp_parser.c | Parses commands and dispatches to handlers |
| Reply System | `addreply()`, `doreply()` | ftpd.c | Manages response queue and output |
| State Manager | Global variables | globals.h | Tracks session state and configuration |

```mermaid
flowchart TD

NETWORK_INPUT["Network Input<br>clientfd"]
SFGETS_BUFFER["Command Buffer<br>cmd[PATH_MAX + 32U]"]
CONTROL_FILTER["Control Character Filter<br>ISCTRLCODE check"]
CMD_NORMALIZE["Command Normalization<br>tolower, argument separation"]
AUTH_CHECK["Authentication Check<br>loggedin != 0"]
TLS_ENFORCEMENT["TLS Enforcement<br>enforce_tls_auth"]
USER_HANDLER["douser"]
PASS_HANDLER["dopass"]
RETR_HANDLER["doretr"]
STOR_HANDLER["dostor"]
LIST_HANDLER["dolist"]
OTHER_HANDLERS["Other Command Handlers"]

CONTROL_FILTER --> CMD_NORMALIZE
TLS_ENFORCEMENT --> USER_HANDLER
TLS_ENFORCEMENT --> PASS_HANDLER
TLS_ENFORCEMENT --> RETR_HANDLER
TLS_ENFORCEMENT --> STOR_HANDLER
TLS_ENFORCEMENT --> LIST_HANDLER
TLS_ENFORCEMENT --> OTHER_HANDLERS

subgraph subGraph2 ["Command Execution"]
    USER_HANDLER
    PASS_HANDLER
    RETR_HANDLER
    STOR_HANDLER
    LIST_HANDLER
    OTHER_HANDLERS
end

subgraph subGraph1 ["Command Parsing"]
    CMD_NORMALIZE
    AUTH_CHECK
    TLS_ENFORCEMENT
    CMD_NORMALIZE --> AUTH_CHECK
    AUTH_CHECK --> TLS_ENFORCEMENT
end

subgraph subGraph0 ["Command Input Processing"]
    NETWORK_INPUT
    SFGETS_BUFFER
    CONTROL_FILTER
    NETWORK_INPUT --> SFGETS_BUFFER
    SFGETS_BUFFER --> CONTROL_FILTER
end
```

**Sources:** [src/ftp_parser.c L70-L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L70-L166)

 [src/ftp_parser.c L224-L824](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L224-L824)

 [src/ftpd.c L671-L768](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L671-L768)

 [src/globals.h L34-L40](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L34-L40)

## Authentication System Architecture

Pure-FTPd implements a flexible authentication architecture that supports multiple backends through a common interface defined in `ftpd_p.h`. The system uses an `Authentication` structure to abstract different authentication methods.

```mermaid
flowchart TD

AUTH_STRUCT["Authentication Structure<br>ftpd_p.h:250-258"]
AUTH_LIST["auth_list[]<br>ftpd_p.h:260-281"]
AUTH_RESULT["AuthResult<br>ftpd.h:257-285"]
UNIX_AUTH["Unix Authentication<br>log_unix"]
PAM_AUTH["PAM Authentication<br>log_pam"]
MYSQL_AUTH["MySQL Authentication<br>log_mysql"]
PGSQL_AUTH["PostgreSQL Authentication<br>log_pgsql"]
LDAP_AUTH["LDAP Authentication<br>log_ldap"]
PUREDB_AUTH["PureDB Authentication<br>log_puredb"]
EXTAUTH["External Authentication<br>log_extauth"]
DOUSER["douser()<br>User Command Handler"]
DOPASS["dopass()<br>Password Command Handler"]
AUTH_CHECK["Authentication Check<br>Backend Selection"]
RESULT_PROCESSING["Result Processing<br>UID/GID, Directory, Quotas"]

AUTH_LIST --> UNIX_AUTH
AUTH_LIST --> PAM_AUTH
AUTH_LIST --> MYSQL_AUTH
AUTH_LIST --> PGSQL_AUTH
AUTH_LIST --> LDAP_AUTH
AUTH_LIST --> PUREDB_AUTH
AUTH_LIST --> EXTAUTH
RESULT_PROCESSING --> AUTH_RESULT

subgraph subGraph2 ["Authentication Flow"]
    DOUSER
    DOPASS
    AUTH_CHECK
    RESULT_PROCESSING
    DOUSER --> AUTH_CHECK
    DOPASS --> AUTH_CHECK
    AUTH_CHECK --> RESULT_PROCESSING
end

subgraph subGraph1 ["Authentication Backends"]
    UNIX_AUTH
    PAM_AUTH
    MYSQL_AUTH
    PGSQL_AUTH
    LDAP_AUTH
    PUREDB_AUTH
    EXTAUTH
end

subgraph subGraph0 ["Authentication Interface"]
    AUTH_STRUCT
    AUTH_LIST
    AUTH_RESULT
    AUTH_STRUCT --> AUTH_LIST
end
```

**Sources:** [src/ftpd_p.h L250-L287](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L250-L287)

 [src/ftpd.h L257-L285](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L257-L285)

 [src/ftpd.c L1269-L1394](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L1269-L1394)

## Data Transfer Architecture

Data transfers in Pure-FTPd use a separate data connection managed through file descriptors `datafd` and `xferfd`, with support for both active and passive modes.

```mermaid
flowchart TD

CLIENTFD["Control Connection<br>clientfd"]
DATAFD["Data Connection<br>datafd"]
XFERFD["Transfer Connection<br>xferfd"]
ACTIVE_MODE["Active Mode<br>doport, doeprt"]
PASSIVE_MODE["Passive Mode<br>dopasv"]
DATA_OPEN["Data Channel Open<br>opendata()"]
DATA_CLOSE["Data Channel Close<br>closedata()"]
UPLOAD["Upload Handler<br>ULHandler"]
DOWNLOAD["Download Handler<br>DLHandler"]
ASCII_BINARY["Type Handling<br>type variable"]
RESTART["Resume Support<br>restartat"]

CLIENTFD --> ACTIVE_MODE
CLIENTFD --> PASSIVE_MODE
DATA_OPEN --> DATAFD
XFERFD --> UPLOAD
XFERFD --> DOWNLOAD
UPLOAD --> DATA_CLOSE
DOWNLOAD --> DATA_CLOSE

subgraph subGraph2 ["Transfer Operations"]
    UPLOAD
    DOWNLOAD
    ASCII_BINARY
    RESTART
    UPLOAD --> ASCII_BINARY
    DOWNLOAD --> ASCII_BINARY
    ASCII_BINARY --> RESTART
end

subgraph subGraph1 ["Transfer Modes"]
    ACTIVE_MODE
    PASSIVE_MODE
    DATA_OPEN
    DATA_CLOSE
    ACTIVE_MODE --> DATA_OPEN
    PASSIVE_MODE --> DATA_OPEN
end

subgraph subGraph0 ["Connection Management"]
    CLIENTFD
    DATAFD
    XFERFD
    DATAFD --> XFERFD
end
```

**Sources:** [src/ftpd_p.h L297-L346](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L297-L346)

 [src/globals.h L28-L29](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L28-L29)

 [src/globals.h L97](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L97-L97)

 [src/globals.h L73](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L73-L73)

## Global State Management

The server maintains its state through global variables defined in `globals.h`, providing a centralized way to track session information, configuration, and transfer state.

| Category | Key Variables | Purpose |
| --- | --- | --- |
| Session State | `loggedin`, `account`, `guest` | User authentication and session status |
| Connection State | `clientfd`, `datafd`, `xferfd` | File descriptors for different connections |
| Transfer State | `uploaded`, `downloaded`, `restartat` | Data transfer tracking and resume support |
| Configuration | `idletime`, `maxusers`, `userchroot` | Server behavior and limits |
| Security State | `chrooted`, `tls_cnx`, `data_protection_level` | Security and encryption status |

**Sources:** [src/globals.h L12-L201](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L12-L201)

## Security Architecture Integration

Pure-FTPd integrates security features throughout its architecture, from TLS encryption to privilege separation and access controls.

```mermaid
flowchart TD

FILE_PERMS["File Permissions<br>u_mask, u_mask_d"]
DOT_FILES["Dot File Access<br>dot_read_ok, dot_write_ok"]
ANONYMOUS_LIMITS["Anonymous Limits<br>anon_only, anon_noupload"]
PRIVSEP_ENABLED["Privilege Separation<br>WITHOUT_PRIVSEP"]
CHROOT_JAIL["Chroot Jail<br>chrooted, root_directory"]
USER_DROPPING["User/Group Dropping<br>useruid, chroot_trustedgid"]
TLS_CNX["TLS Connection<br>tls_cnx"]
TLS_DATA["TLS Data Channel<br>tls_data_cnx"]
CERT_MANAGEMENT["Certificate Management<br>cert_file, key_file"]

subgraph subGraph2 ["Access Control"]
    FILE_PERMS
    DOT_FILES
    ANONYMOUS_LIMITS
    FILE_PERMS --> DOT_FILES
    DOT_FILES --> ANONYMOUS_LIMITS
end

subgraph subGraph1 ["Process Security"]
    PRIVSEP_ENABLED
    CHROOT_JAIL
    USER_DROPPING
    PRIVSEP_ENABLED --> CHROOT_JAIL
    CHROOT_JAIL --> USER_DROPPING
end

subgraph subGraph0 ["Transport Security"]
    TLS_CNX
    TLS_DATA
    CERT_MANAGEMENT
    TLS_CNX --> TLS_DATA
    TLS_DATA --> CERT_MANAGEMENT
end
```

**Sources:** [src/globals.h L174-L184](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L174-L184)

 [src/globals.h L42-L48](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L42-L48)

 [src/globals.h L57-L59](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L57-L59)

 [src/globals.h L110-L111](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L110-L111)

## Reply and Communication System

The server uses a sophisticated reply queuing system to manage responses to FTP clients, supporting multi-line replies and proper FTP protocol formatting.

```mermaid
flowchart TD

ADDREPLY["addreply()<br>Format and queue response"]
ADDREPLY_NOFORMAT["addreply_noformat()<br>Queue literal response"]
REPLY_BUFFER["Reply Buffer<br>replybuf, replybuf_pos"]
REPLY_STRUCT["struct reply<br>ftpd_p.h:57-60"]
FIRST_REPLY["firstreply<br>Queue head"]
LAST_REPLY["lastreply<br>Queue tail"]
DOREPLY["doreply()<br>Send queued replies"]
CLIENT_PRINTF["client_printf()<br>Buffered output"]
CLIENT_FFLUSH["client_fflush()<br>Flush output buffer"]

ADDREPLY --> REPLY_STRUCT
ADDREPLY_NOFORMAT --> REPLY_STRUCT
LAST_REPLY --> DOREPLY
CLIENT_PRINTF --> REPLY_BUFFER

subgraph subGraph2 ["Output Processing"]
    DOREPLY
    CLIENT_PRINTF
    CLIENT_FFLUSH
    DOREPLY --> CLIENT_PRINTF
    CLIENT_PRINTF --> CLIENT_FFLUSH
end

subgraph subGraph1 ["Reply Queue"]
    REPLY_STRUCT
    FIRST_REPLY
    LAST_REPLY
    REPLY_STRUCT --> FIRST_REPLY
    FIRST_REPLY --> LAST_REPLY
end

subgraph subGraph0 ["Reply Generation"]
    ADDREPLY
    ADDREPLY_NOFORMAT
    REPLY_BUFFER
end
```

**Sources:** [src/ftpd.c L671-L768](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L671-L768)

 [src/ftpd.c L305-L351](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L305-L351)

 [src/ftpd_p.h L57-L60](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L57-L60)

 [src/ftpd_p.h L361-L362](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L361-L362)
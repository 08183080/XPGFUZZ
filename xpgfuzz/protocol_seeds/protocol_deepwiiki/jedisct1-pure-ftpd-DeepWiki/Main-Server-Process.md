# Main Server Process

> **Relevant source files**
> * [src/ftp_parser.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c)
> * [src/ftpd.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c)
> * [src/ftpd.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h)
> * [src/main.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c)

This document covers the central server process implementation in Pure-FTPd, focusing on the core FTP server daemon that handles client connections, processes FTP commands, and manages user sessions. The main server process encompasses the program entry point, session lifecycle management, command parsing, and the primary control flow that coordinates all server operations.

For information about specific authentication backends, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management). For details about TLS/SSL implementation, see [TLS/SSL Encryption](/jedisct1/pure-ftpd/3.1-tlsssl-encryption). For configuration and build system details, see [Configuration and Administration](/jedisct1/pure-ftpd/5-configuration-and-administration).

## Architecture Overview

The main server process is implemented across several key source files, with `ftpd.c` serving as the primary coordinator. The architecture follows a single-threaded, event-driven model where each client connection is handled by a separate process (in standalone mode) or managed by inetd/xinetd.

### Core Server Architecture

```mermaid
flowchart TD

main["main()<br>src/main.c"]
pureftpd_start["pureftpd_start()<br>src/ftpd.c"]
parser["parser()<br>src/ftp_parser.c"]
sfgets["sfgets()<br>src/ftp_parser.c"]
cmd_dispatch["Command Dispatch<br>Various do*() functions"]
douser["douser()<br>User Authentication"]
dopass["dopass()<br>Password Verification"]
session_state["Session State<br>Global Variables"]
file_ops["File Operations<br>doretr(), dostor(), dolist()"]
data_conn["Data Connection<br>opendata(), closedata()"]
replies["Reply Management<br>addreply(), doreply()"]
signals["Signal Handling<br>sigalarm(), sigterm()"]
logging["Logging System<br>logfile(), die()"]
security["Security Features<br>Privilege separation"]

pureftpd_start --> parser
cmd_dispatch --> douser
cmd_dispatch --> dopass
cmd_dispatch --> file_ops
cmd_dispatch --> data_conn
parser --> session_state
pureftpd_start --> signals
pureftpd_start --> logging
pureftpd_start --> security

subgraph Infrastructure ["Infrastructure"]
    signals
    logging
    security
end

subgraph subGraph3 ["Core Operations"]
    file_ops
    data_conn
    replies
    file_ops --> replies
end

subgraph subGraph2 ["Session Management"]
    douser
    dopass
    session_state
end

subgraph subGraph1 ["Command Processing"]
    parser
    sfgets
    cmd_dispatch
    parser --> sfgets
    parser --> cmd_dispatch
end

subgraph subGraph0 ["Entry Point"]
    main
    pureftpd_start
    main --> pureftpd_start
end
```

Sources: [src/main.c L1-L8](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c#L1-L8)

 [src/ftpd.c L1-L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L1-L50)

 [src/ftp_parser.c L224-L824](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L224-L824)

 [src/ftpd.h L313-L401](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L313-L401)

## Server Lifecycle and Initialization

The server follows a well-defined lifecycle from startup to client session handling. The process begins with command-line parsing, configuration loading, and network setup.

### Server Startup Flow

```mermaid
flowchart TD

start["Program Start"]
main_entry["main(argc, argv)<br>src/main.c:4"]
pureftpd_start["pureftpd_start(argc, argv, NULL)<br>src/ftpd.c"]
config_parse["Configuration Parsing<br>Command line options"]
init_globals["Global State Initialization<br>src/globals.h variables"]
signal_setup["Signal Handler Setup<br>sigalarm(), sigterm_client()"]
mode_check["Server Mode?"]
standalone["Standalone Server<br>daemons() function"]
inetd_mode["inetd Mode<br>Direct client handling"]
listen_socket["Create Listen Socket<br>bind(), listen()"]
accept_loop["Accept Connection Loop<br>accept(), fork()"]
client_session["Client Session Handler"]
auth_phase["Authentication Phase<br>douser(), dopass()"]
command_loop["Command Processing Loop<br>parser()"]
session_end["Session Cleanup<br>_EXIT()"]

start --> main_entry
main_entry --> pureftpd_start
pureftpd_start --> config_parse
config_parse --> init_globals
init_globals --> signal_setup
signal_setup --> mode_check
mode_check --> standalone
mode_check --> inetd_mode
standalone --> listen_socket
listen_socket --> accept_loop
accept_loop --> client_session
inetd_mode --> client_session
client_session --> auth_phase
auth_phase --> command_loop
command_loop --> session_end
```

Sources: [src/main.c L4-L7](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c#L4-L7)

 [src/ftpd.c L374-L409](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L374-L409)

 [src/ftpd.c L411-L433](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L411-L433)

## Command Processing Pipeline

The command processing system is built around a main parsing loop that reads FTP commands from clients, validates them, and dispatches to appropriate handler functions. This pipeline handles both control and data connections.

### FTP Command Processing Flow

```mermaid
flowchart TD

sfgets["sfgets()<br>src/ftp_parser.c:70"]
poll_read["poll() for client input<br>src/ftp_parser.c:91"]
buffer_mgmt["Command Buffer Management<br>cmd, cmdsize, scanned"]
parser_loop["parser() Main Loop<br>src/ftp_parser.c:236"]
cmd_normalize["Command Normalization<br>tolower(), validation"]
arg_extraction["Argument Extraction<br>src/ftp_parser.c:299-307"]
auth_check["Authentication<br>Required?"]
cmd_lookup["Command Lookup<br>strcmp() checks"]
handler_call["Handler Function Call<br>do*() functions"]
addreply["addreply()<br>src/ftpd.c:697"]
reply_buffer["Reply Buffer Management<br>replybuf, firstreply"]
doreply["doreply()<br>src/ftpd.c:732"]
client_output["Client Output<br>client_printf()"]
auth_required["MSG_NOT_LOGGED_IN<br>src/ftp_parser.c:455"]

buffer_mgmt --> parser_loop
arg_extraction --> auth_check
auth_check --> auth_required
handler_call --> addreply
client_output --> parser_loop

subgraph subGraph3 ["Response Generation"]
    addreply
    reply_buffer
    doreply
    client_output
    addreply --> reply_buffer
    reply_buffer --> doreply
    doreply --> client_output
end

subgraph subGraph2 ["Command Dispatch"]
    auth_check
    cmd_lookup
    handler_call
    auth_check --> cmd_lookup
    cmd_lookup --> handler_call
end

subgraph subGraph1 ["Command Parsing"]
    parser_loop
    cmd_normalize
    arg_extraction
    parser_loop --> cmd_normalize
    cmd_normalize --> arg_extraction
end

subgraph subGraph0 ["Command Reading"]
    sfgets
    poll_read
    buffer_mgmt
    poll_read --> sfgets
    sfgets --> buffer_mgmt
end
```

Sources: [src/ftp_parser.c L70-L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L70-L166)

 [src/ftp_parser.c L224-L824](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L224-L824)

 [src/ftpd.c L689-L768](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L689-L768)

 [src/ftpd.c L315-L351](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L315-L351)

## Session Management

Session management handles the complete lifecycle of an FTP client connection, from initial greeting through authentication to command processing and session termination.

### Session State Management

The server maintains session state through global variables and structures defined in `globals.h`. Key state includes authentication status, current directory, transfer modes, and connection information.

```mermaid
flowchart TD

wd["wd[PATH_MAX]<br>Working directory"]
chrooted["chrooted<br>Chroot status"]
root_directory["root_directory<br>Chroot path"]
loggedin["loggedin<br>Login status flag"]
authresult["authresult<br>AuthResult structure"]
account["account[MAX_USER_LENGTH]<br>Username storage"]
guest["guest<br>Anonymous user flag"]
clientfd["clientfd<br>Control connection FD"]
datafd["datafd<br>Data connection FD"]
xferfd["xferfd<br>Transfer FD"]
ctrlconn["ctrlconn<br>Control connection address"]
type["type<br>Transfer type (A/I)"]
passive["passive<br>Passive mode flag"]
restartat["restartat<br>REST command offset"]
uploaded["uploaded<br>Bytes uploaded"]
downloaded["downloaded<br>Bytes downloaded"]

passive --> datafd

subgraph subGraph2 ["Transfer State"]
    type
    passive
    restartat
    uploaded
    downloaded
    type --> restartat
end

subgraph subGraph1 ["Connection State"]
    clientfd
    datafd
    xferfd
    ctrlconn
    clientfd --> datafd
    datafd --> xferfd
end

subgraph subGraph3 ["Directory State"]
    wd
    chrooted
    root_directory
    wd --> chrooted
    chrooted --> root_directory
end

subgraph subGraph0 ["Authentication State"]
    loggedin
    authresult
    account
    guest
    loggedin --> authresult
    account --> authresult
end
```

Sources: [src/globals.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h)

 [src/ftpd.h L257-L285](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L257-L285)

 [src/ftpd.c L1269-L1480](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L1269-L1480)

## Key Data Structures

The server uses several important data structures to manage client sessions, authentication results, and file operations.

### Core Data Structures

| Structure | Purpose | Key Fields | Source Location |
| --- | --- | --- | --- |
| `AuthResult` | Authentication results and user info | `auth_ok`, `uid`, `gid`, `dir`, quotas, throttling | [src/ftpd.h L257-L285](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L257-L285) |
| `PureFileInfo` | File metadata for listings | `names_pnt`, `size`, `mtime`, `mode`, `uid`, `gid` | [src/ftpd.h L287-L296](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L287-L296) |
| `reply` | Reply chain for client responses | `next`, `line` | [src/ftpd.c L672-L687](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L672-L687) |
| Global variables | Session state management | `loggedin`, `clientfd`, `wd`, `account` | [src/globals.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h) |

### Authentication Result Structure

```mermaid
flowchart TD

auth_ok["auth_ok<br>Authentication status"]
uid_gid["uid, gid<br>User/group IDs"]
dir_info["dir<br>Home directory"]
backend_data["backend_data<br>Backend-specific data"]
throttling["throttling_*<br>Bandwidth limits"]
quotas["user_quota_*<br>Disk quotas"]
ratios["ratio_*<br>Upload/download ratios"]
per_user["per_user_max<br>Connection limits"]

backend_data --> throttling

subgraph subGraph1 ["Optional Features"]
    throttling
    quotas
    ratios
    per_user
    throttling --> quotas
    quotas --> ratios
    ratios --> per_user
end

subgraph subGraph0 ["AuthResult Structure"]
    auth_ok
    uid_gid
    dir_info
    backend_data
    auth_ok --> uid_gid
    uid_gid --> dir_info
    dir_info --> backend_data
end
```

Sources: [src/ftpd.h L257-L285](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.h#L257-L285)

 [src/ftpd.c L1482-L1562](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L1482-L1562)

## Signal Handling and Process Management

The server implements comprehensive signal handling for graceful shutdown, timeout management, and child process cleanup in standalone mode.

### Signal Handler Architecture

```mermaid
flowchart TD

disablesignals["disablesignals()<br>Block all signals<br>src/ftpd.c:50"]
usleep2["usleep2()<br>Signal-safe sleep<br>src/ftpd.c:67"]
enablesignals["enablesignals()<br>Restore signal mask<br>src/ftpd.c:60"]
sigalarm_h["sigalarm()<br>Timeout handling<br>src/ftpd.c:374"]
sigterm_client_h["sigterm_client()<br>Client termination<br>src/ftpd.c:403"]
sigterm_h["sigterm()<br>Server shutdown<br>src/ftpd.c:412"]
sigchild_h["sigchild()<br>Child cleanup<br>src/ftpd.c:382"]
timeout_die["die() with timeout message"]
clean_exit["_EXIT(EXIT_SUCCESS)"]
server_stop["stop_server = 1<br>Close listen sockets"]
child_cleanup["waitpid() cleanup<br>Update nb_children"]

sigalarm_h --> timeout_die
sigterm_client_h --> clean_exit
sigterm_h --> server_stop
sigchild_h --> child_cleanup

subgraph subGraph1 ["Signal Actions"]
    timeout_die
    clean_exit
    server_stop
    child_cleanup
end

subgraph subGraph0 ["Signal Handlers"]
    sigalarm_h
    sigterm_client_h
    sigterm_h
    sigchild_h
end

subgraph subGraph2 ["Signal Management"]
    disablesignals
    usleep2
    enablesignals
    disablesignals --> usleep2
    usleep2 --> enablesignals
end
```

Sources: [src/ftpd.c L50-L72](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L50-L72)

 [src/ftpd.c L374-L433](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L374-L433)

## Error Handling and Logging

The server implements a centralized error handling and logging system that provides detailed information for debugging and security monitoring.

### Error and Logging System

```mermaid
flowchart TD

die_func["die()<br>Fatal error with logging<br>src/ftpd.c:353"]
die_mem["die_mem()<br>Out of memory error<br>src/ftpd.c:369"]
exit_func["_EXIT()<br>Clean process exit<br>src/ftpd.c:296"]
logfile_func["logfile()<br>Syslog integration<br>src/ftpd.c:603"]
log_levels["LOG_INFO, LOG_ERR<br>LOG_WARNING, LOG_DEBUG"]
syslog_out["syslog() output<br>Facility-based routing"]
addreply_func["addreply()<br>Format reply messages<br>src/ftpd.c:697"]
addreply_noformat["addreply_noformat()<br>Direct message<br>src/ftpd.c:689"]
reply_chain["Reply chain management<br>firstreply, lastreply"]
doreply["doreply"]

die_func --> logfile_func
die_func --> addreply_func
reply_chain --> doreply

subgraph subGraph2 ["Reply System"]
    addreply_func
    addreply_noformat
    reply_chain
    addreply_func --> reply_chain
    addreply_noformat --> reply_chain
end

subgraph subGraph1 ["Logging Functions"]
    logfile_func
    log_levels
    syslog_out
    logfile_func --> log_levels
    log_levels --> syslog_out
end

subgraph subGraph0 ["Error Functions"]
    die_func
    die_mem
    exit_func
    die_func --> exit_func
    die_mem --> die_func
end
```

Sources: [src/ftpd.c L296-L372](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L296-L372)

 [src/ftpd.c L603-L650](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L603-L650)

 [src/ftpd.c L689-L768](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L689-L768)

## Process Naming and Management

The server includes functionality for process naming and management, particularly useful for monitoring and debugging in production environments.

### Process Management Features

```mermaid
flowchart TD

setprocessname["setprocessname()<br>Update process title<br>src/ftpd.c:488"]
clearargs["clearargs()<br>Prepare argv for updates<br>src/ftpd.c:435"]
proc_titles["Process Title Updates<br>'pure-ftpd (IDLE)'<br>'pure-ftpd (USER)'"]
argv_lth["argv_lth<br>Argument buffer length"]
argv0["argv0<br>Original argv pointer"]
env_copy["Environment duplication<br>malloc() + strdup()"]

clearargs --> argv_lth
clearargs --> argv0
clearargs --> env_copy
argv_lth --> setprocessname
argv0 --> setprocessname

subgraph subGraph1 ["Memory Management"]
    argv_lth
    argv0
    env_copy
end

subgraph subGraph0 ["Process Control"]
    setprocessname
    clearargs
    proc_titles
    setprocessname --> proc_titles
end
```

Sources: [src/ftpd.c L435-L507](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd.c#L435-L507)

 [src/ftp_parser.c L240-L248](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftp_parser.c#L240-L248)
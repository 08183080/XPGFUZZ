# Transport Mechanisms

> **Relevant source files**
> * [src/src/transports/appendfile.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c)
> * [src/src/transports/autoreply.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c)
> * [src/src/transports/lmtp.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c)
> * [src/src/transports/pipe.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c)
> * [src/src/transports/tf_maildir.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/tf_maildir.c)

This document covers Exim's transport system, which handles the final delivery of messages after routing has determined their destination. Transport drivers implement different delivery methods including local file storage, external commands, and network protocols.

For information about message routing and address resolution, see [Routing System](/Exim/exim/2.2-routing-system). For details about the overall delivery process, see [Transport and Delivery](/Exim/exim/2.3-transport-and-delivery).

## Transport Architecture

Exim's transport system follows a modular driver architecture where each transport type implements a common interface for message delivery. The transport layer receives messages from the routing system and handles the actual delivery mechanism.

### Transport Driver Interface

```mermaid
flowchart TD

TI["transport_instance"]
OPTS["Options Block"]
ENTRY["Entry Function"]
INIT["Init Function"]
SETUP["Setup Function"]
APPEND["appendfile_transport"]
APPENTRY["appendfile_transport_entry()"]
PIPE["pipe_transport"]
PIPEENTRY["pipe_transport_entry()"]
AUTO["autoreply_transport"]
AUTOENTRY["autoreply_transport_entry()"]
LMTP["lmtp_transport"]
LMTPENTRY["lmtp_transport_entry()"]
ADDR["address_item"]
MSG["Message Data"]
ROUTE["Routing Results"]
TCTX["transport_ctx"]
WRITE["transport_write_message()"]

ENTRY --> TCTX
APPENTRY --> WRITE
PIPEENTRY --> WRITE
LMTPENTRY --> WRITE

subgraph subGraph2 ["Delivery Context"]
    ADDR
    MSG
    ROUTE
    TCTX
    WRITE
    ADDR --> MSG
    ADDR --> ROUTE
    TCTX --> ADDR
    TCTX --> WRITE
end

subgraph subGraph1 ["Core Transports"]
    APPEND
    APPENTRY
    PIPE
    PIPEENTRY
    AUTO
    AUTOENTRY
    LMTP
    LMTPENTRY
    APPEND --> APPENTRY
    PIPE --> PIPEENTRY
    AUTO --> AUTOENTRY
    LMTP --> LMTPENTRY
end

subgraph subGraph0 ["Transport Framework"]
    TI
    OPTS
    ENTRY
    INIT
    SETUP
    TI --> OPTS
    TI --> ENTRY
    TI --> INIT
    TI --> SETUP
end
```

Each transport driver provides:

* An options block defining configurable parameters
* An entry function that performs the actual delivery
* An initialization function for setup and validation
* Optional setup function for privileged operations

Sources: [src/src/transports/appendfile.c L1007-L1018](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L1007-L1018)

 [src/src/transports/pipe.c L510-L514](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L510-L514)

 [src/src/transports/autoreply.c L263-L266](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L263-L266)

### Transport Selection and Execution

```mermaid
sequenceDiagram
  participant Router
  participant deliver.c
  participant Transport Driver
  participant transport_write_message()

  Router->>deliver.c: "address_item with transport"
  deliver.c->>Transport Driver: "setup() [privileged]"
  deliver.c->>deliver.c: "Change uid/gid"
  deliver.c->>Transport Driver: "entry() function"
  Transport Driver->>transport_write_message(): "Write message data"
  transport_write_message()->>Transport Driver: "Status/errors"
  Transport Driver->>deliver.c: "Delivery status"
```

The delivery process follows a two-phase approach where setup operations requiring privileges are performed first, followed by the actual delivery under the target user context.

Sources: [src/src/transports/appendfile.c L179-L182](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L179-L182)

 [src/src/transports/pipe.c L125-L127](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L125-L127)

## File Storage Transport (appendfile)

The `appendfile` transport handles delivery to local files and directories, supporting multiple mailbox formats and comprehensive quota management.

### Mailbox Format Support

| Format | Description | Key Features |
| --- | --- | --- |
| Unix mbox | Traditional single-file format | From-line separation, file locking |
| Maildir | Directory-based format | Atomic delivery, no locking required |
| MBX | University of Washington format | Binary headers, efficient indexing |
| Mailstore | Simple directory format | One file per message |

The transport automatically detects and handles format-specific requirements:

```mermaid
flowchart TD

OPTS["maildir_format<br>mbx_format<br>mailstore_format"]
FORMAT["Format Selection"]
DIRS["maildir_ensure_directories()"]
SIZE["maildir_compute_size()"]
QUOTA["Quota Checking"]
LOCK["Lock Strategy"]
FCNTL["fcntl() locking"]
FLOCK["flock() locking"]
DOTLOCK["Dot-file locking"]
MBX["MBX locking"]
DELIVER["File Creation"]

FORMAT --> DIRS
FORMAT --> LOCK
QUOTA --> DELIVER
FCNTL --> DELIVER
FLOCK --> DELIVER
DOTLOCK --> DELIVER
MBX --> DELIVER

subgraph subGraph3 ["Delivery Process"]
    DELIVER
end

subgraph subGraph2 ["File Locking"]
    LOCK
    FCNTL
    FLOCK
    DOTLOCK
    MBX
    LOCK --> FCNTL
    LOCK --> FLOCK
    LOCK --> DOTLOCK
    LOCK --> MBX
end

subgraph subGraph1 ["Maildir Handling"]
    DIRS
    SIZE
    QUOTA
    DIRS --> SIZE
    SIZE --> QUOTA
end

subgraph subGraph0 ["Format Detection"]
    OPTS
    FORMAT
    OPTS --> FORMAT
end
```

Sources: [src/src/transports/appendfile.c L145-L149](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L145-L149)

 [src/src/transports/tf_maildir.c L46-L48](https://github.com/Exim/exim/blob/29568b25/src/src/transports/tf_maildir.c#L46-L48)

 [src/src/transports/appendfile.c L824-L903](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L824-L903)

### Quota Management

The appendfile transport implements sophisticated quota management with multiple enforcement mechanisms:

```mermaid
flowchart TD

QUOTAVAL["quota_value"]
CHECK["Quota Checking"]
QUOTAFC["quota_filecount_value"]
QUOTAWARN["quota_warn_threshold"]
WARN["Warning Generation"]
DIRSIZE["check_dir_size()"]
MAILSIZE["maildir_compute_size()"]
REGEX["quota_size_regex"]
TOTALSIZE["Total Size"]
REJECT["Reject Delivery"]
WARNMSG["Send Warning Message"]
DEFER["Return DEFER"]

CHECK --> DIRSIZE
CHECK --> MAILSIZE
CHECK --> REGEX
TOTALSIZE --> REJECT
WARN --> WARNMSG

subgraph subGraph2 ["Enforcement Actions"]
    REJECT
    WARNMSG
    DEFER
    REJECT --> DEFER
end

subgraph subGraph1 ["Size Calculation Methods"]
    DIRSIZE
    MAILSIZE
    REGEX
    TOTALSIZE
    DIRSIZE --> TOTALSIZE
    MAILSIZE --> TOTALSIZE
    REGEX --> TOTALSIZE
end

subgraph subGraph0 ["Quota Configuration"]
    QUOTAVAL
    CHECK
    QUOTAFC
    QUOTAWARN
    WARN
    QUOTAVAL --> CHECK
    QUOTAFC --> CHECK
    QUOTAWARN --> WARN
end
```

Quota checking supports both size-based and file-count limits, with optional regex-based size extraction from filenames for performance optimization.

Sources: [src/src/transports/appendfile.c L196-L315](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L196-L315)

 [src/src/transports/tf_maildir.c L352-L568](https://github.com/Exim/exim/blob/29568b25/src/src/transports/tf_maildir.c#L352-L568)

 [src/src/transports/appendfile.c L673-L750](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L673-L750)

## Command Execution Transport (pipe)

The `pipe` transport executes external commands for message delivery, supporting both shell and direct command execution with comprehensive security controls.

### Command Execution Modes

```mermaid
flowchart TD

CMD["ob->cmd"]
SHELL["use_shell?"]
SHELLCMD["set_up_shell_command()"]
DIRECTCMD["set_up_direct_command()"]
ALLOW["allow_commands check"]
RESTRICT["restrict_to_path check"]
PATHSEARCH["PATH search"]
EXEC["child_open()"]
PIPE["Pipe I/O"]
MONITOR["Output monitoring"]

DIRECTCMD --> ALLOW
SHELLCMD --> EXEC
PATHSEARCH --> EXEC

subgraph Execution ["Execution"]
    EXEC
    PIPE
    MONITOR
    EXEC --> PIPE
    PIPE --> MONITOR
end

subgraph subGraph1 ["Security Checks"]
    ALLOW
    RESTRICT
    PATHSEARCH
    ALLOW --> RESTRICT
    RESTRICT --> PATHSEARCH
end

subgraph subGraph0 ["Command Setup"]
    CMD
    SHELL
    SHELLCMD
    DIRECTCMD
    CMD --> SHELL
    SHELL --> SHELLCMD
    SHELL --> DIRECTCMD
end
```

The transport provides two execution modes:

* **Shell mode**: Commands executed via `/bin/sh -c`, supporting shell features like pipes and redirection
* **Direct mode**: Direct execution with argument parsing and security restrictions

Sources: [src/src/transports/pipe.c L296-L397](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L296-L397)

 [src/src/transports/pipe.c L419-L497](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L419-L497)

 [src/src/transports/pipe.c L612-L618](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L612-L618)

### Security and Resource Controls

The pipe transport implements multiple layers of security control:

| Control | Purpose | Configuration |
| --- | --- | --- |
| `allow_commands` | Whitelist permitted commands | List of allowed command names |
| `restrict_to_path` | Prevent absolute paths | Boolean flag |
| `timeout` | Limit execution time | Seconds |
| `max_output` | Limit output size | Bytes |
| `permit_coredump` | Allow core dumps | Boolean flag |

```mermaid
flowchart TD

FORK["child_open()"]
PIPES["Setup I/O pipes"]
MONITOR["Output monitoring process"]
LIMIT["max_output enforcement"]
KILL["killpg() on limit"]
TIMEOUT["timeout monitoring"]
SIGKILL["Process group termination"]
ENV["Environment setup"]
LOCALPART["LOCAL_PART"]
DOMAIN["DOMAIN"]
SENDER["SENDER"]
CUSTOM["Custom variables"]

FORK --> TIMEOUT
FORK --> ENV

subgraph Environment ["Environment"]
    ENV
    LOCALPART
    DOMAIN
    SENDER
    CUSTOM
    ENV --> LOCALPART
    ENV --> DOMAIN
    ENV --> SENDER
    ENV --> CUSTOM
end

subgraph subGraph1 ["Timeout Handling"]
    TIMEOUT
    SIGKILL
    TIMEOUT --> SIGKILL
end

subgraph subGraph0 ["Process Management"]
    FORK
    PIPES
    MONITOR
    LIMIT
    KILL
    FORK --> PIPES
    PIPES --> MONITOR
    MONITOR --> LIMIT
    LIMIT --> KILL
end
```

The transport creates a dedicated subprocess for output monitoring to prevent deadlocks and enforce resource limits.

Sources: [src/src/transports/pipe.c L624-L677](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L624-L677)

 [src/src/transports/pipe.c L722-L770](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L722-L770)

 [src/src/transports/pipe.c L125-L166](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L125-L166)

## Autoreply Transport

The `autoreply` transport generates automated responses such as vacation messages, with sophisticated duplicate prevention and content management.

### Response Generation

```mermaid
flowchart TD

ADDR["addr->reply"]
CONTENT["Response Content"]
TRANSPORT["Transport Options"]
FROM["From header"]
SUBJECT["Subject header"]
TEXT["Message text"]
FILE["File inclusion"]
ONCE["once database"]
DBM["DBM file"]
CACHE["Fixed-size cache"]
TIMECHECK["Time-based repeat"]
CHILD["child_open_exim()"]
HEADERS["Generate headers"]
BODY["Message body"]
SEND["Send via Exim"]

CONTENT --> ONCE
TIMECHECK --> CHILD

subgraph Delivery ["Delivery"]
    CHILD
    HEADERS
    BODY
    SEND
    CHILD --> HEADERS
    HEADERS --> BODY
    BODY --> SEND
end

subgraph subGraph1 ["Duplicate Prevention"]
    ONCE
    DBM
    CACHE
    TIMECHECK
    ONCE --> DBM
    ONCE --> CACHE
    DBM --> TIMECHECK
    CACHE --> TIMECHECK
end

subgraph subGraph0 ["Content Sources"]
    ADDR
    CONTENT
    TRANSPORT
    FROM
    SUBJECT
    TEXT
    FILE
    ADDR --> CONTENT
    TRANSPORT --> CONTENT
    CONTENT --> FROM
    CONTENT --> SUBJECT
    CONTENT --> TEXT
    CONTENT --> FILE
end
```

The transport supports both database and file-based tracking of previous responses to prevent mail loops and excessive automation.

Sources: [src/src/transports/autoreply.c L294-L312](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L294-L312)

 [src/src/transports/autoreply.c L403-L530](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L403-L530)

 [src/src/transports/autoreply.c L553-L566](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L553-L566)

### Never Mail and Content Processing

The autoreply transport includes sophisticated recipient filtering and content expansion:

```mermaid
flowchart TD

RECIPIENTS["to/cc/bcc lists"]
NEVER["never_mail check"]
FILTER["check_never_mail()"]
CLEAN["Cleaned recipient list"]
EXPAND["checkexpand()"]
VALIDATE["Character validation"]
SAFE["Safe content"]
INREPLY["In-Reply-To generation"]
REFS["References header"]
AUTO["Auto-Submitted header"]
FINAL["Final message"]

CLEAN --> EXPAND
SAFE --> INREPLY

subgraph subGraph2 ["Message Construction"]
    INREPLY
    REFS
    AUTO
    FINAL
    INREPLY --> REFS
    REFS --> AUTO
    AUTO --> FINAL
end

subgraph subGraph1 ["Content Expansion"]
    EXPAND
    VALIDATE
    SAFE
    EXPAND --> VALIDATE
    VALIDATE --> SAFE
end

subgraph subGraph0 ["Recipient Processing"]
    RECIPIENTS
    NEVER
    FILTER
    CLEAN
    RECIPIENTS --> NEVER
    NEVER --> FILTER
    FILTER --> CLEAN
end
```

Content validation ensures that expanded strings contain only printable characters, preventing injection attacks and malformed messages.

Sources: [src/src/transports/autoreply.c L173-L250](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L173-L250)

 [src/src/transports/autoreply.c L126-L154](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L126-L154)

 [src/src/transports/autoreply.c L584-L594](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L584-L594)

## LMTP Transport

The `lmtp` transport implements the Local Mail Transfer Protocol for communication with local delivery agents that support the LMTP protocol.

### LMTP Protocol Implementation

```mermaid
sequenceDiagram
  participant lmtp_transport
  participant LMTP Server

  lmtp_transport->>LMTP Server: "Connect (socket/command)"
  LMTP Server->>lmtp_transport: "220 Welcome"
  lmtp_transport->>LMTP Server: "LHLO hostname"
  LMTP Server->>lmtp_transport: "250 OK + capabilities"
  lmtp_transport->>LMTP Server: "MAIL FROM:<sender>"
  LMTP Server->>lmtp_transport: "250 OK"
  loop [For each recipient]
    lmtp_transport->>LMTP Server: "RCPT TO:<recipient>"
    LMTP Server->>lmtp_transport: "250 OK / 4xx / 5xx"
    lmtp_transport->>LMTP Server: "DATA"
    LMTP Server->>lmtp_transport: "354 Send data"
    lmtp_transport->>LMTP Server: "Message content + ."
    LMTP Server->>lmtp_transport: "250 OK / 4xx / 5xx"
  end
  lmtp_transport->>LMTP Server: "QUIT"
  LMTP Server->>lmtp_transport: "221 Goodbye"
```

LMTP differs from SMTP in that it returns individual status codes for each recipient after the message data, allowing per-recipient status tracking.

Sources: [src/src/transports/lmtp.c L562-L575](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L562-L575)

 [src/src/transports/lmtp.c L675-L723](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L675-L723)

 [src/src/transports/lmtp.c L229-L252](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L229-L252)

### Connection Methods

The LMTP transport supports two connection methods:

```mermaid
flowchart TD

CONFIG["LMTP Config"]
CMD["command set?"]
CHILDOPEN["child_open()"]
SOCKET["Unix socket"]
ARGV["Command arguments"]
PIPES["stdin/stdout pipes"]
EXPAND["expand_string()"]
CONNECT["Unix domain socket"]
SOCKIO["Socket I/O"]
LMTP["LMTP conversation"]

CHILDOPEN --> ARGV
SOCKET --> EXPAND
PIPES --> LMTP
SOCKIO --> LMTP

subgraph Protocol ["Protocol"]
    LMTP
end

subgraph subGraph2 ["Socket Mode"]
    EXPAND
    CONNECT
    SOCKIO
    EXPAND --> CONNECT
    CONNECT --> SOCKIO
end

subgraph subGraph1 ["Command Mode"]
    ARGV
    PIPES
    ARGV --> PIPES
end

subgraph Configuration ["Configuration"]
    CONFIG
    CMD
    CHILDOPEN
    SOCKET
    CONFIG --> CMD
    CMD --> CHILDOPEN
    CMD --> SOCKET
end
```

Command mode spawns an external LMTP server process, while socket mode connects to an existing server via Unix domain socket.

Sources: [src/src/transports/lmtp.c L492-L516](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L492-L516)

 [src/src/transports/lmtp.c L520-L551](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L520-L551)

 [src/src/transports/lmtp.c L78-L96](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L78-L96)

## Common Transport Patterns

### Message Writing Interface

All transports use the common `transport_write_message()` function for outputting message content:

```mermaid
flowchart TD

TCTX["transport_ctx"]
FD["File descriptor"]
TBLOCK["transport_instance"]
ADDR["address_item"]
OPTIONS["Write options"]
HEADERS["topt_no_headers"]
BODY["topt_no_body"]
CRLF["topt_use_crlf"]
DOT["topt_end_dot"]
ESCAPE["topt_escape_headers"]
WRITE["transport_write_message()"]

OPTIONS --> HEADERS
OPTIONS --> BODY
OPTIONS --> CRLF
OPTIONS --> DOT
OPTIONS --> ESCAPE
FD --> WRITE
HEADERS --> WRITE
BODY --> WRITE
CRLF --> WRITE
DOT --> WRITE
ESCAPE --> WRITE

subgraph subGraph2 ["Output Processing"]
    WRITE
end

subgraph subGraph1 ["Write Options"]
    HEADERS
    BODY
    CRLF
    DOT
    ESCAPE
end

subgraph subGraph0 ["Transport Context"]
    TCTX
    FD
    TBLOCK
    ADDR
    OPTIONS
    TCTX --> FD
    TCTX --> TBLOCK
    TCTX --> ADDR
    TCTX --> OPTIONS
end
```

This provides consistent message formatting across all transport types with configurable options for headers, body, line endings, and content escaping.

Sources: [src/src/transports/appendfile.c L485-L492](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L485-L492)

 [src/src/transports/pipe.c L265-L272](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L265-L272)

 [src/src/transports/lmtp.c L100-L107](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c#L100-L107)

### Error Handling and Status Codes

Transports return standardized status codes through the `address_item` structure:

| Status | Meaning | Action |
| --- | --- | --- |
| `OK` | Successful delivery | Message delivered |
| `DEFER` | Temporary failure | Retry later |
| `FAIL` | Permanent failure | Bounce message |
| `PANIC` | System error | Log and bounce |

```mermaid
flowchart TD

ERROR["Transport Error"]
TEMP["Temporary?"]
DEFER["DEFER status"]
PERM["Permanent?"]
FAIL["FAIL status"]
PANIC["PANIC status"]
QUEUE["Queue for retry"]
BOUNCE["Generate bounce"]
LOG["Error logging"]

DEFER --> QUEUE
FAIL --> BOUNCE
PANIC --> LOG

subgraph subGraph1 ["Error Propagation"]
    QUEUE
    BOUNCE
    LOG
end

subgraph subGraph0 ["Error Classification"]
    ERROR
    TEMP
    DEFER
    PERM
    FAIL
    PANIC
    ERROR --> TEMP
    TEMP --> DEFER
    TEMP --> PERM
    PERM --> FAIL
    PERM --> PANIC
end
```

Each transport implements specific error detection and classification logic appropriate to its delivery mechanism.

Sources: [src/src/transports/appendfile.c L1007-L1018](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c#L1007-L1018)

 [src/src/transports/pipe.c L1123-L1127](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L1123-L1127)

 [src/src/transports/autoreply.c L286-L287](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L286-L287)
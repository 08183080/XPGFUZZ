# Message Reception and SMTP

> **Relevant source files**
> * [doc/doc-docbook/spec.xfpt](https://github.com/Exim/exim/blob/29568b25/doc/doc-docbook/spec.xfpt)
> * [doc/doc-txt/ChangeLog](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/ChangeLog)
> * [doc/doc-txt/NewStuff](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/NewStuff)
> * [doc/doc-txt/OptionLists.txt](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/OptionLists.txt)
> * [src/src/acl.c](https://github.com/Exim/exim/blob/29568b25/src/src/acl.c)
> * [src/src/daemon.c](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c)
> * [src/src/deliver.c](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c)
> * [src/src/exim.c](https://github.com/Exim/exim/blob/29568b25/src/src/exim.c)
> * [src/src/expand.c](https://github.com/Exim/exim/blob/29568b25/src/src/expand.c)
> * [src/src/functions.h](https://github.com/Exim/exim/blob/29568b25/src/src/functions.h)
> * [src/src/globals.c](https://github.com/Exim/exim/blob/29568b25/src/src/globals.c)
> * [src/src/globals.h](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h)
> * [src/src/readconf.c](https://github.com/Exim/exim/blob/29568b25/src/src/readconf.c)
> * [src/src/receive.c](https://github.com/Exim/exim/blob/29568b25/src/src/receive.c)
> * [src/src/smtp_in.c](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c)
> * [src/src/structs.h](https://github.com/Exim/exim/blob/29568b25/src/src/structs.h)
> * [src/src/transport.c](https://github.com/Exim/exim/blob/29568b25/src/src/transport.c)
> * [src/src/transports/smtp.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/smtp.c)
> * [src/src/verify.c](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c)

## Purpose and Scope

This document covers Exim's SMTP server implementation, focusing on how inbound SMTP connections are established, managed, and processed through to message acceptance. It details the protocol handling, command processing, message data reception, and integration with Exim's access control systems.

For information about outbound SMTP delivery, see [Transport and Delivery](/Exim/exim/2.3-transport-and-delivery). For details about Access Control Lists that are invoked during SMTP sessions, see [Access Control Lists (ACLs)](/Exim/exim/2.4-access-control-lists-(acls)).

## Connection Management and Process Architecture

Exim's daemon process accepts incoming SMTP connections and spawns child processes to handle each session. The daemon maintains connection slots and manages process lifecycle.

### Connection Acceptance Flow

```mermaid
flowchart TD

daemon["daemon_go()"]
listen["Listen on SMTP ports"]
accept["accept() new connection"]
slot["Allocate smtp_slot"]
fork["fork() child process"]
parent["Parent: track in smtp_slots[]"]
child["Child: handle_smtp_call()"]
smtp_start["smtp_start_session()"]
command_loop["SMTP command loop"]
exit["Process exit"]
monitor["Monitor child processes"]

daemon --> listen
listen --> accept
accept --> slot
slot --> fork
fork --> parent
fork --> child
child --> smtp_start
smtp_start --> command_loop
command_loop --> exit
parent --> monitor
```

Sources: [src/src/daemon.c L400-L600](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L400-L600)

 [src/src/smtp_in.c L4000-L4100](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L4000-L4100)

The daemon uses a `smtp_slot` structure to track active connections:

```python
typedef struct smtp_slot {
  pid_t pid;              /* pid of the spawned reception process */
  uschar *host_address;   /* address of the client host */
} smtp_slot;
```

Connection limits are enforced through `smtp_accept_max` and `smtp_accept_max_per_host` configuration options.

## SMTP Protocol Implementation

The SMTP protocol implementation centers around command parsing, state management, and response generation within the child process handling each connection.

### SMTP Command Processing Pipeline

```mermaid
flowchart TD

start["smtp_start_session()"]
banner["Send SMTP banner"]
loop["smtp_command_loop()"]
read["Read command line"]
parse["Parse command"]
lookup["Lookup in cmd_list[]"]
sync_check["Synchronization check"]
acl["Run ACLs if applicable"]
execute["Execute command handler"]
response["Send SMTP response"]
state_update["Update session state"]
quit_check["QUIT command?"]
cleanup["Cleanup and exit"]

start --> banner
banner --> loop
loop --> read
read --> parse
parse --> lookup
lookup --> sync_check
sync_check --> acl
acl --> execute
execute --> response
response --> state_update
state_update --> loop
execute --> quit_check
quit_check --> cleanup
quit_check --> loop
```

Sources: [src/src/smtp_in.c L1800-L2000](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L1800-L2000)

 [src/src/smtp_in.c L3500-L4000](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L3500-L4000)

### Command Structure and Recognition

SMTP commands are defined in a structured table that supports efficient lookup and validation:

```mermaid
flowchart TD

cmd_list["cmd_list[]"]
rset["RSET_CMD"]
helo["HELO_CMD"]
ehlo["EHLO_CMD"]
mail["MAIL_CMD"]
rcpt["RCPT_CMD"]
data["DATA_CMD"]
auth["AUTH_CMD"]
starttls["STARTTLS_CMD"]
sync1["Synchronization required"]
args1["Has arguments"]
mail_cmd1["Mail command"]

rset --> sync1
mail --> args1
data --> mail_cmd1

subgraph subGraph1 ["Command Properties"]
    sync1
    args1
    mail_cmd1
end

subgraph subGraph0 ["Command Definition"]
    cmd_list
    rset
    helo
    ehlo
    mail
    rcpt
    data
    auth
    starttls
    cmd_list --> rset
    cmd_list --> helo
    cmd_list --> ehlo
    cmd_list --> mail
    cmd_list --> rcpt
    cmd_list --> data
    cmd_list --> auth
    cmd_list --> starttls
end
```

Sources: [src/src/smtp_in.c L195-L224](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L195-L224)

 [src/src/smtp_in.c L51-L110](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L51-L110)

The `smtp_cmd_list` structure defines each command's properties:

* `name`: Command string (e.g., "mail from:")
* `len`: String length for efficient comparison
* `cmd`: Command code enum value
* `has_arg`: Whether command accepts arguments
* `is_mail_cmd`: Whether command is part of mail transaction

## Message Data Reception

When a `DATA` or `BDAT` command is received, Exim transitions into message reception mode, collecting the message content and storing it in spool files.

### DATA Command Processing Flow

```mermaid
sequenceDiagram
  participant Client
  participant smtp_in.c
  participant receive.c
  participant Spool Files
  participant ACL Engine

  Client->>smtp_in.c: DATA command
  smtp_in.c->>ACL Engine: Run acl_smtp_predata
  ACL Engine->>smtp_in.c: ACL result
  smtp_in.c->>Client: 354 Start mail input
  Client->>smtp_in.c: Message lines
  smtp_in.c->>receive.c: receive_msg()
  receive.c->>Spool Files: Create -D data file
  receive.c->>Spool Files: Create -H header file
  loop [Message Lines]
    Client->>receive.c: Message line
    receive.c->>Spool Files: Write to data file
  end
  Client->>receive.c: . (end marker)
  receive.c->>ACL Engine: Run acl_smtp_data
  ACL Engine->>receive.c: ACL result
  receive.c->>smtp_in.c: Reception result
  smtp_in.c->>Client: 250 OK or error
```

Sources: [src/src/smtp_in.c L5500-L5800](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L5500-L5800)

 [src/src/receive.c L3000-L4000](https://github.com/Exim/exim/blob/29568b25/src/src/receive.c#L3000-L4000)

### Spool File Management

Exim creates two primary spool files during message reception:

| File Type | Purpose | Content |
| --- | --- | --- |
| `-H` file | Message headers and metadata | Envelope information, headers, ACL variables |
| `-D` file | Message body data | Raw message content as received |

The spool file creation process involves:

1. **File Creation**: Generate unique message ID and create spool files
2. **Header Processing**: Parse and validate message headers
3. **Data Storage**: Stream message body to data file with line ending normalization
4. **Metadata Recording**: Store envelope information and processing flags

## Session State Management

SMTP sessions maintain state across multiple commands to enforce protocol correctness and security policies.

### Session State Variables

```mermaid
flowchart TD

count_nonmail["count_nonmail"]
sync_errors["synprot_error_count"]
sender["sender_address"]
recipients["recipients_list"]
rcpt_count["rcpt_count"]
tls_adv["fl.tls_advertised"]
tls_active["tls_in.active"]
esmtp["fl.esmtp"]
helo_seen["fl.helo_seen"]
helo_verify["fl.helo_verify"]
auth_adv["fl.auth_advertised"]
auth_by["authenticated_by"]
auth_id["authenticated_id"]

subgraph subGraph4 ["Connection Control"]
    count_nonmail
    sync_errors
end

subgraph subGraph3 ["Transaction State"]
    sender
    recipients
    rcpt_count
end

subgraph subGraph2 ["TLS State"]
    tls_adv
    tls_active
end

subgraph subGraph1 ["Protocol State"]
    esmtp
    helo_seen
    helo_verify
end

subgraph subGraph0 ["Authentication State"]
    auth_adv
    auth_by
    auth_id
end
```

Sources: [src/src/smtp_in.c L125-L160](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L125-L160)

 [src/src/globals.h L800-L900](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h#L800-L900)

### Command Synchronization

Exim enforces SMTP pipelining rules and command synchronization to prevent protocol abuse:

* **Synchronizing Commands**: HELO, EHLO, DATA, STARTTLS require synchronization
* **Non-sync Commands**: MAIL, RCPT, RSET can be pipelined
* **Error Limits**: Track protocol violations with `smtp_max_synprot_errors`

## Integration with Access Control

SMTP command processing integrates closely with Exim's ACL system, providing multiple enforcement points throughout the session.

### ACL Integration Points

```mermaid
flowchart TD

connect["Connection"]
acl_connect["acl_smtp_connect"]
helo_cmd["HELO/EHLO"]
acl_helo["acl_smtp_helo"]
mail_cmd["MAIL FROM"]
acl_mail["acl_smtp_mail"]
rcpt_cmd["RCPT TO"]
acl_rcpt["acl_smtp_rcpt"]
more_rcpt["More RCPT?"]
data_cmd["DATA"]
acl_predata["acl_smtp_predata"]
message_data["Message Data"]
acl_data["acl_smtp_data"]
accept["Message Accepted"]

connect --> acl_connect
acl_connect --> helo_cmd
helo_cmd --> acl_helo
acl_helo --> mail_cmd
mail_cmd --> acl_mail
acl_mail --> rcpt_cmd
rcpt_cmd --> acl_rcpt
acl_rcpt --> more_rcpt
more_rcpt --> rcpt_cmd
more_rcpt --> data_cmd
data_cmd --> acl_predata
acl_predata --> message_data
message_data --> acl_data
acl_data --> accept
```

Sources: [src/src/smtp_in.c L4500-L5000](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L4500-L5000)

 [src/src/acl.c L3000-L3500](https://github.com/Exim/exim/blob/29568b25/src/src/acl.c#L3000-L3500)

Each ACL checkpoint can:

* **ACCEPT**: Continue processing
* **DENY**: Reject with SMTP error code
* **DEFER**: Temporary failure response
* **DROP**: Close connection immediately
* **WARN**: Log warning but continue

The ACL system provides access to extensive connection and message metadata through expansion variables like `$sender_host_address`, `$authenticated_id`, and `$message_size`.

Sources: [src/src/smtp_in.c L1-L4000](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L1-L4000)

 [src/src/receive.c L1-L4000](https://github.com/Exim/exim/blob/29568b25/src/src/receive.c#L1-L4000)

 [src/src/daemon.c L1-L2000](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L1-L2000)

 [src/src/acl.c L1-L3000](https://github.com/Exim/exim/blob/29568b25/src/src/acl.c#L1-L3000)
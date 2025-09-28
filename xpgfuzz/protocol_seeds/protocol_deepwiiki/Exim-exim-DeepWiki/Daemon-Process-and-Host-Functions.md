# Daemon Process and Host Functions

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
> * [src/src/host.c](https://github.com/Exim/exim/blob/29568b25/src/src/host.c)
> * [src/src/readconf.c](https://github.com/Exim/exim/blob/29568b25/src/src/readconf.c)
> * [src/src/receive.c](https://github.com/Exim/exim/blob/29568b25/src/src/receive.c)
> * [src/src/smtp_in.c](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c)
> * [src/src/structs.h](https://github.com/Exim/exim/blob/29568b25/src/src/structs.h)
> * [src/src/transport.c](https://github.com/Exim/exim/blob/29568b25/src/src/transport.c)
> * [src/src/transports/smtp.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/smtp.c)
> * [src/src/verify.c](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c)

This page documents Exim's daemon process management, host resolution mechanisms, and network interface handling. These foundational systems enable Exim to operate as a persistent mail server, resolve hostnames for mail routing, and manage network connections.

For information about DNS resolution specifics, see [DNS Resolution](/Exim/exim/5.5-dns-resolution). For details about command-line utilities and process control, see [Command-line Utilities](/Exim/exim/5.7-command-line-utilities).

## Daemon Process Architecture

Exim's daemon functionality centers around the `daemon_go()` function, which establishes the main server process and manages child processes for handling SMTP connections and queue operations.

### Daemon Process Flow

```mermaid
flowchart TD

main["main() in exim.c"]
daemon_go["daemon_go() in daemon.c"]
setup_signals["Setup Signal Handlers"]
create_sockets["Create Listening Sockets"]
main_loop["Main Event Loop"]
poll_events["poll() for Events"]
accept_conn["Accept SMTP Connection"]
handle_signals["Handle Signals"]
queue_run["Queue Runner Events"]
spawn_smtp["Spawn SMTP Child Process"]
smtp_slot["Store in smtp_slot Structure"]
spawn_runner["Spawn Queue Runner"]
runner_slot["Store in runner_slot Structure"]
sigchld["SIGCHLD - Child Termination"]
sighup["SIGHUP - Reload Config"]
sigterm["SIGTERM - Shutdown"]

main --> daemon_go
daemon_go --> setup_signals
daemon_go --> create_sockets
daemon_go --> main_loop
main_loop --> poll_events
poll_events --> accept_conn
poll_events --> handle_signals
poll_events --> queue_run
accept_conn --> spawn_smtp
spawn_smtp --> smtp_slot
queue_run --> spawn_runner
spawn_runner --> runner_slot
handle_signals --> sigchld
handle_signals --> sighup
handle_signals --> sigterm
```

The daemon maintains process tracking through two key data structures:

* `smtp_slot`: Tracks SMTP connection handler processes
* `runner_slot`: Tracks queue runner processes

Sources: [src/src/daemon.c L1-L1200](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L1-L1200)

 [src/src/exim.c L1-L6500](https://github.com/Exim/exim/blob/29568b25/src/src/exim.c#L1-L6500)

### Process Management Structures

```mermaid
classDiagram
    class smtp_slot {
        +pid_t pid
        +uschar* host_address
    }
    class runner_slot {
        +pid_t pid
        +uschar* queue_name
    }
    class daemon_state {
        +smtp_slot* smtp_slots
        +runner_slot* queue_runner_slots
        +SIGNAL_BOOL sigchld_seen
        +SIGNAL_BOOL sighup_seen
        +SIGNAL_BOOL sigterm_seen
        +int queue_run_count
    }
    daemon_state --> smtp_slot : manages array
    daemon_state --> runner_slot : manages array
```

Sources: [src/src/daemon.c L18-L31](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L18-L31)

 [src/src/daemon.c L37-L50](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L37-L50)

## Host Resolution System

Exim's host resolution system provides multiple mechanisms for converting hostnames to IP addresses, supporting both traditional gethostbyname() calls and direct DNS queries.

### Host Resolution Flow

```mermaid
flowchart TD

host_lookup["Host Lookup Request"]
host_find["host_find_byname() or host_find_bydns()"]
check_cache["Check Local Cache"]
cache_hit["Cache Hit"]
cache_miss["Cache Miss"]
dns_query["DNS Query"]
gethostbyname["gethostbyname() Call"]
dns_basic_lookup["dns_basic_lookup()"]
dns_response["Parse DNS Response"]
system_resolver["System Resolver"]
host_data["Host Data"]
store_cache["Update Cache"]
return_result["Return host_item List"]

host_lookup --> host_find
host_find --> check_cache
check_cache --> cache_hit
check_cache --> cache_miss
cache_miss --> dns_query
cache_miss --> gethostbyname
dns_query --> dns_basic_lookup
dns_basic_lookup --> dns_response
gethostbyname --> system_resolver
system_resolver --> host_data
cache_hit --> host_data
dns_response --> host_data
host_data --> store_cache
store_cache --> return_result
```

Sources: [src/src/host.c L1-L4500](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L1-L4500)

 [src/src/host.c L2500-L3000](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L2500-L3000)

### Host Data Structures

The host resolution system uses several key structures to manage host information:

```mermaid
classDiagram
    class host_item {
        +uschar* name
        +uschar* address
        +int port
        +int status
        +int why
        +host_item* next
    }
    class dns_answer {
        +uschar* data
        +int length
        +BOOL secure
    }
    class dns_record {
        +uschar* name
        +int type
        +int size
        +uschar* data
    }
    class ip_address_item {
        +uschar* address
        +int port
        +ip_address_item* next
    }
    dns_answer --> dns_record : "linked list"
```

Sources: [src/src/structs.h L200-L250](https://github.com/Exim/exim/blob/29568b25/src/src/structs.h#L200-L250)

 [src/src/structs.h L350-L400](https://github.com/Exim/exim/blob/29568b25/src/src/structs.h#L350-L400)

## Network Interface Management

Exim manages network interfaces for both listening sockets and outbound connections through a centralized interface discovery and management system.

### Interface Discovery and Management

```mermaid
flowchart TD

interface_init["Interface Initialization"]
scan_interfaces["Scan Available Interfaces"]
get_local_ips["Get Local IP Addresses"]
build_interface_list["Build ip_address_item List"]
local_interface_data["Store in local_interface_data"]
daemon_start["Daemon Start"]
create_listeners["Create Listening Sockets"]
bind_interfaces["Bind to Configured Interfaces"]
outbound_conn["Outbound Connection"]
select_interface["Select Source Interface"]
bind_source["Bind Source Address"]

interface_init --> scan_interfaces
scan_interfaces --> get_local_ips
get_local_ips --> build_interface_list
build_interface_list --> local_interface_data
daemon_start --> create_listeners
create_listeners --> bind_interfaces
outbound_conn --> select_interface
select_interface --> bind_source
local_interface_data --> bind_interfaces
local_interface_data --> select_interface
```

Sources: [src/src/host.c L25-L100](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L25-L100)

 [src/src/daemon.c L800-L1000](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L800-L1000)

### Connection State Management

```mermaid
stateDiagram-v2
    [*] --> Listening : "daemon_go()"
    Listening --> Accepting : "incoming connection"
    Accepting --> Connected : "accept()"
    Connected --> Processing : "spawn child process"
    Processing --> Completed : "child exits"
    Completed --> Listening : "SIGCHLD handled"
    Listening --> Shutdown : "accept()"
    Processing --> Shutdown : "SIGTERM"
    Shutdown --> [*] : "cleanup and exit"
    Listening --> Reload : "SIGHUP"
    Reload --> Listening : "config reloaded"
```

Sources: [src/src/daemon.c L1000-L1200](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L1000-L1200)

 [src/src/daemon.c L400-L600](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L400-L600)

## Integration with Core Systems

The daemon and host functions integrate closely with Exim's core mail processing systems:

### System Integration Architecture

```mermaid
flowchart TD

daemon_go["daemon_go()"]
smtp_slots["smtp_slot tracking"]
signal_handlers["Signal Handlers"]
host_find["host_find_byname()"]
dns_lookup["dns_basic_lookup()"]
host_cache["Host Cache"]
listen_sockets["Listening Sockets"]
interface_mgmt["Interface Management"]
connection_accept["Connection Accept"]
smtp_in["smtp_in.c"]
delivery["deliver.c"]
routing["Routing System"]

daemon_go --> listen_sockets
connection_accept --> smtp_in
smtp_in --> host_find
delivery --> host_find
routing --> dns_lookup

subgraph subGraph3 ["Core Mail Processing"]
    smtp_in
    delivery
    routing
end

subgraph subGraph2 ["Network Layer"]
    listen_sockets
    interface_mgmt
    connection_accept
end

subgraph subGraph1 ["Host Resolution"]
    host_find
    dns_lookup
    host_cache
    host_find --> host_cache
    dns_lookup --> host_cache
end

subgraph subGraph0 ["Daemon Layer"]
    daemon_go
    smtp_slots
    signal_handlers
end
```

Sources: [src/src/daemon.c L200-L400](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L200-L400)

 [src/src/host.c L1500-L2000](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L1500-L2000)

 [src/src/smtp_in.c L100-L300](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_in.c#L100-L300)

### Key Global Variables

The daemon and host systems rely on several global variables for state management:

| Variable | Type | Purpose |
| --- | --- | --- |
| `local_interface_data` | `ip_address_item*` | Cached local interface addresses |
| `daemon_listen_port` | `int` | Default listening port |
| `smtp_accept_max` | `int` | Maximum concurrent SMTP connections |
| `queue_run_max` | `int` | Maximum concurrent queue runners |
| `sigchld_seen` | `SIGNAL_BOOL` | Child process termination flag |
| `sighup_seen` | `SIGNAL_BOOL` | Configuration reload flag |

Sources: [src/src/globals.h L400-L600](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h#L400-L600)

 [src/src/globals.c L1000-L1200](https://github.com/Exim/exim/blob/29568b25/src/src/globals.c#L1000-L1200)

 [src/src/daemon.c L37-L50](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L37-L50)

## Error Handling and Recovery

The daemon implements robust error handling for network and process management failures:

### Error Recovery Mechanisms

```mermaid
flowchart TD

error_detected["Error Detected"]
error_type["Determine Error Type"]
network_error["Network Error"]
process_error["Process Error"]
config_error["Configuration Error"]
retry_accept["Retry Accept"]
log_network["Log Network Issue"]
cleanup_child["Cleanup Child Process"]
update_slots["Update Process Slots"]
reload_config["Reload Configuration"]
validate_settings["Validate Settings"]
backoff["Exponential Backoff"]
continue_operation["Continue Operation"]

error_detected --> error_type
error_type --> network_error
error_type --> process_error
error_type --> config_error
network_error --> retry_accept
network_error --> log_network
process_error --> cleanup_child
process_error --> update_slots
config_error --> reload_config
config_error --> validate_settings
retry_accept --> backoff
backoff --> continue_operation
cleanup_child --> continue_operation
reload_config --> continue_operation
```

Sources: [src/src/daemon.c L600-L800](https://github.com/Exim/exim/blob/29568b25/src/src/daemon.c#L600-L800)

 [src/src/host.c L3500-L4000](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L3500-L4000)
# Routing System

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
> * [test/stderr/0275](https://github.com/Exim/exim/blob/29568b25/test/stderr/0275)
> * [test/stderr/0278](https://github.com/Exim/exim/blob/29568b25/test/stderr/0278)
> * [test/stderr/0361](https://github.com/Exim/exim/blob/29568b25/test/stderr/0361)
> * [test/stderr/0386](https://github.com/Exim/exim/blob/29568b25/test/stderr/0386)
> * [test/stderr/0403](https://github.com/Exim/exim/blob/29568b25/test/stderr/0403)
> * [test/stderr/0408](https://github.com/Exim/exim/blob/29568b25/test/stderr/0408)
> * [test/stderr/0487](https://github.com/Exim/exim/blob/29568b25/test/stderr/0487)
> * [test/stderr/2600](https://github.com/Exim/exim/blob/29568b25/test/stderr/2600)
> * [test/stderr/2610](https://github.com/Exim/exim/blob/29568b25/test/stderr/2610)
> * [test/stderr/2620](https://github.com/Exim/exim/blob/29568b25/test/stderr/2620)
> * [test/stderr/5004](https://github.com/Exim/exim/blob/29568b25/test/stderr/5004)
> * [test/stderr/5005](https://github.com/Exim/exim/blob/29568b25/test/stderr/5005)
> * [test/stderr/5006](https://github.com/Exim/exim/blob/29568b25/test/stderr/5006)

## Purpose and Scope

The Routing System is responsible for determining how messages should be delivered by resolving recipient addresses and selecting appropriate transport mechanisms. This system processes incoming addresses through a series of configurable routers that perform address resolution, database lookups, and domain resolution to determine the final delivery method.

For information about message reception before routing, see [Message Reception and SMTP](/Exim/exim/2.1-message-reception-and-smtp). For details about the actual delivery mechanisms after routing, see [Transport and Delivery](/Exim/exim/2.3-transport-and-delivery). For access control during routing, see [Access Control Lists (ACLs)](/Exim/exim/2.4-access-control-lists-(acls)).

## Routing Architecture Overview

The routing system sits between message reception and transport delivery, acting as the decision engine for how messages reach their final destinations.

### Core Routing Components

```mermaid
flowchart TD

RECV["Message Reception<br>smtp_in.c"]
ROUTE["Routing System<br>deliver.c"]
TRANS["Transport Layer<br>transports/"]
EXPAND["String Expansion<br>expand.c"]
DNS["DNS Resolution<br>dns.c"]
DB["Database Lookups<br>Various lookup modules"]
VERIFY["Address Verification<br>verify.c"]
MANUAL["manualroute"]
DNSLOOK["dnslookup"]
REDIRECT["redirect"]
ACCEPT["accept"]

ROUTE --> EXPAND
ROUTE --> DNS
ROUTE --> DB
ROUTE --> VERIFY
ROUTE --> MANUAL
ROUTE --> DNSLOOK
ROUTE --> REDIRECT
ROUTE --> ACCEPT

subgraph subGraph2 ["Router Drivers"]
    MANUAL
    DNSLOOK
    REDIRECT
    ACCEPT
end

subgraph subGraph1 ["Routing Infrastructure"]
    EXPAND
    DNS
    DB
    VERIFY
    EXPAND --> DNS
    EXPAND --> DB
end

subgraph subGraph0 ["Message Processing Pipeline"]
    RECV
    ROUTE
    TRANS
    RECV --> ROUTE
    ROUTE --> TRANS
end
```

Sources: [src/src/deliver.c L1-L100](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L1-L100)

 [src/src/expand.c L1-L100](https://github.com/Exim/exim/blob/29568b25/src/src/expand.c#L1-L100)

 [src/src/dns.c](https://github.com/Exim/exim/blob/29568b25/src/src/dns.c)

 [src/src/verify.c L1-L100](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c#L1-L100)

## Address Resolution Process

The routing system processes addresses through a sequential chain of routers, each attempting to resolve the address until one succeeds or all fail.

### Address Processing Flow

```mermaid
sequenceDiagram
  participant deliver.c
  participant deliver_message()
  participant Router Instance
  participant expand.c
  participant String Expansion
  participant dns.c
  participant DNS Resolution
  participant verify.c
  participant Address Verification
  participant Transport Driver

  deliver.c->>Router Instance: "Process address_item"
  Router Instance->>expand.c: "Expand router options"
  expand.c-->>Router Instance: "Expanded values"
  Router Instance->>dns.c: "Resolve domain (if needed)"
  dns.c-->>Router Instance: "DNS records"
  Router Instance->>verify.c: "Verify address (if required)"
  verify.c-->>Router Instance: "Verification result"
  loop [Address Routed Successfully]
    Router Instance->>Transport Driver: "Assign transport"
    Router Instance-->>deliver.c: "ROUTED"
    Router Instance-->>deliver.c: "PASS/DECLINE"
    deliver.c->>Router Instance: "Try next router"
    Router Instance-->>deliver.c: "DEFER/FAIL"
  end
```

Sources: [src/src/deliver.c L7500-L7600](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L7500-L7600)

 [src/src/expand.c L4000-L4100](https://github.com/Exim/exim/blob/29568b25/src/src/expand.c#L4000-L4100)

 [src/src/verify.c L1000-L1100](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c#L1000-L1100)

## Router Types and Configuration

### Core Router Variables and Data Structures

The routing system maintains several key global variables and data structures for address processing:

```mermaid
flowchart TD

ADDR["address_item"]
DOMAIN["deliver_domain"]
LOCAL["deliver_localpart"]
DATA["address_data"]
RNAME["router_name"]
RVAR["router_var[]"]
SELF["self_hostname"]
HOST["host_list"]
TRANS["transport"]
ERRORS["errors_to"]

RVAR --> DATA
ADDR --> HOST
ADDR --> TRANS
ADDR --> ERRORS

subgraph subGraph2 ["Resolution Results"]
    HOST
    TRANS
    ERRORS
end

subgraph subGraph1 ["Router State"]
    RNAME
    RVAR
    SELF
    RNAME --> RVAR
end

subgraph subGraph0 ["Address Data Structure"]
    ADDR
    DOMAIN
    LOCAL
    DATA
    ADDR --> DOMAIN
    ADDR --> LOCAL
    ADDR --> DATA
end
```

Sources: [src/src/globals.h L600-L700](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h#L600-L700)

 [src/src/structs.h L800-L900](https://github.com/Exim/exim/blob/29568b25/src/src/structs.h#L800-L900)

 [src/src/deliver.c L173-L230](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L173-L230)

### Router Driver Integration

The router system uses a driver-based architecture where different router types handle specific routing scenarios:

| Router Type | Purpose | Key Functions |
| --- | --- | --- |
| `dnslookup` | DNS-based routing for remote domains | MX/A record resolution |
| `manualroute` | Explicit host specification | Static route configuration |
| `redirect` | Address redirection and forwarding | Alias expansion, forwarding |
| `accept` | Local delivery acceptance | Final routing decision |

Sources: [src/src/routers/](https://github.com/Exim/exim/blob/29568b25/src/src/routers/)

 [src/src/deliver.c L5000-L5100](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L5000-L5100)

## Database Integration for Routing

### Routing Database Lookups

The routing system integrates with various database backends for address resolution and routing decisions:

```mermaid
flowchart TD

ROUTER["Router Instance"]
EXPAND["String Expansion"]
HINTS["Hints Database<br>dbfn.c"]
MYSQL["MySQL Lookup<br>mysql.c"]
PGSQL["PostgreSQL Lookup<br>pgsql.c"]
LDAP["LDAP Lookup<br>ldap.c"]
SQLITE["SQLite Lookup<br>sqlite.c"]
DOMAIN["Domain Lookups"]
LOCAL["Local Part Lookups"]
ADDRESS["Address Lookups"]
ALIAS["Alias Resolution"]

EXPAND --> HINTS
EXPAND --> MYSQL
EXPAND --> PGSQL
EXPAND --> LDAP
EXPAND --> SQLITE
HINTS --> DOMAIN
MYSQL --> LOCAL
PGSQL --> ADDRESS
LDAP --> ALIAS
SQLITE --> DOMAIN

subgraph subGraph2 ["Lookup Types"]
    DOMAIN
    LOCAL
    ADDRESS
    ALIAS
end

subgraph subGraph1 ["Database Backends"]
    HINTS
    MYSQL
    PGSQL
    LDAP
    SQLITE
end

subgraph subGraph0 ["Router Processing"]
    ROUTER
    EXPAND
    ROUTER --> EXPAND
end
```

Sources: [src/src/expand.c L3000-L3200](https://github.com/Exim/exim/blob/29568b25/src/src/expand.c#L3000-L3200)

 [src/src/lookups/](https://github.com/Exim/exim/blob/29568b25/src/src/lookups/)

 [test/stderr/2610 L1-L20](https://github.com/Exim/exim/blob/29568b25/test/stderr/2610#L1-L20)

 [test/stderr/2620 L1-L20](https://github.com/Exim/exim/blob/29568b25/test/stderr/2620#L1-L20)

## DNS Resolution in Routing

### DNS Integration Architecture

The routing system heavily relies on DNS resolution for remote domain routing:

```mermaid
flowchart TD

DNSINIT["dns_init()"]
DNSLOOKUP["dns_lookup()"]
DNSBASIC["dns_basic_lookup()"]
DNSSPECIAL["dns_special_lookup()"]
MX["MX Records"]
A["A Records"]
AAAA["AAAA Records"]
CNAME["CNAME Records"]
TXT["TXT Records"]
DNSLOOK_ROUTER["dnslookup router"]
MANUALROUTE_ROUTER["manualroute router"]
VERIFY_CALLOUT["Verification callouts"]

DNSBASIC --> MX
DNSBASIC --> A
DNSBASIC --> AAAA
DNSBASIC --> CNAME
DNSBASIC --> TXT
MX --> DNSLOOK_ROUTER
A --> MANUALROUTE_ROUTER
MX --> VERIFY_CALLOUT

subgraph subGraph2 ["Router Usage"]
    DNSLOOK_ROUTER
    MANUALROUTE_ROUTER
    VERIFY_CALLOUT
end

subgraph subGraph1 ["DNS Record Types"]
    MX
    A
    AAAA
    CNAME
    TXT
end

subgraph subGraph0 ["DNS Resolution Layer"]
    DNSINIT
    DNSLOOKUP
    DNSBASIC
    DNSSPECIAL
    DNSINIT --> DNSLOOKUP
    DNSLOOKUP --> DNSBASIC
    DNSLOOKUP --> DNSSPECIAL
end
```

Sources: [src/src/dns.c L200-L300](https://github.com/Exim/exim/blob/29568b25/src/src/dns.c#L200-L300)

 [src/src/functions.h L205-L220](https://github.com/Exim/exim/blob/29568b25/src/src/functions.h#L205-L220)

 [src/src/verify.c L800-L900](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c#L800-L900)

## Router Configuration and Expansion

### String Expansion in Routing Context

The routing system extensively uses string expansion for dynamic configuration:

```mermaid
flowchart TD

CONDITION["condition"]
DOMAINS["domains"]
LOCALPARTS["local_parts"]
TRANSPORT["transport"]
ROUTE_LIST["route_list"]
DOMAIN_VAR["$domain"]
LOCAL_VAR["$local_part"]
ADDRESS_VAR["$address"]
ROUTER_VAR["$router_name"]
DATA_VAR["$address_data"]
EXPAND_STRING["expand_string()"]
EXPAND_CONDITION["expand_check_condition()"]
VARIABLE_SET["modify_variable()"]

CONDITION --> EXPAND_CONDITION
DOMAINS --> EXPAND_STRING
LOCALPARTS --> EXPAND_STRING
TRANSPORT --> EXPAND_STRING
ROUTE_LIST --> EXPAND_STRING
EXPAND_STRING --> DOMAIN_VAR
EXPAND_STRING --> LOCAL_VAR
EXPAND_STRING --> ADDRESS_VAR
EXPAND_STRING --> ROUTER_VAR
EXPAND_STRING --> DATA_VAR

subgraph subGraph2 ["Expansion Engine"]
    EXPAND_STRING
    EXPAND_CONDITION
    VARIABLE_SET
    EXPAND_CONDITION --> VARIABLE_SET
end

subgraph subGraph1 ["Expansion Variables"]
    DOMAIN_VAR
    LOCAL_VAR
    ADDRESS_VAR
    ROUTER_VAR
    DATA_VAR
end

subgraph subGraph0 ["Router Configuration"]
    CONDITION
    DOMAINS
    LOCALPARTS
    TRANSPORT
    ROUTE_LIST
end
```

Sources: [src/src/expand.c L4500-L4700](https://github.com/Exim/exim/blob/29568b25/src/src/expand.c#L4500-L4700)

 [src/src/readconf.c L3000-L3200](https://github.com/Exim/exim/blob/29568b25/src/src/readconf.c#L3000-L3200)

 [src/src/globals.h L1200-L1300](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h#L1200-L1300)

## Integration with Transport System

### Router to Transport Handoff

The routing system determines which transport will handle the final delivery:

```mermaid
flowchart TD

ROUTER_SUCCESS["Router Success"]
TRANSPORT_ASSIGN["Transport Assignment"]
ADDRESS_PREPARE["Address Preparation"]
SMTP_TRANSPORT["SMTP Transport<br>smtp.c"]
APPENDFILE_TRANSPORT["Appendfile Transport<br>appendfile.c"]
PIPE_TRANSPORT["Pipe Transport<br>pipe.c"]
AUTOREPLY_TRANSPORT["Autoreply Transport<br>autoreply.c"]
ADDR_REMOTE["addr_remote"]
ADDR_LOCAL["addr_local"]
ADDR_PIPE["addr_pipe"]
ADDR_REPLY["addr_reply"]

ADDRESS_PREPARE --> SMTP_TRANSPORT
ADDRESS_PREPARE --> APPENDFILE_TRANSPORT
ADDRESS_PREPARE --> PIPE_TRANSPORT
ADDRESS_PREPARE --> AUTOREPLY_TRANSPORT
SMTP_TRANSPORT --> ADDR_REMOTE
APPENDFILE_TRANSPORT --> ADDR_LOCAL
PIPE_TRANSPORT --> ADDR_PIPE
AUTOREPLY_TRANSPORT --> ADDR_REPLY

subgraph subGraph2 ["Address Classification"]
    ADDR_REMOTE
    ADDR_LOCAL
    ADDR_PIPE
    ADDR_REPLY
end

subgraph subGraph1 ["Transport Types"]
    SMTP_TRANSPORT
    APPENDFILE_TRANSPORT
    PIPE_TRANSPORT
    AUTOREPLY_TRANSPORT
end

subgraph subGraph0 ["Routing Decision Points"]
    ROUTER_SUCCESS
    TRANSPORT_ASSIGN
    ADDRESS_PREPARE
    ROUTER_SUCCESS --> TRANSPORT_ASSIGN
    TRANSPORT_ASSIGN --> ADDRESS_PREPARE
end
```

Sources: [src/src/deliver.c L1000-L1200](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L1000-L1200)

 [src/src/transport.c L1-L100](https://github.com/Exim/exim/blob/29568b25/src/src/transport.c#L1-L100)

 [src/src/transports/smtp.c L1-L100](https://github.com/Exim/exim/blob/29568b25/src/src/transports/smtp.c#L1-L100)

## Router Processing Implementation

### Core Routing Functions

The main routing logic is implemented in several key functions within the delivery system:

| Function | Purpose | File Location |
| --- | --- | --- |
| `deliver_message()` | Main message delivery coordinator | [src/src/deliver.c L8000-L9000](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L8000-L9000) |
| `deliver_split_address()` | Address parsing and preparation | [src/src/deliver.c L4500-L4600](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L4500-L4600) |
| `route_address()` | Core address routing logic | Router-specific files |
| `verify_address()` | Address verification during routing | [src/src/verify.c L500-L600](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c#L500-L600) |

The routing system maintains several address lists during processing:

* `addr_new` - Newly discovered addresses
* `addr_route` - Addresses awaiting routing
* `addr_local` - Addresses routed for local delivery
* `addr_remote` - Addresses routed for remote delivery
* `addr_defer` - Addresses with deferred routing
* `addr_failed` - Addresses with failed routing

Sources: [src/src/deliver.c L63-L71](https://github.com/Exim/exim/blob/29568b25/src/src/deliver.c#L63-L71)

 [src/src/globals.h L500-L600](https://github.com/Exim/exim/blob/29568b25/src/src/globals.h#L500-L600)

 [src/src/verify.c L100-L200](https://github.com/Exim/exim/blob/29568b25/src/src/verify.c#L100-L200)
# Infrastructure and Utilities

> **Relevant source files**
> * [src/exim_monitor/em_hdr.h](https://github.com/Exim/exim/blob/29568b25/src/exim_monitor/em_hdr.h)
> * [src/exim_monitor/em_log.c](https://github.com/Exim/exim/blob/29568b25/src/exim_monitor/em_log.c)
> * [src/src/host.c](https://github.com/Exim/exim/blob/29568b25/src/src/host.c)
> * [src/src/local_scan.h](https://github.com/Exim/exim/blob/29568b25/src/src/local_scan.h)
> * [src/src/mytypes.h](https://github.com/Exim/exim/blob/29568b25/src/src/mytypes.h)
> * [src/src/store.c](https://github.com/Exim/exim/blob/29568b25/src/src/store.c)
> * [src/src/store.h](https://github.com/Exim/exim/blob/29568b25/src/src/store.h)
> * [src/src/string.c](https://github.com/Exim/exim/blob/29568b25/src/src/string.c)

This section covers the foundational systems that support Exim's core functionality. These include memory management, string processing, host resolution, type definitions, and utility interfaces that enable the mail processing pipeline to operate effectively.

For information about the core mail processing systems that use these utilities, see [Core Mail Processing](/Exim/exim/2-core-mail-processing). For details about the command-line utilities and administration tools, see [Command-line Utilities](/Exim/exim/5.7-command-line-utilities).

## Memory Management System

Exim implements a sophisticated pool-based memory allocation system that provides both performance optimization and security through taint tracking. The system manages multiple memory pools with different lifetimes and security characteristics.

### Pool Architecture

```mermaid
flowchart TD

MAIN["POOL_MAIN<br>Short-lived blocks"]
PERM["POOL_PERM<br>Long-lived blocks"]
CONFIG["POOL_CONFIG<br>Configuration data"]
SEARCH["POOL_SEARCH<br>Lookup storage"]
MESSAGE["POOL_MESSAGE<br>Medium-lifetime objects"]
TMAIN["POOL_TAINT_MAIN<br>Tainted short-lived"]
TPERM["POOL_TAINT_PERM<br>Tainted long-lived"]
TCONFIG["POOL_TAINT_CONFIG<br>Tainted config"]
TSEARCH["POOL_TAINT_SEARCH<br>Tainted lookups"]
TMESSAGE["POOL_TAINT_MESSAGE<br>Tainted medium-lifetime"]
QUOTED["Dynamic Quoted Pools<br>Lookup-specific quoting"]
GET["store_get_3()"]
RESET["store_reset_3()"]
MARK["store_mark_3()"]
EXTEND["store_extend_3()"]

GET --> MAIN
GET --> PERM
GET --> TMAIN
GET --> TPERM
GET --> QUOTED

subgraph subGraph3 ["Core Functions"]
    GET
    RESET
    MARK
    EXTEND
    RESET --> MARK
    EXTEND --> GET
end

subgraph subGraph2 ["Quoted Pools"]
    QUOTED
end

subgraph subGraph1 ["Tainted Pools"]
    TMAIN
    TPERM
    TCONFIG
    TSEARCH
    TMESSAGE
end

subgraph subGraph0 ["Untainted Pools"]
    MAIN
    PERM
    CONFIG
    SEARCH
    MESSAGE
end
```

The memory system uses a `pooldesc` structure for each pool, containing chain management, allocation tracking, and statistics:

* `chainbase`: List of memory blocks in the pool
* `current_block`: Active block with free space
* `next_yield`: Next allocation point
* `yield_length`: Remaining space in current block
* `store_block_order`: Log₂ of block allocation size

### Taint Tracking

Exim's taint tracking system prevents untrusted data from being used in security-sensitive contexts. Functions like `is_tainted_fn()` determine whether memory contains tainted data, and `die_tainted()` terminates execution when taint violations occur.

```mermaid
flowchart TD

INPUT["Untrusted Input"]
TAINTED["Tainted Memory"]
TRUSTED["Trusted Data"]
UNTAINTED["Untainted Memory"]
CHECK["is_tainted_fn()"]
SAFE["Safe Operations"]
VIOLATION["Taint Violation"]
DIE["die_tainted()"]

INPUT --> TAINTED
TRUSTED --> UNTAINTED
TAINTED --> CHECK
UNTAINTED --> CHECK
CHECK --> SAFE
CHECK --> VIOLATION
VIOLATION --> DIE
```

Sources: [src/src/store.c L1-L865](https://github.com/Exim/exim/blob/29568b25/src/src/store.c#L1-L865)

 [src/src/store.h L1-L93](https://github.com/Exim/exim/blob/29568b25/src/src/store.h#L1-L93)

## String Processing Utilities

The string processing system provides comprehensive text manipulation capabilities with security-aware operations and specialized formatting functions.

### Core String Functions

```mermaid
flowchart TD

STRCMPIC["strcmpic()"]
STRNCMPIC["strncmpic()"]
STRSTRIC["strstric()"]
SPRINTF["string_sprintf_trc()"]
COPY["string_copy()"]
COPYN["string_copyn()"]
FORMAT["string_format_trc()"]
IPADDR["string_is_ip_address()"]
PRINT["string_printing2()"]
UNPRINT["string_unprinting()"]
CATN["string_catn()"]
APPEND["string_append()"]
DEQUOTE["string_dequote()"]
NEXTLIST["string_nextinlist_trc()"]
BASE62["string_base62_32()/64()"]
ESCAPE["string_interpret_escape()"]
FORMATSIZE["string_format_size()"]
DNSDOMAIN["string_copy_dnsdomain()"]

SPRINTF --> CATN
FORMAT --> CATN
COPY --> CATN
ESCAPE --> UNPRINT

subgraph subGraph4 ["Specialized Functions"]
    BASE62
    ESCAPE
    FORMATSIZE
    DNSDOMAIN
end

subgraph subGraph2 ["String Manipulation"]
    CATN
    APPEND
    DEQUOTE
    NEXTLIST
    NEXTLIST --> DEQUOTE
end

subgraph subGraph1 ["String Validation"]
    IPADDR
    PRINT
    UNPRINT
end

subgraph subGraph0 ["String Creation"]
    SPRINTF
    COPY
    COPYN
    FORMAT
end

subgraph subGraph3 ["String Comparison"]
    STRCMPIC
    STRNCMPIC
    STRSTRIC
end
```

### Growable String System

The `gstring` structure enables efficient string building through dynamic memory management:

```python
typedef struct gstring {
    int size;     // Buffer size
    int ptr;      // Current position
    uschar *s;    // String buffer
} gstring;
```

Functions like `string_catn()` handle buffer growth automatically, using `gstring_grow()` to expand capacity when needed. The system optimizes for common cases while handling edge cases safely.

### List Processing

The `string_nextinlist_trc()` function implements sophisticated list parsing with configurable separators, quoted elements, and taint-aware memory allocation. It supports dynamic separator detection through the `<char` syntax and handles escape sequences within quoted strings.

Sources: [src/src/string.c L1-L2457](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1-L2457)

## Host and Network Functions

The host and network subsystem provides name resolution, interface management, and network address manipulation capabilities.

### Host Resolution Architecture

```mermaid
flowchart TD

BUILDHOST["host_build_hostlist()"]
RANDOM["random_number()"]
GETPORT["host_item_get_port()"]
BUILDIF["host_build_ifacelist()"]
FINDIF["host_find_interfaces()"]
ADDUNIQUE["add_unique_interface()"]
NTOA["host_ntoa()"]
ATON["host_aton()"]
NMTOA["host_nmtoa()"]
FAKE["host_fake_gethostbyname()"]
LOOKUP["dns_lookup_timerwrap()"]
TIMER["get_time_in_ms()"]
SENDERFULL["host_build_sender_fullhost()"]
HOSTIDENT["host_and_ident()"]

NTOA --> SENDERFULL
ATON --> FAKE

subgraph subGraph4 ["Display Functions"]
    SENDERFULL
    HOSTIDENT
end

subgraph subGraph1 ["Host Resolution"]
    FAKE
    LOOKUP
    TIMER
    LOOKUP --> TIMER
    FAKE --> LOOKUP
end

subgraph subGraph0 ["Address Conversion"]
    NTOA
    ATON
    NMTOA
end

subgraph subGraph3 ["Host List Management"]
    BUILDHOST
    RANDOM
    GETPORT
    BUILDHOST --> RANDOM
end

subgraph subGraph2 ["Interface Management"]
    BUILDIF
    FINDIF
    ADDUNIQUE
    BUILDIF --> FINDIF
    FINDIF --> ADDUNIQUE
end
```

### Network Address Processing

The system handles both IPv4 and IPv6 addresses through functions like:

* `host_aton()`: Converts textual addresses to binary format in host byte order
* `host_ntoa()`: Converts binary addresses to textual format
* `string_is_ip_address()`: Validates IP address format with optional netmask parsing

The `host_item` structure represents hosts in lists with fields for name, address, port, MX priority, and status information.

### Interface Discovery

The `host_find_interfaces()` function discovers local network interfaces by:

1. Building interface lists from configuration (`local_interfaces`, `extra_local_interfaces`)
2. Resolving wildcard addresses (0.0.0.0, ::0) to actual interface addresses
3. Removing duplicates while preserving port specifications
4. Caching results for performance

Sources: [src/src/host.c L1-L2075](https://github.com/Exim/exim/blob/29568b25/src/src/host.c#L1-L2075)

## Basic Types and Utilities

The type system provides portable abstractions and utility macros that enable consistent coding practices across the codebase.

### Type Definitions

```mermaid
flowchart TD

USCHAR["uschar (unsigned char)"]
BOOL["BOOL (unsigned)"]
RMARK["rmark (void**)"]
CS["CS (char*)"]
CCS["CCS (const char*)"]
US["US (unsigned char*)"]
CUS["CUS (const unsigned char*)"]
USTR["Ustr* functions"]
UFILE["Ufile* functions"]
UMEM["Umem* functions"]

USCHAR --> CS
USCHAR --> US
CS --> USTR
US --> USTR

subgraph subGraph2 ["Library Wrappers"]
    USTR
    UFILE
    UMEM
    USTR --> UFILE
end

subgraph subGraph1 ["Casting Macros"]
    CS
    CCS
    US
    CUS
end

subgraph subGraph0 ["Basic Types"]
    USCHAR
    BOOL
    RMARK
end
```

### Compiler Attributes

The system defines portable compiler attribute macros:

* `PRINTF_FUNCTION(A,B)`: Printf-style format checking (disabled)
* `ARG_UNUSED`: Mark unused parameters
* `WARN_UNUSED_RESULT`: Warn on ignored return values
* `ALLOC` and `ALLOC_SIZE(A)`: Memory allocation annotations
* `NORETURN`: Functions that don't return

### Library Function Wrappers

Macro wrappers like `Uatoi()`, `Ufopen()`, `Ustrlen()` provide type-safe interfaces to standard library functions while handling the `uschar`/`char` distinction consistently.

Sources: [src/src/mytypes.h L1-L157](https://github.com/Exim/exim/blob/29568b25/src/src/mytypes.h#L1-L157)

 [src/src/local_scan.h L1-L254](https://github.com/Exim/exim/blob/29568b25/src/src/local_scan.h#L1-L254)

## Local Scan and Dynamic Loading Interface

The local scan interface provides a stable ABI for extending Exim with custom message processing logic and dynamic function loading.

### Interface Architecture

```mermaid
flowchart TD

LOCALSCAN["local_scan()"]
DLFUNC["exim_dlfunc_t"]
ACCEPT["LOCAL_SCAN_ACCEPT"]
FREEZE["LOCAL_SCAN_ACCEPT_FREEZE"]
QUEUE["LOCAL_SCAN_ACCEPT_QUEUE"]
REJECT["LOCAL_SCAN_REJECT"]
TEMPREJECT["LOCAL_SCAN_TEMPREJECT"]
HEADER["header_line"]
RECIPIENT["recipient_item"]
OPTION["optionlist"]
EXPAND["expand_string()"]
HEADEROPS["header_add()/remove()"]
MATCH["lss_match_*()"]
B64["lss_b64encode()/decode()"]

LOCALSCAN --> ACCEPT
LOCALSCAN --> REJECT
DLFUNC --> ACCEPT
LOCALSCAN --> HEADER
LOCALSCAN --> RECIPIENT
LOCALSCAN --> EXPAND
LOCALSCAN --> HEADEROPS

subgraph subGraph3 ["Utility Functions"]
    EXPAND
    HEADEROPS
    MATCH
    B64
end

subgraph subGraph2 ["Data Structures"]
    HEADER
    RECIPIENT
    OPTION
end

subgraph subGraph1 ["Return Codes"]
    ACCEPT
    FREEZE
    QUEUE
    REJECT
    TEMPREJECT
end

subgraph subGraph0 ["Entry Points"]
    LOCALSCAN
    DLFUNC
end
```

### Data Structures

The interface exposes key structures for message processing:

* `header_line`: Linked list of message headers with type classification
* `recipient_item`: Recipient addresses with DSN flags and error routing
* `optionlist`: Configuration option definitions with type information

### Global Variables

Key global variables accessible to local_scan functions include:

* `message_id`: Current message identifier
* `sender_address`: Envelope sender
* `recipients_list`: Array of recipients
* `header_list`: Message headers
* `interface_address`: Connection interface

Sources: [src/src/local_scan.h L25-L254](https://github.com/Exim/exim/blob/29568b25/src/src/local_scan.h#L25-L254)

## Monitoring and Debugging Infrastructure

The Exim Monitor provides real-time visualization of mail system activity through specialized data structures and log processing capabilities.

### Monitor Architecture

```mermaid
flowchart TD

QUEUE["queue_item"]
DEST["dest_item"]
SKIP["skip_item"]
PIPE["pipe_item"]
READLOG["read_log()"]
SHOWLOG["show_log()"]
LOGROTATE["Log rotation detection"]
FINDQUEUE["find_queue()"]
FINDDEST["find_dest()"]
QUEUEDISPLAY["queue_display()"]
LOGWIDGET["log_widget"]
QUEUEWIDGET["queue_widget"]
STRIPCHART["Stripcharts"]

FINDQUEUE --> QUEUE
FINDDEST --> DEST
READLOG --> QUEUEDISPLAY
SHOWLOG --> LOGWIDGET

subgraph subGraph3 ["Display Components"]
    LOGWIDGET
    QUEUEWIDGET
    STRIPCHART
end

subgraph subGraph2 ["Queue Management"]
    FINDQUEUE
    FINDDEST
    QUEUEDISPLAY
end

subgraph subGraph1 ["Log Processing"]
    READLOG
    SHOWLOG
    LOGROTATE
    READLOG --> SHOWLOG
end

subgraph subGraph0 ["Data Structures"]
    QUEUE
    DEST
    SKIP
    PIPE
    QUEUE --> DEST
end
```

### Queue Item Management

The `queue_item` structure tracks messages with fields for:

* Message identification (`name`, `dir_char`)
* Timing information (`input_time`, `update_time`)
* Status flags (`seen`, `frozen`)
* Destination list (`destinations`)

The monitor maintains linked lists of queue items and performs efficient lookups using message IDs and state tracking.

Sources: [src/exim_monitor/em_hdr.h L1-L332](https://github.com/Exim/exim/blob/29568b25/src/exim_monitor/em_hdr.h#L1-L332)

 [src/exim_monitor/em_log.c L1-L413](https://github.com/Exim/exim/blob/29568b25/src/exim_monitor/em_log.c#L1-L413)
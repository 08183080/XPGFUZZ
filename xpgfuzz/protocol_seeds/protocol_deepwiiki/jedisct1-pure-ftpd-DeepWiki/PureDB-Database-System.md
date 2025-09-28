# PureDB Database System

> **Relevant source files**
> * [puredb/src/puredb_p.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_p.h)
> * [puredb/src/puredb_read.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c)
> * [puredb/src/puredb_read.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.h)
> * [puredb/src/puredb_write.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c)
> * [puredb/src/puredb_write.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.h)

This document covers the PureDB embedded key-value database system used by Pure-FTPd for storing virtual user information and configuration data. PureDB provides a lightweight, hash-indexed database with optimized read performance through memory mapping and binary search capabilities.

For information about virtual user management using PureDB, see [Virtual Users with PureDB](/jedisct1/pure-ftpd/4.1-virtual-users-with-puredb). For programming examples and API usage, see [PureDB API and Examples](/jedisct1/pure-ftpd/7.1-puredb-api-and-examples).

## Database Architecture

PureDB is a read-optimized, hash-indexed key-value database designed for fast lookups with minimal memory overhead. The system consists of separate read and write components that work together to create and query database files.

### Core Components

```

```

Sources: [puredb/src/puredb_read.h L23-L27](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.h#L23-L27)

 [puredb/src/puredb_write.h L33-L42](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.h#L33-L42)

The database uses a two-level hash table structure with 256 primary buckets, each containing sorted lists of hash collisions for efficient binary search operations.

## File Format and Structure

PureDB files use the "PDB2" format with a specific binary layout optimized for sequential access and memory mapping.

### Database File Layout

```

```

Sources: [puredb/src/puredb_read.c L142-L147](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L142-L147)

 [puredb/src/puredb_write.c L68-L77](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L68-L77)

| Component | Size | Description |
| --- | --- | --- |
| Version Header | 4 bytes | "PDB2" magic string |
| Hash Table 0 | 256 × 4 bytes | Offsets to Hash Table 1 entries |
| Hash Table 0 End | 4 bytes | End offset marker |
| Hash Table 1 | Variable | Sorted hash/offset pairs |
| Data Section | Variable | Key-value pairs with length prefixes |

## Hash-Based Indexing System

The indexing system uses a djb2-based hash function with a two-level lookup structure for optimal performance.

### Hash Function Implementation

The hash function is implemented in both read and write modules with identical logic:

```mermaid
flowchart TD

INPUT["Key String"]
INIT["j = 5381"]
LOOP["For each byte (reverse order)"]
SHIFT["j += (j << 5)"]
XOR["j ^= byte"]
MASK["j &= 0xffffffff"]
BUCKET["Primary Bucket = hash & 0xff"]

subgraph subGraph0 ["Hash Calculation Process"]
    INPUT
    INIT
    LOOP
    SHIFT
    XOR
    MASK
    BUCKET
    INPUT --> INIT
    INIT --> LOOP
    LOOP --> SHIFT
    SHIFT --> XOR
    XOR --> MASK
    MASK --> LOOP
    MASK --> BUCKET
end
```

Sources: [puredb/src/puredb_read.c L9-L21](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L9-L21)

 [puredb/src/puredb_write.c L26-L38](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L26-L38)

### Two-Level Hash Table Structure

```mermaid
flowchart TD

KEYHASH["Key Hash Value"]
PRIMARY["Primary Hash (hash & 0xff)"]
BUCKET["Hash Table 0 Bucket"]
OFFSET["Hash Table 1 Offset"]
ENTRIES["Sorted Hash Table 1 Entries"]
DATAPTR["Data Pointer"]
FULLHASH["Full Hash (32-bit)"]
DATOFFSET["Data Offset (32-bit)"]

ENTRIES --> FULLHASH
ENTRIES --> DATOFFSET

subgraph subGraph1 ["Hash Table 1 Entry"]
    FULLHASH
    DATOFFSET
end

subgraph subGraph0 ["Hash Table Organization"]
    KEYHASH
    PRIMARY
    BUCKET
    OFFSET
    ENTRIES
    DATAPTR
    KEYHASH --> PRIMARY
    PRIMARY --> BUCKET
    BUCKET --> OFFSET
    OFFSET --> ENTRIES
    ENTRIES --> DATAPTR
end
```

Sources: [puredb/src/puredb_read.c L155-L194](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L155-L194)

 [puredb/src/puredb_write.h L23-L31](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.h#L23-L31)

## Memory Management and I/O

PureDB supports multiple I/O strategies depending on platform capabilities and file size.

### Memory Mapping Support

| Platform | Implementation | Macro |
| --- | --- | --- |
| Unix/Linux | `mmap()` | `HAVE_MMAP` |
| Windows | `MapViewOfFile()` | `HAVE_MAPVIEWOFFILE` |
| Fallback | `read()` syscalls | None |

```mermaid
flowchart TD

OPEN["puredb_open()"]
MMAP_CHECK["Memory mapping available?"]
MMAP_CREATE["Create memory mapping"]
USE_READ["Use read() syscalls"]
MMAP_SUCCESS["Mapping successful?"]
USE_MMAP["Use mapped I/O"]
FALLBACK["Set map = NULL"]

subgraph subGraph0 ["I/O Strategy Selection"]
    OPEN
    MMAP_CHECK
    MMAP_CREATE
    USE_READ
    MMAP_SUCCESS
    USE_MMAP
    FALLBACK
    OPEN --> MMAP_CHECK
    MMAP_CHECK --> MMAP_CREATE
    MMAP_CHECK --> USE_READ
    MMAP_CREATE --> MMAP_SUCCESS
    MMAP_SUCCESS --> USE_MMAP
    MMAP_SUCCESS --> FALLBACK
    FALLBACK --> USE_READ
end
```

Sources: [puredb/src/puredb_read.c L119-L141](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L119-L141)

 [puredb/src/puredb_p.h L83-L85](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_p.h#L83-L85)

### Safe I/O Operations

The `safe_read()` function handles interrupted system calls and partial reads:

```mermaid
flowchart TD

START["safe_read()"]
READ_CALL["read() syscall"]
EINTR_CHECK["errno == EINTR?"]
RETRY["Retry read()"]
PARTIAL_CHECK["Partial read?"]
UPDATE_BUFFER["Update buffer pointer"]
CONTINUE["More data needed?"]
COMPLETE["Return total bytes read"]

subgraph subGraph0 ["Safe Read Implementation"]
    START
    READ_CALL
    EINTR_CHECK
    RETRY
    PARTIAL_CHECK
    UPDATE_BUFFER
    CONTINUE
    COMPLETE
    START --> READ_CALL
    READ_CALL --> EINTR_CHECK
    EINTR_CHECK --> RETRY
    RETRY --> READ_CALL
    EINTR_CHECK --> PARTIAL_CHECK
    PARTIAL_CHECK --> UPDATE_BUFFER
    UPDATE_BUFFER --> CONTINUE
    CONTINUE --> READ_CALL
    CONTINUE --> COMPLETE
    PARTIAL_CHECK --> COMPLETE
end
```

Sources: [puredb/src/puredb_read.c L23-L42](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L23-L42)

## Read Operations

The read system provides efficient key lookup through binary search within hash buckets.

### Key Lookup Process

```mermaid
flowchart TD

FIND_START["puredb_find()"]
CALC_HASH["Calculate key hash"]
GET_BUCKET["Get Hash Table 0 bucket"]
READ_OFFSETS["Read Hash Table 1 offsets"]
BINARY_SEARCH["Binary search on hash values"]
KEY_COMPARE["Compare actual key"]
DATA_LOCATION["Locate data section"]
READ_DATA["Read key-value pair"]
RETURN_RESULT["Return offset and length"]

subgraph subGraph0 ["puredb_find() Lookup Flow"]
    FIND_START
    CALC_HASH
    GET_BUCKET
    READ_OFFSETS
    BINARY_SEARCH
    KEY_COMPARE
    DATA_LOCATION
    READ_DATA
    RETURN_RESULT
    FIND_START --> CALC_HASH
    CALC_HASH --> GET_BUCKET
    GET_BUCKET --> READ_OFFSETS
    READ_OFFSETS --> BINARY_SEARCH
    BINARY_SEARCH --> KEY_COMPARE
    KEY_COMPARE --> DATA_LOCATION
    DATA_LOCATION --> READ_DATA
    READ_DATA --> RETURN_RESULT
end
```

Sources: [puredb/src/puredb_read.c L152-L284](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L152-L284)

### Binary Search Optimization

When not compiled with `MINIMAL` or `NO_BINARY_LOOKUP`, PureDB uses binary search to quickly locate matching hash values:

| Search Phase | Description | Code Reference |
| --- | --- | --- |
| Initial Binary Search | Locate hash range in sorted list | [puredb/src/puredb_read.c L195-L234](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L195-L234) |
| Backward Scan | Find first matching hash | [puredb/src/puredb_read.c L206-L217](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L206-L217) |
| Key Verification | Compare actual key content | [puredb/src/puredb_read.c L258-L264](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_read.c#L258-L264) |

## Write Operations

The write system builds databases incrementally using temporary files and a two-phase commit process.

### Database Creation Process

```mermaid
flowchart TD

OPEN["puredbw_open()"]
CREATE_TEMPS["Create temp index/data files"]
WRITE_VERSION["Write version header"]
ADD_LOOP["Add key-value pairs"]
HASH_INSERT["Insert into hash tables"]
WRITE_DATA["Write to data file"]
CLOSE_START["puredbw_close()"]
SORT_HASHES["Sort hash table entries"]
WRITE_INDEX["Write index structure"]
MERGE_FILES["Merge index + data files"]
ATOMIC_RENAME["Atomic rename to final file"]

subgraph subGraph0 ["PureDBW Write Process"]
    OPEN
    CREATE_TEMPS
    WRITE_VERSION
    ADD_LOOP
    HASH_INSERT
    WRITE_DATA
    CLOSE_START
    SORT_HASHES
    WRITE_INDEX
    MERGE_FILES
    ATOMIC_RENAME
    OPEN --> CREATE_TEMPS
    CREATE_TEMPS --> WRITE_VERSION
    WRITE_VERSION --> ADD_LOOP
    ADD_LOOP --> HASH_INSERT
    HASH_INSERT --> WRITE_DATA
    WRITE_DATA --> ADD_LOOP
    ADD_LOOP --> CLOSE_START
    CLOSE_START --> SORT_HASHES
    SORT_HASHES --> WRITE_INDEX
    WRITE_INDEX --> MERGE_FILES
    MERGE_FILES --> ATOMIC_RENAME
end
```

Sources: [puredb/src/puredb_write.c L40-L77](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L40-L77)

 [puredb/src/puredb_write.c L325-L337](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L325-L337)

### Hash Table Management

The write system maintains in-memory hash tables during database construction:

| Data Structure | Purpose | Code Reference |
| --- | --- | --- |
| `Hash0[256]` | Primary hash buckets | [puredb/src/puredb_write.h L41](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.h#L41-L41) |
| `Hash1` lists | Secondary hash entries | [puredb/src/puredb_write.h L23-L26](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.h#L23-L26) |
| Dynamic allocation | Growing hash chains | [puredb/src/puredb_write.c L89-L103](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L89-L103) |

## Platform Support

PureDB provides cross-platform compatibility through conditional compilation and feature detection.

### Platform-Specific Features

| Platform | Memory Mapping | Binary Mode | File Locking |
| --- | --- | --- | --- |
| Unix/Linux | `mmap()` | Not needed | `fsync()` |
| Windows | `MapViewOfFile()` | `O_BINARY` | `F_FULLFSYNC` |
| Generic | `malloc()`+`read()` | Handled automatically | Basic `fsync()` |

```mermaid
flowchart TD

CONFIG["config.h"]
FEATURE_DETECT["Feature Detection"]
MMAP_UNIX["Unix mmap()"]
MMAP_WIN["Windows MapViewOfFile()"]
FALLBACK_IO["Standard I/O"]
BINARY_MODE["O_BINARY flag"]
SYNC_OPS["File sync operations"]

subgraph subGraph0 ["Platform Abstraction Layer"]
    CONFIG
    FEATURE_DETECT
    MMAP_UNIX
    MMAP_WIN
    FALLBACK_IO
    BINARY_MODE
    SYNC_OPS
    CONFIG --> FEATURE_DETECT
    FEATURE_DETECT --> MMAP_UNIX
    FEATURE_DETECT --> MMAP_WIN
    FEATURE_DETECT --> FALLBACK_IO
    FEATURE_DETECT --> BINARY_MODE
    FEATURE_DETECT --> SYNC_OPS
end
```

Sources: [puredb/src/puredb_p.h L35-L85](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_p.h#L35-L85)

 [puredb/src/puredb_write.c L279-L284](https://github.com/jedisct1/pure-ftpd/blob/3818577a/puredb/src/puredb_write.c#L279-L284)

The system automatically detects available platform features at compile time and selects the most efficient I/O strategy for each environment.
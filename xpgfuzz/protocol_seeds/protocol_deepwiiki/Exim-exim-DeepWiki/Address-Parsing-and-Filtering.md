# Address Parsing and Filtering

> **Relevant source files**
> * [doc/doc-txt/README.SIEVE](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/README.SIEVE)
> * [src/src/parse.c](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c)
> * [src/src/rda.c](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c)

This document covers Exim's address parsing infrastructure and filtering mechanisms, including RFC 822/2822 address extraction, forward file processing, and filter interpretation. For information about string expansion used within filters, see [String Expansion](/Exim/exim/2.5-string-expansion). For details on routing decisions that use parsed addresses, see [Routing System](/Exim/exim/2.2-routing-system).

## Core Address Parsing

Exim's address parsing functionality is primarily implemented in `parse.c`, providing RFC 822/2822 compliant address extraction and validation. The parsing system handles complex address formats including source routing, group notation, and quoted strings.

### Address Extraction Pipeline

```mermaid
flowchart TD

Input["Raw Address String"]
FindEnd["parse_find_address_end()"]
ExtractAddr["parse_extract_address()"]
ValidateFormat["Valid Format?"]
ParseLocal["read_local_part()"]
Error["Return Error"]
ParseDomain["read_domain()"]
Result["Extracted Address"]
HandleRoute["Handle Source Routes"]
CollapseRoute["Collapse to Final Address"]
HandleGroup["Handle Group Notation"]
IgnorePhrase["Ignore Phrase Part"]

Input --> FindEnd
FindEnd --> ExtractAddr
ExtractAddr --> ValidateFormat
ValidateFormat --> ParseLocal
ValidateFormat --> Error
ParseLocal --> ParseDomain
ParseDomain --> Result
ExtractAddr --> HandleRoute
HandleRoute --> CollapseRoute
CollapseRoute --> ParseLocal
ExtractAddr --> HandleGroup
HandleGroup --> IgnorePhrase
IgnorePhrase --> ParseLocal
```

**Sources:** [src/src/parse.c L634-L850](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L634-L850)

The `parse_extract_address()` function serves as the main entry point for address extraction, handling various RFC 822 constructs:

| Component | Function | Purpose |
| --- | --- | --- |
| **Address End Detection** | `parse_find_address_end()` | Locates comma or string termination |
| **Domain Parsing** | `read_domain()` | Extracts domain with UTF-8 support |
| **Local Part Parsing** | `read_local_part()` | Handles quoted strings and atoms |
| **Route Processing** | `read_route()` | Processes source routes (ignored) |
| **Comment Handling** | `skip_comment()` | Strips RFC 822 comments |

**Sources:** [src/src/parse.c L71-L133](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L71-L133)

 [src/src/parse.c L247-L396](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L247-L396)

 [src/src/parse.c L425-L491](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L425-L491)

### Address Format Support

Exim's parser handles multiple address formats while maintaining strict RFC compliance:

```

```

**Sources:** [src/src/parse.c L645-L697](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L645-L697)

 [src/src/parse.c L709-L770](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L709-L770)

## Forward List Processing

The `parse_forward_list()` function processes alias files and `.forward` files, extracting multiple addresses and handling special directives.

### Forward List Pipeline

```mermaid
flowchart TD

Input["Forward List String"]
ParseLoop["Parse Each Address"]
CheckSpecial["Special Address?"]
Include["Process Include File"]
Defer["Return FF_DEFER"]
Blackhole["Return FF_BLACKHOLE"]
Fail["Return FF_FAIL"]
Skip["Skip (Compatibility)"]
ValidateAddr["Validate Address"]
CheckPipe["Pipe/File?"]
CreatePipe["Create Pipe/File Address"]
CreateAddr["Create Mail Address"]
ReadFile["Read Include File"]
Security["Security Checks"]
RecursiveCall["Recursive parse_forward_list()"]
AddToChain["Add to Address Chain"]
Complete["Return FF_DELIVERED"]

Input --> ParseLoop
ParseLoop --> CheckSpecial
CheckSpecial --> Include
CheckSpecial --> Defer
CheckSpecial --> Blackhole
CheckSpecial --> Fail
CheckSpecial --> Skip
CheckSpecial --> ValidateAddr
ValidateAddr --> CheckPipe
CheckPipe --> CreatePipe
CheckPipe --> CreateAddr
Include --> ReadFile
ReadFile --> Security
Security --> RecursiveCall
CreatePipe --> AddToChain
CreateAddr --> AddToChain
RecursiveCall --> AddToChain
AddToChain --> ParseLoop
ParseLoop --> Complete
```

**Sources:** [src/src/parse.c L1248-L1707](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L1248-L1707)

### Special Address Handling

| Special Address | Action | Purpose |
| --- | --- | --- |
| **`:defer:`** | Temporary failure | Defer delivery with message |
| **`:blackhole:`** | Discard silently | Delete message without trace |
| **`:fail:`** | Permanent failure | Reject with error message |
| **`:include:path`** | Include file | Process addresses from file |
| **`:unknown:`** | Skip | Backward compatibility |

**Sources:** [src/src/parse.c L1342-L1369](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L1342-L1369)

### Include File Security

Include file processing implements strict security controls:

```mermaid
flowchart TD

IncludeReq["Include Request"]
ValidatePath["Validate File Path"]
CheckTaint["Path Tainted?"]
RejectTaint["Reject Tainted Path"]
CheckAbsolute["Absolute Path?"]
RejectRelative["Reject Relative Path"]
CheckDirectory["Directory Restriction?"]
ValidateDir["Validate Directory Constraint"]
OpenFile["Open File"]
CheckSymlinks["Check for Symlinks"]
ReadContents["Read File Contents"]
RecursiveProcess["Process Recursively"]

IncludeReq --> ValidatePath
ValidatePath --> CheckTaint
CheckTaint --> RejectTaint
CheckTaint --> CheckAbsolute
CheckAbsolute --> RejectRelative
CheckAbsolute --> CheckDirectory
CheckDirectory --> ValidateDir
CheckDirectory --> OpenFile
ValidateDir --> CheckSymlinks
CheckSymlinks --> OpenFile
OpenFile --> ReadContents
ReadContents --> RecursiveProcess
```

**Sources:** [src/src/parse.c L1376-L1574](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L1376-L1574)

## Filter Detection and Types

The `rda.c` module provides filter detection and routing between different filter types. The `rda_is_filter()` function determines the filter type based on the first line.

### Filter Type Detection

```mermaid
flowchart TD

Input["Input String"]
SkipWhitespace["Skip Leading Whitespace"]
CheckExim["Starts with '# Exim filter'?"]
EximFilter["FILTER_EXIM"]
CheckSieve["Starts with '# Sieve filter'?"]
SieveFilter["FILTER_SIEVE"]
ForwardList["FILTER_FORWARD"]
EximModule["Load exim_filter module"]
SieveModule["Load sieve_filter module"]
ParseForward["parse_forward_list()"]

Input --> SkipWhitespace
SkipWhitespace --> CheckExim
CheckExim --> EximFilter
CheckExim --> CheckSieve
CheckSieve --> SieveFilter
CheckSieve --> ForwardList
EximFilter --> EximModule
SieveFilter --> SieveModule
ForwardList --> ParseForward
```

**Sources:** [src/src/rda.c L43-L68](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L43-L68)

### Filter Processing Architecture

| Filter Type | Module | Handler | Purpose |
| --- | --- | --- | --- |
| **FILTER_EXIM** | `exim_filter` | `EXIM_INTERPRET` | Native Exim filtering |
| **FILTER_SIEVE** | `sieve_filter` | `SIEVE_INTERPRET` | RFC 5228 Sieve filtering |
| **FILTER_FORWARD** | Built-in | `parse_forward_list()` | Traditional forward files |

**Sources:** [src/src/rda.c L384-L424](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L384-L424)

## Sieve Filter Implementation

Exim implements RFC 5228 Sieve filtering with several extensions. The Sieve implementation is modular and supports standard Sieve commands plus Exim-specific extensions.

### Sieve Feature Support

```mermaid
flowchart TD

Address["address"]
Header["header"]
Size["size"]
Exists["exists"]
True["true/false"]
Vacation["vacation (RFC 5230)"]
Envelope["envelope test"]
Copy["copy parameter"]
Notify["notify action"]
Subaddr["subaddress"]
Keep["keep"]
Discard["discard"]
Redirect["redirect"]
FileInto["fileinto"]
If["if/elsif/else"]

subgraph Tests ["Tests"]
    Address
    Header
    Size
    Exists
    True
end

subgraph Extensions ["Extensions"]
    Vacation
    Envelope
    Copy
    Notify
    Subaddr
end

subgraph subGraph0 ["Core RFC 5228"]
    Keep
    Discard
    Redirect
    FileInto
    If
end
```

**Sources:** [doc/doc-txt/README.SIEVE L21-L27](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/README.SIEVE#L21-L27)

### Sieve Configuration Integration

The Sieve filter integrates with Exim's routing system through specific router and transport configurations:

```mermaid
flowchart TD

Router["redirect router"]
CheckFile["Sieve file exists?"]
DetectType["Detect filter type"]
SieveProc["Process Sieve script"]
Keep["keep action"]
FileInto["fileinto action"]
Vacation["vacation action"]
Redirect["redirect action"]
LocalTransport["localuser transport"]
VacationTransport["vacation transport"]
NewAddress["Create new address"]
AddressFile["$address_file variable"]
AutoReply["autoreply transport"]

Router --> CheckFile
CheckFile --> DetectType
DetectType --> SieveProc
SieveProc --> Keep
SieveProc --> FileInto
SieveProc --> Vacation
SieveProc --> Redirect
Keep --> LocalTransport
FileInto --> LocalTransport
Vacation --> VacationTransport
Redirect --> NewAddress
LocalTransport --> AddressFile
VacationTransport --> AutoReply
```

**Sources:** [doc/doc-txt/README.SIEVE L36-L87](https://github.com/Exim/exim/blob/29568b25/doc/doc-txt/README.SIEVE#L36-L87)

## Security and Validation

Address parsing and filtering include multiple security layers to prevent abuse and maintain system integrity.

### Security Control Points

```mermaid
flowchart TD

Input["Input Data"]
TaintCheck["Taint Validation"]
PathValid["Path Validation"]
PermCheck["Permission Checks"]
Reject["Reject Processing"]
FileOwner["Owner Validation"]
FileMode["Mode Validation"]
SizeLimit["Size Limits"]
ProcessSafe["Safe Processing"]
Fork["Fork Process"]
SetUID["Set UID/GID"]
Chroot["Directory Restrictions"]
ProcessFilter["Process Filter"]

Input --> TaintCheck
TaintCheck --> PathValid
PathValid --> PermCheck
TaintCheck --> Reject
PathValid --> Reject
PermCheck --> Reject
PermCheck --> FileOwner
FileOwner --> FileMode
FileMode --> SizeLimit
SizeLimit --> ProcessSafe

subgraph subGraph0 ["Subprocess Security"]
    ProcessSafe
    Fork
    SetUID
    Chroot
    ProcessFilter
    ProcessSafe --> Fork
    Fork --> SetUID
    SetUID --> Chroot
    Chroot --> ProcessFilter
end
```

**Sources:** [src/src/rda.c L183-L189](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L183-L189)

 [src/src/rda.c L246-L284](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L246-L284)

 [src/src/parse.c L1417-L1423](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L1417-L1423)

### File Access Controls

| Control | Implementation | Purpose |
| --- | --- | --- |
| **Taint Checking** | `is_tainted()` validation | Prevent attacker-controlled paths |
| **Owner Validation** | UID/GID verification | Ensure proper file ownership |
| **Mode Checking** | Permission bit validation | Prevent world-writable files |
| **Size Limits** | `MAX_FILTER_SIZE` enforcement | Prevent resource exhaustion |
| **Symlink Prevention** | Path component checking | Avoid directory traversal |

**Sources:** [src/src/rda.c L256-L274](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L256-L274)

 [src/src/parse.c L1442-L1508](https://github.com/Exim/exim/blob/29568b25/src/src/parse.c#L1442-L1508)

## Integration with Mail Processing

The address parsing and filtering system integrates with Exim's core mail processing pipeline through the routing system and transport mechanisms.

### Processing Flow Integration

```mermaid
flowchart TD

Router["redirect router"]
ReadFile["Read .forward file"]
RDAInterpret["rda_interpret()"]
FilterDetect["Filter Detection"]
ProcessFilter["Process Filter/Forward"]
GenerateAddrs["Generate Addresses"]
LocalDel["Local Delivery"]
RemoteDel["Remote Delivery"]
PipeDel["Pipe Delivery"]
VacationDel["Vacation Response"]
AppendFile["appendfile transport"]
SMTPTransport["smtp transport"]
PipeTransport["pipe transport"]
AutoReply["autoreply transport"]

RDAInterpret --> FilterDetect
GenerateAddrs --> LocalDel
GenerateAddrs --> RemoteDel
GenerateAddrs --> PipeDel
GenerateAddrs --> VacationDel
LocalDel --> AppendFile
RemoteDel --> SMTPTransport
PipeDel --> PipeTransport
VacationDel --> AutoReply

subgraph subGraph3 ["Transport Selection"]
    AppendFile
    SMTPTransport
    PipeTransport
    AutoReply
end

subgraph subGraph2 ["Delivery Phase"]
    LocalDel
    RemoteDel
    PipeDel
    VacationDel
end

subgraph subGraph1 ["Filter Processing"]
    FilterDetect
    ProcessFilter
    GenerateAddrs
    FilterDetect --> ProcessFilter
    ProcessFilter --> GenerateAddrs
end

subgraph subGraph0 ["Routing Phase"]
    Router
    ReadFile
    RDAInterpret
    Router --> ReadFile
    ReadFile --> RDAInterpret
end
```

**Sources:** [src/src/rda.c L554-L603](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L554-L603)

 [src/src/rda.c L654-L777](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L654-L777)

The system maintains strict separation between privileged parsing operations and user-controlled filter execution through subprocess isolation and capability dropping.

**Sources:** [src/src/rda.c L625-L650](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L625-L650)

 [src/src/rda.c L792-L985](https://github.com/Exim/exim/blob/29568b25/src/src/rda.c#L792-L985)
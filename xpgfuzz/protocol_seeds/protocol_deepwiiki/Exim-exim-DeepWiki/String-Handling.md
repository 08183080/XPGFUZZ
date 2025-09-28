# String Handling

> **Relevant source files**
> * [src/src/macros.h](https://github.com/Exim/exim/blob/29568b25/src/src/macros.h)
> * [src/src/string.c](https://github.com/Exim/exim/blob/29568b25/src/src/string.c)

This document covers Exim's comprehensive string processing utilities, character handling functions, and formatting mechanisms. These foundational components provide safe, efficient string manipulation capabilities throughout the mail processing pipeline.

For information about string expansion (variable substitution and expression evaluation), see [String Expansion](/Exim/exim/2.5-string-expansion). For memory management details, see [Memory Management](/Exim/exim/5.2-memory-management).

## Purpose and Scope

Exim's string handling system provides:

* **Safe String Manipulation**: Functions for copying, concatenating, and modifying strings with taint tracking
* **Character Validation**: Utilities for validating IP addresses, email addresses, and character sets
* **Format Conversion**: Base62 encoding, size formatting, and escape sequence processing
* **String Building**: Growable string (`gstring`) infrastructure for efficient string construction
* **List Processing**: Functions for parsing and manipulating separated lists
* **Security Features**: Taint tracking and validation to prevent security vulnerabilities

## String Handling Architecture

```

```

**Sources:** [src/src/string.c L1-L2092](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1-L2092)

 [src/src/macros.h L1-L1234](https://github.com/Exim/exim/blob/29568b25/src/src/macros.h#L1-L1234)

## Core String Functions

### String Copying and Memory Management

The string copying functions handle memory allocation and taint tracking:

| Function | Purpose | Taint Handling |
| --- | --- | --- |
| `string_copy_function()` | Copy string with same taint status | Preserves taint |
| `string_copy_taint_function()` | Copy with explicit taint status | Explicit control |
| `string_copy_malloc()` | Copy to malloc'd memory | No taint tracking |
| `string_copy_dnsdomain()` | Copy DNS domain with unescaping | Always tainted |

```mermaid
flowchart TD

INPUT["Input_String"]
COPY["string_copy_*"]
TAINT_CHECK["Taint_Compatible?"]
MEMCPY["memcpy_to_new_block"]
REBUFFER["gstring_rebuffer"]
OUTPUT["Output_String"]
TAINT_PROTO["Taint_Prototype"]
SOURCE_TAINT["Source_Taint_Status"]

INPUT --> COPY
COPY --> TAINT_CHECK
TAINT_CHECK --> MEMCPY
TAINT_CHECK --> REBUFFER
REBUFFER --> MEMCPY
MEMCPY --> OUTPUT
TAINT_PROTO --> COPY
SOURCE_TAINT --> TAINT_CHECK
```

**Sources:** [src/src/string.c L466-L522](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L466-L522)

### Growable String Infrastructure

The `gstring` structure enables efficient string building:

```

```

**Sources:** [src/src/string.c L1173-L1315](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1173-L1315)

## String Formatting and Printing

### Format String Processing

The `string_vformat_trc()` function provides printf-style formatting with Exim-specific extensions:

| Format | Description | Example |
| --- | --- | --- |
| `%D` | Daily datestamp | `20240315` |
| `%M` | Monthly datestamp | `202403` |
| `%S` | Lowercase string | `hello` |
| `%T` | Uppercase string | `HELLO` |
| `%Y` | gstring pointer | Variable content |
| `%b` | blob pointer | Binary data |
| `%H` | Hex print with precision | `48656c6c6f` |
| `%Z` | Quote-print format | `{SP}Hello{LF}` |

```mermaid
flowchart TD

FORMAT["Format_String"]
PARSE["Parse_Specifiers"]
WIDTH["Extract_Width"]
PRECISION["Extract_Precision"]
MODIFIERS["Extract_Modifiers"]
NUMERIC["Numeric_Types"]
STRING["String_Types"]
SPECIAL["Special_Types"]
SPRINTF["sprintf_formatting"]
CUSTOM["Custom_Processing"]
DATESTAMP["Datestamp_Generation"]
BUFFER["Output_Buffer"]

WIDTH --> NUMERIC
PRECISION --> STRING
MODIFIERS --> SPECIAL
NUMERIC --> SPRINTF
STRING --> CUSTOM
SPECIAL --> DATESTAMP
SPRINTF --> BUFFER
CUSTOM --> BUFFER
DATESTAMP --> BUFFER

subgraph subGraph2 ["Output Generation"]
    SPRINTF
    CUSTOM
    DATESTAMP
end

subgraph subGraph1 ["Type Handling"]
    NUMERIC
    STRING
    SPECIAL
end

subgraph subGraph0 ["Format Processing"]
    FORMAT
    PARSE
    WIDTH
    PRECISION
    MODIFIERS
    FORMAT --> PARSE
    PARSE --> WIDTH
    PARSE --> PRECISION
    PARSE --> MODIFIERS
end
```

**Sources:** [src/src/string.c L1399-L1873](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1399-L1873)

## Character Handling and Validation

### Character Classification Macros

Exim defines custom character classification macros for consistent behavior across platforms:

```

```

**Sources:** [src/src/macros.h L71-L83](https://github.com/Exim/exim/blob/29568b25/src/src/macros.h#L71-L83)

### IP Address Validation

The `string_is_ip_addressX()` function validates IPv4 and IPv6 addresses:

```mermaid
flowchart TD

INPUT["IP_Address_String"]
SLASH["Contains_Slash?"]
MASK["Parse_Netmask"]
PERCENT["Contains_Percent?"]
VALIDATE_MASK["Validate_Mask_Value"]
INTERFACE["Parse_Interface_ID"]
DETECT["Detect_Address_Family"]
COLON["Contains_Colon?"]
IPV6["inet_pton_AF_INET6"]
IPV4["inet_pton_AF_INET"]
VALIDATE6["Validate_IPv6_Constraints"]
VALIDATE4["Validate_IPv4_Constraints"]
RETURN6["Return_6"]
RETURN4["Return_4"]
ERROR["Return_0"]

INPUT --> SLASH
SLASH --> MASK
SLASH --> PERCENT
MASK --> VALIDATE_MASK
VALIDATE_MASK --> PERCENT
PERCENT --> INTERFACE
PERCENT --> DETECT
INTERFACE --> DETECT
DETECT --> COLON
COLON --> IPV6
COLON --> IPV4
IPV6 --> VALIDATE6
IPV4 --> VALIDATE4
VALIDATE6 --> RETURN6
VALIDATE4 --> RETURN4
VALIDATE_MASK --> ERROR
IPV6 --> ERROR
IPV4 --> ERROR
```

**Sources:** [src/src/string.c L44-L166](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L44-L166)

## List Processing Functions

### Separated List Parsing

The `string_nextinlist_trc()` function parses separated lists with dynamic separator detection:

```mermaid
flowchart TD

LISTPTR["List_Pointer"]
SEPARATOR["Separator_Detection"]
DYNAMIC["Dynamic_Separator?"]
PARSE_DELIM["Parse_Delimiter"]
USE_DEFAULT["Use_Default_Separator"]
VALIDATE_TAINT["Validate_Taint_Status"]
BUFFER["Buffer_Provided?"]
COPY_TO_BUFFER["Copy_to_Buffer"]
ALLOCATE_GSTRING["Allocate_gstring"]
DOUBLE_CHECK["Check_Doubled_Separators"]
TRIM["Trim_Whitespace"]
RETURN["Return_Item"]

DYNAMIC --> PARSE_DELIM
DYNAMIC --> USE_DEFAULT
VALIDATE_TAINT --> BUFFER
COPY_TO_BUFFER --> DOUBLE_CHECK
ALLOCATE_GSTRING --> DOUBLE_CHECK

subgraph subGraph3 ["Special Character Handling"]
    DOUBLE_CHECK
    TRIM
    RETURN
    DOUBLE_CHECK --> TRIM
    TRIM --> RETURN
end

subgraph subGraph2 ["Item Extraction"]
    BUFFER
    COPY_TO_BUFFER
    ALLOCATE_GSTRING
    BUFFER --> COPY_TO_BUFFER
    BUFFER --> ALLOCATE_GSTRING
end

subgraph subGraph1 ["Separator Handling"]
    PARSE_DELIM
    USE_DEFAULT
    VALIDATE_TAINT
    PARSE_DELIM --> VALIDATE_TAINT
    USE_DEFAULT --> VALIDATE_TAINT
end

subgraph subGraph0 ["List Processing"]
    LISTPTR
    SEPARATOR
    DYNAMIC
    LISTPTR --> SEPARATOR
    SEPARATOR --> DYNAMIC
end
```

**Sources:** [src/src/string.c L913-L1018](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L913-L1018)

## String Security and Taint Tracking

### Taint Compatibility Checking

String functions perform taint compatibility checks to prevent security vulnerabilities:

```mermaid
flowchart TD

SOURCE["Source_String"]
DEST["Destination_String"]
CHECK["is_incompatible?"]
PROCEED["Proceed_with_Operation"]
REBUFFER["Rebuffer_Allowed?"]
GSTRING_REBUFFER["gstring_rebuffer"]
DIE_TAINTED["die_tainted"]
NEW_BUFFER["Allocate_New_Buffer"]
COPY_DATA["Copy_Existing_Data"]
PANIC["Panic_with_Location"]
EXIT["Exit_Process"]

CHECK --> PROCEED
CHECK --> REBUFFER
REBUFFER --> GSTRING_REBUFFER
REBUFFER --> DIE_TAINTED
COPY_DATA --> PROCEED
DIE_TAINTED --> PANIC

subgraph subGraph3 ["Security Enforcement"]
    PANIC
    EXIT
    PANIC --> EXIT
end

subgraph subGraph2 ["Rebuffering Process"]
    GSTRING_REBUFFER
    DIE_TAINTED
    NEW_BUFFER
    COPY_DATA
    GSTRING_REBUFFER --> NEW_BUFFER
    NEW_BUFFER --> COPY_DATA
end

subgraph subGraph1 ["Compatibility Results"]
    PROCEED
    REBUFFER
end

subgraph subGraph0 ["Taint Validation"]
    SOURCE
    DEST
    CHECK
    SOURCE --> DEST
    SOURCE --> CHECK
    DEST --> CHECK
end
```

**Sources:** [src/src/string.c L1263-L1267](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1263-L1267)

 [src/src/string.c L1420-L1428](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1420-L1428)

## Utility Functions

### Escape Sequence Processing

The `string_interpret_escape()` function handles C-style escape sequences:

| Escape | Description | Result |
| --- | --- | --- |
| `\n` | Newline | `0x0A` |
| `\r` | Carriage return | `0x0D` |
| `\t` | Tab | `0x09` |
| `\nnn` | Octal value | Byte value |
| `\xHH` | Hex value | Byte value |

### String Comparison Functions

Case-insensitive string comparison functions:

| Function | Purpose | Behavior |
| --- | --- | --- |
| `strcmpic()` | Compare strings | Case-insensitive |
| `strncmpic()` | Compare n characters | Case-insensitive |
| `strstric()` | Find substring | Case-insensitive search |

**Sources:** [src/src/string.c L268-L309](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L268-L309)

 [src/src/string.c L758-L852](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L758-L852)

## Integration with Core Systems

String handling integrates with Exim's core systems through:

* **Memory Management**: Uses pool-based allocation from [Memory Management](/Exim/exim/5.2-memory-management)
* **String Expansion**: Provides foundation for [String Expansion](/Exim/exim/2.5-string-expansion)
* **Logging**: Supports safe string formatting for log output
* **Configuration**: Handles configuration string processing
* **Protocol Handling**: Supports SMTP command and header processing

The string handling system ensures consistent, secure, and efficient string operations throughout Exim's mail processing pipeline.

**Sources:** [src/src/string.c L1-L2092](https://github.com/Exim/exim/blob/29568b25/src/src/string.c#L1-L2092)

 [src/src/macros.h L46-L83](https://github.com/Exim/exim/blob/29568b25/src/src/macros.h#L46-L83)
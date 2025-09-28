# Content Scanning

> **Relevant source files**
> * [src/src/ip.c](https://github.com/Exim/exim/blob/29568b25/src/src/ip.c)
> * [src/src/malware.c](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c)
> * [src/src/mime.c](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c)
> * [src/src/mime.h](https://github.com/Exim/exim/blob/29568b25/src/src/mime.h)
> * [src/src/regex.c](https://github.com/Exim/exim/blob/29568b25/src/src/regex.c)
> * [src/src/smtp_out.c](https://github.com/Exim/exim/blob/29568b25/src/src/smtp_out.c)
> * [src/src/spam.c](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c)
> * [src/src/spool_mbox.c](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c)
> * [test/confs/4000](https://github.com/Exim/exim/blob/29568b25/test/confs/4000)
> * [test/log/4000](https://github.com/Exim/exim/blob/29568b25/test/log/4000)
> * [test/mail/4000.userx](https://github.com/Exim/exim/blob/29568b25/test/mail/4000.userx)
> * [test/rejectlog/4000](https://github.com/Exim/exim/blob/29568b25/test/rejectlog/4000)
> * [test/scripts/4000-scanning/4000](https://github.com/Exim/exim/blob/29568b25/test/scripts/4000-scanning/4000)
> * [test/stdout/4000](https://github.com/Exim/exim/blob/29568b25/test/stdout/4000)

Exim's content scanning subsystem provides comprehensive analysis of email messages for malware, spam, MIME structure violations, and pattern-based content filtering. This system operates during the SMTP DATA phase and integrates with Exim's Access Control List (ACL) framework to enable policy-driven message handling.

For information about DKIM signature verification, see [DKIM Verification](/Exim/exim/3.2-dkim-verification). For general ACL processing, see [Access Control Lists (ACLs)](/Exim/exim/2.4-access-control-lists-(acls)).

## Architecture Overview

The content scanning system consists of several interconnected components that work together to analyze message content:

```mermaid
flowchart TD

ACL["acl_smtp_data / acl_smtp_mime"]
SPOOL["spool_mbox()"]
MBOX["Temporary .eml Files"]
MALWARE["malware_internal()"]
SPAM["spam()"]
MIME["mime_acl_check()"]
REGEX["regex() / mime_regex()"]
CLAM["ClamAV"]
SPAMASS["SpamAssassin"]
RSPAMD["rspamd"]
FSEC["F-Secure"]
DRWEB["DrWeb"]
OTHER["Other AV Engines"]
DECODE["mime_decode()"]
PARSE["Header Parsing"]
BOUNDARY["Boundary Detection"]

MBOX --> MALWARE
MBOX --> SPAM
MBOX --> REGEX
ACL --> MIME
MIME --> DECODE
MIME --> PARSE
MIME --> BOUNDARY
MALWARE --> CLAM
MALWARE --> FSEC
MALWARE --> DRWEB
MALWARE --> OTHER
SPAM --> SPAMASS
SPAM --> RSPAMD

subgraph subGraph3 ["MIME Processing"]
    DECODE
    PARSE
    BOUNDARY
end

subgraph subGraph2 ["External Scanners"]
    CLAM
    SPAMASS
    RSPAMD
    FSEC
    DRWEB
    OTHER
end

subgraph subGraph1 ["Analysis Engines"]
    MALWARE
    SPAM
    MIME
    REGEX
end

subgraph subGraph0 ["Content Scanning Core"]
    ACL
    SPOOL
    MBOX
    ACL --> SPOOL
    SPOOL --> MBOX
end
```

**Sources:** [src/src/malware.c L1-L102](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L1-L102)

 [src/src/spam.c L177-L604](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L177-L604)

 [src/src/mime.c L491-L604](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L491-L604)

 [src/src/spool_mbox.c L32-L198](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c#L32-L198)

## Message Flow and Processing Pipeline

The content scanning process follows a well-defined pipeline that transforms the raw SMTP message into analyzable formats:

```mermaid
sequenceDiagram
  participant SMTP Session
  participant ACL Engine
  participant spool_mbox()
  participant Scanner Engines
  participant External Tools

  SMTP Session->>ACL Engine: DATA command received
  ACL Engine->>spool_mbox(): Create temporary mbox file
  spool_mbox()->>spool_mbox(): Generate scan/msgid/msgid.eml
  loop [Malware Scanning]
    ACL Engine->>Scanner Engines: malware_internal()
    Scanner Engines->>External Tools: Connect to AV daemon
    External Tools->>Scanner Engines: Scan results
    Scanner Engines->>ACL Engine: malware_name / OK/FAIL
    ACL Engine->>Scanner Engines: spam()
    Scanner Engines->>External Tools: Send to SpamAssassin/rspamd
    External Tools->>Scanner Engines: Spam score/action
    Scanner Engines->>ACL Engine: spam_score / spam_action
    ACL Engine->>Scanner Engines: mime_acl_check()
    Scanner Engines->>Scanner Engines: Parse headers/boundaries
    Scanner Engines->>Scanner Engines: Decode attachments
    Scanner Engines->>ACL Engine: MIME variables set
  end
  ACL Engine->>SMTP Session: Accept/Reject/Defer
```

**Sources:** [src/src/spool_mbox.c L32-L198](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c#L32-L198)

 [src/src/malware.c L573-L683](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L573-L683)

 [src/src/spam.c L214-L604](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L214-L604)

 [src/src/mime.c L491-L604](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L491-L604)

## Malware Detection System

The malware detection subsystem supports multiple antivirus engines through a unified interface. Each scanner type is defined with specific connection parameters and communication protocols.

### Scanner Configuration and Selection

The scanner infrastructure is built around the `m_scans[]` array which defines supported scanner types:

| Scanner Type | Connection | Default Options | Protocol |
| --- | --- | --- | --- |
| ClamAV (`M_CLAMD`) | MC_NONE | `/tmp/clamd` | Custom TCP/Unix |
| F-Secure (`M_FSEC`) | MC_UNIX | `/var/run/.fsav` | Unix socket |
| DrWeb (`M_DRWEB`) | MC_STRM | `/usr/local/drweb/run/drwebd.sock` | Stream socket |
| Kaspersky (`M_KAVD`) | MC_UNIX | `/var/run/AvpCtl` | Unix socket |
| Avast (`M_AVAST`) | MC_STRM | `/var/run/avast/scan.sock` | Stream socket |

**Sources:** [src/src/malware.c L57-L102](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L57-L102)

 [src/src/malware.c L652-L682](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L652-L682)

### Malware Scanning Process

```mermaid
flowchart TD

CONFIG["av_scanner config"]
PARSE["string_nextinlist()"]
FIND["Find scanner in m_scans[]"]
SOCKET["Create socket"]
CONNECT["ip_tcpsocket/ip_unixsocket"]
PROTO["Scanner-specific protocol"]
SEND["Send scan request"]
MBOX["Transmit mbox file"]
RECV["Receive results"]
PARSE_RESULT["Parse virus names"]

FIND --> SOCKET
PROTO --> SEND

subgraph subGraph2 ["Scanning Execution"]
    SEND
    MBOX
    RECV
    PARSE_RESULT
    SEND --> MBOX
    MBOX --> RECV
    RECV --> PARSE_RESULT
end

subgraph subGraph1 ["Connection Management"]
    SOCKET
    CONNECT
    PROTO
    SOCKET --> CONNECT
    CONNECT --> PROTO
end

subgraph subGraph0 ["Scanner Selection"]
    CONFIG
    PARSE
    FIND
    CONFIG --> PARSE
    PARSE --> FIND
end
```

**Sources:** [src/src/malware.c L646-L682](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L646-L682)

 [src/src/malware.c L687-L984](https://github.com/Exim/exim/blob/29568b25/src/src/malware.c#L687-L984)

The `malware_internal()` function coordinates the entire scanning process, handling scanner selection, connection establishment, and result parsing. Different scanner types implement distinct communication protocols - for example, ClamAV uses a simple text protocol while DrWeb uses a binary protocol with network byte order integers.

## Spam Detection Integration

The spam detection system primarily integrates with SpamAssassin and rspamd through network protocols. The implementation handles server selection, load balancing, and result interpretation.

### SpamAssassin vs rspamd Protocol Handling

```mermaid
flowchart TD

VARIANT["Check spamd->is_rspamd"]
SA_REQ["REPORT SPAMC/1.2"]
SA_USER["User: username"]
SA_LEN["Content-length: size"]
SA_BODY["Message body"]
SA_RESP["SPAMD response + score"]
RS_REQ["CHECK RSPAMC/1.3"]
RS_META["Queue-Id, From, Recipients"]
RS_HELO["Helo, Hostname, IP"]
RS_BODY["Message body"]
RS_RESP["Metric: default; score"]

VARIANT --> SA_REQ
VARIANT --> RS_REQ

subgraph subGraph2 ["rspamd Protocol"]
    RS_REQ
    RS_META
    RS_HELO
    RS_BODY
    RS_RESP
    RS_REQ --> RS_META
    RS_META --> RS_HELO
    RS_HELO --> RS_BODY
    RS_BODY --> RS_RESP
end

subgraph subGraph1 ["SpamAssassin Protocol"]
    SA_REQ
    SA_USER
    SA_LEN
    SA_BODY
    SA_RESP
    SA_REQ --> SA_USER
    SA_USER --> SA_LEN
    SA_LEN --> SA_BODY
    SA_BODY --> SA_RESP
end

subgraph subGraph0 ["Protocol Selection"]
    VARIANT
end
```

**Sources:** [src/src/spam.c L343-L376](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L343-L376)

 [src/src/spam.c L480-L527](https://github.com/Exim/exim/blob/29568b25/src/src/spam.c#L480-L527)

The spam detection system uses a server selection algorithm based on priority and weight, allowing for load distribution across multiple spamd instances. Failed servers are marked and excluded from subsequent selections within the same message processing session.

## MIME Processing and Structure Analysis

The MIME processing engine handles message structure parsing, header analysis, and content decoding. It operates through the `mime_acl_check()` function which processes each MIME part individually.

### MIME Header Processing Pipeline

```mermaid
flowchart TD

READ["mime_get_header()"]
MATCH["Match against mime_header_list[]"]
EXTRACT["Extract header values"]
PARAMS["Parse parameters"]
RFC2231["RFC 2231 encoded params"]
RFC2047["RFC 2047 encoded filenames"]
QUOTED["Quoted string handling"]
CONTINUE["Parameter continuation"]
QP["Quoted-Printable (mime_decode_qp)"]
B64["Base64 (mime_decode_base64)"]
ASIS["As-is (mime_decode_asis)"]

PARAMS --> RFC2231
PARAMS --> RFC2047
PARAMS --> QUOTED
PARAMS --> CONTINUE
EXTRACT --> QP
EXTRACT --> B64
EXTRACT --> ASIS

subgraph subGraph2 ["Content Decoding"]
    QP
    B64
    ASIS
end

subgraph subGraph1 ["Parameter Parsing"]
    RFC2231
    RFC2047
    QUOTED
    CONTINUE
end

subgraph subGraph0 ["Header Processing"]
    READ
    MATCH
    EXTRACT
    PARAMS
    READ --> MATCH
    MATCH --> EXTRACT
    EXTRACT --> PARAMS
end
```

**Sources:** [src/src/mime.c L304-L395](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L304-L395)

 [src/src/mime.c L573-L689](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L573-L689)

 [src/src/mime.c L220-L301](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L220-L301)

The MIME processor maintains state through a series of expansion variables (`mime_content_type`, `mime_filename`, `mime_charset`, etc.) that are accessible within ACL conditions. These variables are reset for each MIME part processed.

### Boundary Detection and Multipart Handling

The boundary detection mechanism in `mime_acl_check()` handles nested multipart structures by maintaining a context stack. Each boundary context tracks the current boundary string and processes parts until an end boundary is encountered.

**Sources:** [src/src/mime.c L524-L550](https://github.com/Exim/exim/blob/29568b25/src/src/mime.c#L524-L550)

## Regular Expression Content Filtering

The regex scanning system provides pattern-based content analysis through two primary functions: `regex()` for full message scanning and `mime_regex()` for decoded MIME part analysis.

### Pattern Compilation and Matching

```mermaid
flowchart TD

LIST["Pattern list"]
COMPILE["compile()"]
CACHE["Cacheable regex_compile()"]
STORE["pcre_list structure"]
BUFFER["Message buffer"]
MATCHER["matcher()"]
PCRE["pcre2_match()"]
VARS["regex_vars[] population"]

STORE --> MATCHER

subgraph subGraph1 ["Matching Phase"]
    BUFFER
    MATCHER
    PCRE
    VARS
    BUFFER --> MATCHER
    MATCHER --> PCRE
    PCRE --> VARS
end

subgraph subGraph0 ["Compilation Phase"]
    LIST
    COMPILE
    CACHE
    STORE
    LIST --> COMPILE
    COMPILE --> CACHE
    CACHE --> STORE
end
```

**Sources:** [src/src/regex.c L31-L62](https://github.com/Exim/exim/blob/29568b25/src/src/regex.c#L31-L62)

 [src/src/regex.c L71-L102](https://github.com/Exim/exim/blob/29568b25/src/src/regex.c#L71-L102)

The regex system uses a reset point mechanism to manage memory during large message processing, preventing memory exhaustion when scanning large messages with multiple patterns.

## Temporary File Management

The `spool_mbox()` function creates temporary MBOX-format files that scanners can process. These files are created in the `scan/message_id/` directory structure within the spool directory.

### MBOX File Generation Process

The MBOX file generation process constructs a complete email message including envelope information, headers, and body content. Wire format conversion is handled for messages that use CRLF line endings.

**Sources:** [src/src/spool_mbox.c L51-L181](https://github.com/Exim/exim/blob/29568b25/src/src/spool_mbox.c#L51-L181)

The temporary files are automatically cleaned up by `unspool_mbox()` unless the `no_mbox_unspool` flag is set, which is useful for debugging scanner interactions.

## Integration with ACL Framework

Content scanning integrates with Exim's ACL system through specific ACL verbs and conditions. The scanning functions are typically called from `acl_smtp_data` or `acl_smtp_mime` ACLs.

### ACL Integration Points

| ACL Condition | Function | Purpose |
| --- | --- | --- |
| `malware` | `malware_internal()` | Virus scanning |
| `spam` | `spam()` | Spam detection |
| `mime_regex` | `mime_regex()` | MIME part pattern matching |
| `regex` | `regex()` | Full message pattern matching |

**Sources:** [test/confs/4000 L19-L38](https://github.com/Exim/exim/blob/29568b25/test/confs/4000#L19-L38)

The content scanning system sets various expansion variables that can be used in subsequent ACL processing, allowing for sophisticated policy decisions based on scan results.
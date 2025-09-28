# Global Configuration System

> **Relevant source files**
> * [pure-ftpd.conf.in](https://github.com/jedisct1/pure-ftpd/blob/3818577a/pure-ftpd.conf.in)
> * [src/ftpd_p.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h)
> * [src/globals.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h)
> * [src/simpleconf.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c)
> * [src/simpleconf.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.h)
> * [src/simpleconf_ftpd.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h)

This document covers Pure-FTPd's global configuration management system, including how configuration variables are defined, parsed, and managed throughout the server's lifecycle. The system handles both compile-time configuration options and runtime configuration from files and command-line arguments.

For information about build-time configuration and compilation options, see [Build System and Compilation](/jedisct1/pure-ftpd/5.1-build-system-and-compilation). For runtime configuration files and administrative setup, see [Runtime Configuration](/jedisct1/pure-ftpd/5.2-runtime-configuration).

## Overview

Pure-FTPd's global configuration system consists of several interconnected components:

* **Global Variables System**: Centralized variable definitions using preprocessor macros
* **SimpleConf Parser**: Configuration file parser that converts `.conf` files to command-line arguments
* **Authentication Chain Configuration**: Dynamic configuration of authentication methods
* **Runtime State Management**: Management of server state and configuration updates

```mermaid
flowchart TD

CONF_FILE["pure-ftpd.conf<br>Configuration File"]
CMD_LINE["Command Line<br>Arguments"]
COMPILE_TIME["Compile-time<br>Defines"]
SIMPLECONF["simpleconf_parser<br>(simpleconf.c)"]
GETOPT["getopt_long<br>Parser"]
GLOBALS_H["globals.h<br>GLOBAL() Macros"]
AUTH_CHAIN["Authentication<br>Chain"]
RUNTIME_STATE["Runtime State<br>Variables"]
FTPD_MAIN["ftpd.c<br>Main Server"]
TLS_SYSTEM["TLS System"]
AUTH_BACKENDS["Auth Backends"]

CONF_FILE --> SIMPLECONF
SIMPLECONF --> CMD_LINE
CMD_LINE --> GETOPT
COMPILE_TIME --> GLOBALS_H
GETOPT --> GLOBALS_H
RUNTIME_STATE --> FTPD_MAIN
RUNTIME_STATE --> TLS_SYSTEM
AUTH_CHAIN --> AUTH_BACKENDS

subgraph subGraph3 ["Server Components"]
    FTPD_MAIN
    TLS_SYSTEM
    AUTH_BACKENDS
end

subgraph subGraph2 ["Global State"]
    GLOBALS_H
    AUTH_CHAIN
    RUNTIME_STATE
    GLOBALS_H --> AUTH_CHAIN
    GLOBALS_H --> RUNTIME_STATE
end

subgraph subGraph1 ["Parsing Layer"]
    SIMPLECONF
    GETOPT
end

subgraph subGraph0 ["Configuration Sources"]
    CONF_FILE
    CMD_LINE
    COMPILE_TIME
end
```

Sources: [src/globals.h L1-L201](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L1-L201)

 [src/simpleconf.c L1-L743](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L1-L743)

 [src/ftpd_p.h L1-L365](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L1-L365)

## Global Variables System

Pure-FTPd uses a centralized global variables system defined in `globals.h` that employs preprocessor macros to declare variables that can be shared across the entire codebase.

### GLOBAL Macro System

The system uses two primary macros to define global variables:

| Macro | Purpose | Usage |
| --- | --- | --- |
| `GLOBAL0(A)` | Declares uninitialized global variable | `GLOBAL0(signed char debug);` |
| `GLOBAL(A, B)` | Declares initialized global variable | `GLOBAL(int clientfd, 0);` |

The macros expand differently based on the `DEFINE_GLOBALS` preprocessor flag:

```mermaid
flowchart TD

DEFINE_SET["#ifdef DEFINE_GLOBALS<br>GLOBAL0(A) → A<br>GLOBAL(A,B) → A = B"]
DEFINE_UNSET["#else<br>GLOBAL0(A) → extern A<br>GLOBAL(A,B) → extern A"]
CONNECTION["Connection State<br>clientfd, datafd<br>ctrlconn, peer"]
USER_AUTH["User Authentication<br>account, loggedin<br>useruid, guest"]
SERVER_CONFIG["Server Configuration<br>serverport, userchroot<br>maxusers, maxload"]
TRANSFER_STATE["Transfer State<br>downloaded, uploaded<br>restartat, type"]

DEFINE_SET --> CONNECTION
DEFINE_SET --> USER_AUTH
DEFINE_SET --> SERVER_CONFIG
DEFINE_SET --> TRANSFER_STATE
DEFINE_UNSET --> CONNECTION
DEFINE_UNSET --> USER_AUTH
DEFINE_UNSET --> SERVER_CONFIG
DEFINE_UNSET --> TRANSFER_STATE

subgraph subGraph1 ["Variable Categories"]
    CONNECTION
    USER_AUTH
    SERVER_CONFIG
    TRANSFER_STATE
end

subgraph subGraph0 ["Compilation Context"]
    DEFINE_SET
    DEFINE_UNSET
end
```

Sources: [src/globals.h L4-L10](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L4-L10)

 [src/globals.h L12-L200](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L12-L200)

### Key Configuration Categories

The global variables are organized into several functional categories:

**Connection Management:**

* `clientfd` - Command connection file descriptor [src/globals.h L28](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L28-L28)
* `datafd` - Data connection file descriptor [src/globals.h L29](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L29-L29)
* `ctrlconn` - Control connection socket address [src/globals.h L30-L31](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L30-L31)

**User Authentication and Authorization:**

* `account` - User login name [src/globals.h L39](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L39-L39)
* `loggedin` - Login status flag [src/globals.h L38](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L38-L38)
* `useruid` - Minimum allowed UID [src/globals.h L51](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L51-L51)
* `guest` - Guest user flag [src/globals.h L50](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L50-L50)

**Server Behavior Configuration:**

* `userchroot` - Chroot behavior settings [src/globals.h L42-L44](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L42-L44)
* `maxusers` - Maximum concurrent users [src/globals.h L90](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L90-L90)
* `maxload` - Maximum system load threshold [src/globals.h L89](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L89-L89)
* `throttling_bandwidth_dl/ul` - Bandwidth limits [src/globals.h L24-L25](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L24-L25)

Sources: [src/globals.h L28-L200](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L28-L200)

## Configuration File Parsing

Pure-FTPd uses the SimpleConf system to parse configuration files and convert them into command-line arguments that can be processed by the standard `getopt_long` parser.

### SimpleConf Parser Architecture

```mermaid
flowchart TD

CONF_DIRECTIVES["Configuration Directives<br>MaxClientsNumber 50<br>ChrootEveryone yes<br>VerboseLog no"]
INCLUDES["Include Directives<br>Include additional.conf"]
COMMENTS["Comments<br># This is a comment"]
LEXER["Lexical Analysis<br>State Machine"]
MATCHER["Pattern Matching<br>simpleconf_options[]"]
TEMPLATE["Template Engine<br>$0, $1, $* substitution"]
CMDLINE_ARGS["Command Line Arguments<br>--maxclientsnumber=50<br>--chrooteveryone<br>--verboselog"]
GETOPT_PROCESSING["getopt_long()<br>Processing"]

CONF_DIRECTIVES --> LEXER
INCLUDES --> LEXER
COMMENTS --> LEXER
TEMPLATE --> CMDLINE_ARGS

subgraph subGraph2 ["Output Generation"]
    CMDLINE_ARGS
    GETOPT_PROCESSING
    CMDLINE_ARGS --> GETOPT_PROCESSING
end

subgraph subGraph1 ["Parser Components"]
    LEXER
    MATCHER
    TEMPLATE
    LEXER --> MATCHER
    MATCHER --> TEMPLATE
end

subgraph subGraph0 ["Configuration File Structure"]
    CONF_DIRECTIVES
    INCLUDES
    COMMENTS
end
```

Sources: [src/simpleconf.c L115-L542](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L115-L542)

 [src/simpleconf_ftpd.h L6-L122](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L6-L122)

### Pattern Matching System

The SimpleConf parser uses a sophisticated pattern matching system defined in `simpleconf_ftpd.h`. Each configuration directive is mapped to its corresponding command-line option:

**Pattern Syntax:**

* `<bool>` - Boolean values (yes/no, true/false, on/off, 1/0)
* `<digits>` - Numeric values
* `<nospace>` - Non-whitespace strings
* `<any>` - Any printable characters
* `<any*>` - Any characters including spaces
* `(<pattern>)` - Capture groups for template substitution

**Template Variables:**

* `$0`, `$1`, etc. - Captured groups
* `$*` - Entire matched portion
* Boolean directives with `?` suffix are only processed if enabled

Sources: [src/simpleconf.c L17-L43](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L17-L43)

 [src/simpleconf_ftpd.h L6-L122](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L6-L122)

### Configuration Directive Examples

| Configuration Directive | Command Line Equivalent | Pattern |
| --- | --- | --- |
| `MaxClientsNumber 50` | `--maxclientsnumber=50` | `"MaxClientsNumber (<digits>)"` |
| `ChrootEveryone yes` | `--chrooteveryone` | `"ChrootEveryone? <bool>"` |
| `MySQLConfigFile /etc/mysql.conf` | `--login=mysql:/etc/mysql.conf` | `"MySQLConfigFile (<any*>)"` |
| `Umask 133:022` | `--umask=133:022` | `"Umask (<digits>):(<digits>)"` |

Sources: [src/simpleconf_ftpd.h L75-L109](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L75-L109)

 [pure-ftpd.conf.in L40-L383](https://github.com/jedisct1/pure-ftpd/blob/3818577a/pure-ftpd.conf.in#L40-L383)

## Authentication Chain Configuration

The global configuration system manages a dynamic authentication chain that allows multiple authentication methods to be configured and chained together.

### Authentication Structure

```mermaid
flowchart TD

AUTH_LIST["auth_list[]<br>Static Registry"]
UNIX_AUTH["unix: pw_unix_*"]
PAM_AUTH["pam: pw_pam_*"]
MYSQL_AUTH["mysql: pw_mysql_*"]
PGSQL_AUTH["pgsql: pw_pgsql_*"]
LDAP_AUTH["ldap: pw_ldap_*"]
PUREDB_AUTH["puredb: pw_puredb_*"]
EXTAUTH_AUTH["extauth: pw_extauth_*"]
FIRST_AUTH["first_authentications"]
LINKED_CHAIN["Authentications*<br>Linked List"]
LAST_AUTH["last_authentications"]
PAM_CONF["PAMAuthentication yes"]
MYSQL_CONF["MySQLConfigFile /path"]
UNIX_CONF["UnixAuthentication yes"]

PAM_CONF --> FIRST_AUTH
MYSQL_CONF --> LINKED_CHAIN
UNIX_CONF --> LAST_AUTH

subgraph subGraph2 ["Configuration Sources"]
    PAM_CONF
    MYSQL_CONF
    UNIX_CONF
end

subgraph subGraph1 ["Runtime Chain"]
    FIRST_AUTH
    LINKED_CHAIN
    LAST_AUTH
    FIRST_AUTH --> LINKED_CHAIN
    LINKED_CHAIN --> LAST_AUTH
end

subgraph subGraph0 ["Authentication Registry"]
    AUTH_LIST
    UNIX_AUTH
    PAM_AUTH
    MYSQL_AUTH
    PGSQL_AUTH
    LDAP_AUTH
    PUREDB_AUTH
    EXTAUTH_AUTH
    AUTH_LIST --> UNIX_AUTH
    AUTH_LIST --> PAM_AUTH
    AUTH_LIST --> MYSQL_AUTH
    AUTH_LIST --> PGSQL_AUTH
    AUTH_LIST --> LDAP_AUTH
    AUTH_LIST --> PUREDB_AUTH
    AUTH_LIST --> EXTAUTH_AUTH
end
```

Sources: [src/ftpd_p.h L250-L296](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L250-L296)

 [src/ftpd_p.h L260-L281](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L260-L281)

### Authentication Handler Interface

Each authentication method implements a standardized interface with three functions:

```mermaid
flowchart TD

PARSE["parse()<br>Parse config file<br>Initialize backend"]
CHECK["check()<br>Validate credentials<br>Fill AuthResult"]
EXIT["exit()<br>Cleanup resources<br>Close connections"]
CONFIG_FILE["Authentication<br>Config File"]
AUTH_RESULT["AuthResult<br>Structure"]
CLEANUP["Resource<br>Cleanup"]

CONFIG_FILE --> PARSE
CHECK --> AUTH_RESULT
EXIT --> CLEANUP

subgraph subGraph1 ["Configuration Flow"]
    CONFIG_FILE
    AUTH_RESULT
    CLEANUP
end

subgraph subGraph0 ["Authentication Lifecycle"]
    PARSE
    CHECK
    EXIT
    PARSE --> CHECK
    CHECK --> EXIT
end
```

**Authentication Structure Definition:**

* `name` - Authentication method identifier
* `parse` - Configuration file parser function pointer
* `check` - Credential validation function pointer
* `exit` - Cleanup function pointer

Sources: [src/ftpd_p.h L250-L258](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L250-L258)

 [src/ftpd_p.h L283-L287](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L283-L287)

## Runtime Configuration Management

The global configuration system maintains runtime state and handles configuration updates during server operation.

### Configuration State Categories

```mermaid
flowchart TD

COMPILE_TIME["Compile-time Options<br>#ifdef WITH_TLS<br>#ifdef QUOTAS"]
STARTUP_CONFIG["Startup Configuration<br>Command line args<br>Config file parsing"]
CONNECTION_STATE["Connection State<br>clientfd, datafd<br>loggedin, account"]
SESSION_STATE["Session State<br>downloaded, uploaded<br>wd, restartat"]
DYNAMIC_LIMITS["Dynamic Limits<br>maxload, maxusers<br>throttling settings"]
USER_CONTEXT["User Context<br>useruid, guest<br>userchroot, authresult"]
TRANSFER_CONTEXT["Transfer Context<br>type, restartat<br>xferfd, load"]

STARTUP_CONFIG --> CONNECTION_STATE
STARTUP_CONFIG --> DYNAMIC_LIMITS
CONNECTION_STATE --> USER_CONTEXT
SESSION_STATE --> TRANSFER_CONTEXT

subgraph subGraph2 ["Per-Session Variables"]
    USER_CONTEXT
    TRANSFER_CONTEXT
end

subgraph subGraph1 ["Mutable Runtime State"]
    CONNECTION_STATE
    SESSION_STATE
    DYNAMIC_LIMITS
    DYNAMIC_LIMITS --> SESSION_STATE
end

subgraph subGraph0 ["Immutable Configuration"]
    COMPILE_TIME
    STARTUP_CONFIG
    COMPILE_TIME --> STARTUP_CONFIG
end
```

Sources: [src/globals.h L12-L200](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L12-L200)

 [src/ftpd_p.h L48-L55](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L48-L55)

### Configuration Update Mechanisms

The system includes several mechanisms for updating configuration during runtime:

**State Update Flag:**

* `state_needs_update` - Indicates when global state requires refresh [src/globals.h L112](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L112-L112)

**Session Management:**

* `session_start_time` - Session initialization timestamp [src/globals.h L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L166-L166)
* `deferred_quit` - Delayed session termination flag [src/globals.h L83](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L83-L83)

**Load-based Configuration:**

* `load` - Current system load for dynamic limits [src/globals.h L53](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L53-L53)
* `maxload` - Maximum allowed system load threshold [src/globals.h L89](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L89-L89)

Sources: [src/globals.h L53](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L53-L53)

 [src/globals.h L83](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L83-L83)

 [src/globals.h L112](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L112-L112)

 [src/globals.h L166](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L166-L166)

## Feature-Specific Configuration

The global configuration system supports conditional compilation and feature-specific configuration through preprocessor directives.

### Conditional Feature Configuration

```mermaid
flowchart TD

THROTTLE_ENABLED["#ifdef THROTTLING"]
THROTTLE_VARS["throttling_delay<br>throttling<br>throttling_bandwidth_*"]
UPLOAD_ENABLED["#ifdef WITH_UPLOAD_SCRIPT"]
UPLOAD_VARS["do_upload_script<br>upload_pipe_fd<br>upload_pipe_lock"]
QUOTA_ENABLED["#ifdef QUOTAS"]
QUOTA_VARS["user_quota_size<br>user_quota_files"]
TLS_ENABLED["#ifdef WITH_TLS"]
TLS_VARS["enforce_tls_auth<br>tlsciphersuite<br>cert_file, key_file"]
TLS_DISABLED["#ifndef WITH_TLS<br>void *tls_cnx<br>void *tls_data_cnx"]

subgraph subGraph3 ["Throttling Configuration"]
    THROTTLE_ENABLED
    THROTTLE_VARS
    THROTTLE_ENABLED --> THROTTLE_VARS
end

subgraph subGraph2 ["Upload Script Configuration"]
    UPLOAD_ENABLED
    UPLOAD_VARS
    UPLOAD_ENABLED --> UPLOAD_VARS
end

subgraph subGraph1 ["Quota Configuration"]
    QUOTA_ENABLED
    QUOTA_VARS
    QUOTA_ENABLED --> QUOTA_VARS
end

subgraph subGraph0 ["TLS Configuration"]
    TLS_ENABLED
    TLS_VARS
    TLS_DISABLED
    TLS_ENABLED --> TLS_VARS
    TLS_ENABLED --> TLS_DISABLED
end
```

**Key Feature Flags:**

* `WITH_TLS` - TLS/SSL encryption support [src/globals.h L174-L184](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L174-L184)
* `QUOTAS` - Disk quota management [src/globals.h L150-L153](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L150-L153)
* `WITH_UPLOAD_SCRIPT` - Post-upload script execution [src/globals.h L131-L135](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L131-L135)
* `THROTTLING` - Bandwidth throttling [src/globals.h L19-L22](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L19-L22)
* `FTPWHO` - Session monitoring [src/globals.h L124-L129](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L124-L129)

Sources: [src/globals.h L19-L200](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/globals.h#L19-L200)

 [src/ftpd_p.h L62-L110](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/ftpd_p.h#L62-L110)
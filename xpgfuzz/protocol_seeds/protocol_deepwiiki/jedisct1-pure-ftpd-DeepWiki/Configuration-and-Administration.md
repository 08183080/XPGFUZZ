# Configuration and Administration

> **Relevant source files**
> * [README](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README)
> * [configure.ac](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac)
> * [pure-ftpd.conf.in](https://github.com/jedisct1/pure-ftpd/blob/3818577a/pure-ftpd.conf.in)
> * [src/simpleconf.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c)
> * [src/simpleconf.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.h)
> * [src/simpleconf_ftpd.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h)

This document covers Pure-FTPd's comprehensive configuration system and administrative tools. Pure-FTPd provides multiple layers of configuration: build-time compilation options, runtime configuration files, and command-line parameters. The system uses a flexible configuration parser that supports various authentication backends, security features, and operational settings.

For information about specific authentication methods, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management). For details about helper utilities and monitoring tools, see [Administrative Utilities](/jedisct1/pure-ftpd/5.3-administrative-utilities).

## Configuration Architecture Overview

Pure-FTPd employs a multi-layered configuration architecture that separates build-time feature selection from runtime operational settings:

```mermaid
flowchart TD

CONFIGURE["configure.ac<br>Autoconf Script"]
BUILD_OPTS["Build Options<br>--with-mysql, --with-tls, etc."]
COMPILE["Compilation<br>Feature Selection"]
CONF_FILE["pure-ftpd.conf<br>Configuration File"]
SIMPLECONF["simpleconf.c<br>Configuration Parser"]
CMD_LINE["Command Line Options<br>Runtime Parameters"]
MAPPING["simpleconf_ftpd.h<br>Directive Mapping"]
VALIDATION["Configuration Validation<br>Type Checking & Parsing"]
FTPD_MAIN["pure-ftpd Process<br>Server Execution"]
FEATURE_FLAGS["Compiled Features<br>Authentication, TLS, etc."]

COMPILE --> FEATURE_FLAGS
SIMPLECONF --> MAPPING
VALIDATION --> CMD_LINE
CMD_LINE --> FTPD_MAIN

subgraph subGraph3 ["Server Runtime"]
    FTPD_MAIN
    FEATURE_FLAGS
    FEATURE_FLAGS --> FTPD_MAIN
end

subgraph subGraph2 ["Configuration Mapping"]
    MAPPING
    VALIDATION
    MAPPING --> VALIDATION
end

subgraph subGraph1 ["Runtime Configuration System"]
    CONF_FILE
    SIMPLECONF
    CMD_LINE
    CONF_FILE --> SIMPLECONF
end

subgraph subGraph0 ["Build-Time Configuration"]
    CONFIGURE
    BUILD_OPTS
    COMPILE
    CONFIGURE --> BUILD_OPTS
    BUILD_OPTS --> COMPILE
end
```

**Sources:** [configure.ac L1-L1500](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac#L1-L1500)

 [src/simpleconf.c L1-L743](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L1-L743)

 [src/simpleconf_ftpd.h L1-L125](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L1-L125)

 [pure-ftpd.conf.in L1-L468](https://github.com/jedisct1/pure-ftpd/blob/3818577a/pure-ftpd.conf.in#L1-L468)

## Build System Configuration

### Autoconf Build System

The primary build configuration is handled through GNU Autoconf via the `configure.ac` script. This system performs feature detection, dependency checking, and compilation option processing:

```mermaid
flowchart TD

SYS_HEADERS["System Headers<br>AC_CHECK_HEADERS"]
SYS_FUNCS["System Functions<br>AC_CHECK_FUNCS"]
LIB_DETECT["Library Detection<br>AC_CHECK_LIB"]
WITH_OPTS["--with-* Options<br>Feature Enablement"]
WITHOUT_OPTS["--without-* Options<br>Feature Disablement"]
SECURITY_OPTS["Security Options<br>Stack Protection, PIE"]
AUTH_MYSQL["--with-mysql<br>MySQL Authentication"]
AUTH_PGSQL["--with-pgsql<br>PostgreSQL Authentication"]
AUTH_LDAP["--with-ldap<br>LDAP Authentication"]
AUTH_PUREDB["--with-puredb<br>Virtual Users"]
AUTH_PAM["--with-pam<br>PAM Authentication"]
CONFIG_H["config.h<br>Preprocessor Definitions"]
MAKEFILE["Makefile<br>Build Rules"]
FEATURE_MACROS["Feature Macros<br>WITH_MYSQL, WITH_TLS, etc."]

SYS_HEADERS --> CONFIG_H
SYS_FUNCS --> CONFIG_H
LIB_DETECT --> CONFIG_H
WITH_OPTS --> FEATURE_MACROS
WITHOUT_OPTS --> FEATURE_MACROS
SECURITY_OPTS --> FEATURE_MACROS
AUTH_MYSQL --> FEATURE_MACROS
AUTH_PGSQL --> FEATURE_MACROS
AUTH_LDAP --> FEATURE_MACROS
AUTH_PUREDB --> FEATURE_MACROS
AUTH_PAM --> FEATURE_MACROS

subgraph subGraph3 ["Generated Output"]
    CONFIG_H
    MAKEFILE
    FEATURE_MACROS
    FEATURE_MACROS --> CONFIG_H
    CONFIG_H --> MAKEFILE
end

subgraph subGraph2 ["Authentication Backends"]
    AUTH_MYSQL
    AUTH_PGSQL
    AUTH_LDAP
    AUTH_PUREDB
    AUTH_PAM
end

subgraph subGraph1 ["Compilation Options"]
    WITH_OPTS
    WITHOUT_OPTS
    SECURITY_OPTS
end

subgraph subGraph0 ["Feature Detection"]
    SYS_HEADERS
    SYS_FUNCS
    LIB_DETECT
end
```

**Sources:** [configure.ac L1-L100](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac#L1-L100)

 [configure.ac L283-L436](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac#L283-L436)

 [configure.ac L1207-L1500](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac#L1207-L1500)

### Key Build-Time Options

The build system supports extensive customization through configure options:

| Option Category | Key Options | Purpose |
| --- | --- | --- |
| Authentication | `--with-mysql`, `--with-pgsql`, `--with-ldap` | Enable database authentication backends |
| Security | `--with-tls`, `--with-privsep` | SSL/TLS support and privilege separation |
| Virtual Users | `--with-puredb`, `--with-extauth` | Virtual user authentication systems |
| Features | `--with-quotas`, `--with-throttling`, `--with-ratios` | Quota management and bandwidth control |
| Logging | `--with-altlog`, `--with-ftpwho` | Alternative logging formats and monitoring |
| Minimal Build | `--with-minimal`, `--without-*` | Reduced feature set for embedded systems |

**Sources:** [configure.ac L305-L436](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac#L305-L436)

 [README L102-L360](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README#L102-L360)

## Runtime Configuration System

### SimpleConf Configuration Parser

Pure-FTPd uses a custom configuration parser called SimpleConf that provides flexible configuration file processing with type validation and template substitution:

```mermaid
flowchart TD

CONF_READ["Configuration File<br>pure-ftpd.conf"]
LINE_PARSE["Line-by-Line Parsing<br>try_entry()"]
PATTERN_MATCH["Pattern Matching<br>Property Recognition"]
STATE_PROP["STATE_PROPNAME<br>Property Name Recognition"]
STATE_VALUE["STATE_MATCH_*<br>Value Type Validation"]
STATE_TEMPLATE["STATE_TEMPLATE_*<br>Output Generation"]
ENTRY_MAP["SimpleConfEntry[]<br>Directive Definitions"]
TYPE_VALID["Type Validation<br><bool>, <digits>, <any>"]
CMD_GEN["Command Line Generation<br>Template Substitution"]
ARG_ARRAY["char* argv[]<br>Command Line Arguments"]
VALIDATION["Argument Validation<br>Range & Format Checks"]
SERVER_OPTS["Server Options<br>Runtime Configuration"]

PATTERN_MATCH --> STATE_PROP
CMD_GEN --> STATE_TEMPLATE
STATE_TEMPLATE --> ARG_ARRAY

subgraph subGraph3 ["Output Processing"]
    ARG_ARRAY
    VALIDATION
    SERVER_OPTS
    ARG_ARRAY --> VALIDATION
    VALIDATION --> SERVER_OPTS
end

subgraph subGraph2 ["Configuration Mapping"]
    ENTRY_MAP
    TYPE_VALID
    CMD_GEN
    ENTRY_MAP --> TYPE_VALID
    TYPE_VALID --> CMD_GEN
end

subgraph subGraph1 ["State Machine Parser"]
    STATE_PROP
    STATE_VALUE
    STATE_TEMPLATE
    STATE_PROP --> STATE_VALUE
    STATE_VALUE --> STATE_TEMPLATE
end

subgraph subGraph0 ["Configuration File Processing"]
    CONF_READ
    LINE_PARSE
    PATTERN_MATCH
    CONF_READ --> LINE_PARSE
    LINE_PARSE --> PATTERN_MATCH
end
```

**Sources:** [src/simpleconf.c L115-L542](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L115-L542)

 [src/simpleconf.c L577-L742](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L577-L742)

### Configuration File Format

The configuration file uses a simple key-value format with type validation and boolean support:

```mermaid
flowchart TD

INCLUDE_DIR["Include Directive<br>Include additional_file.conf"]
TEMPLATE_SUB["Template Substitution<br>$* and $0-$9 variables"]
BOOLEAN_OPT["Boolean Properties<br>Property? syntax"]
PROPERTY["Property Name<br>ChrootEveryone"]
SEPARATOR["Separator<br>= or :"]
VALUE["Value<br>yes/no, numbers, strings"]
COMMENT["Comments<br># prefix"]
BOOL_TYPE["Boolean Values<br>yes/no, true/false, on/off"]
NUM_TYPE["Numeric Values<br><digits>, ranges"]
STR_TYPE["String Values<br><any>, <nospace>"]
PATH_TYPE["File Paths<br>Quoted strings"]

VALUE --> BOOL_TYPE
VALUE --> NUM_TYPE
VALUE --> STR_TYPE
VALUE --> PATH_TYPE

subgraph subGraph1 ["Value Types"]
    BOOL_TYPE
    NUM_TYPE
    STR_TYPE
    PATH_TYPE
end

subgraph subGraph0 ["Configuration Syntax"]
    PROPERTY
    SEPARATOR
    VALUE
    COMMENT
    PROPERTY --> SEPARATOR
    SEPARATOR --> VALUE
end

subgraph subGraph2 ["Special Features"]
    INCLUDE_DIR
    TEMPLATE_SUB
    BOOLEAN_OPT
    INCLUDE_DIR --> TEMPLATE_SUB
    BOOLEAN_OPT --> TEMPLATE_SUB
end
```

**Sources:** [src/simpleconf.c L172-L443](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L172-L443)

 [src/simpleconf_ftpd.h L6-L122](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L6-L122)

 [pure-ftpd.conf.in L18-L468](https://github.com/jedisct1/pure-ftpd/blob/3818577a/pure-ftpd.conf.in#L18-L468)

## Configuration Directive Mapping

The `simpleconf_ftpd.h` file defines the mapping between configuration file directives and command-line options:

### Authentication Configuration Mapping

| Configuration Directive | Command Line Option | Compilation Requirement |
| --- | --- | --- |
| `MySQLConfigFile <path>` | `--login=mysql:<path>` | `WITH_MYSQL` |
| `PGSQLConfigFile <path>` | `--login=pgsql:<path>` | `WITH_PGSQL` |
| `LDAPConfigFile <path>` | `--login=ldap:<path>` | `WITH_LDAP` |
| `PureDB <path>` | `--login=puredb:<path>` | `WITH_PUREDB` |
| `PAMAuthentication <bool>` | `--login=pam` | `USE_PAM` |
| `ExtAuth <path>` | `--login=extauth:<path>` | `WITH_EXTAUTH` |

### Security and Access Control Mapping

| Configuration Directive | Command Line Option | Purpose |
| --- | --- | --- |
| `ChrootEveryone <bool>` | `--chrooteveryone` | Chroot all users |
| `TLS <digits>` | `--tls=<level>` | TLS security level |
| `TrustedGID <digits>` | `--trustedgid=<gid>` | Trusted group ID |
| `MinUID <digits>` | `--minuid=<uid>` | Minimum user ID |
| `MaxClientsNumber <digits>` | `--maxclientsnumber=<num>` | Connection limits |

**Sources:** [src/simpleconf_ftpd.h L56-L73](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L56-L73)

 [src/simpleconf_ftpd.h L36-L108](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf_ftpd.h#L36-L108)

## Configuration File Processing Flow

The configuration system processes files through multiple stages with error handling and validation:

```mermaid
flowchart TD

FILE_OPEN["fopen() Configuration File<br>append_to_command_line_from_file()"]
LINE_READ["fgets() Line Reading<br>SC_MAX_ARG_LENGTH buffer"]
LINE_CHOMP["chomp() Line Cleaning<br>Remove trailing whitespace"]
ENTRY_LOOP["Entry Matching Loop<br>try_entry() for each directive"]
PATTERN_MATCH["Pattern Matching<br>Property name recognition"]
VALUE_PARSE["Value Parsing<br>Type-specific validation"]
TYPE_CHECK["Type Validation<br>Boolean, numeric, string checks"]
RANGE_CHECK["Range Validation<br>Numeric bounds checking"]
SYNTAX_CHECK["Syntax Validation<br>Format compliance"]
TEMPLATE_PROC["Template Processing<br>$* variable substitution"]
ARG_BUILD["Argument Building<br>Command line construction"]
ARGV_ARRAY["argv[] Array<br>Final argument list"]
SYNTAX_ERR["Syntax Errors<br>Line number reporting"]
TYPE_ERR["Type Errors<br>Invalid value format"]
FILE_ERR["File Errors<br>Missing includes"]

LINE_CHOMP --> ENTRY_LOOP
VALUE_PARSE --> TYPE_CHECK
SYNTAX_CHECK --> TEMPLATE_PROC
VALUE_PARSE --> SYNTAX_ERR
TYPE_CHECK --> TYPE_ERR
FILE_OPEN --> FILE_ERR

subgraph subGraph4 ["Error Handling"]
    SYNTAX_ERR
    TYPE_ERR
    FILE_ERR
end

subgraph subGraph3 ["Output Generation"]
    TEMPLATE_PROC
    ARG_BUILD
    ARGV_ARRAY
    TEMPLATE_PROC --> ARG_BUILD
    ARG_BUILD --> ARGV_ARRAY
end

subgraph subGraph2 ["Validation Stage"]
    TYPE_CHECK
    RANGE_CHECK
    SYNTAX_CHECK
    TYPE_CHECK --> RANGE_CHECK
    RANGE_CHECK --> SYNTAX_CHECK
end

subgraph subGraph1 ["Parsing Stage"]
    ENTRY_LOOP
    PATTERN_MATCH
    VALUE_PARSE
    ENTRY_LOOP --> PATTERN_MATCH
    PATTERN_MATCH --> VALUE_PARSE
end

subgraph subGraph0 ["File Reading Stage"]
    FILE_OPEN
    LINE_READ
    LINE_CHOMP
    FILE_OPEN --> LINE_READ
    LINE_READ --> LINE_CHOMP
end
```

**Sources:** [src/simpleconf.c L577-L712](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L577-L712)

 [src/simpleconf.c L606-L637](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L606-L637)

 [src/simpleconf.c L696-L707](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L696-L707)

## Administrative Configuration Patterns

### Multi-Level Configuration Hierarchy

Pure-FTPd implements a hierarchical configuration system where settings can be specified at different levels with appropriate precedence:

| Configuration Level | Source | Precedence | Purpose |
| --- | --- | --- | --- |
| Compile-time | `configure` options | Lowest | Feature availability |
| Configuration file | `pure-ftpd.conf` | Medium | Default settings |
| Command line | Runtime arguments | Highest | Override settings |
| Include files | Additional `.conf` files | Variable | Modular configuration |

### Configuration Validation and Error Reporting

The SimpleConf parser provides comprehensive error reporting with line-number precision:

```mermaid
flowchart TD

SYNTAX_ERROR["Syntax Errors<br>Invalid format"]
TYPE_ERROR["Type Errors<br>Wrong value type"]
PROP_ERROR["Property Errors<br>Unknown directive"]
LINE_NUM["Line Number<br>Precise location"]
COLUMN_POS["Column Position<br>Character offset"]
ERROR_MSG["Error Message<br>Descriptive text"]
PARSE_ABORT["Parse Abortion<br>Stop processing"]
ERROR_LOG["Error Logging<br>stderr output"]
EXIT_CODE["Exit Code<br>Non-zero return"]

SYNTAX_ERROR --> LINE_NUM
TYPE_ERROR --> LINE_NUM
PROP_ERROR --> LINE_NUM
ERROR_MSG --> PARSE_ABORT

subgraph subGraph2 ["Error Handling"]
    PARSE_ABORT
    ERROR_LOG
    EXIT_CODE
    PARSE_ABORT --> ERROR_LOG
    ERROR_LOG --> EXIT_CODE
end

subgraph subGraph1 ["Error Reporting"]
    LINE_NUM
    COLUMN_POS
    ERROR_MSG
    LINE_NUM --> COLUMN_POS
    COLUMN_POS --> ERROR_MSG
end

subgraph subGraph0 ["Error Detection"]
    SYNTAX_ERROR
    TYPE_ERROR
    PROP_ERROR
end
```

**Sources:** [src/simpleconf.c L624-L638](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L624-L638)

 [src/simpleconf.c L696-L707](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/simpleconf.c#L696-L707)

This configuration and administration system provides Pure-FTPd with flexible, type-safe configuration management that supports complex deployment scenarios while maintaining ease of use for basic setups.
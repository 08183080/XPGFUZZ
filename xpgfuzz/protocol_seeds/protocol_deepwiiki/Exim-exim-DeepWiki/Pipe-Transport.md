# Pipe Transport

> **Relevant source files**
> * [src/src/transports/appendfile.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c)
> * [src/src/transports/autoreply.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c)
> * [src/src/transports/lmtp.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c)
> * [src/src/transports/pipe.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c)
> * [src/src/transports/tf_maildir.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/tf_maildir.c)

## Purpose and Scope

The pipe transport provides a mechanism for delivering messages to external commands or programs. It enables Exim to hand off messages to scripts, filters, or other mail processing utilities by executing commands and feeding message data through standard input pipes.

For information about file-based delivery, see [File and Directory Storage](/Exim/exim/6.1-file-and-directory-storage). For SMTP-based delivery, see [SMTP Transport](/Exim/exim/6.2-smtp-transport). For automated responses, see [Autoreply Transport](/Exim/exim/6.4-autoreply-transport).

## Architecture Overview

The pipe transport creates child processes to execute external commands, managing input/output pipes and implementing proper security controls. It supports both direct command execution and shell-based execution modes.

### Core Components

```mermaid
flowchart TD

PTE["pipe_transport_entry()"]
PTI["pipe_transport_init()"]
PTS["pipe_transport_setup()"]
PTOB["pipe_transport_options_block"]
SDC["set_up_direct_command()"]
SSC["set_up_shell_command()"]
TSC["transport_set_up_command()"]
CO["child_open()"]
CC["child_close()"]
TWM["transport_write_message()"]
CMD["cmd option"]
ENV["environment option"]
PATH["path option"]
TO["timeout option"]

PTE --> SDC
PTE --> SSC
PTE --> TWM
PTE --> CO
PTE --> CC
PTOB --> CMD
PTOB --> ENV
PTOB --> PATH
PTOB --> TO

subgraph Configuration ["Configuration"]
    CMD
    ENV
    PATH
    TO
end

subgraph subGraph2 ["Process Management"]
    CO
    CC
    TWM
end

subgraph subGraph1 ["Command Setup"]
    SDC
    SSC
    TSC
    SDC --> TSC
    SSC --> TSC
end

subgraph subGraph0 ["Pipe Transport Core"]
    PTE
    PTI
    PTS
    PTOB
    PTI --> PTOB
end
```

Sources: [src/src/transports/pipe.c L510-L1127](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L510-L1127)

### Transport Registration

```mermaid
flowchart TD

TI["transport_info"]
PTI_STRUCT["pipe_transport_info"]
DI["drinfo"]
DN["driver_name: 'pipe'"]
OPT["options"]
OPTC["options_count"]
OPTB["options_block"]
INIT["init function"]
CODE["code function"]

DI --> DN
DI --> OPT
DI --> OPTC
DI --> OPTB
DI --> INIT
TI --> CODE

subgraph subGraph1 ["Driver Interface"]
    DN
    OPT
    OPTC
    OPTB
    INIT
    CODE
end

subgraph subGraph0 ["Transport System"]
    TI
    PTI_STRUCT
    DI
    PTI_STRUCT --> TI
    TI --> DI
end
```

Sources: [src/src/transports/pipe.c L1136-L1152](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L1136-L1152)

## Configuration and Options

The pipe transport is configured through the `pipe_transport_options_block` structure, which contains numerous options controlling command execution, environment, and behavior.

### Key Configuration Options

| Option | Type | Purpose |
| --- | --- | --- |
| `command` | string | The command to execute |
| `path` | string | Search path for command resolution |
| `environment` | string | Additional environment variables |
| `timeout` | time | Maximum execution time |
| `use_shell` | bool | Whether to use shell for command execution |
| `allow_commands` | string | List of permitted commands |
| `restrict_to_path` | bool | Restrict commands to PATH directories |
| `use_bsmtp` | bool | Use BSMTP format for output |

Sources: [src/src/transports/pipe.c L30-L72](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L30-L72)

### Default Configuration

```mermaid
flowchart TD

NO_SHELL["use_shell: FALSE"]
NO_ALLOW["allow_commands: NULL"]
NO_RESTRICT["restrict_to_path: FALSE"]
PATH_DEF["path: '/bin:/usr/bin'"]
TEMP_ERR["temp_errors: EX_TEMPFAIL:EX_CANTCREAT"]
UMASK_DEF["umask: 022"]
MAX_OUT["max_output: 20480"]
TIMEOUT_DEF["timeout: 3600 seconds"]

subgraph subGraph1 ["Security Defaults"]
    NO_SHELL
    NO_ALLOW
    NO_RESTRICT
end

subgraph subGraph0 ["Default Values"]
    PATH_DEF
    TEMP_ERR
    UMASK_DEF
    MAX_OUT
    TIMEOUT_DEF
end
```

Sources: [src/src/transports/pipe.c L93-L101](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L93-L101)

## Command Execution Modes

The pipe transport supports two primary modes of command execution: direct command execution and shell-based execution.

### Direct Command Mode

In direct mode, the transport parses the command line and executes the program directly without involving a shell.

```mermaid
sequenceDiagram
  participant Pipe Transport
  participant set_up_direct_command()
  participant transport_set_up_command()
  participant Process System

  Pipe Transport->>set_up_direct_command(): Parse command line
  set_up_direct_command()->>transport_set_up_command(): Create argument vector
  transport_set_up_command()->>set_up_direct_command(): Return argv array
  set_up_direct_command()->>set_up_direct_command(): Check command permissions
  set_up_direct_command()->>set_up_direct_command(): Resolve command path
  set_up_direct_command()->>Process System: Execute via child_open()
```

Sources: [src/src/transports/pipe.c L296-L397](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L296-L397)

### Shell Command Mode

When `use_shell` is enabled, commands are executed through `/bin/sh -c`.

```mermaid
sequenceDiagram
  participant Pipe Transport
  participant set_up_shell_command()
  participant Expansion Engine
  participant Process System

  Pipe Transport->>set_up_shell_command(): Setup shell execution
  set_up_shell_command()->>Expansion Engine: Expand $pipe_addresses
  Expansion Engine->>set_up_shell_command(): Return expanded command
  set_up_shell_command()->>set_up_shell_command(): Build argv["/bin/sh", "-c", command]
  set_up_shell_command()->>Process System: Execute via child_open()
```

Sources: [src/src/transports/pipe.c L419-L497](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L419-L497)

### Command Resolution Process

```mermaid
flowchart TD

START["Command String"]
EXPAND["Expand Variables"]
CHECK_PERMS["Check Permissions"]
RESOLVE_PATH["Resolve Path"]
EXECUTE["Execute Command"]
ALLOW_CMD["allow_commands check"]
RESTRICT_PATH["restrict_to_path check"]
PATH_SEARCH["PATH directory search"]

CHECK_PERMS --> ALLOW_CMD
CHECK_PERMS --> RESTRICT_PATH
RESOLVE_PATH --> PATH_SEARCH
PATH_SEARCH --> EXECUTE

subgraph subGraph1 ["Permission Checks"]
    ALLOW_CMD
    RESTRICT_PATH
    PATH_SEARCH
end

subgraph subGraph0 ["Command Processing"]
    START
    EXPAND
    CHECK_PERMS
    RESOLVE_PATH
    EXECUTE
    START --> EXPAND
    EXPAND --> CHECK_PERMS
    CHECK_PERMS --> RESOLVE_PATH
end
```

Sources: [src/src/transports/pipe.c L318-L397](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L318-L397)

## Process Lifecycle Management

The pipe transport manages complex process lifecycles involving multiple child processes for command execution and output handling.

### Process Creation and Management

```mermaid
flowchart TD

ENTRY["pipe_transport_entry()"]
SETUP["Setup environment"]
CREATE["Create child processes"]
MONITOR["Monitor execution"]
CLEANUP["Cleanup resources"]
CMD_PROC["Command Execution"]
STDIN["Read from pipe"]
PROCESS["Process message"]
EXIT["Exit with status"]
OUT_PROC["Output Handler"]
STDOUT["Read command output"]
LIMIT["Check output limits"]
WRITE["Write to return file"]

CREATE --> CMD_PROC
CREATE --> OUT_PROC

subgraph subGraph2 ["Child Process (Output Handler)"]
    OUT_PROC
    STDOUT
    LIMIT
    WRITE
end

subgraph subGraph1 ["Child Process (Command)"]
    CMD_PROC
    STDIN
    PROCESS
    EXIT
end

subgraph subGraph0 ["Main Process"]
    ENTRY
    SETUP
    CREATE
    MONITOR
    CLEANUP
    ENTRY --> SETUP
    SETUP --> CREATE
    CREATE --> MONITOR
    MONITOR --> CLEANUP
end
```

Sources: [src/src/transports/pipe.c L705-L770](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L705-L770)

### Output Handling Architecture

```mermaid
sequenceDiagram
  participant Main Transport
  participant Command Process
  participant Output Process
  participant Return File

  Main Transport->>Command Process: Create with pipes
  Main Transport->>Output Process: Fork output handler
  Main Transport->>Command Process: Write message via stdin
  Command Process->>Output Process: Send output via stdout
  Output Process->>Output Process: Check output limits
  Output Process->>Return File: Write to return file
  Output Process->>Main Transport: Signal completion
  Main Transport->>Command Process: Wait for completion
```

Sources: [src/src/transports/pipe.c L742-L769](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L742-L769)

## Environment and Security

The pipe transport implements comprehensive security controls and environment management.

### Environment Variable Setup

```mermaid
flowchart TD

HOST_VAR["HOST (if host_list)"]
TZ_VAR["TZ (timezone)"]
CUSTOM["Custom environment"]
PATH_VAR["PATH"]
SHELL_VAR["SHELL=/bin/sh"]
LOCAL_PART["LOCAL_PART"]
LOGNAME["LOGNAME"]
USER["USER"]
DOMAIN["DOMAIN"]
HOME["HOME"]
MESSAGE_ID["MESSAGE_ID"]
RECIPIENT["RECIPIENT"]
SENDER["SENDER"]

subgraph subGraph2 ["Optional Variables"]
    HOST_VAR
    TZ_VAR
    CUSTOM
end

subgraph subGraph1 ["Path Variables"]
    PATH_VAR
    SHELL_VAR
end

subgraph subGraph0 ["Standard Variables"]
    LOCAL_PART
    LOGNAME
    USER
    DOMAIN
    HOME
    MESSAGE_ID
    RECIPIENT
    SENDER
end
```

Sources: [src/src/transports/pipe.c L625-L677](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L625-L677)

### Security Controls

```mermaid
flowchart TD

MAX_OUTPUT["max_output limits"]
OUTPUT_MONITORING["Output size monitoring"]
KILL_RUNAWAY["Kill runaway processes"]
UMASK_SET["umask setting"]
PROCESS_GROUP["Process group leadership"]
TIMEOUT_KILL["Timeout-based termination"]
RESOURCE_LIMITS["Resource limit enforcement"]
ALLOW_LIST["allow_commands whitelist"]
PATH_RESTRICT["restrict_to_path enforcement"]
SHELL_DISABLE["use_shell restrictions"]
TAINT_CHECK["Tainted command detection"]

subgraph subGraph2 ["Output Security"]
    MAX_OUTPUT
    OUTPUT_MONITORING
    KILL_RUNAWAY
end

subgraph subGraph1 ["Process Security"]
    UMASK_SET
    PROCESS_GROUP
    TIMEOUT_KILL
    RESOURCE_LIMITS
end

subgraph subGraph0 ["Command Security"]
    ALLOW_LIST
    PATH_RESTRICT
    SHELL_DISABLE
    TAINT_CHECK
end
```

Sources: [src/src/transports/pipe.c L318-L366](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L318-L366)

 [src/src/transports/pipe.c L582-L589](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L582-L589)

 [src/src/transports/pipe.c L753-L765](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L753-L765)

## Error Handling and Status Codes

The pipe transport implements sophisticated error handling with configurable temporary failure codes and comprehensive status reporting.

### Error Classification

```mermaid
flowchart TD

CMD_PERM["Command permission denied"]
PATH_ERR["Command not in PATH"]
ENV_ERR["Environment expansion failure"]
WRITE_FAIL["Write errors to stdin"]
EPIPE_ERR["EPIPE: Broken pipe"]
OUTPUT_LIMIT["Output size exceeded"]
EXEC_FAIL["EX_EXECFAILED: Command not found"]
TIMEOUT_ERR["Timeout: Process exceeded limit"]
SIGNAL_TERM["Signal termination"]
WAIT_FAIL["Wait() system call failure"]

subgraph subGraph2 ["Configuration Errors"]
    CMD_PERM
    PATH_ERR
    ENV_ERR
end

subgraph subGraph1 ["I/O Errors"]
    WRITE_FAIL
    EPIPE_ERR
    OUTPUT_LIMIT
end

subgraph subGraph0 ["Process Errors"]
    EXEC_FAIL
    TIMEOUT_ERR
    SIGNAL_TERM
    WAIT_FAIL
end
```

Sources: [src/src/transports/pipe.c L883-L1109](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L883-L1109)

### Status Code Handling

The transport uses the `temp_errors` configuration to determine which exit codes should be treated as temporary failures versus permanent failures.

```mermaid
flowchart TD

RC["Return Code"]
TEMP_CHECK["Check temp_errors"]
DEFER_STATUS["DEFER status"]
FAIL_STATUS["FAIL status"]

subgraph subGraph0 ["Exit Code Processing"]
    RC
    TEMP_CHECK
    DEFER_STATUS
    FAIL_STATUS
    RC --> TEMP_CHECK
    TEMP_CHECK --> DEFER_STATUS
    TEMP_CHECK --> FAIL_STATUS
end
```

Sources: [src/src/transports/pipe.c L1043-L1060](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L1043-L1060)

## BSMTP Support

When `use_bsmtp` is enabled, the pipe transport formats output as Batch SMTP, useful for feeding messages to SMTP-aware programs.

### BSMTP Message Flow

```mermaid
sequenceDiagram
  participant Pipe Transport
  participant External Command

  Pipe Transport->>External Command: MAIL FROM:<sender>
  Pipe Transport->>External Command: RCPT TO:<recipient1>
  Pipe Transport->>External Command: RCPT TO:<recipient2>
  Pipe Transport->>External Command: DATA
  Pipe Transport->>External Command: Message Headers
  Pipe Transport->>External Command: Message Body
  Pipe Transport->>External Command: .
```

Sources: [src/src/transports/pipe.c L821-L862](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L821-L862)

The transport automatically sets appropriate check and escape strings for BSMTP format and enables header escaping when this mode is activated.

Sources: [src/src/transports/pipe.c L231-L236](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c#L231-L236)
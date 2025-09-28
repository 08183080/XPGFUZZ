# Autoreply Transport

> **Relevant source files**
> * [src/src/transports/appendfile.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/appendfile.c)
> * [src/src/transports/autoreply.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c)
> * [src/src/transports/lmtp.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/lmtp.c)
> * [src/src/transports/pipe.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/pipe.c)
> * [src/src/transports/tf_maildir.c](https://github.com/Exim/exim/blob/29568b25/src/src/transports/tf_maildir.c)

## Purpose and Scope

The Autoreply Transport in Exim is a specialized transport mechanism designed to generate and send automatic replies to incoming emails. It provides a comprehensive solution for implementing vacation messages, auto-acknowledgments, and other automated response systems directly within Exim.

Unlike other transports that deliver messages to final destinations, the Autoreply Transport creates entirely new messages in response to received ones. It features mechanisms to prevent mail loops and excessive responses through its once-only capabilities.

## Architecture and Integration

#### Autoreply Transport Code Structure

```mermaid
flowchart TD

autoreply_transport_info["autoreply_transport_info"]
autoreply_transport_entry["autoreply_transport_entry()"]
addr_reply_check["addr->reply exists?"]
addr_reply_data["addr->reply fields"]
transport_config["autoreply_transport_options"]
checkexpand["checkexpand()"]
check_never_mail["check_never_mail()"]
once_processing["Once-only Processing"]
dbm_mode["exim_dbopen()"]
cache_mode["Fixed-size Cache File"]
child_open_exim["child_open_exim()"]
message_generation["Message Generation"]
transport_write_message["transport_write_message()"]
database_update["Database Update"]
child_close["child_close()"]

autoreply_transport_info --> autoreply_transport_entry
autoreply_transport_entry --> addr_reply_check
addr_reply_check --> addr_reply_data
addr_reply_check --> transport_config
addr_reply_data --> checkexpand
transport_config --> checkexpand
checkexpand --> check_never_mail
check_never_mail --> once_processing
once_processing --> dbm_mode
once_processing --> cache_mode
dbm_mode --> child_open_exim
cache_mode --> child_open_exim
child_open_exim --> message_generation
message_generation --> transport_write_message
transport_write_message --> database_update
database_update --> child_close
```

#### Transport Registration in Exim System

```mermaid
flowchart TD

transport_system["Transport System"]
appendfile_transport_info["appendfile_transport_info"]
pipe_transport_info["pipe_transport_info"]
autoreply_transport_info["autoreply_transport_info"]
lmtp_transport_info["lmtp_transport_info"]
driver_name["driver_name: 'autoreply'"]
init_func["init: autoreply_transport_init"]
entry_func["code: autoreply_transport_entry"]
local_flag["local: TRUE"]
options_block["options_block: autoreply_transport_option_defaults"]

transport_system --> appendfile_transport_info
transport_system --> pipe_transport_info
transport_system --> autoreply_transport_info
transport_system --> lmtp_transport_info
autoreply_transport_info --> driver_name
autoreply_transport_info --> init_func
autoreply_transport_info --> entry_func
autoreply_transport_info --> local_flag
autoreply_transport_info --> options_block
```

Sources: [src/src/transports/autoreply.c L262-L802](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L262-L802)

 [src/src/transports/autoreply.c L811-L827](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L811-L827)

 [src/src/transports/autoreply.c L86-L100](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L86-L100)

## Transport Registration

The Autoreply Transport is registered with Exim's transport system using the `autoreply_transport_info` structure, which defines its characteristics, capabilities, and integration points:

```mermaid
flowchart TD

AutoreplyTransportInfo["autoreply_transport_info"]
DriverName["driver_name: \autoreply\"]
OptionsDefinition["options: autoreply_transport_options"]
InitFunction["init: autoreply_transport_init()"]
EntryPoint["code: autoreply_transport_entry()"]
LocalFlag["local: TRUE"]

AutoreplyTransportInfo --> DriverName
AutoreplyTransportInfo --> OptionsDefinition
AutoreplyTransportInfo --> InitFunction
AutoreplyTransportInfo --> EntryPoint
AutoreplyTransportInfo --> LocalFlag
```

Sources: [src/src/transports/autoreply.c L811-L827](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L811-L827)

## Configuration Options

The autoreply transport uses the `autoreply_transport_options[]` array and `autoreply_transport_options_block` structure:

### Message Content Options from autoreply_transport_options[]

| Option | Type | Structure Field | Description |
| --- | --- | --- | --- |
| `from` | `opt_stringptr` | `LOFF(from)` | Sets the From: header in the reply message |
| `reply_to` | `opt_stringptr` | `LOFF(reply_to)` | Sets the Reply-To: header |
| `to` | `opt_stringptr` | `LOFF(to)` | Specifies recipient(s) in the To: header |
| `cc` | `opt_stringptr` | `LOFF(cc)` | Specifies carbon copy recipient(s) |
| `bcc` | `opt_stringptr` | `LOFF(bcc)` | Specifies blind carbon copy recipient(s) |
| `subject` | `opt_stringptr` | `LOFF(subject)` | Sets the Subject: header |
| `headers` | `opt_stringptr` | `LOFF(headers)` | Specifies additional headers to add |
| `text` | `opt_stringptr` | `LOFF(text)` | Specifies the message body content |

Sources: [src/src/transports/autoreply.c L25-L43](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L25-L43)

 [src/src/transports/autoreply.c L23](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L23-L23)

### File and Control Options from autoreply_transport_options[]

| Option | Type | Structure Field | Default | Description |
| --- | --- | --- | --- | --- |
| `file` | `opt_stringptr` | `LOFF(file)` | NULL | Path to file whose contents are included in reply |
| `file_expand` | `opt_bool` | `LOFF(file_expand)` | FALSE | Expands variables in file contents when TRUE |
| `file_optional` | `opt_bool` | `LOFF(file_optional)` | FALSE | Doesn't fail if file missing when TRUE |
| `log` | `opt_stringptr` | `LOFF(logfile)` | NULL | Path to file for logging autoreply activity |
| `never_mail` | `opt_stringptr` | `LOFF(never_mail)` | NULL | Address list to never send autoreplies to |
| `return_message` | `opt_bool` | `LOFF(return_message)` | FALSE | Includes original message in reply when TRUE |
| `mode` | `opt_octint` | `LOFF(mode)` | 0600 | File mode for created files (octal) |

### Once-Only Options from autoreply_transport_options[]

| Option | Type | Structure Field | Description |
| --- | --- | --- | --- |
| `once` | `opt_stringptr` | `LOFF(oncelog)` | Path to file/database tracking sent replies |
| `once_file_size` | `opt_int` | `LOFF(once_file_size)` | Size of cache file (0 = use DBM) |
| `once_repeat` | `opt_stringptr` | `LOFF(once_repeat)` | Time period before allowing repeat replies |

The default values are defined in `autoreply_transport_option_defaults` with only `mode = 0600` explicitly set.

Sources: [src/src/transports/autoreply.c L25-L43](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L25-L43)

 [src/src/transports/autoreply.c L66-L68](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L66-L68)

 [src/src/transports/autoreply.c L612-L625](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L612-L625)

## Message Processing Flow

#### Main Entry Point Flow with Function Calls

```mermaid
flowchart TD

autoreply_transport_entry["autoreply_transport_entry()"]
reply_data_check["addr->reply != NULL?"]
use_addr_reply["Use addr->reply fields"]
use_transport_config["Use ob->* options"]
checkexpand_calls["checkexpand() calls"]
never_mail_expand["expand_string(ob->never_mail)"]
check_never_mail_func["check_never_mail(to, never_mail)"]
dont_deliver_check["f.dont_deliver check"]
return_false["return FALSE"]
once_processing["Once processing"]
once_file_size_check["ob->once_file_size > 0?"]
cache_file_mode["Uopen() cache file"]
dbm_mode["exim_dbopen()"]
time_check["Check cached timestamp"]
log_skip["Log skip & goto END_OFF"]
child_open_exim_call["child_open_exim()"]
fdopen_call["fdopen(fd, 'wb')"]
message_headers["fprintf() headers"]
moan_write_references["moan_write_references()"]
text_output["Text & file output"]
return_message_check["return_message?"]
transport_write_message_call["transport_write_message()"]
fclose_call["fclose(fp)"]
child_close_call["child_close(pid, 0)"]
update_database["Update once database"]
exim_dbput_or_write["exim_dbput() or write()"]
log_activity["Log to file if configured"]
return_false_success["return FALSE"]

autoreply_transport_entry --> reply_data_check
reply_data_check --> use_addr_reply
reply_data_check --> use_transport_config
use_addr_reply --> checkexpand_calls
use_transport_config --> checkexpand_calls
checkexpand_calls --> never_mail_expand
never_mail_expand --> check_never_mail_func
check_never_mail_func --> dont_deliver_check
dont_deliver_check --> return_false
dont_deliver_check --> once_processing
once_processing --> once_file_size_check
once_file_size_check --> cache_file_mode
once_file_size_check --> dbm_mode
cache_file_mode --> time_check
dbm_mode --> time_check
time_check --> log_skip
time_check --> child_open_exim_call
child_open_exim_call --> fdopen_call
fdopen_call --> message_headers
message_headers --> moan_write_references
moan_write_references --> text_output
text_output --> return_message_check
return_message_check --> transport_write_message_call
return_message_check --> fclose_call
transport_write_message_call --> fclose_call
fclose_call --> child_close_call
child_close_call --> update_database
update_database --> exim_dbput_or_write
exim_dbput_or_write --> log_activity
log_activity --> return_false_success
```

Sources: [src/src/transports/autoreply.c L262-L802](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L262-L802)

 [src/src/transports/autoreply.c L126-L154](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L126-L154)

 [src/src/transports/autoreply.c L173-L250](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L173-L250)

## Data Sources

The `autoreply_transport_entry()` function determines data source using `addr->reply` pointer:

#### Data Source Selection Logic

```mermaid
flowchart TD

addr_reply_check["if (addr->reply)"]
use_reply_block["Use addr->reply fields"]
use_transport_options["Use transport ob->* options"]
reply_fields["reply->from, reply->to, reply->subject, etc."]
direct_assignment["Direct assignment (no expansion)"]
expand_forbid_set["expand_forbid = addr->reply->expand_forbid"]
get_option_calls["GET_OPTION() macro calls"]
ob_from["ob->from"]
ob_to["ob->to"]
ob_subject["ob->subject"]
checkexpand_from["checkexpand(from, addr, trname, cke_hdr)"]
checkexpand_to["checkexpand(to, addr, trname, cke_hdr)"]
checkexpand_subject["checkexpand(subject, addr, trname, cke_hdr)"]
expansion_validation["String expansion & validation"]
proceed_processing["Proceed with autoreply processing"]

addr_reply_check --> use_reply_block
addr_reply_check --> use_transport_options
use_reply_block --> reply_fields
reply_fields --> direct_assignment
direct_assignment --> expand_forbid_set
use_transport_options --> get_option_calls
get_option_calls --> ob_from
get_option_calls --> ob_to
get_option_calls --> ob_subject
ob_from --> checkexpand_from
ob_to --> checkexpand_to
ob_subject --> checkexpand_subject
checkexpand_from --> expansion_validation
checkexpand_to --> expansion_validation
checkexpand_subject --> expansion_validation
direct_assignment --> proceed_processing
expansion_validation --> proceed_processing
```

1. **Address Reply Block** (`addr->reply != NULL`): Data comes from mail filter setup, no expansion needed
2. **Transport Configuration** (`addr->reply == NULL`): Uses `ob->*` options with `checkexpand()` validation

Sources: [src/src/transports/autoreply.c L294-L356](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L294-L356)

 [src/src/transports/autoreply.c L318-L345](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L318-L345)

## Once-Only Implementation

The once-only mechanism prevents duplicate replies using either DBM database or fixed-size cache files:

#### Once-Only Database Operations

```mermaid
flowchart TD

oncelog_check["oncelog && *oncelog && to"]
file_size_check["ob->once_file_size > 0?"]
cache_mode["Fixed-size Cache Mode"]
dbm_mode["DBM Mode"]
uopen_cache["Uopen(oncelog, O_CREAT|O_RDWR)"]
read_cache["read(cache_fd, cache_buff, cache_size)"]
scan_cache["Scan cache entries"]
ustrcmp_check["Ustrcmp(to, cached_address)"]
memcpy_time["memcpy(&then, p, sizeof(time_t))"]
then_zero["then = 0"]
exim_dbopen_call["exim_dbopen(oncelog, dirname, O_RDWR|O_CREAT)"]
exim_datum_init["exim_datum_init(&key_datum)"]
exim_dbget_call["exim_dbget(dbm_file, &key_datum, &result_datum)"]
memcpy_dbm_time["memcpy(&then, datum_data, sizeof(time_t))"]
then_zero_dbm["then = 0"]
time_comparison["now - then < once_repeat_sec"]
send_reply["Send reply"]
skip_reply["Skip reply (goto END_OFF)"]
update_time["memcpy(cache_time, &now, sizeof(time_t))"]
write_update["write() or exim_dbput()"]

oncelog_check --> file_size_check
file_size_check --> cache_mode
file_size_check --> dbm_mode
cache_mode --> uopen_cache
uopen_cache --> read_cache
read_cache --> scan_cache
scan_cache --> ustrcmp_check
ustrcmp_check --> memcpy_time
ustrcmp_check --> then_zero
dbm_mode --> exim_dbopen_call
exim_dbopen_call --> exim_datum_init
exim_datum_init --> exim_dbget_call
exim_dbget_call --> memcpy_dbm_time
exim_dbget_call --> then_zero_dbm
memcpy_time --> time_comparison
memcpy_dbm_time --> time_comparison
then_zero --> send_reply
then_zero_dbm --> send_reply
time_comparison --> skip_reply
time_comparison --> send_reply
send_reply --> update_time
update_time --> write_update
```

#### Database Update After Successful Send

```mermaid
flowchart TD

successful_send["Successful child_close()"]
cache_fd_check["cache_fd >= 0?"]
lseek_start["lseek(cache_fd, 0, SEEK_SET)"]
dbm_file_check["dbm_file != NULL?"]
cache_update_logic["Update cache logic"]
memcpy_now["memcpy(cache_time, &now, sizeof(time_t))"]
write_cache["write(cache_fd, from, size)"]
exim_datum_data_set["exim_datum_data_set(&value_datum, &now)"]
exim_dbput_call["exim_dbput(dbm_file, &key_datum, &value_datum)"]
close_cache["close(cache_fd)"]
exim_dbclose["exim_dbclose(dbm_file)"]

successful_send --> cache_fd_check
cache_fd_check --> lseek_start
cache_fd_check --> dbm_file_check
lseek_start --> cache_update_logic
cache_update_logic --> memcpy_now
memcpy_now --> write_cache
dbm_file_check --> exim_datum_data_set
exim_datum_data_set --> exim_dbput_call
write_cache --> close_cache
exim_dbput_call --> exim_dbclose
```

Sources: [src/src/transports/autoreply.c L402-L530](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L402-L530)

 [src/src/transports/autoreply.c L690-L734](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L690-L734)

## Never Mail Mechanism

The `never_mail` option prevents autoreplies to specific addresses using the `check_never_mail()` function:

#### Never Mail Processing Algorithm

```mermaid
flowchart TD

expand_never_mail["expand_string(ob->never_mail)"]
check_recipients["Check to, cc, bcc headers"]
check_never_mail_func["check_never_mail(list, never_mail)"]
store_mark["store_mark()"]
string_copy_newlist["string_copy(list)"]
parse_loop["Parse address loop"]
parse_find_address_end["parse_find_address_end(s, FALSE)"]
parse_extract_address["parse_extract_address(s, &error, ...)"]
match_address_list["match_address_list(next, TRUE, FALSE, &never_mail, ...)"]
remove_address["Remove address (memmove)"]
keep_address["Keep address"]
hit_true["hit = TRUE"]
continue_loop["Continue to next address"]
more_addresses["More addresses?"]
check_hit["hit == TRUE?"]
return_newlist["return newlist"]
store_reset["store_reset(reset_point)"]
return_original["return original list"]

expand_never_mail --> check_recipients
check_recipients --> check_never_mail_func
check_never_mail_func --> store_mark
store_mark --> string_copy_newlist
string_copy_newlist --> parse_loop
parse_loop --> parse_find_address_end
parse_find_address_end --> parse_extract_address
parse_extract_address --> match_address_list
match_address_list --> remove_address
match_address_list --> keep_address
remove_address --> hit_true
keep_address --> continue_loop
hit_true --> continue_loop
continue_loop --> more_addresses
more_addresses --> parse_loop
more_addresses --> check_hit
check_hit --> return_newlist
check_hit --> store_reset
store_reset --> return_original
```

The function uses `match_address_list()` to check each recipient against the never_mail patterns, removing matches to prevent mail loops.

Sources: [src/src/transports/autoreply.c L173-L250](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L173-L250)

 [src/src/transports/autoreply.c L361-L383](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L361-L383)

## Child Process Handling

The autoreply transport uses `child_open_exim()` to create a subprocess for message delivery:

#### Child Process Creation and Communication

```mermaid
flowchart TD

child_open_exim_call["child_open_exim(&fd, US'autoreply')"]
fdopen_call["fdopen(fd, 'wb')"]
set_defer["addr->transport_return = DEFER"]
fprintf_headers["fprintf() - Generate headers"]
from_header["fprintf(fp, 'From: %s<br>', from)"]
to_header["fprintf(fp, 'To: %s<br>', to)"]
subject_header["fprintf(fp, 'Subject: %s<br>', subject)"]
in_reply_to["Generate In-Reply-To from header_list"]
moan_write_references_call["moan_write_references(fp, message_id)"]
auto_submitted["fprintf(fp, 'Auto-Submitted: auto-replied<br>')"]
message_body["Message body output"]
text_output["text exists?"]
fprintf_text["fprintf(fp, '%s', CS text)"]
file_check["file exists?"]
ufgets_loop["Ufgets() file reading loop"]
return_message_check["return_message?"]
file_expand_check["file_expand?"]
expand_string_line["expand_string(big_buffer)"]
fprintf_line["fprintf(fp, '%s', CS big_buffer)"]
transport_write_message_call["transport_write_message(&tctx, bounce_return_size_limit)"]
fclose_fp["fclose(fp)"]
child_close_call["child_close(pid, 0)"]
success_path["Success"]
no_recipients["No recipients (success)"]
defer_result["addr->transport_return = DEFER"]

child_open_exim_call --> fdopen_call
child_open_exim_call --> set_defer
fdopen_call --> fprintf_headers
fprintf_headers --> from_header
from_header --> to_header
to_header --> subject_header
subject_header --> in_reply_to
in_reply_to --> moan_write_references_call
moan_write_references_call --> auto_submitted
auto_submitted --> message_body
message_body --> text_output
text_output --> fprintf_text
text_output --> file_check
fprintf_text --> file_check
file_check --> ufgets_loop
file_check --> return_message_check
ufgets_loop --> file_expand_check
file_expand_check --> expand_string_line
file_expand_check --> fprintf_line
expand_string_line --> fprintf_line
fprintf_line --> return_message_check
return_message_check --> transport_write_message_call
return_message_check --> fclose_fp
transport_write_message_call --> fclose_fp
fclose_fp --> child_close_call
child_close_call --> success_path
child_close_call --> no_recipients
child_close_call --> defer_result
```

The child process handles the actual SMTP delivery, while the parent process constructs the message content and manages the once-only database.

Sources: [src/src/transports/autoreply.c L555-L676](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L555-L676)

 [src/src/transports/autoreply.c L594-L595](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L594-L595)

 [src/src/transports/autoreply.c L630-L671](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L630-L671)

## Reply Message Generation

The reply message is constructed with:

1. Standard headers (From, Reply-To, To, Cc, Bcc, Subject)
2. In-Reply-To header referencing the original message
3. References header preserving the References chain
4. Auto-Submitted: auto-replied header to mark as an automatic response
5. Additional custom headers
6. Message body text
7. Contents of the specified file (if any)
8. Original message (if return_message is enabled)

Sources: [src/src/transports/autoreply.c L574-L671](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L574-L671)

## Logging

If the `log` option is set, information about sent messages is logged:

1. Each log entry includes a timestamp
2. Sender information is recorded
3. Recipients (To, Cc, Bcc) are logged
4. Subject is logged
5. Any additional headers are recorded

The log file is opened with O_APPEND, ensuring that concurrent autoreplies don't overwrite each other.

Sources: [src/src/transports/autoreply.c L762-L793](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L762-L793)

## Error Handling

The Autoreply Transport handles various error conditions:

1. **Configuration Errors**: Failures in string expansion are reported with detailed messages
2. **File Access Errors**: Problems with opening or reading files are reported
3. **Child Process Errors**: Non-zero exit status from the child process is handled
4. **Database Errors**: Problems with the once-only database are managed

In most cases, errors result in DEFER status to allow retrying later.

Sources: [src/src/transports/autoreply.c L333-L345](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L333-L345)

 [src/src/transports/autoreply.c L533-L552](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L533-L552)

 [src/src/transports/autoreply.c L746-L752](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L746-L752)

## Integration with Mail Filters

The Autoreply Transport works particularly well with mail filtering systems:

```mermaid
flowchart TD

IncomingMail["Incoming Mail"]
MailFilter["Mail Filter"]
SetupReply["Setup Reply Block"]
Router["Router with autoreply Transport"]
AutoreplyTransport["Autoreply Transport"]
CheckAddressReply["Check addr->reply"]
UseReplyBlock["Use Filter-Supplied Data"]
UseConfig["Use Transport Config"]

IncomingMail --> MailFilter
MailFilter --> SetupReply
SetupReply --> Router
Router --> AutoreplyTransport
AutoreplyTransport --> CheckAddressReply
CheckAddressReply --> UseReplyBlock
CheckAddressReply --> UseConfig
```

Sources: [src/src/transports/autoreply.c L292-L313](https://github.com/Exim/exim/blob/29568b25/src/src/transports/autoreply.c#L292-L313)

## Configuration Example

Here's an example transport configuration for an autoreply transport:

```python
vacation_autoreply:
  driver = autoreply
  to = $sender_address
  subject = Auto: $h_subject
  text = "I am currently away from the office and will reply when I return.\n\nYour message about \"$h_subject\" has been received."
  file = /home/$local_part/.vacation.msg
  file_optional = true
  log = /home/$local_part/.vacation.log
  once = /home/$local_part/.vacation.db
  once_repeat = 7d
  never_mail = <, postmaster@*, mailer-daemon@*
  return_message = false
```

## Related Transportation Systems

The Autoreply Transport is one of several transports in Exim, each serving a distinct purpose:

| Transport | Purpose |
| --- | --- |
| Autoreply | Generates automatic responses to incoming mail |
| Appendfile | Delivers mail to files or directories (mailboxes) |
| Pipe | Delivers mail to an external command |
| LMTP | Delivers mail using the Local Mail Transfer Protocol |
| SMTP | Delivers mail to remote servers using SMTP |

For more information on other transport mechanisms, see [Transport Mechanisms](/Exim/exim/6-transport-mechanisms).
# Key Modules

> **Relevant source files**
> * [src/modules/dialog/Makefile](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/Makefile)
> * [src/modules/dialog/dialog.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dialog.c)
> * [src/modules/dialog/dlg_cb.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_cb.c)
> * [src/modules/dialog/dlg_cb.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_cb.h)
> * [src/modules/dialog/dlg_cseq.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_cseq.c)
> * [src/modules/dialog/dlg_cseq.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_cseq.h)
> * [src/modules/dialog/dlg_db_handler.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_db_handler.c)
> * [src/modules/dialog/dlg_db_handler.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_db_handler.h)
> * [src/modules/dialog/dlg_dmq.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_dmq.c)
> * [src/modules/dialog/dlg_dmq.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_dmq.h)
> * [src/modules/dialog/dlg_handlers.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_handlers.c)
> * [src/modules/dialog/dlg_handlers.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_handlers.h)
> * [src/modules/dialog/dlg_hash.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_hash.c)
> * [src/modules/dialog/dlg_hash.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_hash.h)
> * [src/modules/dialog/dlg_load.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_load.h)
> * [src/modules/dialog/dlg_profile.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_profile.c)
> * [src/modules/dialog/dlg_profile.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_profile.h)
> * [src/modules/dialog/dlg_req_within.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_req_within.c)
> * [src/modules/dialog/dlg_req_within.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_req_within.h)
> * [src/modules/dialog/dlg_timer.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_timer.c)
> * [src/modules/dialog/dlg_timer.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_timer.h)
> * [src/modules/dialog/dlg_var.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_var.c)
> * [src/modules/dialog/dlg_var.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_var.h)
> * [src/modules/dialog/doc/dialog.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/doc/dialog.xml)
> * [src/modules/dialog/doc/dialog_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/doc/dialog_admin.xml)
> * [src/modules/dialog/doc/dialog_devel.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/doc/dialog_devel.xml)
> * [src/modules/dispatcher/Makefile](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/Makefile)
> * [src/modules/dispatcher/api.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/api.h)
> * [src/modules/dispatcher/config.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/config.c)
> * [src/modules/dispatcher/config.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/config.h)
> * [src/modules/dispatcher/dispatch.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatch.c)
> * [src/modules/dispatcher/dispatch.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatch.h)
> * [src/modules/dispatcher/dispatcher.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatcher.c)
> * [src/modules/dispatcher/doc/dispatcher.cfg](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/doc/dispatcher.cfg)
> * [src/modules/dispatcher/doc/dispatcher.list](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/doc/dispatcher.list)
> * [src/modules/dispatcher/doc/dispatcher.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/doc/dispatcher.xml)
> * [src/modules/dispatcher/doc/dispatcher_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/doc/dispatcher_admin.xml)
> * [src/modules/dispatcher/ds_ht.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/ds_ht.c)
> * [src/modules/dispatcher/ds_ht.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/ds_ht.h)
> * [src/modules/rtpengine/Makefile](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/Makefile)
> * [src/modules/rtpengine/api.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/api.h)
> * [src/modules/rtpengine/bencode.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/bencode.c)
> * [src/modules/rtpengine/bencode.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/bencode.h)
> * [src/modules/rtpengine/compat.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/compat.h)
> * [src/modules/rtpengine/config.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/config.c)
> * [src/modules/rtpengine/config.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/config.h)
> * [src/modules/rtpengine/doc/rtpengine.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/doc/rtpengine.xml)
> * [src/modules/rtpengine/doc/rtpengine_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/doc/rtpengine_admin.xml)
> * [src/modules/rtpengine/rtpengine.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine.c)
> * [src/modules/rtpengine/rtpengine.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine.h)
> * [src/modules/rtpengine/rtpengine_db.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_db.c)
> * [src/modules/rtpengine/rtpengine_funcs.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_funcs.c)
> * [src/modules/rtpengine/rtpengine_funcs.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_funcs.h)
> * [src/modules/rtpengine/rtpengine_hash.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_hash.c)
> * [src/modules/rtpengine/rtpengine_hash.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_hash.h)

This page provides an overview of the most essential and commonly used modules in Kamailio. These key modules form the backbone of many SIP server deployments and enable critical functionality like load balancing, dialog tracking, and media handling.

The modules covered here represent core functionality that's often needed in production deployments. For information about the Kamailio core architecture, see [Core Architecture](/kamailio/kamailio/2-core-architecture), and for module compilation details, see [Module Compilation System](/kamailio/kamailio/5.2-module-compilation-system).

## Dispatcher Module

The Dispatcher module provides SIP load balancing and traffic distribution capabilities for Kamailio. It allows SIP requests to be distributed across multiple destinations using various algorithms and provides failover capabilities.

### Architecture and Components

![Dispatcher Module Architecture](https://github.com/kamailio/kamailio/blob/2b4e9f8b/Dispatcher Module Architecture)

"]
Alg --> Alg2["Round-robin"]
Alg --> Alg3["Weight-based"]
Alg --> Alg4["Call-load"]
Alg --> Alg5["Latency-based"]
end

```
subgraph "Destinations"
    DSSelect --> Sets["Destination Sets"]
    Sets --> Set1["Set 1"]
    Sets --> Set2["Set 2"]
    
    Set1 --> Dest1["Destination 1"]
    Set1 --> Dest2["Destination 2"]
    
    Dest1 --> DestAttr["URI, Flags, Priority, Weight"]
end

subgraph "State Management"
    DSHealth --> States["States"]
    States --> Active["Active"]
    States --> Inactive["Inactive"]
    States --> Probing["Probing"]
    States --> Disabled["Disabled"]
end
```

)

Sources: [src/modules/dispatcher/dispatcher.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatcher.c)

 [src/modules/dispatcher/dispatch.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatch.c)

 [src/modules/dispatcher/dispatch.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatch.h)

 [src/modules/dispatcher/doc/dispatcher_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/doc/dispatcher_admin.xml)

### Key Features

1. **Destination Sets**: Allows grouping of SIP servers into sets for organized load balancing.
2. **Load Balancing Algorithms**: Supports multiple algorithms including: * Round-robin * Hash-based (Call-ID, From URI, To URI, etc.) * Weight-based * Call-load distribution * Latency-based
3. **Health Monitoring**: Proactively monitors destination health and automatically: * Detects failures * Marks destinations as inactive * Performs periodic health checks * Returns destinations to active state when recovered
4. **Failover Handling**: Provides automatic failover to alternative destinations when a selected target is unreachable.
5. **Destination States**: * Active: Fully operational * Inactive: Temporarily unavailable * Probing: Under health check * Disabled: Administratively disabled

### Configuration

The module can load destinations from a configuration file (`dispatcher.list`) or from a database. Here's a sample dispatcher list file format:

```markdown
# setid(integer) destination(sip uri) flags(integer) priority(integer) attributes(string)
1 sip:192.168.1.1:5060 0 0 weight=10
1 sip:192.168.1.2:5060 0 0 weight=20
2 sip:192.168.2.1:5060 0 0
```

Common module parameters include:

| Parameter | Description |
| --- | --- |
| list_file | Path to the file with destination sets |
| db_url | Database URL if loading from database |
| flags | Bitmask to control module behavior |
| ds_ping_interval | Interval for pinging inactive gateways |
| ds_probing_threshold | Number of failed requests before marking destination inactive |

### Usage Example

```python
# Load the dispatcher module with file-based configuration
loadmodule "dispatcher.so"
modparam("dispatcher", "list_file", "/etc/kamailio/dispatcher.list")

# Select destination from set 1 using round-robin algorithm
if (is_method("INVITE")) {
    ds_select("1", "4");  # set 1, round-robin algorithm
}
```

## Dialog Module

The Dialog module provides dialog state tracking for Kamailio, enabling awareness of complete SIP dialogs (calls) beyond individual transactions. This is essential for call counting, monitoring, and many advanced services.

### Architecture and Components

![Dialog Module Architecture](https://github.com/kamailio/kamailio/blob/2b4e9f8b/Dialog Module Architecture)

"]
States --> S3["3: Confirmed (waiting ACK)"]
States --> S4["4: Confirmed (active)"]
States --> S5["5: Deleted"]
end

```
subgraph "Storage"
    DlgCreate --> Hash["Hash Table"]
    Hash --> Entry1["Entry 1"]
    Hash --> Entry2["Entry 2"]
    
    Entry1 --> Dialog1["Dialog 1"]
    Entry1 --> Dialog2["Dialog 2"]
    
    Dialog1 --> DlgData["Call-ID, Tags, Route, State, Timeout"]
end

subgraph "Profiling"
    DlgProfile --> ProfTypes["Profile Types"]
    ProfTypes --> NoVal["No Value Profiles"]
    ProfTypes --> WithVal["Value Profiles (key/value)"]
end
```

)

Sources: [src/modules/dialog/dialog.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dialog.c)

 [src/modules/dialog/dlg_hash.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_hash.c)

 [src/modules/dialog/dlg_handlers.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_handlers.c)

 [src/modules/dialog/dlg_db_handler.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_db_handler.c)

 [src/modules/dialog/dlg_req_within.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_req_within.c)

 [src/modules/dialog/dlg_var.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_var.c)

 [src/modules/dialog/doc/dialog_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/doc/dialog_admin.xml)

### Key Features

1. **Dialog State Tracking**: Tracks the complete lifecycle of SIP dialogs: * Unconfirmed (initial) * Early (ringing) * Confirmed (waiting for ACK) * Confirmed (active call) * Deleted (terminated)
2. **Dialog Profiling**: Enables classification and tracking of dialogs based on: * Simple membership (no-value profiles) * Key-value attributes (value profiles)
3. **Dialog Variables**: Allows storing and retrieving variables associated with dialogs
4. **Database Integration**: Provides persistence of dialog information for: * Survivability across restarts * Replication across nodes * Reporting and analytics
5. **Dialog Timers**: Manages dialog timeouts with: * Default timeouts * Custom timeouts via AVPs * Early dialog timeouts * No-ACK timeouts

### Configuration

Common module parameters include:

| Parameter | Description |
| --- | --- |
| hash_size | Size of dialog hash table |
| rr_param | Parameter used in Record-Route for dialog matching |
| default_timeout | Default dialog timeout in seconds |
| profiles_with_value | Defined profiles with values |
| profiles_no_value | Defined profiles without values |
| db_url | Database URL for persistence |
| db_mode | Database mode (0=none, 1=read, 2=write) |

### Usage Example

```sql
# Load the dialog module
loadmodule "dialog.so"
modparam("dialog", "default_timeout", 43200)
modparam("dialog", "profiles_no_value", "caller")
modparam("dialog", "profiles_with_value", "account")

route {
    if (is_method("INVITE") && !has_totag()) {
        # Create dialog
        dlg_manage();
        
        # Add dialog to profiles
        set_dlg_profile("caller");
        set_dlg_profile("account", "$avp(account_id)");
        
        # Set custom timeout
        dlg_set_timeout("3600");
    }
}
```

## RTPEngine Module

The RTPEngine module integrates Kamailio with the Sipwise rtpengine media proxy, allowing control of media streams. It enables media manipulation, NAT traversal, transcoding, and recording.

### Architecture and Components

![RTPEngine Module Architecture](https://github.com/kamailio/kamailio/blob/2b4e9f8b/RTPEngine Module Architecture)

Sources: [src/modules/rtpengine/rtpengine.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine.c)

 [src/modules/rtpengine/rtpengine.h](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine.h)

 [src/modules/rtpengine/rtpengine_funcs.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine_funcs.c)

 [src/modules/rtpengine/doc/rtpengine_admin.xml](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/doc/rtpengine_admin.xml)

### Key Features

1. **Media Handling Operations**: Supports various operations: * SDP offer/answer processing * Session deletion * Recording control * Media blocking/unblocking * DTMF handling
2. **Multiple RTPEngine Instance Support**: * Configurable sets of RTPEngine instances * Load balancing within sets * Automatic failover
3. **Health Monitoring**: * Regular ping checks * Automatic disabling of failed instances * Re-enabling recovered instances
4. **Transport Options**: * UDP (IPv4/IPv6) * WebSocket * Secure WebSocket (WSS)
5. **Media Manipulation**: * Transcoding * ICE/SRTP handling * Codec filtering * IP address translation

### Configuration

Common module parameters include:

| Parameter | Description |
| --- | --- |
| rtpengine_sock | Definition of socket(s) to connect to RTPEngine instance(s) |
| rtpengine_disable_tout | Time (seconds) before retrying a disabled RTPEngine |
| rtpengine_tout_ms | Timeout (milliseconds) waiting for reply |
| ping_interval | Interval for health checks |
| setid_avp | AVP to store set ID |

### Usage Example

```css
# Load the RTPEngine module
loadmodule "rtpengine.so"

# Define a set of RTPEngines with weights for load balancing
modparam("rtpengine", "rtpengine_sock", "udp:192.168.1.1:22222=2 udp:192.168.1.2:22222=1")

# Process INVITE with SDP
route {
    if (is_method("INVITE") && has_body("application/sdp")) {
        rtpengine_offer();
    }
    
    if (is_method("ACK") && has_body("application/sdp")) {
        rtpengine_answer();
    }
}

# When dialog terminates
onreply_route {
    if (status=~"(487|486|480|603)") {
        rtpengine_delete();
    }
}
```

## Module Interactions

The Dispatcher, Dialog, and RTPEngine modules work together to form a complete SIP processing pipeline in Kamailio. Each module handles a distinct aspect of SIP processing.

![Module Interactions](https://github.com/kamailio/kamailio/blob/2b4e9f8b/Module Interactions)

Sources: [src/modules/dispatcher/dispatch.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dispatcher/dispatch.c)

 [src/modules/dialog/dlg_handlers.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/dialog/dlg_handlers.c)

 [src/modules/rtpengine/rtpengine.c](https://github.com/kamailio/kamailio/blob/2b4e9f8b/src/modules/rtpengine/rtpengine.c)

### Typical Call Flow

1. **Call Setup Phase**: * Dispatcher module selects appropriate destination for the INVITE * Dialog module creates and tracks the new dialog * RTPEngine module processes SDP in INVITE offer * Dispatcher forwards modified request to selected destination
2. **In-Dialog Messaging**: * Dialog module matches requests to existing dialog * RTPEngine processes SDP answers and updates * Dispatcher might be involved for sending to the right destination
3. **Call Termination**: * Dialog module detects dialog termination (BYE) * RTPEngine module deletes media session * Dialog module removes the dialog from its tables

### Common Use Cases

1. **Load-Balanced SIP Proxy**: ```sql # Route incoming INVITE if (is_method("INVITE") && !has_totag()) {     # Create dialog     dlg_manage();          # Select destination using weight-based algorithm     ds_select("1", "9");  # set 1, weight-based algorithm          # Process media with RTPEngine     rtpengine_offer("ICE=remove");          # Forward to selected destination     t_relay(); } ```
2. **High-Availability SIP Server**: ```markdown # Process INVITE with failover if (is_method("INVITE") && !has_totag()) {     dlg_manage();          # Try destinations with failover     if (!ds_select_dst("1", "4")) {         send_reply("500", "No destination available");         exit;     }          rtpengine_offer();     t_relay(); } # Handle failover failure_route[MANAGE_FAILURE] {     if (t_is_canceled()) {         exit;     }          # Try next destination in set     if (ds_next_dst()) {         rtpengine_offer();         t_relay();     } } ```
3. **Media Recording**: ```markdown # Start recording all calls if (is_method("INVITE") && !has_totag()) {     dlg_manage();          $var(flags) = "ICE=remove recording_method=pcap";     rtpengine_offer("$var(flags)");          ds_select("1", "4");     t_relay(); } ```

This integration enables Kamailio to function as a complete, highly available SIP server with load balancing, dialog awareness, and media handling capabilities.
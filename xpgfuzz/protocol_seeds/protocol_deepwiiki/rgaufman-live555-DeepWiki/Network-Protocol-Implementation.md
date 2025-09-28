# Network Protocol Implementation

> **Relevant source files**
> * [groupsock/GroupsockHelper.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp)
> * [groupsock/include/GroupsockHelper.hh](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/include/GroupsockHelper.hh)
> * [liveMedia/RTCP.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp)
> * [liveMedia/RTPInterface.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp)
> * [liveMedia/RTPSink.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp)
> * [liveMedia/include/RTPInterface.hh](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/RTPInterface.hh)

This page documents the network protocol implementation in the live555 streaming media library, focusing on how the system implements and manages RTP (Real-time Transport Protocol) and RTCP (RTP Control Protocol) networking. This covers the transport layer of the library, which is responsible for sending and receiving media packets over networks. For information about higher-level media session management, see [Media Session Management](/rgaufman/live555/4-media-session-management).

## Network Protocol Architecture Overview

The live555 library implements the RTP/RTCP protocols as defined in RFC 3550, with several extensions for flexibility and robustness. The implementation is centered around a few key classes that manage the network transport:

```mermaid
flowchart TD

RTPInterface["RTPInterface"]
RTCPInstance["RTCPInstance"]
RTPSink["RTPSink"]
Groupsock["Groupsock"]
GroupsockHelper["GroupsockHelper"]
UDP["UDP Transport"]
TCP["TCP Transport (RFC 2326)"]
SRTP["SRTP/SRTCP (Secure)"]

RTPInterface --> Groupsock
Groupsock --> UDP
RTPInterface --> TCP
RTPSink --> SRTP
RTCPInstance --> SRTP

subgraph subGraph2 ["Transport Options"]
    UDP
    TCP
    SRTP
end

subgraph subGraph1 ["Socket Layer"]
    Groupsock
    GroupsockHelper
    GroupsockHelper --> Groupsock
end

subgraph subGraph0 ["Network Protocol Layer"]
    RTPInterface
    RTCPInstance
    RTPSink
    RTPSink --> RTPInterface
    RTCPInstance --> RTPInterface
end
```

Sources:

* [liveMedia/include/RTPInterface.hh L45-L85](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/RTPInterface.hh#L45-L85)
* [liveMedia/RTPInterface.cpp L132-L193](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L132-L193)
* [liveMedia/RTCP.cpp L112-L178](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L112-L178)
* [liveMedia/RTPSink.cpp L25-L71](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L25-L71)

## RTP Interface

The `RTPInterface` class provides an abstraction layer for sending and receiving RTP/RTCP packets over network connections. It's designed to hide the complexity of supporting both UDP and TCP transports.

### Key Capabilities

* Sending RTP/RTCP packets over UDP (via `Groupsock`)
* Supporting RTP-over-TCP (interleaved mode as defined in RFC 2326, section 10.12)
* Handling multiple concurrent TCP connections
* Supporting IPv4 and IPv6 networking
* Providing auxiliary handlers for custom packet processing

```mermaid
classDiagram
    class RTPInterface {
        -Medium* fOwner
        -Groupsock* fGS
        -tcpStreamRecord* fTCPStreams
        +sendPacket()
        +startNetworkReading()
        +handleRead()
        +stopNetworkReading()
        +setStreamSocket()
        +addStreamSocket()
        +removeStreamSocket()
    }
    class tcpStreamRecord {
        -int fStreamSocketNum
        -unsigned char fStreamChannelId
        -TLSState* fTLSState
        -tcpStreamRecord* fNext
    }
    class SocketDescriptor {
        -HashTable* fSubChannelHashTable
        -tcpReadHandler()
        -registerRTPInterface()
        -lookupRTPInterface()
        -deregisterRTPInterface()
    }
    RTPInterface --> tcpStreamRecord : contains
    RTPInterface --> SocketDescriptor : uses
```

Sources:

* [liveMedia/include/RTPInterface.hh L45-L113](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/RTPInterface.hh#L45-L113)
* [liveMedia/RTPInterface.cpp L133-L193](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L133-L193)
* [liveMedia/RTPInterface.cpp L342-L375](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L342-L375)

### RTP-over-TCP Implementation

The library provides a specialized implementation of RTP-over-TCP following the encoding defined in RFC 2326 (section 10.12), which uses the format:

```html
$<streamChannelId><packetSize><packet>
```

This allows RTP/RTCP packets to be interleaved with RTSP control messages on the same TCP connection. The implementation uses a two-level hash table system:

1. A top-level hash table that maps TCP socket numbers to `SocketDescriptor` objects
2. Each `SocketDescriptor` contains a sub-channel hash table mapping channel IDs to `RTPInterface` instances

```mermaid
sequenceDiagram
  participant Application
  participant RTPInterface
  participant TCP Connection
  participant Groupsock

  Application->>RTPInterface: sendPacket(data)
  loop [UDP Transport]
    RTPInterface->>Groupsock: output(data)
    RTPInterface->>RTPInterface: sendRTPorRTCPPacketOverTCP()
    RTPInterface->>TCP Connection: send("$")
    RTPInterface->>TCP Connection: send(channelId)
    RTPInterface->>TCP Connection: send(packetSize)
    RTPInterface->>TCP Connection: send(data)
    TCP Connection-->>RTPInterface: readHandler triggered
    RTPInterface-->>RTPInterface: tcpReadHandler1()
    RTPInterface-->>RTPInterface: Parse $<channelId><size>
    RTPInterface-->>Application: deliver packet
  end
```

Sources:

* [liveMedia/RTPInterface.cpp L342-L375](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L342-L375)
* [liveMedia/RTPInterface.cpp L429-L648](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L429-L648)

## RTCP Implementation

RTCP (RTP Control Protocol) is implemented in the `RTCPInstance` class, which handles the creation, sending, and receiving of RTCP packets. It supports all standard RTCP packet types including Sender Reports (SR), Receiver Reports (RR), Source Description (SDES), and BYE packets.

### RTCP Packet Types and Processing

The implementation supports the following RTCP packet types:

| Packet Type | Purpose | Implementation |
| --- | --- | --- |
| SR (200) | Sender Report | Provides sending and receiving statistics for participants that are active senders |
| RR (201) | Receiver Report | Provides reception statistics from participants that are not active senders |
| SDES (202) | Source Description | Contains items describing the source, including CNAME, NAME, EMAIL, etc. |
| BYE (203) | Goodbye | Indicates end of participation |
| APP (204) | Application-specific | For application-specific functions |
| RTPFB (205) | RTP Feedback | Generic RTP-level feedback |
| PSFB (206) | Payload-specific Feedback | Feedback messages related to the payload |

```mermaid
flowchart TD

RTCPInstance["RTCPInstance"]
RTCPMemberDB["RTCPMemberDatabase"]
OutBuf["OutPacketBuffer"]
Incoming["Incoming RTCP Packet"]
Process["processIncomingReport()"]
Handler["Handler Functions"]
Outgoing["Outgoing RTCP Packet"]
SRHandler["SR Handler"]
RRHandler["RR Handler"]
ByeHandler["BYE Handler"]
AppHandler["APP Handler"]

Process --> RTCPInstance
RTCPInstance --> Handler
RTCPInstance --> Outgoing
Handler --> SRHandler
Handler --> RRHandler
Handler --> ByeHandler
Handler --> AppHandler

subgraph subGraph2 ["Handler Types"]
    SRHandler
    RRHandler
    ByeHandler
    AppHandler
end

subgraph subGraph1 ["RTCP Packet Flow"]
    Incoming
    Process
    Handler
    Outgoing
    Incoming --> Process
end

subgraph subGraph0 ["RTCPInstance Components"]
    RTCPInstance
    RTCPMemberDB
    OutBuf
    RTCPInstance --> RTCPMemberDB
    RTCPInstance --> OutBuf
end
```

Sources:

* [liveMedia/RTCP.cpp L112-L178](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L112-L178)
* [liveMedia/RTCP.cpp L544-L867](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L544-L867)
* [liveMedia/RTCP.cpp L916-L951](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L916-L951)

### RTCP Statistics Tracking

The RTCP implementation includes comprehensive statistics tracking for both sent and received packets:

1. `RTPTransmissionStatsDB` - Maintains statistics for transmitted packets
2. `RTCPMemberDatabase` - Tracks active RTCP members in the session

These components enable:

* Round-trip time estimation
* Packet loss detection
* Jitter calculation
* Participant timeout detection

Sources:

* [liveMedia/RTCP.cpp L28-L108](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L28-L108)
* [liveMedia/RTPSink.cpp L219-L410](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L219-L410)

## Socket Management

The library provides a comprehensive set of utilities for socket management through the `GroupsockHelper` functions. These functions abstract the platform-specific details of socket operations.

### Socket Creation and Configuration

Socket creation is handled by specialized functions for both UDP (datagram) and TCP (stream) sockets:

```mermaid
flowchart TD

setupSocket["setupDatagramSocket()/setupStreamSocket()"]
createSocket["createSocket()"]
setOptions["Set Socket Options"]
bind["bind()"]
reuseAddr["SO_REUSEADDR"]
reusePort["SO_REUSEPORT"]
nonBlocking["Non-blocking Mode"]
bufferSize["Buffer Size"]
multicastOptions["Multicast Options"]

setupSocket --> createSocket
createSocket --> setOptions
setOptions --> bind
setOptions --> reuseAddr
setOptions --> reusePort
setOptions --> nonBlocking
setOptions --> bufferSize
setOptions --> multicastOptions

subgraph subGraph0 ["Socket Options"]
    reuseAddr
    reusePort
    nonBlocking
    bufferSize
    multicastOptions
end
```

Sources:

* [groupsock/GroupsockHelper.cpp L90-L214](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp#L90-L214)
* [groupsock/GroupsockHelper.cpp L293-L388](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp#L293-L388)

### Multicast Support

The library provides robust support for IP multicast, including:

* Joining and leaving multicast groups
* Source-specific multicast (SSM)
* TTL (Time-To-Live) configuration
* Multicast loop control
* Support for both IPv4 and IPv6 multicast

```mermaid
flowchart TD

join["socketJoinGroup()"]
leave["socketLeaveGroup()"]
joinSSM["socketJoinGroupSSM()"]
leaveSSM["socketLeaveGroupSSM()"]
setsockopt["setsockopt()"]
ip_mreq["struct ip_mreq (IPv4)"]
ipv6_mreq["struct ipv6_mreq (IPv6)"]
ip_mreq_source["struct ip_mreq_source (SSM)"]

join --> setsockopt
leave --> setsockopt
joinSSM --> setsockopt
leaveSSM --> setsockopt

subgraph subGraph1 ["Internal Implementation"]
    setsockopt
    ip_mreq
    ipv6_mreq
    ip_mreq_source
    setsockopt --> ip_mreq
    setsockopt --> ipv6_mreq
    setsockopt --> ip_mreq_source
end

subgraph subGraph0 ["Multicast Functions"]
    join
    leave
    joinSSM
    leaveSSM
end
```

Sources:

* [groupsock/GroupsockHelper.cpp L557-L724](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp#L557-L724)

## Network Data Flow

The network data flow in live555 involves several layers of abstractions from the application level down to the socket level.

### Sending Flow

```mermaid
sequenceDiagram
  participant Application
  participant RTPSink
  participant RTPInterface
  participant Socket Layer

  Application->>RTPSink: doSpecialFrameHandling()
  RTPSink->>RTPInterface: sendPacket(packet, packetSize)
  loop [UDP Transport]
    RTPInterface->>Socket Layer: output() via Groupsock
    RTPInterface->>RTPInterface: sendRTPorRTCPPacketOverTCP()
    RTPInterface->>Socket Layer: send() with header + packet
    RTPInterface->>RTPInterface: Apply SRTP encryption
    RTPInterface->>Socket Layer: Send encrypted packet
  end
```

Sources:

* [liveMedia/RTPInterface.cpp L233-L251](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L233-L251)
* [liveMedia/RTPSink.cpp L43-L66](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L43-L66)

### Receiving Flow

```mermaid
sequenceDiagram
  participant Socket Layer
  participant RTPInterface
  participant RTPSource
  participant Application

  Socket Layer->>RTPInterface: incomingPacketHandler()
  loop [UDP Transport]
    RTPInterface->>RTPInterface: handleRead() from Groupsock
    RTPInterface->>RTPInterface: tcpReadHandler()
    RTPInterface->>RTPInterface: Parse framing protocol
    RTPInterface->>RTPInterface: Decrypt SRTP packet
  end
  RTPInterface->>RTPSource: processReceivedPayload()
  RTPSource->>Application: deliverFrame()
```

Sources:

* [liveMedia/RTPInterface.cpp L271-L327](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L271-L327)
* [liveMedia/RTPInterface.cpp L431-L466](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPInterface.cpp#L431-L466)

## Security in Network Transport

The library supports secure RTP (SRTP) and SRTCP through the `SRTPCryptographicContext` class and related components. This implementation provides:

* Packet encryption and authentication
* Support for MIKEY (Multimedia Internet KEYing) for key management
* Integration with standard RTP/RTCP implementation

```mermaid
flowchart TD

SRTP["SRTP/SRTCP"]
MIKEY["MIKEYState"]
Crypto["SRTPCryptographicContext"]
RTPSink["RTPSink"]
RTCPInstance["RTCPInstance"]
SDP["SDP with key-mgmt"]

RTPSink --> Crypto
RTCPInstance --> Crypto
RTPSink --> SRTP
RTCPInstance --> SRTP
MIKEY --> SDP

subgraph subGraph1 ["Integration Points"]
    RTPSink
    RTCPInstance
    SDP
end

subgraph subGraph0 ["Security Components"]
    SRTP
    MIKEY
    Crypto
    SRTP --> MIKEY
    MIKEY --> Crypto
end
```

Sources:

* [liveMedia/RTPSink.cpp L43-L66](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L43-L66)
* [liveMedia/RTPSink.cpp L98-L117](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L98-L117)
* [liveMedia/RTCP.cpp L286-L290](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L286-L290)

## Cross-Platform Considerations

The library implements several abstractions and helper functions to ensure cross-platform compatibility:

* Socket functions that abstract platform-specific details
* Portable time functions (e.g., custom implementation of `gettimeofday()` for Windows)
* Platform-specific handling of non-blocking I/O
* Support for both IPv4 and IPv6 on all platforms

These abstractions are primarily implemented in the `GroupsockHelper` module.

Sources:

* [groupsock/GroupsockHelper.cpp L216-L257](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp#L216-L257)
* [groupsock/GroupsockHelper.cpp L1013-L1096](https://github.com/rgaufman/live555/blob/a0eb8f91/groupsock/GroupsockHelper.cpp#L1013-L1096)

## Network Protocol Implementation Class Hierarchy

The following diagram shows the class hierarchy and relationships among the key network protocol implementation components:

```mermaid
classDiagram
    class Medium {
        #UsageEnvironment& fEnv
        +lookupByName()
    }
    class RTCPInstance {
        -RTCPMemberDatabase* fKnownMembers
        -RTPSink* fSink
        -RTPSource* fSource
        -OutPacketBuffer* fOutBuf
        +incomingReportHandler()
        +sendReport()
        +sendBYE()
    }
    class RTPSink {
        -RTPInterface fRTPInterface
        -unsigned fPacketCount
        -unsigned fOctetCount
        -u_int32_t fTimestampBase
        -u_int16_t fSeqNo
        -RTPTransmissionStatsDB* fTransmissionStatsDB
        +sendPacket()
        +convertToRTPTimestamp()
    }
    class RTPInterface {
        -Medium* fOwner
        -Groupsock* fGS
        -tcpStreamRecord* fTCPStreams
        +sendPacket()
        +handleRead()
        +startNetworkReading()
        +stopNetworkReading()
    }
    class Groupsock {
        -UsageEnvironment& fEnv
        -int fSocketNum
        -struct sockaddr_storage fSourceAddr
        +output()
        +handleRead()
        +socketNum()
    }
    Medium <|-- RTCPInstance : contains
    Medium <|-- RTPSink : uses
    RTPSink --> RTPInterface : uses
    RTCPInstance --> RTPInterface
    RTPInterface --> Groupsock
```

Sources:

* [liveMedia/include/RTPInterface.hh L46-L113](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/RTPInterface.hh#L46-L113)
* [liveMedia/RTCP.cpp L112-L178](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTCP.cpp#L112-L178)
* [liveMedia/RTPSink.cpp L162-L193](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/RTPSink.cpp#L162-L193)
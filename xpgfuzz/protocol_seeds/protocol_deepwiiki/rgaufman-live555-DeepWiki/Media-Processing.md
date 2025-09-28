# Media Processing

> **Relevant source files**
> * [liveMedia/BitVector.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/BitVector.cpp)
> * [liveMedia/H263plusVideoStreamParser.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H263plusVideoStreamParser.cpp)
> * [liveMedia/H264or5VideoStreamFramer.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp)
> * [liveMedia/MPEG2IndexFromTransportStream.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MPEG2IndexFromTransportStream.cpp)
> * [liveMedia/MatroskaFileParser.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp)
> * [liveMedia/include/BitVector.hh](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/BitVector.hh)
> * [testProgs/testMKVStreamer.cpp](https://github.com/rgaufman/live555/blob/a0eb8f91/testProgs/testMKVStreamer.cpp)

This document covers the media processing components in the LIVE555 Streaming Media library that handle parsing, framing, and transforming various media formats for network streaming. These components serve as the crucial bridge between raw media data and network-ready RTP packets. For information about the network protocol implementation used to transmit processed media, see [Network Protocol Implementation](/rgaufman/live555/5-network-protocol-implementation).

## Overview of Media Processing Architecture

The LIVE555 library implements a modular approach to media processing, with specialized components for different media formats. The architecture is designed to efficiently parse media files or streams, extract frames, and prepare them for transmission over networks using RTP.

```mermaid
flowchart TD

Files["Media Files"]
LiveSources["Live Sources"]
Parsers["File/Stream Parsers"]
Framers["Stream Framers"]
FramedSources["FramedSource Subclasses"]
Filters["Processing Filters"]
RTPSinks["RTPSink Subclasses"]

Files --> Parsers
LiveSources --> Framers
Parsers --> FramedSources
Framers --> FramedSources
Filters --> RTPSinks
FramedSources --> RTPSinks

subgraph subGraph3 ["Media Streaming Layer"]
    RTPSinks
end

subgraph subGraph2 ["Media Processing Layer"]
    FramedSources
    Filters
    FramedSources --> Filters
end

subgraph subGraph1 ["Media Parsing Layer"]
    Parsers
    Framers
end

subgraph subGraph0 ["Media Source"]
    Files
    LiveSources
end
```

Sources: [liveMedia/H264or5VideoStreamFramer.cpp L1-L190](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp#L1-L190)

 [liveMedia/MatroskaFileParser.cpp L20-L99](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp#L20-L99)

## Media Source Processing Hierarchy

The LIVE555 library implements a class hierarchy for processing different media sources, with `FramedSource` as the base class. Specialized subclasses handle specific media formats and protocols.

```mermaid
classDiagram
    class Medium {
        +createNew()
        +close()
    }
    class FramedSource {
        +doGetNextFrame()
        +afterGetting()
    }
    class StreamParser {
        +parseFrame()
        +restoreSavedParserState()
    }
    class MPEGVideoStreamFramer {
        +continueReadProcessing()
    }
    class H264or5VideoStreamFramer {
        +saveCopyOfVPS()
        +saveCopyOfSPS()
        +saveCopyOfPPS()
    }
    class MatroskaFileParser {
        +parseStartOfFile()
        +parseTrack()
        +parseBlock()
    }
    class H263plusVideoStreamParser {
    }
    class MPEG2IndexFromTransportStream {
    }
    Medium --> FramedSource
    FramedSource <|-- StreamParser
    FramedSource <|-- MPEGVideoStreamFramer
    MPEGVideoStreamFramer <|-- H264or5VideoStreamFramer
    StreamParser <|-- MatroskaFileParser
    StreamParser <|-- H263plusVideoStreamParser
    StreamParser <|-- MPEG2IndexFromTransportStream
```

Sources: [liveMedia/H264or5VideoStreamFramer.cpp L74-L189](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp#L74-L189)

 [liveMedia/MatroskaFileParser.cpp L26-L115](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp#L26-L115)

## Video Parsing and Framing

### H.264/H.265 Video Processing

The `H264or5VideoStreamFramer` class is a specialized framer for H.264 and H.265 video streams. It handles the complexities of these video formats, including:

1. NAL unit parsing
2. Parameter sets management (VPS, SPS, PPS)
3. Frame boundary detection
4. Timestamp calculation

```mermaid
flowchart TD

Raw["Raw H.264/H.265<br>Bitstream"]
Framer["H264or5VideoStreamFramer"]
Parser["H264or5VideoStreamParser"]
BitVector["BitVector<br>(Bit-level parsing)"]
VPS["VPS Storage"]
SPS["SPS Storage"]
PPS["PPS Storage"]
RTPPackets["RTP Packets"]

Raw --> Framer
Framer --> Parser
Parser --> BitVector
Framer --> VPS
Framer --> SPS
Framer --> PPS
Framer --> RTPPackets

subgraph subGraph0 ["Parameter Sets Management"]
    VPS
    SPS
    PPS
end
```

Sources: [liveMedia/H264or5VideoStreamFramer.cpp L27-L189](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp#L27-L189)

 [liveMedia/BitVector.cpp L20-L55](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/BitVector.cpp#L20-L55)

The `H264or5VideoStreamFramer` implements methods to save and manage parameter sets which are essential for video decoding:

* `saveCopyOfVPS()`: Stores the Video Parameter Set (H.265 only)
* `saveCopyOfSPS()`: Stores the Sequence Parameter Set
* `saveCopyOfPPS()`: Stores the Picture Parameter Set

These parameter sets are then made available to the decoder through the RTP stream.

### Bit-Level Parsing

Media formats require bit-level precision for parsing headers and other metadata. The `BitVector` class provides utilities for manipulating bits within byte streams:

```mermaid
flowchart TD

GetBits["getBits()<br>get1Bit()"]
SkipBits["skipBits()"]
ExpGolomb["get_expGolomb()<br>get_expGolombSigned()"]
RawBytes["Raw Byte Stream"]
BitVector["BitVector"]
ParsedData["Parsed Header Data"]

RawBytes --> BitVector

subgraph Operations ["Operations"]
    GetBits
    SkipBits
    ExpGolomb
end
```

Sources: [liveMedia/include/BitVector.hh L29-L60](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/include/BitVector.hh#L29-L60)

 [liveMedia/BitVector.cpp L23-L150](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/BitVector.cpp#L23-L150)

The `BitVector` class supports various operations:

| Operation | Description | Use Case |
| --- | --- | --- |
| `getBits()` | Extract multiple bits | Reading multi-bit fields in headers |
| `get1Bit()` | Extract a single bit | Reading flags |
| `skipBits()` | Skip over bits | Navigating through the bitstream |
| `get_expGolomb()` | Parse exponential-Golomb codes | H.264/H.265 parameter decoding |

## Container Format Processing

### Matroska (MKV) File Parsing

The `MatroskaFileParser` class handles the parsing of Matroska (MKV) files, which can contain various types of media including video, audio, and subtitles.

```mermaid
flowchart TD

MKVFile["Matroska File"]
MatroskaFileParser["MatroskaFileParser"]
ParseHeader["parseStartOfFile()<br>(EBML Header)"]
ParseTracks["parseTrack()<br>(Media Tracks)"]
ParseCues["parseCues()<br>(Index Data)"]
ParseClusters["lookForNextBlock()<br>(Media Clusters)"]
ParseBlock["parseBlock()<br>(Media Frames)"]
DeliveredFrames["Frames Ready<br>for Streaming"]

MKVFile --> MatroskaFileParser
MatroskaFileParser --> ParseHeader
ParseBlock --> DeliveredFrames

subgraph subGraph0 ["Parsing Stages"]
    ParseHeader
    ParseTracks
    ParseCues
    ParseClusters
    ParseBlock
    ParseHeader --> ParseTracks
    ParseTracks --> ParseCues
    ParseCues --> ParseClusters
    ParseClusters --> ParseBlock
end
```

Sources: [liveMedia/MatroskaFileParser.cpp L95-L290](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp#L95-L290)

 [liveMedia/MatroskaFileParser.cpp L315-L754](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp#L315-L754)

The Matroska parser processes the file in stages:

1. **EBML Header Parsing**: The first stage parses the EBML header that identifies the file as a Matroska container.
2. **Track Parsing**: Identifies and extracts information about media tracks in the file.
3. **Cue Parsing**: Processes indexing information for seeking.
4. **Cluster and Block Parsing**: Extracts actual media data from clusters and blocks.

The parser identifies track types and their appropriate MIME types:

| Track Type | MIME Type | Codec ID Prefix |
| --- | --- | --- |
| H.264 Video | video/H264 | V_MPEG4/ISO/AVC |
| H.265 Video | video/H265 | V_MPEGH/ISO/HEVC |
| AAC Audio | audio/AAC | A_AAC |
| Opus Audio | audio/OPUS | A_OPUS |
| VP9 Video | video/VP9 | V_VP9 |

Sources: [liveMedia/MatroskaFileParser.cpp L445-L483](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MatroskaFileParser.cpp#L445-L483)

## Media Format Detection and Adaptation

LIVE555 implements automatic format detection and provides appropriate adapters for various media types. The library identifies formats based on signatures or header information, and then creates appropriate parser/framer objects.

```mermaid
flowchart TD

H264["H.264 Stream"]
H265["H.265 Stream"]
H263["H.263 Stream"]
MKV["Matroska File"]
TS["MPEG Transport Stream"]
Detect["Format Detection"]
H264Framer["H264or5VideoStreamFramer"]
H263Framer["H263plusVideoStreamParser"]
MKVParser["MatroskaFileParser"]
TSParser["MPEG2IndexFromTransportStream"]

H264 --> Detect
H265 --> Detect
H263 --> Detect
MKV --> Detect
TS --> Detect
Detect --> H264Framer
Detect --> H263Framer
Detect --> MKVParser
Detect --> TSParser

subgraph subGraph2 ["Media Processors"]
    H264Framer
    H263Framer
    MKVParser
    TSParser
end

subgraph subGraph1 ["Format Detection"]
    Detect
end

subgraph subGraph0 ["Input Media"]
    H264
    H265
    H263
    MKV
    TS
end
```

Sources: [liveMedia/H264or5VideoStreamFramer.cpp L138-L189](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp#L138-L189)

 [liveMedia/H263plusVideoStreamParser.cpp L28-L45](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H263plusVideoStreamParser.cpp#L28-L45)

 [liveMedia/MPEG2IndexFromTransportStream.cpp L109-L143](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/MPEG2IndexFromTransportStream.cpp#L109-L143)

## Timestamp Management

The LIVE555 library implements careful tracking of media timestamps for proper synchronization during streaming:

```mermaid
flowchart TD

PCR["Program Clock Reference<br>(Transport Stream)"]
PTS["Presentation Timestamp<br>(PES Packets)"]
TR["Temporal Reference<br>(Video Frames)"]
MKVTime["Matroska Timecode"]
Convert["Convert to<br>Common Timebase"]
Normalize["Normalize<br>Timestamps"]
Calculate["Calculate<br>Frame Duration"]
RTPTimestamp["RTP<br>Timestamp"]
RTCP["RTCP<br>Synchronization"]

PCR --> Convert
PTS --> Convert
TR --> Convert
MKVTime --> Convert
Calculate --> RTPTimestamp

subgraph subGraph2 ["RTP Timing"]
    RTPTimestamp
    RTCP
    RTPTimestamp --> RTCP
end

subgraph subGraph1 ["Timestamp Processing"]
    Convert
    Normalize
    Calculate
    Convert --> Normalize
    Normalize --> Calculate
end

subgraph subGraph0 ["Timestamp Sources"]
    PCR
    PTS
    TR
    MKVTime
end
```

Sources: [liveMedia/H264or5VideoStreamFramer.cpp L129-L136](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H264or5VideoStreamFramer.cpp#L129-L136)

 [liveMedia/H263plusVideoStreamParser.cpp L329-L343](https://github.com/rgaufman/live555/blob/a0eb8f91/liveMedia/H263plusVideoStreamParser.cpp#L329-L343)

The library handles various timing mechanisms:

* For H.264/H.265: Extracts timing information from NAL units
* For H.263: Calculates timing based on temporal reference values
* For Matroska: Uses the container's internal timecode system
* For MPEG Transport Streams: Utilizes PCR (Program Clock Reference) values

## Example: MKV Streaming Pipeline

The following is an example of the processing pipeline for streaming a Matroska file:

```mermaid
sequenceDiagram
  participant Matroska File
  participant MatroskaFileParser
  participant MatroskaDemux
  participant FramedSource
  participant RTPSink
  participant Network

  Matroska File->>MatroskaFileParser: createNew()
  MatroskaFileParser->>MatroskaDemux: newDemux()
  MatroskaDemux->>FramedSource: newDemuxedTrack()
  FramedSource->>MatroskaFileParser: getNextFrame()
  MatroskaFileParser->>Matroska File: read data
  MatroskaFileParser->>MatroskaFileParser: parseBlock()
  MatroskaFileParser->>FramedSource: deliver frame
  FramedSource->>RTPSink: pass frame
  RTPSink->>Network: send RTP packet
```

Sources: [testProgs/testMKVStreamer.cpp L68-L115](https://github.com/rgaufman/live555/blob/a0eb8f91/testProgs/testMKVStreamer.cpp#L68-L115)

## Integration with RTP Streaming

The media processing components integrate with the RTP subsystem to enable network streaming:

```mermaid
flowchart TD

FramedSource["FramedSource<br>(Base class)"]
MediaFramer["Stream Framers"]
FileParser["File Parsers"]
RTPSink["RTPSink<br>(Base class)"]
RTPPacketizing["RTP Packetizing"]
RTCPReporting["RTCP Reporting"]
Network["Network<br>Transmission"]

FramedSource --> RTPSink
RTPPacketizing --> Network
RTCPReporting --> Network

subgraph subGraph1 ["RTP Subsystem"]
    RTPSink
    RTPPacketizing
    RTCPReporting
    RTPSink --> RTPPacketizing
    RTPSink --> RTCPReporting
end

subgraph subGraph0 ["Media Processing"]
    FramedSource
    MediaFramer
    FileParser
    MediaFramer --> FramedSource
    FileParser --> FramedSource
end
```

Sources: [testProgs/testMKVStreamer.cpp L94-L123](https://github.com/rgaufman/live555/blob/a0eb8f91/testProgs/testMKVStreamer.cpp#L94-L123)

## Summary

The Media Processing subsystem in LIVE555 provides a comprehensive framework for handling various media formats and preparing them for network streaming. The key components include:

1. **Stream Framers**: Parse and frame raw media streams (H.264/H.265, H.263, etc.)
2. **File Parsers**: Extract media data from container formats (Matroska, MPEG-TS)
3. **Bit-Level Utilities**: Support precise parsing of media headers and metadata
4. **Timestamp Management**: Ensure proper synchronization during streaming

These components work together to transform raw media data into properly formatted packets suitable for RTP streaming over networks.
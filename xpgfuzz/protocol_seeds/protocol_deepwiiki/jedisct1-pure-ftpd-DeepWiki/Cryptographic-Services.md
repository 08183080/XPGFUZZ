# Cryptographic Services

> **Relevant source files**
> * [src/Makefile.am](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/Makefile.am)
> * [src/alt_arc4random.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c)
> * [src/alt_arc4random.h](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.h)

This document covers Pure-FTPd's cryptographic random number generation system, which provides secure pseudorandom number generation across different platforms. The system implements a ChaCha20-based PRNG (Pseudorandom Number Generator) as an alternative to system-provided random number generators on platforms that lack robust implementations.

For information about TLS/SSL encryption services, see [TLS/SSL Encryption](/jedisct1/pure-ftpd/3.1-tlsssl-encryption). For authentication-related cryptographic functions, see [Authentication and User Management](/jedisct1/pure-ftpd/4-authentication-and-user-management).

## Overview

Pure-FTPd's cryptographic services are primarily focused on providing high-quality random number generation through the `alt_arc4random` implementation. This system serves as a cross-platform replacement for OpenBSD's `arc4random()` family of functions, using the ChaCha20 stream cipher to generate cryptographically secure random numbers.

The implementation is used throughout Pure-FTPd for security-sensitive operations such as:

* Password salt generation in the virtual user system
* Session token generation
* Cryptographic nonce generation
* General-purpose secure random number generation

## System Architecture

```mermaid
flowchart TD

PUREPW["pure-pw<br>Password Manager"]
FTPD["pure-ftpd<br>Main Server"]
OTHER["Other Components"]
API_RANDOM["alt_arc4random()"]
API_BUF["alt_arc4random_buf()"]
API_UNIFORM["alt_arc4random_uniform()"]
API_STIR["alt_arc4random_stir()"]
RNG_STATE["struct rng_state"]
CHACHA20["ChaCha20 Implementation"]
KEY_MGMT["Key Management"]
RESERVE["Reserve Buffer"]
URANDOM["/dev/urandom"]
RANDOM["/dev/random"]
DEVICE_OPEN["random_dev_open()"]

PUREPW --> API_RANDOM
FTPD --> API_BUF
OTHER --> API_UNIFORM
API_RANDOM --> RNG_STATE
API_BUF --> RNG_STATE
API_UNIFORM --> RNG_STATE
API_STIR --> RNG_STATE
CHACHA20 --> DEVICE_OPEN
KEY_MGMT --> DEVICE_OPEN

subgraph subGraph3 ["Entropy Sources"]
    URANDOM
    RANDOM
    DEVICE_OPEN
    DEVICE_OPEN --> URANDOM
    DEVICE_OPEN --> RANDOM
end

subgraph subGraph2 ["ChaCha20 PRNG Core"]
    RNG_STATE
    CHACHA20
    KEY_MGMT
    RESERVE
    RNG_STATE --> CHACHA20
    RNG_STATE --> KEY_MGMT
    RNG_STATE --> RESERVE
end

subgraph subGraph1 ["Cryptographic API"]
    API_RANDOM
    API_BUF
    API_UNIFORM
    API_STIR
end

subgraph subGraph0 ["Application Layer"]
    PUREPW
    FTPD
    OTHER
end
```

Sources: [src/alt_arc4random.c L1-L240](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L1-L240)

 [src/alt_arc4random.h L1-L31](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.h#L1-L31)

 [src/Makefile.am L25-L29](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/Makefile.am#L25-L29)

 [src/Makefile.am L213-L228](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/Makefile.am#L213-L228)

## ChaCha20 Implementation

The core of the cryptographic system is a ChaCha20 stream cipher implementation that provides the pseudorandom number generation. ChaCha20 is a modern, high-security cipher designed by Daniel J. Bernstein.

```mermaid
flowchart TD

INIT["chacha20_init()"]
UPDATE["chacha20_update()"]
ROUNDS["CHACHA20_ROUNDS()"]
QUARTER["CHACHA20_QUARTERROUND()"]
STATE["uint32_t st[16]"]
CONSTANTS["constants[4]"]
KEY["key[32 bytes]"]
COUNTER["counter[4 bytes]"]
KEYSTREAM["64-byte blocks"]
RNG_FUNC["chacha20_rng()"]
OUTPUT["Random output"]

INIT --> STATE
STATE --> UPDATE
RNG_FUNC --> INIT
RNG_FUNC --> UPDATE
UPDATE --> KEYSTREAM

subgraph subGraph2 ["Output Generation"]
    KEYSTREAM
    RNG_FUNC
    OUTPUT
    KEYSTREAM --> OUTPUT
end

subgraph subGraph1 ["State Management"]
    STATE
    CONSTANTS
    KEY
    COUNTER
    CONSTANTS --> STATE
    KEY --> STATE
    COUNTER --> STATE
end

subgraph subGraph0 ["ChaCha20 Core Functions"]
    INIT
    UPDATE
    ROUNDS
    QUARTER
    UPDATE --> ROUNDS
    ROUNDS --> QUARTER
end
```

Sources: [src/alt_arc4random.c L28-L88](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L28-L88)

 [src/alt_arc4random.c L13-L26](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L13-L26)

### ChaCha20 Constants and Parameters

The implementation uses standard ChaCha20 parameters:

| Parameter | Value | Purpose |
| --- | --- | --- |
| Key Size | 32 bytes | ChaCha20 key length |
| Block Size | 64 bytes | Output block size |
| Rounds | 20 | Security rounds |
| Constants | "expand 32-byte k" | ChaCha20 magic constants |

The magic constants are implemented as [src/alt_arc4random.c L60-L62](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L60-L62)

:

```javascript
static const uint32_t constants[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};
```

## Random Number Generator State

The system maintains global state through the `rng_state` structure, which manages the PRNG's internal state and entropy source connections.

```mermaid
flowchart TD

INITIALIZED["initialized<br>int"]
FD["fd<br>int"]
OFF["off<br>size_t"]
KEY["key[32]<br>uint8_t"]
RESERVE["reserve[512]<br>uint8_t"]
STIR["alt_arc4random_stir()"]
REFILL["Refill Reserve"]
RESEED["Reseed Key"]
DEV_URANDOM["/dev/urandom"]
DEV_RANDOM["/dev/random"]

INITIALIZED --> STIR
FD --> DEV_URANDOM
FD --> DEV_RANDOM
KEY --> RESEED
RESERVE --> REFILL

subgraph subGraph2 ["Entropy Sources"]
    DEV_URANDOM
    DEV_RANDOM
end

subgraph subGraph1 ["State Operations"]
    STIR
    REFILL
    RESEED
    STIR --> RESEED
    REFILL --> RESEED
end

subgraph subGraph0 ["struct rng_state"]
    INITIALIZED
    FD
    OFF
    KEY
    RESERVE
    OFF --> RESERVE
end
```

Sources: [src/alt_arc4random.c L90-L98](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L90-L98)

 [src/alt_arc4random.c L138-L153](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L138-L153)

### Reserve Buffer System

The implementation uses a 512-byte reserve buffer (`RNG_RESERVE_LEN`) to optimize performance by reducing the frequency of ChaCha20 operations and system calls to entropy sources.

## Platform Compatibility Layer

The system provides a compatibility layer that automatically selects between the native system implementation and the custom ChaCha20 implementation based on the target platform.

```mermaid
flowchart TD

PLATFORM_CHECK["Platform Check"]
OPENBSD["OpenBSD/CloudABI/WASI"]
OTHER_PLATFORMS["Other Platforms"]
NATIVE_ARC4["arc4random()"]
NATIVE_STIR["arc4random_stir()"]
NATIVE_BUF["arc4random_buf()"]
NATIVE_UNIFORM["arc4random_uniform()"]
ALT_ARC4["alt_arc4random()"]
ALT_STIR["alt_arc4random_stir()"]
ALT_BUF["alt_arc4random_buf()"]
ALT_UNIFORM["alt_arc4random_uniform()"]

OPENBSD --> NATIVE_ARC4
OPENBSD --> NATIVE_STIR
OPENBSD --> NATIVE_BUF
OPENBSD --> NATIVE_UNIFORM
OTHER_PLATFORMS --> ALT_ARC4
OTHER_PLATFORMS --> ALT_STIR
OTHER_PLATFORMS --> ALT_BUF
OTHER_PLATFORMS --> ALT_UNIFORM

subgraph subGraph2 ["Alternative Path"]
    ALT_ARC4
    ALT_STIR
    ALT_BUF
    ALT_UNIFORM
end

subgraph subGraph1 ["OpenBSD Path"]
    NATIVE_ARC4
    NATIVE_STIR
    NATIVE_BUF
    NATIVE_UNIFORM
end

subgraph subGraph0 ["Compile-Time Selection"]
    PLATFORM_CHECK
    OPENBSD
    OTHER_PLATFORMS
    PLATFORM_CHECK --> OPENBSD
    PLATFORM_CHECK --> OTHER_PLATFORMS
end
```

Sources: [src/alt_arc4random.h L8-L26](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.h#L8-L26)

 [src/alt_arc4random.c L3-L4](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L3-L4)

## API Functions

The cryptographic services provide a complete API compatible with OpenBSD's `arc4random()` family:

### Core Functions

| Function | Purpose | Return Type |
| --- | --- | --- |
| `alt_arc4random()` | Generate 32-bit random number | `uint32_t` |
| `alt_arc4random_buf()` | Fill buffer with random data | `void` |
| `alt_arc4random_uniform()` | Generate uniform random in range | `uint32_t` |
| `alt_arc4random_stir()` | Force reseeding from entropy source | `void` |
| `alt_arc4random_close()` | Cleanup and close resources | `int` |

### Function Implementation Details

```mermaid
flowchart TD

BUF_START["alt_arc4random_buf(buffer, len)"]
CHECK_INIT["initialized?"]
STIR_CALL["alt_arc4random_stir()"]
MAIN_LOOP["Main generation loop"]
RESERVE_CHECK["reserve empty?"]
DIRECT_GEN["Direct generation<br>(len >= 512)"]
COPY_DATA["Copy from reserve"]
REFILL_RESERVE["Refill reserve buffer"]

subgraph subGraph0 ["alt_arc4random_buf() Flow"]
    BUF_START
    CHECK_INIT
    STIR_CALL
    MAIN_LOOP
    RESERVE_CHECK
    DIRECT_GEN
    COPY_DATA
    REFILL_RESERVE
    BUF_START --> CHECK_INIT
    CHECK_INIT --> STIR_CALL
    CHECK_INIT --> MAIN_LOOP
    STIR_CALL --> MAIN_LOOP
    MAIN_LOOP --> RESERVE_CHECK
    RESERVE_CHECK --> DIRECT_GEN
    RESERVE_CHECK --> COPY_DATA
    DIRECT_GEN --> MAIN_LOOP
    REFILL_RESERVE --> COPY_DATA
    COPY_DATA --> MAIN_LOOP
end
```

Sources: [src/alt_arc4random.c L155-L190](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L155-L190)

 [src/alt_arc4random.c L192-L199](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L192-L199)

 [src/alt_arc4random.c L201-L217](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L201-L217)

## Entropy Source Management

The system manages entropy sources through a robust device selection and management system that prioritizes `/dev/urandom` over `/dev/random` for better performance while maintaining security.

### Device Selection Logic

```mermaid
flowchart TD

OPEN_START["random_dev_open()"]
DEVICES["Device list:<br>/dev/urandom<br>/dev/random"]
TRY_OPEN["open(device, O_RDONLY)"]
FSTAT_CHECK["fstat() validation"]
CHAR_DEV_CHECK["Character device?"]
SET_CLOEXEC["Set FD_CLOEXEC"]
SUCCESS["Return fd"]
NEXT_DEVICE["Try next device"]
FAIL["Return -1, errno=EIO"]

OPEN_START --> DEVICES
DEVICES --> TRY_OPEN
TRY_OPEN --> FSTAT_CHECK
FSTAT_CHECK --> CHAR_DEV_CHECK
CHAR_DEV_CHECK --> SET_CLOEXEC
CHAR_DEV_CHECK --> NEXT_DEVICE
SET_CLOEXEC --> SUCCESS
NEXT_DEVICE --> TRY_OPEN
NEXT_DEVICE --> FAIL
```

Sources: [src/alt_arc4random.c L100-L136](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L100-L136)

### Error Handling

The system implements robust error handling with fallback mechanisms:

* If entropy source access fails, the process aborts to prevent weak randomness
* File descriptor management includes proper cleanup and close-on-exec flags
* Safe read operations ensure complete entropy collection

## Integration Points

The cryptographic services integrate with several Pure-FTPd components:

### Build System Integration

The `alt_arc4random` module is included in multiple build targets:

* `libpureftpd.a` - Main server library
* `pure-pw` - Virtual user password management tool

### Usage in Components

| Component | Usage Pattern |
| --- | --- |
| `pure-pw` | Password salt generation |
| Main server | Session tokens, nonces |
| Authentication | Random challenge generation |

Sources: [src/Makefile.am L28](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/Makefile.am#L28-L28)

 [src/Makefile.am L214](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/Makefile.am#L214-L214)

## Security Considerations

The implementation includes several security features:

1. **Forward Secrecy**: Key rotation after each use prevents state recovery attacks
2. **Secure Cleanup**: `pure_memzero()` ensures sensitive data is cleared [src/alt_arc4random.c L222](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L222-L222)
3. **Entropy Validation**: Device validation ensures only proper entropy sources are used
4. **Fail-Safe Design**: System aborts if entropy sources are unavailable rather than falling back to weak randomness

The ChaCha20-based design provides security advantages over traditional linear congruential generators and provides consistent security properties across all supported platforms.

Sources: [src/alt_arc4random.c L219-L227](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L219-L227)

 [src/alt_arc4random.c L144-L150](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/alt_arc4random.c#L144-L150)
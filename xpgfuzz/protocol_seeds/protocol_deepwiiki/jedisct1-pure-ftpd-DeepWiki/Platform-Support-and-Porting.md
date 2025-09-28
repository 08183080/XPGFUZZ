# Platform Support and Porting

> **Relevant source files**
> * [README.Donations](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Donations)
> * [README.MacOS-X](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.MacOS-X)
> * [README.Windows](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Windows)
> * [src/main.c](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c)

This document covers Pure-FTPd's cross-platform compatibility, platform-specific build configurations, and considerations for running the server on different operating systems. It includes detailed information about supported platforms, compilation requirements, and platform-specific limitations.

For information about build system configuration and compilation options, see [Build System and Compilation](/jedisct1/pure-ftpd/5.1-build-system-and-compilation). For runtime configuration that may vary by platform, see [Runtime Configuration](/jedisct1/pure-ftpd/5.2-runtime-configuration).

## Supported Platforms Overview

Pure-FTPd is primarily designed for Unix-like systems but provides experimental support for Windows through Cygwin. The codebase includes platform-specific adaptations and build configurations to ensure compatibility across different operating systems.

### Platform Support Matrix

| Platform | Support Level | Key Components | Special Requirements |
| --- | --- | --- | --- |
| Linux | Full | All features | Standard build tools |
| FreeBSD/OpenBSD/NetBSD | Full | All features | BSD-specific adaptations |
| macOS | Full | All features + Bonjour | PAM integration, Homebrew |
| Solaris | Full | All features | Platform-specific configure |
| Windows (Cygwin) | Experimental | Limited features | Static linking, special paths |
| AIX/HP-UX | Basic | Core features | Legacy Unix adaptations |

**Platform Detection and Adaptation Architecture**

```mermaid
flowchart TD

configure["configure.ac"]
config_h["config.h.in"]
makefile["Makefile.am"]
autoconf["Autoconf Macros"]
platform_tests["Platform Feature Tests"]
compiler_tests["Compiler Detection"]
config_header["config.h"]
build_flags["CFLAGS/LDFLAGS"]
feature_defines["Feature Defines"]
main_c["src/main.c"]
ftpd_h["src/ftpd.h"]
platform_code["Platform-specific Code"]

configure --> autoconf
configure --> platform_tests
configure --> compiler_tests
autoconf --> config_header
platform_tests --> feature_defines
compiler_tests --> build_flags
config_header --> main_c
feature_defines --> ftpd_h
build_flags --> platform_code

subgraph subGraph3 ["Source Adaptations"]
    main_c
    ftpd_h
    platform_code
end

subgraph subGraph2 ["Generated Configuration"]
    config_header
    build_flags
    feature_defines
end

subgraph subGraph1 ["Platform Detection"]
    autoconf
    platform_tests
    compiler_tests
end

subgraph subGraph0 ["Build System"]
    configure
    config_h
    makefile
end
```

Sources: [configure.ac](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac)

 [config.h.in](https://github.com/jedisct1/pure-ftpd/blob/3818577a/config.h.in)

 [src/main.c L1-L8](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c#L1-L8)

## Windows Support (Cygwin)

Pure-FTPd provides experimental Windows support through the Cygwin compatibility layer. This implementation requires specific build configurations and has several platform-specific limitations.

### Build Configuration for Windows

The Windows build requires specific configure flags and static linking to create self-contained executables:

```
env LDFLAGS="-static -s" \
    ./configure --with-everything --with-brokenrealpath \
                --without-shadow  --with-nonroot --with-tls
```

### Required Build Options

| Option | Purpose | Reason |
| --- | --- | --- |
| `--with-brokenrealpath` | Handle path resolution issues | Cygwin realpath() limitations |
| `--without-shadow` | Disable shadow password support | Not available on Windows |
| `--with-nonroot` | Allow non-root execution | Windows security model differences |
| `-static` | Static linking | Reduce DLL dependencies |

### Windows-Specific File System Mapping

```mermaid
flowchart TD

cygwin_root["C:\CYGWIN"]
etc_dir["C:\CYGWIN\etc"]
ftp_dir["C:\CYGWIN\FTP"]
pureftpd_dir["C:\CYGWIN\PURE-FTPD"]
config_files["/etc/pureftpd.pdb"]
anon_ftp["/ftp"]
vhost_dirs["/pure-ftpd/"]
log_files["Log Files"]
ftp_anon_dir["FTP_ANON_DIR"]

config_files --> etc_dir
anon_ftp --> ftp_dir
vhost_dirs --> pureftpd_dir
log_files --> cygwin_root
ftp_anon_dir --> ftp_dir

subgraph subGraph2 ["Environment Variables"]
    ftp_anon_dir
end

subgraph subGraph1 ["Pure-FTPd References"]
    config_files
    anon_ftp
    vhost_dirs
    log_files
end

subgraph subGraph0 ["Windows Paths"]
    cygwin_root
    etc_dir
    ftp_dir
    pureftpd_dir
end
```

### Windows Installation Requirements

**Required Files:**

* `pure-ftpd.exe` - Main server executable
* `pure-pw.exe` - User management utility
* `cygwin1.dll` - Cygwin runtime library

**Directory Structure:**

```yaml
C:\CYGWIN\           (required base directory)
C:\CYGWIN\etc\       (configuration files)
C:\CYGWIN\FTP\       (anonymous FTP root)
C:\etc\              (alternative config location)
```

### Windows Limitations

* **User Management**: System users (`/etc/passwd`) not supported; must use PureDB virtual users
* **UID/GID**: All users share same UID/GID; chroot recommended for security
* **Service Integration**: Can run as Windows service using third-party tools like Firedaemon
* **File Permissions**: Limited Unix-style permission support

Sources: [README.Windows L1-L91](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Windows#L1-L91)

## macOS Support

macOS support includes full feature compatibility with additional platform-specific integrations like Bonjour service discovery and native PAM authentication.

### Installation Methods

**Homebrew Installation:**

```
brew install pure-ftpd
```

**Available Homebrew Options:**

* `--with-mysql` - MySQL authentication support
* `--with-postgresql` - PostgreSQL authentication support
* `--with-virtualchroot` - Symbolic link following for chrooted accounts

### macOS Authentication Setup

Pure-FTPd integrates with macOS's OpenDirectory system through PAM configuration:

**Build Configuration:**

```
./configure --with-pam --with-everything
make install-strip
```

**PAM Configuration File (`/etc/pam.d/pure-ftpd`):**

```markdown
# pure-ftpd: auth account password session
auth       required       pam_opendirectory.so
account    required       pam_permit.so
password   required       pam_deny.so
session    required       pam_permit.so
```

### Bonjour Integration

macOS systems can advertise Pure-FTPd services through Bonjour:

**Build with Bonjour:**

```
./configure --with-bonjour
```

**Runtime Bonjour Service:**

```
/usr/local/sbin/pure-ftpd -lpam -B -v "My FTP Server"
```

Sources: [README.MacOS-X L1-L41](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.MacOS-X#L1-L41)

## Unix/Linux Platform Variations

### Platform-Specific Build Adaptations

```mermaid
flowchart TD

linux["Linux"]
freebsd["FreeBSD"]
openbsd["OpenBSD"]
netbsd["NetBSD"]
solaris["Solaris"]
aix["AIX"]
hpux["HP-UX"]
sendfile["sendfile()"]
kqueue["kqueue"]
epoll["epoll"]
shadow["Shadow Passwords"]
pam["PAM"]
capabilities["Linux Capabilities"]
autoconf_macros["AC_CHECK_FUNC"]
header_checks["AC_CHECK_HEADERS"]
library_checks["AC_CHECK_LIB"]
feature_availability["Runtime Feature Flags"]

linux --> epoll
linux --> sendfile
linux --> capabilities
linux --> shadow
freebsd --> kqueue
freebsd --> sendfile
openbsd --> kqueue
netbsd --> kqueue
solaris --> sendfile
aix --> pam
hpux --> pam
autoconf_macros --> feature_availability
header_checks --> feature_availability
library_checks --> feature_availability

subgraph subGraph2 ["Build System Detection"]
    autoconf_macros
    header_checks
    library_checks
end

subgraph subGraph1 ["Feature Availability"]
    sendfile
    kqueue
    epoll
    shadow
    pam
    capabilities
end

subgraph subGraph0 ["Unix Variants"]
    linux
    freebsd
    openbsd
    netbsd
    solaris
    aix
    hpux
end
```

### Common Unix Build Patterns

**Standard Unix Build:**

```
./configure --with-everything
make install-strip
```

**Security-Enhanced Build:**

```
./configure --with-everything --with-paranoidmsg --with-tls
```

**Minimal Build:**

```
./configure --with-minimal
```

Sources: [configure.ac](https://github.com/jedisct1/pure-ftpd/blob/3818577a/configure.ac)

 [README.MacOS-X L22-L24](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.MacOS-X#L22-L24)

## Entry Point and Platform Abstraction

The Pure-FTPd entry point maintains platform independence through a simple abstraction layer:

**Main Entry Point Architecture**

```mermaid
flowchart TD

main_func["main()"]
argc_argv["argc, argv"]
pureftpd_start["pureftpd_start()"]
platform_init["Platform Initialization"]
config_parse["Configuration Parsing"]
unix_init["Unix Signal Handling"]
windows_init["Windows Cygwin Setup"]
macos_init["macOS Bonjour Setup"]

main_func --> pureftpd_start
argc_argv --> pureftpd_start
platform_init --> unix_init
platform_init --> windows_init
platform_init --> macos_init

subgraph subGraph2 ["Platform-Specific Init"]
    unix_init
    windows_init
    macos_init
end

subgraph subGraph1 ["Core Abstraction"]
    pureftpd_start
    platform_init
    config_parse
    pureftpd_start --> platform_init
    pureftpd_start --> config_parse
end

subgraph subGraph0 ["Platform Entry"]
    main_func
    argc_argv
end
```

**Entry Point Implementation:**

```c
int main(int argc, char *argv[])
{
    return pureftpd_start(argc, argv, NULL);
}
```

The `pureftpd_start()` function handles all platform-specific initialization, allowing the main entry point to remain clean and portable.

Sources: [src/main.c L4-L7](https://github.com/jedisct1/pure-ftpd/blob/3818577a/src/main.c#L4-L7)

## Platform-Specific Limitations and Considerations

### Windows (Cygwin) Limitations

| Feature | Status | Limitation |
| --- | --- | --- |
| System Users | Not Supported | Must use PureDB virtual users |
| File Permissions | Limited | Unix permissions partially emulated |
| Process Security | Modified | No privilege separation |
| Anonymous FTP | Supported | Fixed path `C:\CYGWIN\FTP` |
| Virtual Hosting | Supported | Uses `C:\CYGWIN\PURE-FTPD\<ip>\` |

### macOS Considerations

| Feature | Implementation | Notes |
| --- | --- | --- |
| Authentication | PAM + OpenDirectory | Native system integration |
| Service Discovery | Bonjour | Optional compile-time feature |
| File Systems | HFS+/APFS | Full Unicode support |
| Package Management | Homebrew | Preferred installation method |

### Unix Variations

| Platform | Special Considerations | Build Notes |
| --- | --- | --- |
| Linux | Full feature support | Standard reference platform |
| FreeBSD | kqueue, jails support | Native BSD features |
| OpenBSD | Security focus | Enhanced privilege separation |
| Solaris | Legacy compatibility | Traditional Unix patterns |

## Porting Guidelines

### Adding New Platform Support

1. **Configure Detection**: Add platform detection to `configure.ac`
2. **Feature Testing**: Implement feature availability tests
3. **Header Adaptation**: Update platform-specific includes in `ftpd.h`
4. **Build Integration**: Add platform-specific build flags
5. **Documentation**: Create platform-specific README file

### Platform Testing Checklist

* Basic FTP operations (LIST, RETR, STOR)
* Authentication methods (system, virtual, database)
* TLS/SSL encryption
* File permission handling
* Process security features
* Service integration (init scripts, systemd, etc.)

Sources: [README.Windows L77-L91](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.Windows#L77-L91)

 [README.MacOS-X L5-L17](https://github.com/jedisct1/pure-ftpd/blob/3818577a/README.MacOS-X#L5-L17)
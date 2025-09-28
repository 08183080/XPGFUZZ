# Project History and Community

> **Relevant source files**
> * [src/ACKNOWLEDGMENTS](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS)
> * [src/README.UPDATING](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING)

This document covers the historical development of the Exim Mail Transfer Agent project, its community structure, and the evolution of its codebase architecture. It provides context on how the project has evolved from its origins through major version releases and community transitions.

For information about the current technical architecture, see [Architecture Overview](/Exim/exim/1.2-architecture-overview). For details about building and configuring Exim, see [Build-time Configuration](/Exim/exim/4.1-build-time-configuration).

## Project Origins and Evolution

Exim was originally conceived and developed by Philip Hazel, who served as the project's architect and primary maintainer for many years before retiring. The project has undergone significant architectural evolution, with two major upheavals in its design philosophy.

The most significant architectural change occurred with the release of version 4.00, which completely reworked the mail routing system. The previous "directors" concept was abolished in favor of a unified router-based approach, and policy controls were moved from separate options to Access Control Lists (ACLs). An earlier major change occurred at version 3.00, which reworked domain/host/net/address list handling.

**Project Development Timeline**

```css
#mermaid-0lr5tt5kdhyl{font-family:ui-sans-serif,-apple-system,system-ui,Segoe UI,Helvetica;font-size:16px;fill:#ccc;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-0lr5tt5kdhyl .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-0lr5tt5kdhyl .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-0lr5tt5kdhyl .error-icon{fill:#333;}#mermaid-0lr5tt5kdhyl .error-text{fill:#cccccc;stroke:#cccccc;}#mermaid-0lr5tt5kdhyl .edge-thickness-normal{stroke-width:1px;}#mermaid-0lr5tt5kdhyl .edge-thickness-thick{stroke-width:3.5px;}#mermaid-0lr5tt5kdhyl .edge-pattern-solid{stroke-dasharray:0;}#mermaid-0lr5tt5kdhyl .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-0lr5tt5kdhyl .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-0lr5tt5kdhyl .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-0lr5tt5kdhyl .marker{fill:#666;stroke:#666;}#mermaid-0lr5tt5kdhyl .marker.cross{stroke:#666;}#mermaid-0lr5tt5kdhyl svg{font-family:ui-sans-serif,-apple-system,system-ui,Segoe UI,Helvetica;font-size:16px;}#mermaid-0lr5tt5kdhyl p{margin:0;}#mermaid-0lr5tt5kdhyl .edge{stroke-width:3;}#mermaid-0lr5tt5kdhyl .section--1 rect,#mermaid-0lr5tt5kdhyl .section--1 path,#mermaid-0lr5tt5kdhyl .section--1 circle,#mermaid-0lr5tt5kdhyl .section--1 path{fill:hsl(0, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section--1 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon--1{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge--1{stroke:hsl(0, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth--1{stroke-width:17;}#mermaid-0lr5tt5kdhyl .section--1 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-0 rect,#mermaid-0lr5tt5kdhyl .section-0 path,#mermaid-0lr5tt5kdhyl .section-0 circle,#mermaid-0lr5tt5kdhyl .section-0 path{fill:hsl(-120, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-0 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-0{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-0{stroke:hsl(-120, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-0{stroke-width:14;}#mermaid-0lr5tt5kdhyl .section-0 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-1 rect,#mermaid-0lr5tt5kdhyl .section-1 path,#mermaid-0lr5tt5kdhyl .section-1 circle,#mermaid-0lr5tt5kdhyl .section-1 path{fill:hsl(0, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-1 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-1{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-1{stroke:hsl(0, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-1{stroke-width:11;}#mermaid-0lr5tt5kdhyl .section-1 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-2 rect,#mermaid-0lr5tt5kdhyl .section-2 path,#mermaid-0lr5tt5kdhyl .section-2 circle,#mermaid-0lr5tt5kdhyl .section-2 path{fill:hsl(30, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-2 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-2{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-2{stroke:hsl(30, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-2{stroke-width:8;}#mermaid-0lr5tt5kdhyl .section-2 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-3 rect,#mermaid-0lr5tt5kdhyl .section-3 path,#mermaid-0lr5tt5kdhyl .section-3 circle,#mermaid-0lr5tt5kdhyl .section-3 path{fill:hsl(60, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-3 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-3{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-3{stroke:hsl(60, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-3{stroke-width:5;}#mermaid-0lr5tt5kdhyl .section-3 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-4 rect,#mermaid-0lr5tt5kdhyl .section-4 path,#mermaid-0lr5tt5kdhyl .section-4 circle,#mermaid-0lr5tt5kdhyl .section-4 path{fill:hsl(90, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-4 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-4{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-4{stroke:hsl(90, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-4{stroke-width:2;}#mermaid-0lr5tt5kdhyl .section-4 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-5 rect,#mermaid-0lr5tt5kdhyl .section-5 path,#mermaid-0lr5tt5kdhyl .section-5 circle,#mermaid-0lr5tt5kdhyl .section-5 path{fill:hsl(120, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-5 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-5{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-5{stroke:hsl(120, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-5{stroke-width:-1;}#mermaid-0lr5tt5kdhyl .section-5 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-6 rect,#mermaid-0lr5tt5kdhyl .section-6 path,#mermaid-0lr5tt5kdhyl .section-6 circle,#mermaid-0lr5tt5kdhyl .section-6 path{fill:hsl(150, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-6 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-6{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-6{stroke:hsl(150, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-6{stroke-width:-4;}#mermaid-0lr5tt5kdhyl .section-6 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-7 rect,#mermaid-0lr5tt5kdhyl .section-7 path,#mermaid-0lr5tt5kdhyl .section-7 circle,#mermaid-0lr5tt5kdhyl .section-7 path{fill:hsl(210, 0%, 75%);}#mermaid-0lr5tt5kdhyl .section-7 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-7{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-7{stroke:hsl(210, 0%, 75%);}#mermaid-0lr5tt5kdhyl .edge-depth-7{stroke-width:-7;}#mermaid-0lr5tt5kdhyl .section-7 line{stroke:rgb(63.75, 63.75, 63.75);stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-8 rect,#mermaid-0lr5tt5kdhyl .section-8 path,#mermaid-0lr5tt5kdhyl .section-8 circle,#mermaid-0lr5tt5kdhyl .section-8 path{fill:hsl(270, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-8 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-8{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-8{stroke:hsl(270, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-8{stroke-width:-10;}#mermaid-0lr5tt5kdhyl .section-8 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-9 rect,#mermaid-0lr5tt5kdhyl .section-9 path,#mermaid-0lr5tt5kdhyl .section-9 circle,#mermaid-0lr5tt5kdhyl .section-9 path{fill:hsl(300, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-9 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-9{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-9{stroke:hsl(300, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-9{stroke-width:-13;}#mermaid-0lr5tt5kdhyl .section-9 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-10 rect,#mermaid-0lr5tt5kdhyl .section-10 path,#mermaid-0lr5tt5kdhyl .section-10 circle,#mermaid-0lr5tt5kdhyl .section-10 path{fill:hsl(330, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-10 text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .node-icon-10{font-size:40px;color:#ccc;}#mermaid-0lr5tt5kdhyl .section-edge-10{stroke:hsl(330, 0%, 0%);}#mermaid-0lr5tt5kdhyl .edge-depth-10{stroke-width:-16;}#mermaid-0lr5tt5kdhyl .section-10 line{stroke:#ffffff;stroke-width:3;}#mermaid-0lr5tt5kdhyl .lineWrapper line{stroke:#ccc;}#mermaid-0lr5tt5kdhyl .disabled,#mermaid-0lr5tt5kdhyl .disabled circle,#mermaid-0lr5tt5kdhyl .disabled text{fill:lightgray;}#mermaid-0lr5tt5kdhyl .disabled text{fill:#efefef;}#mermaid-0lr5tt5kdhyl .section-root rect,#mermaid-0lr5tt5kdhyl .section-root path,#mermaid-0lr5tt5kdhyl .section-root circle{fill:hsl(0, 0%, 0%);}#mermaid-0lr5tt5kdhyl .section-root text{fill:#ccc;}#mermaid-0lr5tt5kdhyl .icon-container{height:100%;display:flex;justify-content:center;align-items:center;}#mermaid-0lr5tt5kdhyl .edge{fill:none;}#mermaid-0lr5tt5kdhyl .eventWrapper{filter:brightness(120%);}#mermaid-0lr5tt5kdhyl :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}Early   DevelopmentPre-3.00Initial   architectureBasic   domain/hosthandlingFirst   Major   Rework3.00Domain/host/net/addresslists   reworkedFoundation   formodern   listhandlingSecond   MajorRework4.00"Directors"abolishedUnifiedrouter-basedsystemIntroduction   ofACLsPolicy   controlcentralizationModern   Era4.70+DKIMimplementationEmbedded   PCREremovedExternal   librarydependencies4.80+TLS   architectureoverhaulGnuTLS   APImodernizationSecurity   hardening4.90+TaintingmechanismEnhanced   securitymodelDynamic   modulesupport"Exim Major Architecture Evolution"
```

Sources: [src/README.UPDATING L8-L22](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING#L8-L22)

## Community Structure

The Exim project operates with a multi-tiered community structure consisting of maintainers with commit access, contributors who provide patches and improvements, and packagers who handle distribution-specific packaging.

**Current Community Organization**

```mermaid
flowchart TD

maintainers["Maintainers<br>Commit Access"]
retired["Retired Maintainers<br>Philip Hazel (Original Author)"]
contributors["Contributors<br>Patches & Bug Reports"]
packagers["Distribution Packagers<br>OS Integration"]
testers["Test Suite Contributors<br>Quality Assurance"]
eximstats_maint["eximstats Maintainer<br>Steve Campbell"]
webmaster["Webmaster<br>Mike Cardwell"]
security["Security Focus<br>Phil Pennock, David Woodhouse"]
dkim_content["DKIM & Content Scanning<br>Tom Kistner, Michael Haardt"]

maintainers --> contributors
maintainers --> packagers
maintainers --> eximstats_maint
maintainers --> webmaster
maintainers --> security
maintainers --> dkim_content

subgraph subGraph2 ["Specialized Roles"]
    eximstats_maint
    webmaster
    security
    dkim_content
end

subgraph subGraph1 ["Extended Community"]
    contributors
    packagers
    testers
    contributors --> testers
end

subgraph subGraph0 ["Core Team"]
    maintainers
    retired
    retired --> maintainers
end
```

Sources: [src/ACKNOWLEDGMENTS L339-L355](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L339-L355)

 [src/ACKNOWLEDGMENTS L463-L476](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L463-L476)

### Active Maintainers and Focus Areas

The current maintainer team includes individuals with specialized focus areas:

| Maintainer | Primary Focus Area |
| --- | --- |
| Steve Campbell | eximstats maintenance |
| Mike Cardwell | Exim webmaster |
| Tony Finch | Ratelimit code, general fixes |
| Graeme Fowler | General maintenance |
| Michael Haardt | Sieve support, DKIM |
| Jeremy Harris | General development |
| John Jetmore | Testing and utilities |
| Tom Kistner | DKIM, content scanning, SPA |
| Todd Lyons | General maintenance |
| Nigel Metheringham | Transitioning from active role |
| Phil Pennock | Security, mostly idle status |
| David Woodhouse | Dynamic modules, security |

Sources: [src/ACKNOWLEDGMENTS L341-L354](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L341-L354)

## Contributor Recognition System

The project maintains detailed acknowledgment records divided into historical periods, recognizing different types of contributions beyond just code patches.

**Contribution Categories**

```mermaid
flowchart TD

code["Code Patches<br>Feature Implementation"]
bugs["Bug Reports<br>Issue Diagnosis"]
docs["Documentation<br>Fixes & Improvements"]
testing["Testing<br>Suite Development"]
analysis["Code Analysis<br>Security Review"]
integration["Platform Integration<br>OS-specific Support"]
maintainer["Maintainer Status<br>Commit Access"]
regular["Regular Contributor<br>Multiple Contributions"]
occasional["Occasional Contributor<br>Specific Fixes"]
old_list["Original List<br>Pre-retirement Era"]
new_list["Modern List<br>Post-2007 Era"]
changelog["ChangeLog Integration<br>Ongoing Attribution"]

code --> maintainer
bugs --> regular
docs --> regular
testing --> maintainer
analysis --> regular
integration --> occasional
regular --> old_list
regular --> new_list
maintainer --> changelog

subgraph subGraph2 ["Historical Tracking"]
    old_list
    new_list
    changelog
end

subgraph subGraph1 ["Recognition Levels"]
    maintainer
    regular
    occasional
end

subgraph subGraph0 ["Types of Contributions"]
    code
    bugs
    docs
    testing
    analysis
    integration
end
```

The project explicitly acknowledges that contribution tracking has gaps, particularly during a five-year transition period, and actively encourages contributors to report omissions.

Sources: [src/ACKNOWLEDGMENTS L10-L14](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L10-L14)

 [src/ACKNOWLEDGMENTS L24-L42](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L24-L42)

## Major Version Milestones

The evolution of Exim's architecture can be tracked through significant version releases that introduced breaking changes or major new capabilities.

### Security and Architecture Milestones

| Version | Key Changes | Impact |
| --- | --- | --- |
| 4.73 | Root execution prohibited | Security hardening |
| 4.80 | TLS architecture overhaul | GnuTLS API modernization |
| 4.88 | Deprecated features removed | Code cleanup |
| 4.91 | DANE/SPF promoted to supported | Feature stabilization |
| 4.93 | Tainting mechanism introduced | Memory safety |
| 4.94 | Transport taint checking | Configuration compatibility |

**Version Evolution Security Focus**

```mermaid
flowchart TD

embedded_pcre["Embedded PCRE Removed<br>External Dependencies"]
ssl_changes["SSL/TLS Overhaul<br>Modern Cipher Support"]
gnutls_api["GnuTLS API Update<br>Deprecation Removal"]
root_prohibition["Root Execution Banned<br>CONFIGURE_OWNER Enforcement"]
config_security["Configuration Security<br>TRUSTED_CONFIG_LIST"]
tls_validation["TLS Validation at Startup<br>Early Error Detection"]
tainting["Tainting Mechanism<br>Untrusted Data Tracking"]
transport_security["Transport Taint Checking<br>Delivery Path Security"]
dynamic_modules["Dynamic Module Support<br>Runtime Extensibility"]

gnutls_api --> root_prohibition
tls_validation --> tainting

subgraph subGraph2 ["4.90+ Modern Era"]
    tainting
    transport_security
    dynamic_modules
    tainting --> transport_security
    transport_security --> dynamic_modules
end

subgraph subGraph1 ["4.80-4.90 Era"]
    root_prohibition
    config_security
    tls_validation
    root_prohibition --> config_security
    config_security --> tls_validation
end

subgraph subGraph0 ["4.70-4.80 Era"]
    embedded_pcre
    ssl_changes
    gnutls_api
    embedded_pcre --> ssl_changes
    ssl_changes --> gnutls_api
end
```

Sources: [src/README.UPDATING L29-L94](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING#L29-L94)

 [src/README.UPDATING L375-L409](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING#L375-L409)

### Feature Lifecycle Management

The project follows a structured approach to feature development, moving capabilities through experimental, supported, and deprecated phases:

```mermaid
stateDiagram-v2
    [*] --> Experimental : "Proven Stable"
    Experimental --> Supported : "Proven Stable"
    Experimental --> Removed : "Unsuccessful"
    Supported --> Deprecated : "Unsuccessful"
    Deprecated --> Removed : "End of Life"
```

Examples of this lifecycle include the promotion of DANE and SPF from experimental to supported status in version 4.91, and the removal of deprecated GnuTLS options in version 4.88.

Sources: [src/README.UPDATING L106-L122](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING#L106-L122)

 [src/README.UPDATING L149-L155](https://github.com/Exim/exim/blob/29568b25/src/README.UPDATING#L149-L155)

## Development Process Evolution

The project has evolved from a single-maintainer model to a distributed maintenance approach, with changes tracked through multiple mechanisms:

* **ChangeLog entries**: Detailed change tracking with maintainer initials
* **Git commit history**: Modern version control integration
* **Acknowledgment updates**: Community contribution tracking
* **Test suite expansion**: Quality assurance evolution

The transition from Philip Hazel's direct maintenance to the current maintainer team represents a significant evolution in project governance, with specialized roles emerging for different aspects of the codebase.

Sources: [src/ACKNOWLEDGMENTS L320-L327](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L320-L327)

 [src/ACKNOWLEDGMENTS L347-L348](https://github.com/Exim/exim/blob/29568b25/src/ACKNOWLEDGMENTS#L347-L348)
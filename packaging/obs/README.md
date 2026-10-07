# OBS packaging for Libre DT-Lab

This directory holds the packaging files used by the
[Open Build Service](https://build.opensuse.org) (OBS) to build `.rpm` and
`.deb` packages from the same source tree.

## Files

| File | Purpose |
|---|---|
| `libre-dt-lab.spec` | RPM spec used for openSUSE and Fedora builds |
| `libre-dt-lab.changes` | RPM changelog (required by OBS) |
| `libre-dt-lab-rpmlintrc` | rpmlint filters |
| `libre-dt-lab.dsc` + `debian.*` | Debian source package inputs (debtransform), see below |
| `_service` | OBS source service: fetches the git tree (incl. submodules) and generates the tarball |

## RPM

The spec is a fork-adapted version of the openSUSE darktable spec. Notable
differences:

* everything is rebranded to `libre-dt-lab` (binary, data dir, desktop id
  `org.libredtlab.libredtlab`, man pages, bash completion);
* the bundled OpenCL headers (`src/external/OpenCL`) are **kept** and used,
  the fork relies on them;
* GraphicsMagick is used instead of ImageMagick;
* the version is set by the OBS service from the `libre-X.Y.Z` git tags.

## Debian

OBS builds Debian packages from a Debian source package. When `debian.*`
files and a `.dsc` are present, OBS runs `debtransform` to assemble them.

## Versioning / source service

The `_service` uses `obs_scm` (git, submodules enabled) + `set_version` +
`tar`. The tarball is named `libre-dt-lab-<version>.tar.xz`, matching
`Source0` in the spec. `<version>` is derived from the `libre-*` git tags.

## Useful commands

```bash
# checkout the OBS package locally
osc checkout home:<user>:libre-dt-lab libre-dt-lab

# local build (rpm, openSUSE Tumbleweed)
osc build openSUSE_Tumbleweed x86_64

# local build (deb, Debian)
osc build Debian_13 x86_64
```

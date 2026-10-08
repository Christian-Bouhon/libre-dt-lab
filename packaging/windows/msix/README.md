# Microsoft Store (MSIX) packaging for Libre DT-Lab

Libre DT-Lab is a GTK3/Win32 desktop application. For the Microsoft Store it is
packaged as a **full-trust MSIX** (Desktop Bridge): the complete install tree
(`bin/`, `lib/`, `share/`) is packed together with a manifest declaring the
`runFullTrust` capability.

## Files

| File | Purpose |
|---|---|
| `AppxManifest.xml` | MSIX manifest template (tokens replaced at build time) |
| `make-layout.sh` | Builds the layout dir, generates the tile assets and the manifest |
| `pack-msix.ps1` | Packs the layout into a `.msix` with the Windows SDK `MakeAppx` |
| `file-associations.txt` | Extensions registered as file type associations (raw + images) |

## How CI builds it

The `Windows` job in `.github/workflows/main.yml` adds, for the x64 (UCRT64)
build, a layout step (MSYS2, `magick` available) and a pack step (PowerShell).
The resulting `.msix` is uploaded with the other Windows artifacts.

The package identity is taken from GitHub **repository variables** (Settings →
Secrets and variables → Actions → Variables):

| Variable | Default (reserved app) |
|---|---|
| `MSIX_IDENTITY_NAME` | `Christian-B.6348823D232E9` |
| `MSIX_PUBLISHER` | `CN=9866BDE7-54D1-43A5-ACDF-81A156743408` |
| `MSIX_PUBLISHER_DISPLAY_NAME` | `Christian-B` |
| `MSIX_DISPLAY_NAME` | `Libre DT-Lab` |

These defaults are the identity reserved in Partner Center, so the CI produces
a Store-ready package out of the box. Override the repository variables
(Settings → Secrets and variables → Actions → Variables) if the identity
changes.

Reserved app:
* Store URL: <https://apps.microsoft.com/detail/9P6L69MWW85C>
* Store ID: `9P6L69MWW85C`

## Local build

```bash
# inside the MSYS2 UCRT64 shell, with an install tree in /opt/libre-dt-lab
packaging/windows/msix/make-layout.sh \
  /opt/libre-dt-lab ./msix-layout \
  data/pixmaps/256x256/libre-dt-lab.png \
  packaging/windows/msix/AppxManifest.xml \
  1.0.0.0 x64 "Christian-B.6348823D232E9" "CN=9866BDE7-54D1-43A5-ACDF-81A156743408" \
  packaging/windows/msix/file-associations.txt

# from PowerShell
packaging/windows/msix/pack-msix.ps1 -LayoutDir .\msix-layout -OutputFile .\libre-dt-lab.msix
```

Local testing / sideloading requires either a trusted signing certificate or
Developer Mode (`Add-AppxPackage -Path .\libre-dt-lab.msix`).

## Microsoft Store submission

1. **Developer account** (free, individual or company):
   https://storedeveloper.microsoft.com → verify identity.
2. **Reserve the app name** in Partner Center, then copy the values from
   *Product management → Product identity*:
   - *Package/Identity/Name* → repo variable `MSIX_IDENTITY_NAME`
   - *Package/Identity/Publisher* → repo variable `MSIX_PUBLISHER`
   - *Package/Properties/PublisherDisplayName* → repo variable `MSIX_PUBLISHER_DISPLAY_NAME`
3. **Store listing**: description, category, at least one screenshot
   (see `../store/screenshots/`), store logos, support/website, and the
   **privacy policy URL**:
   `https://christian-bouhon.github.io/libre-dt-lab/privacy-policy.html`
   (the policy is in `docs/privacy-policy.md`, hosted via GitHub Pages).
4. **Age rating** questionnaire (IARC).
5. **Package**: upload the generated `.msix`. The Store re-signs it with a
   Microsoft certificate, so no code-signing certificate is needed.
6. **Certification** (a few business days for a new app). Each update is a new
   submission with an incremented package `Version` (4-part).

### Package version

The manifest version must be a 4-part dotted number and must increase at each
submission. The CI derives `MAJOR.MINOR.PATCH.0` from the `libre-X.Y.Z` tag
(e.g. tag `libre-1.0.0` → `1.0.0.0`).

### gphoto2 / camera tethering

The install tree ships the libgphoto2 drivers under `lib/libgphoto2*`. Unlike
the Inno/NSIS installers, an MSIX package cannot set `CAMLIBS`/`IOLIBS`
machine-wide, so the application sets them at startup from its own location
(see `_win_set_gphoto2_env()` in `src/common/file_location.c`).

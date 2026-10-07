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

| Variable | Meaning |
|---|---|
| `MSIX_IDENTITY_NAME` | Package/Identity `Name` from Partner Center |
| `MSIX_PUBLISHER` | Package/Identity `Publisher` (`CN=...`) from Partner Center |

If unset, defaults (`LibreDTLab`, `CN=Libre DT-Lab`) are used so that a package
can still be built for local testing / sideloading.

## Local build

```bash
# inside the MSYS2 UCRT64 shell, with an install tree in /opt/libre-dt-lab
packaging/windows/msix/make-layout.sh \
  /opt/libre-dt-lab ./msix-layout \
  data/pixmaps/256x256/libre-dt-lab.png \
  packaging/windows/msix/AppxManifest.xml \
  1.0.0.0 x64 LibreDTLab "CN=Libre DT-Lab" \
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
   - *Package/Properties/PublisherDisplayName* (already `Libre DT-Lab`)
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

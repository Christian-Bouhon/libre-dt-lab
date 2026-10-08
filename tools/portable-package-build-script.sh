#!/usr/bin/env bash
#
# Build self-contained (portable) .deb and .rpm packages for Libre DT-Lab.
#
# The application and *all* of its shared-library dependencies are bundled
# into an AppDir (reusing the AppImage pipeline), then installed under
# /opt/libre-dt-lab. A small launcher in /usr/bin plus a desktop entry provide
# the application menu integration.
#
# The only runtime dependency kept is the C library (glibc), which is why the
# package is built on an old base (Ubuntu 22.04 / glibc 2.35) for compatibility.
#
# Usage: run from the repository root (or anywhere; the script locates the root).
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# 1) Build the application and bundle its dependencies into ./AppDir.
#    Set SKIP_BUILD=1 to reuse an existing ./AppDir (used for testing the packaging).
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  bash tools/appimage-build-script.sh "$@"
fi

APP="$ROOT/AppDir"
if [ ! -x "$APP/usr/bin/libre-dt-lab" ]; then
  echo "Error: AppDir not built as expected ($APP)" >&2
  exit 1
fi

# 2) Version and architecture.
VERSION="$(sh tools/get_git_version_string.sh)"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  DEBARCH=amd64; RPMARCH=x86_64 ;;
  aarch64) DEBARCH=arm64; RPMARCH=aarch64 ;;
  *) echo "Error: unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
# rpm Versions may not contain '-'
RPM_VERSION="${VERSION//-/_}"
DEB_VERSION="${VERSION}-1"
DEB_FILE="libre-dt-lab_${DEB_VERSION}_${DEBARCH}.deb"

DIST="$ROOT/dist"
ROOTFS="$DIST/rootfs"
rm -rf "$DIST"
mkdir -p "$ROOTFS/opt" "$ROOTFS/usr/bin" \
         "$ROOTFS/usr/share/applications" "$ROOTFS/usr/share/icons"

# 3) Install the AppDir under /opt/libre-dt-lab (keeps bin/, lib/, share/ layout).
cp -a "$APP" "$ROOTFS/opt/libre-dt-lab"

# Launcher for the application menu / PATH.
cat > "$ROOTFS/usr/bin/libre-dt-lab" <<'EOF'
#!/bin/sh
exec /opt/libre-dt-lab/AppRun "$@"
EOF
chmod 0755 "$ROOTFS/usr/bin/libre-dt-lab"

# Desktop entry (built by the AppImage pipeline, so tokens are already resolved).
DESKTOP_SRC="$APP/usr/share/applications/org.libredtlab.libredtlab.desktop"
DESKTOP_DST="$ROOTFS/usr/share/applications/org.libredtlab.libredtlab.desktop"
if [ -f "$DESKTOP_SRC" ]; then
  cp "$DESKTOP_SRC" "$DESKTOP_DST"
else
  cat > "$DESKTOP_DST" <<'EOF'
[Desktop Entry]
Name=Libre DT-Lab
GenericName=Virtual Lighttable and Darkroom
Comment=Organize and develop images from digital cameras
Exec=libre-dt-lab %U
TryExec=libre-dt-lab
Icon=libre-dt-lab
Terminal=false
Type=Application
Categories=Graphics;Photography;GTK;
StartupNotify=true
EOF
fi
# Make sure the launcher is resolved through the PATH.
sed -i -E 's#^Exec=.*#Exec=libre-dt-lab %U#; s#^TryExec=.*#TryExec=libre-dt-lab#' "$DESKTOP_DST"

# Icons (from the bundle, else from the source tree).
if [ -d "$APP/usr/share/icons/hicolor" ]; then
  cp -a "$APP/usr/share/icons/hicolor" "$ROOTFS/usr/share/icons/"
else
  for size in 16x16 22x22 24x24 32x32 48x48 64x64 128x128 256x256; do
    src="$ROOT/data/pixmaps/$size/libre-dt-lab.png"
    [ -f "$src" ] || continue
    mkdir -p "$ROOTFS/usr/share/icons/hicolor/$size/apps"
    cp "$src" "$ROOTFS/usr/share/icons/hicolor/$size/apps/libre-dt-lab.png"
  done
fi

# 4) Build the .deb
DEB="$DIST/deb"
mkdir -p "$DEB/DEBIAN"
cp -a "$ROOTFS/." "$DEB/"
INSTALLED_SIZE="$(du -sk "$ROOTFS" | cut -f1)"
cat > "$DEB/DEBIAN/control" <<EOF
Package: libre-dt-lab
Version: ${DEB_VERSION}
Section: graphics
Priority: optional
Architecture: ${DEBARCH}
Depends: libc6 (>= 2.35), libstdc++6, libgcc-s1
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Christian Bouhon <christian.bouhon@outlook.be>
Homepage: https://github.com/Christian-Bouhon/libre-dt-lab
Description: Libre DT-Lab - virtual lighttable and darkroom (portable build)
 Libre DT-Lab is an experimental fork of darktable. This self-contained build
 bundles the application and its libraries and installs them under
 /opt/libre-dt-lab, with a launcher and a menu entry.
EOF
cp packaging/linux/postinst "$DEB/DEBIAN/postinst"; chmod 0755 "$DEB/DEBIAN/postinst"
cp packaging/linux/postrm   "$DEB/DEBIAN/postrm";   chmod 0755 "$DEB/DEBIAN/postrm"
dpkg-deb --build --root-owner-group "$DEB" "$DIST/$DEB_FILE"

# 5) Build the .rpm
RPMTOP="$DIST/rpmbuild"
mkdir -p "$RPMTOP/BUILD" "$RPMTOP/RPMS" "$RPMTOP/SOURCES" "$RPMTOP/SPECS" "$RPMTOP/SRPMS"
cp -a "$ROOTFS" "$RPMTOP/SOURCES/root"
cat > "$RPMTOP/SPECS/libre-dt-lab.spec" <<EOF
Name:           libre-dt-lab
Version:        ${RPM_VERSION}
Release:        1
Summary:        Libre DT-Lab - virtual lighttable and darkroom (portable build)
License:        GPL-3.0-or-later
Group:          Applications/Graphics
URL:            https://github.com/Christian-Bouhon/libre-dt-lab
AutoReqProv:    no

%description
Libre DT-Lab is an experimental fork of darktable. This self-contained build
bundles the application and its libraries and installs them under
/opt/libre-dt-lab, with a launcher and a menu entry.

%install
mkdir -p %{buildroot}
cp -a %{_sourcedir}/root/. %{buildroot}/

%post
if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database -q /usr/share/applications >/dev/null 2>&1 || :; fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor >/dev/null 2>&1 || :; fi

%postun
if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database -q /usr/share/applications >/dev/null 2>&1 || :; fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor >/dev/null 2>&1 || :; fi

%files
/opt/libre-dt-lab
/usr/bin/libre-dt-lab
/usr/share/applications/org.libredtlab.libredtlab.desktop
/usr/share/icons/hicolor

%changelog
EOF
rpmbuild -bb --define "_topdir $RPMTOP" "$RPMTOP/SPECS/libre-dt-lab.spec"
find "$RPMTOP/RPMS" -name '*.rpm' -exec cp {} "$DIST/" \;

echo ""
echo "Built packages:"
ls -1 "$DIST"/*.deb "$DIST"/*.rpm

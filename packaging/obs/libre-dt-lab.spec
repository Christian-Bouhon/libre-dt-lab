#
# spec file for package libre-dt-lab
#
# Copyright (c) 2026 Libre DT-Lab contributors
#
# Based on the openSUSE darktable spec. All modifications and additions
# remain under the license of the pristine package unless stated otherwise.
#
# Please submit bugfixes or comments at
# https://github.com/Christian-Bouhon/libre-dt-lab/issues
#

%global desktop_filename org.libredtlab.libredtlab

# use the system lua interpreter library on suse/fedora, else the intree copy
%if 0%{?suse_version} || 0%{?fedora}
%global _dont_use_intree_lua ON
%else
%global _dont_use_intree_lua OFF
%endif

# use the system LibRaw on suse/fedora (the intree copy is always shipped)
%if 0%{?suse_version} || 0%{?fedora}
%bcond_without libraw
%global _use_system_libraw ON
%else
%bcond_with    libraw
%global _use_system_libraw OFF
%endif

%bcond_without openmp
%bcond_without opencl
%bcond_without gmic
%bcond_without avif
%bcond_without jxl
%bcond_without libheif
%bcond_without osmgpsmap
%bcond_without flickcurl
%bcond_without translated_manpages

# The OpenCL kernels don't compile/run reliably on ppc64le
%ifarch ppc64le
%bcond_with opencl
%endif

%if %{with openmp}
%global _use_openmp ON
%else
%global _use_openmp OFF
%endif

%if %{with opencl}
%global _use_opencl ON
%else
%global _use_opencl OFF
%endif

%if %{with gmic}
%global _use_gmic ON
%else
%global _use_gmic OFF
%endif

%if %{with avif}
%global _use_avif ON
%else
%global _use_avif OFF
%endif

%if %{with jxl}
%global _use_jxl ON
%else
%global _use_jxl OFF
%endif

%if %{with libheif}
%global _use_libheif ON
%else
%global _use_libheif OFF
%endif

%if %{with osmgpsmap}
%global _use_map ON
%else
%global _use_map OFF
%endif

%if %{with libraw}
%global _use_libraw ON
%else
%global _use_libraw OFF
%endif

Name:           libre-dt-lab
Version:        0
Release:        0
Summary:        A virtual Lighttable and Darkroom
License:        GPL-3.0-or-later
Group:          Productivity/Graphics/Viewers
URL:            https://github.com/Christian-Bouhon/libre-dt-lab
Source0:        %{name}-%{version}.tar.xz
Source2:        %{name}-rpmlintrc
ExclusiveArch:  x86_64 aarch64 ppc64le riscv64

# build tools
BuildRequires:  cmake >= 3.18
BuildRequires:  gcc-c++
%if 0%{?fedora}
BuildRequires:  ninja-build
%else
BuildRequires:  ninja
%endif
BuildRequires:  fdupes
BuildRequires:  intltool
BuildRequires:  libxslt
BuildRequires:  perl
%if 0%{?fedora}
BuildRequires:  perl-FindBin
%endif
%if %{with translated_manpages}
BuildRequires:  po4a
%endif
%if 0%{?suse_version}
BuildRequires:  update-desktop-files
%endif
BuildRequires:  desktop-file-utils
BuildRequires:  hicolor-icon-theme
BuildRequires:  xz
BuildRequires:  pkgconfig

# library dependencies
BuildRequires:  pkgconfig(gtk+-3.0) >= 3.24.15
BuildRequires:  pkgconfig(glib-2.0)
BuildRequires:  pkgconfig(gio-2.0)
BuildRequires:  pkgconfig(gdk-pixbuf-2.0)
BuildRequires:  pkgconfig(pango)
BuildRequires:  pkgconfig(atk)
BuildRequires:  pkgconfig(libxml-2.0)
BuildRequires:  pkgconfig(libtiff-4)
BuildRequires:  pkgconfig(libjpeg)
BuildRequires:  pkgconfig(libpng)
BuildRequires:  pkgconfig(zlib)
BuildRequires:  pkgconfig(lcms2)
BuildRequires:  pkgconfig(lensfun) >= 0.3.2
BuildRequires:  pkgconfig(libcurl)
BuildRequires:  pkgconfig(exiv2)
BuildRequires:  pkgconfig(pugixml)
BuildRequires:  pkgconfig(sqlite3)
BuildRequires:  pkgconfig(json-glib-1.0)
BuildRequires:  pkgconfig(librsvg-2.0)
BuildRequires:  pkgconfig(libsecret-1)
BuildRequires:  pkgconfig(libwebp)
BuildRequires:  pkgconfig(libopenjp2)
BuildRequires:  pkgconfig(sdl2)
BuildRequires:  pkgconfig(tinfo)
BuildRequires:  pkgconfig(iso-codes)
BuildRequires:  pkgconfig(libgphoto2)
BuildRequires:  pkgconfig(cups)
BuildRequires:  pkgconfig(Imath)
BuildRequires:  pkgconfig(OpenEXR)
BuildRequires:  pkgconfig(GraphicsMagick)
BuildRequires:  pkgconfig(colord)
BuildRequires:  pkgconfig(colord-gtk)
BuildRequires:  pkgconfig(icu-uc)
BuildRequires:  portmidi-devel
BuildRequires:  potrace-devel
BuildRequires:  libarchive-devel
%if 0%{?suse_version} >= 1550
BuildRequires:  pkgconfig(lua5.4)
%else
BuildRequires:  pkgconfig(lua)
%endif
%if %{with flickcurl}
BuildRequires:  pkgconfig(flickcurl)
%endif
%if %{with libraw}
BuildRequires:  pkgconfig(libraw) >= 0.21
%endif
%if %{with libheif}
BuildRequires:  pkgconfig(libheif)
%endif
%if %{with jxl}
BuildRequires:  pkgconfig(libjxl) >= 0.7.0
%endif
%if %{with avif}
BuildRequires:  pkgconfig(libavif) >= 0.9.0
%endif
%if %{with osmgpsmap}
BuildRequires:  pkgconfig(osmgpsmap-1.0)
%endif
%if %{with gmic}
%if 0%{?is_opensuse}
BuildRequires:  libgmic-devel
%else
BuildRequires:  gmic-devel
%endif
%endif

Requires:       iso-codes
%if 0%{?fedora}
Recommends:     roboto-fontface-fonts
%else
Recommends:     google-roboto-fonts
%endif

%description
Libre DT-Lab is an experimental fork of darktable, a virtual lighttable and
darkroom for photographers. It manages digital negatives in a database and can
show them through a zoomable lighttable. It also enables developing raw images
and enhancing them.

%package tools-basecurve
Summary:        The basecurve tool from tools/basecurve/
Group:          Productivity/Graphics/Viewers
Requires:       GraphicsMagick
Requires:       dcraw
Requires:       exiftool

%description tools-basecurve
Libre DT-Lab is an experimental fork of darktable, a virtual lighttable and
darkroom for photographers.

This package provides the basecurve tool from tools/basecurve/.

%package tools-noise
Summary:        Noise profiling tools to support new cameras
Group:          Productivity/Graphics/Viewers
Requires:       GraphicsMagick
Requires:       ghostscript
Requires:       gnuplot

%description tools-noise
Libre DT-Lab is an experimental fork of darktable, a virtual lighttable and
darkroom for photographers.

This package provides the noise profiling tools to add support for new cameras.

%prep
%autosetup -n %{name}-%{version}

# The bundled OpenCL headers (src/external/OpenCL) are used as-is; the intree
# lua and LibRaw copies are simply ignored when the system versions are used.

%build
%cmake \
  -DCMAKE_INSTALL_LIBDIR=%{_lib} \
  -DCMAKE_INSTALL_DATAROOTDIR=share \
  -DCMAKE_INSTALL_LIBEXECDIR=%{_libexecdir} \
  -DCMAKE_INSTALL_DOCDIR=%{_defaultdocdir}/%{name} \
  -DCOMPILER_SUPPORTS_SPLIT_DEBUG_INFO=OFF \
  -DBINARY_PACKAGE_BUILD=1 \
  -DRAWSPEED_ENABLE_LTO=ON \
  -DDONT_USE_INTERNAL_LUA=%{_dont_use_intree_lua} \
  -DDONT_USE_INTERNAL_LIBRAW=%{_use_system_libraw} \
  -DUSE_OPENCL=%{_use_opencl} \
  -DUSE_OPENMP=%{_use_openmp} \
  -DUSE_GMIC=%{_use_gmic} \
  -DUSE_AVIF=%{_use_avif} \
  -DUSE_JXL=%{_use_jxl} \
  -DUSE_HEIF=%{_use_libheif} \
  -DUSE_LIBRAW=%{_use_libraw} \
  -DUSE_MAP=%{_use_map} \
  -DBUILD_NOISE_TOOLS=ON \
  -DBUILD_CURVE_TOOLS=ON
%cmake_build

%install
%cmake_install
%find_lang %{name}

%if 0%{?suse_version}
%suse_update_desktop_file %{desktop_filename}
%endif

%fdupes %{buildroot}/%{_prefix}

%if ! 0%{?suse_version}
%post
touch --no-create %{_datadir}/icons/hicolor >/dev/null 2>/dev/null || :

%postun
update-desktop-database >/dev/null 2>/dev/null || :
if [ $1 -eq 0 ] ; then
    touch --no-create %{_datadir}/icons/hicolor >/dev/null 2>/dev/null
    gtk-update-icon-cache %{_datadir}/icons/hicolor >/dev/null 2>/dev/null || :
fi

%posttrans
gtk-update-icon-cache %{_datadir}/icons/hicolor >/dev/null 2>/dev/null || :
%endif

%files -f %{name}.lang
%license %{_defaultdocdir}/%{name}/LICENSE
%doc %{_defaultdocdir}/%{name}/AUTHORS
%exclude %{_defaultdocdir}/%{name}/README.tools.basecurve.md
%{_bindir}/libre-dt-lab
%{_bindir}/libre-dt-lab-cli
%{_bindir}/libre-dt-lab-chart
%{_bindir}/libre-dt-lab-cmstest
%{_bindir}/libre-dt-lab-generate-cache
%{_bindir}/libre-dt-lab-rs-identify
%if %{with opencl}
%{_bindir}/libre-dt-lab-cltest
%endif
%{_libdir}/libre-dt-lab
%{_datadir}/applications/%{desktop_filename}.desktop
%{_datadir}/libre-dt-lab
%dir %{_datadir}/metainfo
%{_datadir}/metainfo/%{desktop_filename}.appdata.xml
%{_datadir}/icons/hicolor/*/apps/libre-dt-lab.*
%{_datadir}/bash-completion/completions/libre-dt-lab
%{_datadir}/bash-completion/completions/libre-dt-lab-*
%{_mandir}/man1/libre-dt-lab*.1*
%if %{with translated_manpages}
%{_mandir}/*/man1/libre-dt-lab*.1*
%endif

%files tools-basecurve
%{_libexecdir}/libre-dt-lab/tools/libre-dt-lab-curve-tool
%{_libexecdir}/libre-dt-lab/tools/darktable-curve-tool-helper
%{_datadir}/libre-dt-lab/tools/basecurve/
%doc %{_defaultdocdir}/%{name}/README.tools.basecurve.md

%files tools-noise
%{_libexecdir}/libre-dt-lab/tools/libre-dt-lab-noiseprofile
%{_libexecdir}/libre-dt-lab/tools/darktable-gen-noiseprofile
%{_libexecdir}/libre-dt-lab/tools/profiling-shot.xmp
%{_libexecdir}/libre-dt-lab/tools/subr.sh

%changelog

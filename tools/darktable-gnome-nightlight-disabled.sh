#!/bin/bash
# command line: put this file in path before Libre DT-Lab as: /usr/local/bin/libre-dt-lab 
# desktop icon: edit /usr/share/applications/org.libredtlab.libredtlab.desktop: Exec and TryExec pointing to /usr/local/bin/libre-dt-lab  
# (ubuntu18.04, dt from git)
#

[ "${FLOCKER}" != "$0" ] && exec env FLOCKER="$0" flock -en "$0" "$0" "$@" || :

gnomenightlight="org.gnome.settings-daemon.plugins.color night-light-enabled"

trap "gsettings set ${gnomenightlight} $(gsettings get ${gnomenightlight})" EXIT

gsettings set ${gnomenightlight} false

/opt/libre-dt-lab/bin/libre-dt-lab "$@"

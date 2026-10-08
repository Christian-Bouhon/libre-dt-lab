#!/bin/sh

set -e;

input="$1"
authors="$2"
output="$3"
version="$4"
mandate="$5"

r=$(sed -n 's,.*\$Release: \(.*\)\$$,\1,p' "$input")
if [ -n "$version" ]; then
  r="$version"
fi
d=$(sed -n 's,/,-,g;s,.*\$Date: \(..........\).*,\1,p' "$input")
if [ -n "$mandate" ]; then
  d="$mandate"
fi
D=""
if [ -n "$d" ]; then
  D="--date=$d"
fi

# derive the man page name from the output file name (libre-dt-lab-cli.1 -> LIBRE-DT-LAB-CLI)
name=$(basename "$output" .1 | tr '[:lower:]' '[:upper:]')

# pass the optional --date argument without ever passing an empty argument
if [ -n "$D" ]; then
  set -- "$D"
else
  set --
fi

pod2man --utf8 --name="$name" --release="libre-dt-lab $r" --center="libre-dt-lab" "$@" "$input" \
  | sed -e '/.*DREGGNAUTHORS.*/r '"$authors" | sed -e '/.*DREGGNAUTHORS.*/d' \
  > "$output" || rm "$output"

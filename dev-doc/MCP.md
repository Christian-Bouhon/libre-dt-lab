# MCP server (libre-dt-lab-mcp)

A headless [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes the raw-development engine to AI agents over stdio (JSON-RPC 2.0).

It is a sibling binary to `libre-dt-lab-cli`, linking `lib_darktable` directly,
so it gets real module introspection, the in-process pixelpipe, and native
styles/history. It does not drive a live GUI.

The upstream implementation and the full tool reference live in
[`src/mcp/README.md`](../src/mcp/README.md). This page is the fork-specific,
user-facing setup guide.

## Build

`USE_MCP` is enabled by default. Disable it with `-DUSE_MCP=OFF` or
`./build.sh --disable-mcp`.

```sh
./build.sh
# or
cmake -B build -DUSE_MCP=ON
cmake --build build --target libre-dt-lab-mcp -j
```

The binary is written to `build/bin/libre-dt-lab-mcp`. After installation it is
`/opt/libre-dt-lab/bin/libre-dt-lab-mcp` or the equivalent `$BINDIR`.

`mcp_tools.json` is installed to `<datadir>/libre-dt-lab/mcp_tools.json`.
It contains the tool descriptions and schemas and can be edited without a
rebuild.

## Why a dedicated config directory?

MCP clients launch the server themselves. Give the server its own darktable
config and cache directory so it never locks the library the GUI is using:

```sh
mkdir -p ~/.config/libre-dt-lab-mcp ~/.cache/libre-dt-lab-mcp
```

The server uses `--configdir` to select its own `library.db`, as everywhere else
in darktable. With neither `--library` nor `--configdir`, it injects
`--library :memory:` and `--conf write_sidecar_files=never`, so loose files are
imported ad hoc and nothing persists.

## Quick start: Claude Code

```sh
claude mcp add libre-dt-lab -- \
  /path/to/build/bin/libre-dt-lab-mcp \
  --core --configdir ~/.config/libre-dt-lab-mcp \
         --cachedir  ~/.cache/libre-dt-lab-mcp
```

Use `-s user` to make it available in all projects, or `-s project` to write a
shared `.mcp.json`. Check with `claude mcp list` or `/mcp`.

The helper script prints the same command and the Claude Desktop JSON for a
given binary:

```sh
tools/mcp/print-client-config.sh /path/to/build/bin/libre-dt-lab-mcp
```

## Quick start: Claude Desktop

Add this to `claude_desktop_config.json` (Settings -> Developer -> Edit Config)
and restart the app:

```json
{
  "mcpServers": {
    "libre-dt-lab": {
      "command": "/path/to/build/bin/libre-dt-lab-mcp",
      "args": [
        "--core",
        "--configdir", "/home/you/.config/libre-dt-lab-mcp",
        "--cachedir",  "/home/you/.cache/libre-dt-lab-mcp"
      ]
    }
  }
}
```

To work on a real catalog, replace the config/cache arguments with
`--core --library /path/to/library.db`. The GUI must not be holding that
library.

On Windows, use `libre-dt-lab-mcp.exe` and a dedicated directory such as
`%LOCALAPPDATA%\libre-dt-lab-mcp`; the arguments are otherwise identical.

## Read-only mode

Renders carrying a `stack` write the image history, and the library tools can
rate, label, import, or reset images. Use `--read-only` before `--core` to
refuse every write while keeping reads and plain renders:

```sh
libre-dt-lab-mcp --read-only --core --library /path/to/library.db
```

`--read-only` is the server's own flag and must come before `--core`.
Everything after `--core` is forwarded verbatim to darktable's `dt_init`, so
`--configdir`, `--cachedir`, `--library`, `--conf`, `-d`, etc. work as usual.

## What it can do

- **Introspection**: list modules, read schemas, decode/encode `op_params`.
- **Develop**: `render` (base64 PNG), `image_stats` (min/max/mean/percentiles).
- **Library**: import, list film rolls/images, metadata, history, rating,
  colour labels, styles, export.
- **Export**: darktable's own format modules (jpeg, png, tiff, webp, jxl, avif,
  exr, ...).

See [`src/mcp/README.md`](../src/mcp/README.md) for the full list and the
behavior of `imgid` vs `path`, `stack`, `history_end`, and `--read-only`.

## Examples

Render a raw and show it:

> Using libre-dt-lab, render this raw and show it.

Measure a module parameter:

> Measure image_stats with agx target_black at 0.008 vs 0.0008.

## Troubleshooting

- **`library.db` is locked**: close the GUI for that library, or use a
  dedicated `--configdir` for the server.
- **No tools available**: check `mcp_tools.json` is installed next to the
  fork's data files (`<datadir>/libre-dt-lab/mcp_tools.json`).
- **ONNX / AI tools fail**: the server uses the same AI configuration as the
  app. Install/enable a model first, or run without AI-dependent tools.
- **stdout is not JSON**: the server redirects darktable logging to stderr;
  only JSON-RPC is written to stdout. Run with `-d` after `--core` if you need
  logs.
- **First render is slow**: demosaic + the full pipe can take a few seconds.

## Security and privacy

- The server is a local stdio process; it does not open a network port.
- It runs with the same permissions as your user and can modify any library you
  point it at, unless `--read-only` is used.
- Prefer a dedicated config directory or `--read-only` when an agent is
  exploring.
- AI model inference stays local; MCP itself sends nothing to a cloud service.

## Maintenance

This is a fork-specific integration of the upstream MCP work. The server code
was ported from upstream commits `e07bcd25bb`, `07a85d8890`, `1335549059` and
`f82ef243a4`, with the small allocator fix from `87274529db`.

Two local adaptations are required:

- `dt_variables_expand_path()` is backported into `src/common/variables.c/.h`
  (it was added upstream after this fork's base).
- The target/output binary and the data install path use the fork branding:
  `libre-dt-lab-mcp` and `<datadir>/libre-dt-lab/mcp_tools.json`.

Because this fork does not track upstream, keep these adaptations in mind if
the MCP code is ever refreshed from a newer upstream release.

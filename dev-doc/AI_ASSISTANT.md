# AI assistant (darkroom)

A native, photographer-friendly assistant that proposes **slider adjustments**
for the image currently open in the darkroom. It is not an MCP client and needs
no command line or external agent.

The assistant can adjust modules that are already active, and can **activate
allowed modules** that are available but inactive. The photographer reviews the
proposal and applies or ignores it.

## Privacy

By default, **no image is sent**. The model receives text and numbers only:

- camera maker/model, lens, ISO, shutter, aperture, focal length, dimensions;
- current slider values for active modules and default values for available
  modules;
- local image statistics: luminance percentiles, clipping, contrast spread;
- the user's request.

It does **not** receive file names, GPS coordinates, serial numbers or the
original file.

An optional **preview image** can be enabled in the assistant: a 768 px JPEG of
the **current darkroom render**, generated in memory, with no metadata. It is
opt-in and off by default. All API providers attempt it only when enabled; the
selected model must support vision. If preview capture fails or the provider
rejects the image, the assistant continues text-only and logs the fallback.

Ollama runs locally by default. Use a local Ollama model for a fully offline
setup.

## Setup

Preferences -> **Assistant**.

### Ollama (local, recommended)

1. Install Ollama and pull a model. `llama3.1:8b` is a reliable baseline;
   `qwen2.5:7b` and `qwen2.5-coder:7b` are worth testing for strict JSON.
   Avoid very large models on a 22 GB machine.
2. In the Assistant preferences:
   - provider: **Ollama (local)**
   - endpoint: `http://localhost:11434/api/chat`
   - model: click **detect** and choose an installed local model.

No API key is needed. The model field is a dropdown with a free-text entry.
Ollama cloud models (names ending in `-cloud` or `:cloud`) are opt-in:
enable **include Ollama cloud models**, then click **detect**. Text is then sent
to Ollama cloud. If the optional preview image is enabled and the selected
Ollama model supports vision, it is sent as well. Ollama cloud usually requires
signing in with Ollama outside darktable.

### Optional preview image

A persisted checkbox **send preview image** is available in both the darkroom
assistant panel and Preferences -> Assistant. When enabled:

- a 768 px JPEG of the current darkroom render is generated in memory;
- no file is written, and no metadata, filename or GPS is included;
- all provider adapters can send it; the selected model must support vision;
- the text request and local statistics are still sent;
- if the model does not support images, or capture/upload fails, the assistant
  falls back to text-only and logs it.

The setting applies to all following photos until unchecked.

### API providers (cloud, optional)

Provider list:
- **DeepSeek**, **OpenAI**, **OpenRouter**, **Groq**, **Mistral**, **xAI Grok**,
  **Together AI**, **Fireworks AI**, **Perplexity**;
- **Google Gemini** (native API);
- **Anthropic Claude** (native Messages API + JSON tool use);
- **Local server** (LM Studio, vLLM, llama.cpp server);
- **Custom API**.

For each provider:
1. choose the provider;
2. the endpoint and default model are starting points;
3. paste the API key and click **store**;
4. click **detect** to list the models currently available for that key;
5. choose a model, then click **verify configuration**.

Each provider has its own endpoint, model and key. Switching provider no longer
overwrites another provider's settings.

API keys are stored only in the system secure store (GNOME Keyring/libsecret,
KWallet, macOS Keychain or Windows Credential Manager), one slot per provider.
There is no plaintext fallback. If no keyring backend is available or unlocked,
the key is kept for the current session only and must be re-entered after a
restart. The Assistant tab shows the storage state. Configure the backend in
**Preferences -> Security -> password storage backend** if needed.

Presets are not authoritative: provider model lists change often, so use
**detect**. If the current model is absent from the provider list, the assistant
keeps it rather than silently replacing it.

Cloud endpoints must use HTTPS, except localhost. Only text leaves the machine.

These are **API providers** for model inference, not MCP servers. The separate
`libre-dt-lab-mcp` binary exposes darktable tools to MCP clients such as Claude
Desktop/Code; it does not replace the provider layer.

### Assistant modules

The tab contains a **assistant modules** list:

- the assistant can only use modules in this list;
- the list is prefilled with the v1 modules;
- use **add** / **remove** to choose modules, and **reset** to restore the
  default list;
- an allowed module may be active or inactive; if inactive, the assistant can
  activate it.

The list is stored in `plugins/assistant/modules`.

A **tone & texture preference** chooses which module the assistant favours for
local tonal work:

- **Tone & Texture** (default): `contrast` is preferred when allowed;
- **Tone Equalizer**: `toneequal` is preferred when allowed;
- **Automatic**: the model chooses, preferring the module already active.

The assistant never changes the tone equalizer's internal mask parameters
(`blending`, `feathering`, `contrast_boost`, `exposure_boost`, `iterations`).
When it activates Tone Equalizer, it reloads the module defaults and only moves
the nine tonal-band sliders, so the fork's default luminance mask is preserved.
When the module is already active, its current mask is left untouched.

Other role preferences are available in the same tab:

- **Tone mapper**: **default** (follows `plugins/darkroom/workflow`), then
  `3DCF`, `ACES 2.0`, `AgX`, `Filmic RGB`, `Sigmoid`.
- **Color**: **automatic** (default), `Color Balance RGB`, `Color Harmonizer`,
  `RGB Primaries`. This is a soft preference: other color modules may still be
  used when the request needs them.
- **Detail**: **automatic** (default), `Diffuse or Sharpen`, `Sharpen`. This is
  also a soft preference.

A role preference never overrides the allowed-module list. If the selected
module is not allowed, the assistant ignores the preference.

For global exposure corrections, the assistant prefers the **Exposure** module
(`exposure.exposure`) rather than a tone mapper's exposure control. A tone
mapper's exposure is reserved for small refinements of the tone mapping.

The **exposure bias (EV)** preference (default `0.0`, range `-2.0` to `+2.0`,
step `0.1`) is added to the assistant's exposure suggestion. It is additive: a
base suggestion of `-0.5 EV` with a `+0.7 EV` bias becomes `+0.2 EV`, not
`+0.7 EV`. A positive bias is skipped when highlights are clipping
(`clip_high_pct > 0.5` or `p99 > 0.995`); a negative bias is still applied. The
value is included in `suggested_exposure_ev`, so the model is told not to add it
twice. If the model omits an exposure change even though the preference applies,
the assistant adds one `exposure.exposure` change based on
`suggested_exposure_ev`; it still appears in the plan and can be unticked. No
change is written to the history until **Apply**.

## Using it

Open an image in the darkroom, then open the **AI assistant** panel on the right.

- **Analyze**: sends metadata, current/default slider values and local image
  statistics, then shows a proposal. If the model proposes an invalid or
  out-of-range value, the assistant automatically retries once with the error
  and the allowed ranges.
- Review the table: module, slider, current value, proposed value, reason.
- A module that will be activated is marked `· enable`.
- Untick any change you do not want.
- **Apply**: writes the selected changes as normal darktable history entries.
  Inactive allowed modules are activated.
- **Revert**: restores a temporary history snapshot taken just before Apply.
  Slider values, activation state, module order and masks return to the
  pre-apply state, and the assistant history entries are removed so the history
  is not lengthened. Edits made manually after Apply are also discarded.
- **Stop**: cancels a request in progress.
- **Settings**: opens the Assistant preferences tab.

The optional chat field lets the photographer add intent, for example:

> moins d'accentuation et de contraste, style portrait

The assistant still replies with a slider plan; nothing is applied until the
photographer clicks **Apply**.

## Allowed modules

The list is controlled by Preferences -> Assistant. The v1 set is:

| Module | Sliders |
|---|---|
| `exposure` | `exposure`, `black` |
| `toneequal` | the nine tonal bands (`noise` to `speculars`) |
| `agx` | `look_brightness`, `look_saturation`, `curve_contrast_around_pivot` |
| `filmicrgb` | `contrast`, `saturation`, `balance` |
| `sigmoid` | `middle_grey_contrast`, `contrast_skewness`, `display_white_target`, `display_black_target`, `input_exposure` |
| `3dcf` | `contrast`, `gamma`, `vibrance`, `input_exposure`, `contrast_pivot`, `toe_power`, `shoulder_power`, `hl_desaturation`, `hl_detail_recovery` |
| `aces20` | `exposure_ev`, `peak_luminance` |
| `contrast` | `global_scale`, `micro/fine/local/broad/coarse_scale`, `gain_highlights`, `gain_shadows`, `colorful_contrast`, `contrast_balance`, `color_balance` |
| `colorbalancergb` | `contrast`, `vibrance`, `saturation_global` |
| `colorharmonizer` | `anchor_hue`, `pull_strength`, `neutral_protection`, `pull_width`, `smoothing` |
| `primaries` | RGB hue/purity and achromatic tint sliders |
| `sharpen` | `amount`, `threshold` |
| `diffuse` | `sharpness` |

Each parameter has a maximum allowed change per proposal. A model proposal
outside that delta (or outside the module range) is clamped to the nearest safe
value, marked `(safety-limited)` in its reason, and logged; the rest of the plan
is kept. Proposals are capped at five changes and at most two newly activated
modules.

### Tone mapper

Only one tone mapper may be active. The assistant uses:

1. the tone mapper already active in the history, if any;
2. otherwise the explicit **tone mapper preference**, when one is selected and
   allowed;
3. otherwise the one selected by `plugins/darkroom/workflow` (the **default**
   choice, currently `3DCF`).

It never stacks a second tone mapper, and it does not switch an already active
tone mapper. The darktable workflow preference itself is not changed.

## Local image statistics

The assistant exports the current image at 512 px **in memory**, through the user
pipeline, and computes:

- luminance percentiles p1, p5, p25, p50, p75, p95, p99;
- low/high clipping percentages, globally and per channel;
- mean luminance and contrast spread;
- mean RGB, saturation mean/percentiles and warm/cool color cast;
- deterministic hints (exposure, shadows, highlights, contrast, muted colors,
  color separation, color cast);
- an `suggested_exposure_ev` that includes the configured exposure bias when
  highlights allow.

No file is written. The image is sent only when the optional preview checkbox
is enabled and the provider supports vision. The hints help the model make a
first-pass proposal instead of guessing from metadata alone.

## Styles and looks

Named looks (Portrait, Landscape, ...) are **not implemented yet**. They will be
slider recipes, not darktable `.dtstyle` presets. A look will only change the
same allowed sliders, never the history stack, masks, or module order.

## Troubleshooting

- **"Assistant not configured"**: open Preferences -> Assistant and fill in the
  endpoint/model (and key for a cloud provider).
- **Ollama connection error**: check that Ollama is running. In Preferences ->
  Assistant, click **detect** to refresh the list of local models.
- **Configured model is not installed**: click **detect**; the assistant will
  select an installed model and tell you which one was missing.
- **No local model found**: pull a model with Ollama, then click **detect**
  again. To list Ollama cloud models, enable **include Ollama cloud models**.
- **Cloud model unavailable or retired**: click **detect** and choose a model
  currently offered by the provider. Static defaults are only starting points.
- **"module is not allowed in preferences"**: add the module in Preferences ->
  Assistant.
- **"`module.param` is not an allowed slider"**: the model invented a slider
  that is not in the catalog. The assistant ignores that change, keeps the valid
  ones, and logs it. If no valid change remains, it retries once.
- **"another tone mapper is already active"**: the plan tried to activate a
  second tone mapper. Use the active one or adjust it manually.
- **"preferred tone mapper is ..."**: no tone mapper is active, and the plan
  tried a different one than the selected tone mapper preference (or the
  `plugins/darkroom/workflow` default). Use the preferred one, or choose
  **default** in the tone mapper preference.
- **"proposed X is outside [min, max]" / "change is too large"**: the proposal
  exceeded a safety limit. It is normally clamped to the limit, marked
  `safety-limited`, and the rest of the plan is kept. If clamping cannot produce
  a valid change, the assistant retries once; for a tone mapper's exposure
  control it suggests `exposure.exposure` for a larger global correction.
- **HTTP 401/403**: the provider rejected the key. Check that the key belongs to
  the selected provider/endpoint and that it is the secret key, not a key ID or
  label. Use **verify configuration**; no key is ever logged.
- **Preview image unavailable or rejected**: the assistant automatically retries
  text-only and logs the reason. Check the provider/model supports images, or
  disable **send preview image**.
- **Key not remembered after restart**: the secure storage backend is
  unavailable or locked; with secure-only storage the key is session-only.
  Configure a keyring backend (Preferences -> Security) or re-enter the key.
- **Provider model rejected**: click **detect**; provider names and defaults
  change frequently.
- **Cloud endpoint rejected**: use HTTPS, or a localhost address.

## Implementation

- `src/libs/ai_assistant.c` - darkroom panel, chat, apply/revert UI.
- `src/common/assistant/assistant_provider.c` - text-only HTTP providers.
- `src/common/assistant/assistant_catalog.c` - module catalog loading.
- `src/common/assistant/assistant_context.c` - metadata, modules, tone mapper.
- `src/common/assistant/assistant_actions.c` - allowlist, activation,
  validation, apply/revert.
- `src/common/assistant/assistant_stats.c` - local 512 px statistics and hints.
- `src/common/assistant/assistant_image.c` - optional in-memory JPEG preview.
- `src/gui/preferences_assistant.c` - Assistant preferences tab and module list.
- `data/ai_assistant_modules.json` - editable module catalog.

The provider layer supports Ollama, standard chat APIs, Gemini native and
Anthropic native APIs. Provider definitions, per-provider settings and secure
key slots live in `assistant_provider.c/.h`. Model detection uses each API's
native model-list endpoint.

## Extending the module catalog

`data/ai_assistant_modules.json` is installed to
`<datadir>/libre-dt-lab/ai_assistant_modules.json`. Add a module entry with its
operation, description, role and allowed params. The assistant validates every
parameter against the runtime introspection, so the catalog cannot apply
unknown fields. The Preferences list is where the user chooses which catalog
modules the assistant may use.

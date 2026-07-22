---
name: write-darktable-iop
description: Generate a new darktable image-processing module (IOP) from scratch. Use when the user asks to create a new module, add a processing feature, or implement a pixelpipe filter. Covers file creation, registration, pipeline ordering, parameter introspection, GUI, CPU and OpenCL processing, tiling, and colour management.
---

# Darktable IOP Module Writer

## Module Naming

- **File name**: `src/iop/<shortname>.c` — short, lowercase, no hyphens (use underscore if needed)
- **Operation name**: returned by `name()` — user-visible, translatable via `_(...)`, ≤ 19 characters (enforced by CMake macro)
- **Order name**: `"<shortname>"` — used in `iop_order.c` and the introspection system

## Required Boilerplate

### Includes

```c
#include "bauhaus/bauhaus.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
```

### Introspection macro (required, near top of file)

```c
DT_MODULE_INTROSPECTION(1, dt_iop_<shortname>_params_t)
```

`1` = version number; bump when `params_t` layout changes.

## Structs to Define

### `dt_iop_<shortname>_params_t` — serialized parameters

- **No pointers** — this struct is persisted as a raw binary blob in the database and XMP
- Use `gboolean` not `bool` (4-byte aligned)
- Use standard C types: `int`, `float`, `gboolean`, `enum` (4 bytes)
- Add introspection tags as comments:
  ```c
  typedef struct dt_iop_<shortname>_params_t
  {
    float factor;     // $MIN: -5.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "scale factor"
    int method;       // $MIN: 0 $MAX: 2 $DEFAULT: 0 $DESCRIPTION: "method"
    gboolean invert;  // $DESCRIPTION: "invert"
    float padding;    // fill to multiple of 4 bytes if needed
  } dt_iop_<shortname>_params_t;
  ```

### `dt_iop_<shortname>_gui_data_t` — GUI widget references

```c
typedef struct dt_iop_<shortname>_gui_data_t
{
  GtkWidget *factor_slider;
  GtkWidget *method_combo;
  GtkWidget *invert_check;
  // … any custom GtkWidgets
} dt_iop_<shortname>_gui_data_t;
```

### `dt_iop_<shortname>_data_t` — per-pipe processing data (optional)

Populated in `commit_params()`, used in `process()`. Store precomputed coefficients here:

```c
typedef struct dt_iop_<shortname>_data_t
{
  float factor;
  int method;
  gboolean invert;
  /* precomputed LUT, coefficients, etc. */
} dt_iop_<shortname>_data_t;
```

### `dt_iop_<shortname>_global_data_t` — process-wide shared data (optional)

Allocated in `init_global()`, used by all module instances:

```c
typedef struct dt_iop_<shortname>_global_data_t
{
  /* lookup tables, compiled kernels, etc. */
} dt_iop_<shortname>_global_data_t;
```

## REQUIRED Callbacks

### `const char *name(void)`

```c
const char *name(void)
{
  return _("shortname");
}
```

### `dt_iop_colorspace_type_t default_colorspace(...)`

Return one of: `IOP_CS_RAW`, `IOP_CS_LAB`, `IOP_CS_RGB`, `IOP_CS_LCH`, `IOP_CS_HSL`, `IOP_CS_JZCZHZ`.

Most creative modules work in `IOP_CS_RGB`. If the module processes raw data directly, use `IOP_CS_RAW`.

### `void process(...)` — the pixel processing function

```c
void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  dt_iop_<shortname>_data_t *d = piece->data;
  const size_t ch = piece->colors;

  if(!dt_iop_have_required_input_format(4, self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  DT_OMP_FOR()
  for(size_t j = 0; j < roi_out->height; j++)
  {
    const float *in = ((const float *)ivoid) + (size_t)ch * roi_in->width * j;
    float *out = ((float *)ovoid) + (size_t)ch * roi_out->width * j;
    for(size_t i = 0; i < roi_out->width; i++)
    {
      for_each_channel(c, aligned(in, out))
        out[c] = in[c] * d->factor;
      in += ch;
      out += ch;
    }
  }
}
```

**Key rules:**
- Never use GTK API here; this runs on worker threads
- Use flat indexing (`in[k * ch + c]`) or well-structured inner loops — avoid pointer carries
- Use `DT_OMP_FOR()` for OpenMP; add `collapse(...)` if nested loops
- Use `float *const restrict` on image pointers
- `dt_iop_have_required_input_format()` handles the common case of passthrough on mismatch
- For raster mask support, see the pattern in the boilerplate (`dt_iop_piece_is_raster_mask_used` / `dt_iop_alloc_image_buffers`)

## OPTIONAL Callbacks — When to Implement

### Lifecycle

| Callback | When needed |
|----------|-------------|
| `init()` | Set `self->params_size`, call `dt_iop_default_init(self)`. Override `self->default_enabled` if module should be off initially. |
| `cleanup()` | Free per-instance allocations. Call `dt_iop_default_cleanup(self)`. |
| `init_global()` | Allocate `self->data = calloc(...)` for cross-instance data (LUTs, compiled kernels). |
| `cleanup_global()` | Free `self->data`. |
| `reload_defaults()` | Override defaults per image class (e.g., different params for raw vs JPEG). |

### Parameter Management

| Callback | When needed |
|----------|-------------|
| `commit_params()` | Always, unless default memcpy is sufficient. Copy `self->params` into `piece->data` with precomputed coefficients. Use `dt_iop_<shortname>_data_t *d = piece->data;`. |
| `legacy_params()` | When `params_t` layout changes across versions. Chain migrations for every intermediate version. |

### Pixelpipe geometry

| Callback | When needed |
|----------|-------------|
| `modify_roi_in()` | Module needs extra border pixels for convolution, or changes input scale. |
| `modify_roi_out()` | Module changes output dimensions relative to input. |
| `tiling_callback()` | Module sets `IOP_FLAGS_ALLOW_TILING` — report memory needs to the tiling system. |

### Distortion

| Callback | When needed |
|----------|-------------|
| `distort_transform()` | Module warps coordinates (lens correction, perspective). |
| `distort_backtransform()` | Inverse of the above for the pipe's inverse transform. |
| `distort_mask()` | Distort raster masks to match the warp. |

### GUI callbacks

| Callback | When needed |
|----------|-------------|
| `gui_init()` | Always — build the module's widget tree. |
| `gui_cleanup()` | Always — free GUI allocations. |
| `gui_update()` | Push `self->params` to GUI widgets. |
| `gui_changed()` | Show/hide widgets based on param values, or update derived widgets. |
| `color_picker_apply()` | Module accepts color picker input. |

### OpenCL

| Callback | When needed |
|----------|-------------|
| `process_cl()` | GPU implementation. Return `DT_OPENCL_PROCESS_CL` on success, `DT_OPENCL_PROCESS_PENDING_CL` for tiled fallback. |
| `process_tiling_cl()` | Tiled GPU processing. |

## Other Required Functions

### `int flags(void)`

Common flag combinations:
```c
int flags(void)
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}
```

See the flags table in the reference for all options (`IOP_FLAGS_ALLOW_TILING`, `IOP_FLAGS_ONE_INSTANCE`, etc.).

### `int default_group(void)`

Return one or more groups combined with `|`:
```c
int default_group(void)
{
  return IOP_GROUP_COLOR | IOP_GROUP_GRADING;
}
```

Groups: `IOP_GROUP_BASIC`, `IOP_GROUP_TONE`, `IOP_GROUP_COLOR`, `IOP_GROUP_CORRECT`, `IOP_GROUP_EFFECT` (pre-3.4), `IOP_GROUP_TECHNICAL`, `IOP_GROUP_GRADING`, `IOP_GROUP_EFFECTS` (post-3.4).

### `const char **description(dt_iop_module_t *self)`

```c
const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
    _("what the module does"),
    _("corrective or creative"),
    _("linear, RGB, scene-referred"),
    _("linear, RGB"),
    _("linear, RGB, scene-referred"));
}
```

## GUI Patterns

### `gui_init()` — build the widget tree

Use `dt_bauhaus_*_from_params()` for introspection-driven widgets (auto-connected to params):

```c
void gui_init(dt_iop_module_t *self)
{
  dt_iop_<shortname>_gui_data_t *g = IOP_GUI_ALLOC(<shortname>);

  self->widget = dt_gui_vbox();

  // Slider with color picker:
  g->factor_slider = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                       dt_bauhaus_slider_from_params(self, N_("factor")));

  // Combobox from enum/int:
  g->method_combo = dt_bauhaus_combobox_from_params(self, "method");

  // Toggle button:
  g->invert_check = dt_bauhaus_toggle_from_params(self, "invert");

  // Manual slider (not introspection-driven, needs signal handler):
  g->extra_slider = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, 0.001, 0.0, 2);
  dt_bauhaus_widget_set_label(g->extra_slider, NULL, N_("extra adjustment"));
  g_signal_connect(G_OBJECT(g->extra_slider), "value-changed",
                   G_CALLBACK(extra_callback), self);
}
```

### Callback guard

Use `DT_GUARD_GUI_UPDATE()` at the start of any signal callback that modifies `self->params` from a GUI widget, to prevent feedback loops.

### History commit

After modifying `self->params` from a GUI callback, call:

```c
dt_dev_add_history_item(darktable.develop, self, TRUE);
```

## Coding Style

Follow the official developer's guide conventions (enforced by `tools/beautify_style.sh` via `clang-format`):

- **American English spelling**, especially for user-visible strings
- **Spaces, not tabs**; `shiftwidth=2`
- **No trailing whitespace**
- **Braces `{` and `}` on their own lines**
- **Line length ≤ 90 characters**
- **Function parameters each on their own line**
- **Boolean operators** (`||` / `&&`) at line start, one per line
- **SQL formatting** preserved as multi-line string literals
- **Comments**: use `//` for single-line, `/* */` only for multi-line blocks or documentation headers

## Writing Efficient Processing Code

Follow the official handbook rules:

1. **Index-based addressing only** — address depends on loop counters, not prior iteration
2. **Flat indexing** — `for(k = 0; k < ch * w * h; k += ch)` over nested height/width/ch loops
3. **Unpack structs** — copy struct members to local variables before the loop
4. **`const restrict`** — on all image pointers; `*out` must never alias `*in`
5. **Ternary over `if`** — `(x > 0) ? x : -x` compiles to SIMD masks
6. **Align data** — `DT_ALIGNED_PIXEL` on pixel arrays, 64-byte alignment on buffers
7. **`DT_OMP_FOR()`** — OpenMP parallelization; add `collapse(2)` for nested loops
8. **`for_each_channel`** macro — compiler hint for auto-vectorization across channels

## Registration

### `src/iop/CMakeLists.txt`

```cmake
add_iop(<shortname> "<shortname>.c")
# add DEFAULT_VISIBLE if the module should appear in the default UI panel
```

### `src/common/iop_order.c`

Insert an entry in all **five** order tables:

1. `legacy_order[]`
2. `v30_order[]`
3. `v30_jpg_order[]`
4. `v50_order[]` (current default for raw)
5. `v50_jpg_order[]` (current default for JPEG)

Use the `@@_NEW_MODULE` comment markers to find insertion areas:

```c
{ {XX.0f }, "<shortname>", 0},
```

Pick a position value that places the module semantically. Modules at the same value share a position; use fractional values (e.g., `45.3`) to insert between existing modules.

## OpenCL Path

### `process_cl()` pattern

```c
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_<shortname>_data_t *d = piece->data;

  if(!dt_iop_have_required_input_format(4, self, piece->colors,
                                        dev_in, dev_out, roi_in, roi_out))
    return DT_OPENCL_PROCESS_CL;

  const int devid = piece->pipe->devid;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const size_t ch = piece->colors;

  float factor = d->factor;

  size_t sizes[] = { ROUNDUPDWD(width, devid), ROUNDUPDWD(height, devid), 1 };
  dt_opencl_local_buffer_t locopt = { 0 };

  dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), &dev_in);
  dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(cl_mem), &dev_out);
  dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(int), &width);
  dt_opencl_set_kernel_arg(devid, kernel, 3, sizeof(int), &height);
  dt_opencl_set_kernel_arg(devid, kernel, 4, sizeof(float), &factor);

  return dt_opencl_enqueue_kernel(devid, kernel, 2, sizes, NULL, locopt);
}
```

- Return `DT_OPENCL_PROCESS_CL` on success
- Return `DT_OPENCL_PROCESS_PENDING_CL` if tiled execution is needed
- Never return `DT_OPENCL_PROCESS_CL` from a path that failed — the caller interprets the return as "processed successfully"

### OpenCL kernel file

Kernels go in `src/iop/<shortname>.cl`:

```opencl
kernel void kernel_name(read_only image2d_t in, write_only image2d_t out,
                        const int width, const int height, const float factor)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int4 coords = (int4)(x, y, 0, 0);
  const float4 pixel = read_imagef(in, sampleri, coords);

  float4 result = pixel * factor;

  write_imagef(out, coords, result);
}
```

## Colour Management Checklist

1. Choose the right `default_colorspace()` — most creative modules use `IOP_CS_RGB` (scene-linear)
2. If the module operates on raw data, use `IOP_CS_RAW` and verify it runs before `colorin` in the pipe
3. If the module modifies hue, use `IOP_CS_LCH` or `IOP_CS_HSL`
4. If processing in RGB, verify white point handling and chromatic adaptation are correct
5. Use `dt_iop_have_required_input_format()` to skip on unexpected channel counts
6. Preserve the 4th channel (alpha) unless the module explicitly manipulates it

## Parameter Versioning

1. Start with `DT_MODULE_INTROSPECTION(1, ...)`
2. On any change to `params_t`: bump the version number
3. Write `legacy_params()` for every intermediate version — chain them:

```c
int legacy_params(dt_iop_module_t *self, const void *const old_params,
                  const int old_version, void **new_params,
                  int32_t *new_params_size, int *new_version)
{
  if(old_version == 1) {
    dt_iop_<shortname>_params_v1_t *old = (dt_iop_<shortname>_params_v1_t *)old_params;
    // map old fields to new, 0-fill padding
    *new_params_size = sizeof(dt_iop_<shortname>_params_t);
    *new_params = malloc(*new_params_size);
    dt_iop_<shortname>_params_t *new = (dt_iop_<shortname>_params_t *)*new_params;
    *new = (dt_iop_<shortname>_params_t){ 0 };  // zero-init
    new->factor = old->old_factor;  // renamed or recomputed
    new->method = old->method;
    *new_version = 2;
    return 0;
  }
  return 1;  // unknown version
}
```

## Full Module Checklist

- [ ] `DT_MODULE_INTROSPECTION()` at top of file
- [ ] `params_t` struct with introspection tags (`$MIN`, `$MAX`, `$DEFAULT`, `$DESCRIPTION`)
- [ ] `data_t` struct for pipe-local processing data
- [ ] `gui_data_t` struct for widget references
- [ ] `name()` returns translatable `_("...")` ≤ 19 chars
- [ ] `default_colorspace()` returns appropriate space
- [ ] `process()` — correct, vectorized, with input format guard
- [ ] `flags()` — appropriate IOP flags set
- [ ] `default_group()` — correct module group
- [ ] `description()` — 5-field description
- [ ] `init()` → `dt_iop_default_init()`, set `self->params_size`
- [ ] `cleanup()` → `dt_iop_default_cleanup()`
- [ ] `commit_params()` — precompute coefficients in `piece->data`
- [ ] `gui_init()` — build widget tree with `IOP_GUI_ALLOC()`
- [ ] `gui_cleanup()` — free GUI data
- [ ] `gui_update()` — sync params to widgets
- [ ] Registration in `src/iop/CMakeLists.txt` via `add_iop()`
- [ ] Pipeline order entries in all 5 tables in `src/common/iop_order.c`
- [ ] If OpenCL: `.cl` kernel file + `process_cl()` returning proper status
- [ ] If tiling: `IOP_FLAGS_ALLOW_TILING` + `tiling_callback()`
- [ ] If serialization changes: `legacy_params()` with proper migration

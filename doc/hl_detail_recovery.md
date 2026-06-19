# HL Detail Recovery in 3DCF

The `hl_detail_recovery` parameter in the 3DCF module re-injects fine texture
lost during tone mapping by computing the difference between the original
scene-linear luminance and its edge-preserving smoothed version (the *base*).

## Algorithm

1. **Luminance extraction** — scene-linear luminance is computed from the
   sanitized RGB channels using the working-space luma coefficients.

2. **Bilateral grid smoothing** — the luminance is scaled by 20× (to map the
   typical scene-linear range [0, 5] onto [0, 100], matching the Lab L range
   the bilateral grid was designed for), then processed through a 3D bilateral
   grid:

   - `sigma_s = 50` (spatial radius, in pixels)
   - `sigma_r = 5` (range sigma — edge sensitivity in luminance space)
   - `detail = -1.0` (pure smoothing = base extraction)

   The bilateral grid preserves strong edges natively (via the range term),
   so no additional edge-attenuation heuristic is needed.

3. **Detail = luminance − base** — the difference represents texture that was
   present in the original but removed by smoothing.

4. **Gain-compensated reinjection** — `detail` is scaled by the user's
   `hl_detail_recovery` slider and multiplied by a gain factor
   `gain = lum_tm / lum_orig` to compensate for the tone mapper's attenuation.
   The final luminance is `lum_final = lum_tm + detail * gain * hl_detail_recovery`,
   clamped to a minimum scale of 0.25× per pixel to prevent black clipping.

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sigma_s` | 50 | Spatial sigma of bilateral grid (pixels) |
| `sigma_r` | 5 | Range sigma of bilateral grid (luminance units, scaled) |
| `detail`  | −1.0 | Slice parameter: −1 = pure smoothing (base) |
| `bilateral_scale` | 20.0 | Maps scene-linear lum [0,5] → [0,100] for bilateral grid |

## CPU / GPU consistency

Both the CPU (`process`) and OpenCL (`process_cl`) paths use the same
parameters and produce matching results. The bilateral grid OpenCL kernels
splat and slice from `image2d_t` objects; the intermediate luminance buffer
and the scaled-luminance RGBA image are allocated with
`dt_opencl_alloc_device` (which creates `image2d_t` objects matching the
requested bytes-per-pixel).

## History

- **Guided filter (original)** — used `gf_radius` proportional to sensor
  diagonal; required an explicit `edge_atten` term to suppress halos on
  high-contrast edges. Removed in favor of the bilateral grid because the
  guided filter's edge awareness depends on a global epsilon parameter,
  causing residual halos on very sharp scene-referred edges.
- **Bilateral grid (current)** — edge-preserving by construction; eliminates
  the `edge_atten` workaround and the resolution-dependent radius
  normalisation.

/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.
    Libre DT-lab Edition (C) 2026 Christian Bouhon.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.

    ---------------------------------------------------------------------------
    OpenCL kernels for the "contrast & texture" module. The heavy lifting (the
    edge-aware multi-scale blur) is done by the shared EIGF kernels in eigf.cl
    orchestrated host-side; here we only compute the guide luminance and apply
    the final multi-scale contrast maths. Both kernels mirror their CPU
    counterparts (compute_pixel_luminance_mask / apply_local_contrast) 1:1.

    The detail-mask visualisation path is intentionally not ported: process_cl
    falls back to the CPU when a mask is being displayed.
*/

#include "common.h"

#define CONTRAST_MIN_FLOAT 1.52587890625e-05f  // exp2f(-16), matches MIN_FLOAT

/* Pixel-wise guide luminance, DT_TONEEQ_NORM_2 with exposure=1, fulcrum=0,
 * contrast=1 → max(sqrt(r^2+g^2+b^2), MIN_FLOAT). */
__kernel void contrast_luminance(read_only image2d_t in,
                                 global float *luminance,
                                 const int width,
                                 const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 px = read_imagef(in, sampleri, (int2)(x, y));
  const float l = sqrt(px.x * px.x + px.y * px.y + px.z * px.z);
  luminance[mad24(y, width, x)] = fmax(l, CONTRAST_MIN_FLOAT);
}

/* Apply the multi-scale contrast. Inactive scales are fed the lum_pixel buffer
 * as a harmless placeholder by the host: log2(lum_pixel/lum_pixel) = 0, and the
 * (gain-1) factor is 0 too, so they contribute exactly nothing (same result as
 * the CPU which skips them). */
__kernel void contrast_apply(read_only image2d_t in,
                             write_only image2d_t out,
                             global const float *lum_pixel,
                             global const float *lum_local,
                             global const float *lum_coarse,
                             global const float *lum_broad,
                             global const float *lum_fine,
                             global const float *lum_micro,
                             const int width,
                             const int height,
                             const float gain_local,
                             const float gain_coarse,
                             const float gain_broad,
                             const float gain_fine,
                             const float gain_micro,
                             const float gain_global,
                             const float w_local,
                             const float w_global,
                             const float noise_threshold,
                             const float csf_adaptation,
                             const float color_balance,
                             const float colorful_contrast,
                             const float green_compensation)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int k = mad24(y, width, x);
  const int2 pos = (int2)(x, y);

  const float lp = fmax(lum_pixel[k], CONTRAST_MIN_FLOAT);

  // Sum of (gain - 1) * detail_ev over the five spatial scales.
  float correction_ev = 0.0f;
  correction_ev += (gain_local  - 1.0f) * log2(lp / fmax(lum_local[k],  CONTRAST_MIN_FLOAT));
  correction_ev += (gain_coarse - 1.0f) * log2(lp / fmax(lum_coarse[k], CONTRAST_MIN_FLOAT));
  correction_ev += (gain_broad  - 1.0f) * log2(lp / fmax(lum_broad[k],  CONTRAST_MIN_FLOAT));
  correction_ev += (gain_fine   - 1.0f) * log2(lp / fmax(lum_fine[k],   CONTRAST_MIN_FLOAT));
  correction_ev += (gain_micro  - 1.0f) * log2(lp / fmax(lum_micro[k],  CONTRAST_MIN_FLOAT));

  // Balance weighting (global <-> spatial)
  correction_ev *= w_local;

  // Noise protection — attenuate local contrast gain in low luminance
  if(noise_threshold > 1e-6f)
  {
    const float floor_v = noise_threshold * 8.0f;
    const float ratio = fmin(lp / fmax(floor_v, 1e-6f), 1.0f);
    const float attenuation = ratio * ratio * (3.0f - 2.0f * ratio);
    correction_ev *= attenuation;
  }

  // Global contrast with CSF weighting, centred on middle gray (0.1845)
  const float log_lum = log2(lp / 0.1845f);
  const float csf_weight = exp(-(log_lum * log_lum) / 12.5f);
  const float effective_csf_weight = (1.0f - csf_adaptation) + csf_adaptation * csf_weight;
  const float global_term = (gain_global - 1.0f) * effective_csf_weight * log_lum * w_global;

  // Colorimetric contrast factor (red vs blue)
  const float4 px = read_imagef(in, sampleri, pos);
  float factor = 1.0f;
  if(fabs(color_balance) > 0.001f)
  {
    const float avg = fmax((px.x + px.y + px.z) / 3.0f, 1e-6f);
    const float mix = (color_balance * 0.5f) * (px.x - px.z);
    factor = fmax(1.0f + mix / avg, 0.0f);
  }

  const float multiplier = exp2(correction_ev + global_term) * factor;
  const float L_final = lp * multiplier;

  float ratio = L_final / fmax(lp, 1e-6f);
  ratio = fmin(ratio, 8.0f);

  float4 o = px * ratio;
  o.w = px.w; // carry alpha through unchanged

  // Saturation boost tied to CSF weight, active only above 50%
  if(csf_adaptation > 0.5f)
  {
    const float saturation_boost = 1.0f + (csf_adaptation - 0.5f) * csf_weight * 0.1f;
    o.x *= saturation_boost;
    o.y *= saturation_boost;
    o.z *= saturation_boost;
  }

  // Colorful contrast — warm/cool separation, luminance-compensated on green
  if(fabs(colorful_contrast) > 0.001f)
  {
    const float chroma_gain = colorful_contrast * 0.15f;
    const float chroma_diff = (o.x - o.z) * chroma_gain * effective_csf_weight;

    o.x += chroma_diff;
    o.z -= chroma_diff;
    o.y -= chroma_diff * green_compensation;

    if(o.x < 0.0f || o.y < 0.0f || o.z < 0.0f)
    {
      float t = 1.0f;
      if(o.x < 0.0f) t = fmin(t, L_final / (L_final - o.x));
      if(o.y < 0.0f) t = fmin(t, L_final / (L_final - o.y));
      if(o.z < 0.0f) t = fmin(t, L_final / (L_final - o.z));
      o.x = L_final + t * (o.x - L_final);
      o.y = L_final + t * (o.y - L_final);
      o.z = L_final + t * (o.z - L_final);
    }
  }

  write_imagef(out, pos, o);
}

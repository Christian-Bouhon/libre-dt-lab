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
    OpenCL port of the ACES 2.0 SSTS spectral tone pipeline. The CPU reference
    (src/iop/spectral_tone.c, dt_st_pipeline_eval) computes the per-pixel maths
    in double precision; this kernel uses single precision throughout. The
    visual difference is below 8-bit quantisation for tone mapping, and every
    code path mirrors the CPU version 1:1 so the GPU and CPU previews match.
*/

#include "common.h"
#include "colorspace.h"

/* Must match dt_st_cl_params_t in src/iop/spectral_tone.c byte for byte.
 * Only 4-byte members (float / int) are used so the host and device layouts
 * are identical with no padding — the struct is passed by value via CLARG. */
typedef struct dt_st_cl_params_t
{
  float input_matrix[9];     // D50-adapted Rec.2020 RGB -> D50 XYZ
  float output_matrix[9];    // D50 XYZ -> working-space RGB
  float luma_coeff[3];       // luminance weights for the output color space
  float color_look_mat[9];   // selected color look 3x3 (identity when none)
  float exposure_factor;
  float contrast;
  float contrast_pivot;
  float hl_desat;
  float hl_rotation;
  float abney_cos;
  float abney_sin;
  float white_chroma_x;
  float white_chroma_z;
  float gray_point;
  float gray_gamma;
  float vibrance;
  float gamut_knee;
  float gamut_steepness;
  float ssts_s_2;
  float ssts_m_2;
  float ssts_g;
  float ssts_t_1;
  float ssts_n_r;
  float ssts_n;
  float look_opacity;
  int   look_idx;            // 0 = no look, 1..10 = color_look_mat is active
} dt_st_cl_params_t;

/* ACES 2.0 SSTS forward — Michaelis-Menten segment + flare compensation */
static inline float st_ssts_fwd(const dt_st_cl_params_t *p, const float x)
{
  if(x <= 0.0f) return 0.0f;
  const float f = p->ssts_m_2 * pow(x / (x + p->ssts_s_2), p->ssts_g);
  const float h = (f * f) / (f + p->ssts_t_1);
  return h * p->ssts_n_r;
}

/* Tone-mapped Y from scene-linear Y (SSTS + BT.1886 + contrast S-curve) */
static inline float st_compute_y_tm(const float y_scene, const dt_st_cl_params_t *p)
{
  if(p->ssts_n <= 0.0f) return 0.0f;

  float y_tm = st_ssts_fwd(p, y_scene * p->exposure_factor) / p->ssts_n;

  /* BT.1886 OETF */
  y_tm = pow(fmax(y_tm, 0.0f), 1.0f / 2.4f);

  /* Post-SSTS contrast S-curve pivoted at contrast_pivot */
  if(p->contrast != 1.0f)
  {
    const float c = p->contrast;
    const float pv = p->contrast_pivot;
    if(y_tm <= pv)
      y_tm = pv * pow(fmax(y_tm / pv, 0.0f), c);
    else
      y_tm = 1.0f - (1.0f - pv) * pow(fmax((1.0f - y_tm) / (1.0f - pv), 0.0f), c);
  }
  return y_tm;
}

/* Highlight desaturation weight */
static inline float st_desat_weight(const float y_norm, const float hl_desat)
{
  if(hl_desat <= 0.0f || y_norm <= 0.7f) return 0.0f;
  const float t = fmax(y_norm - 0.7f, 0.0f) / y_norm;
  const float x = fmin(t * hl_desat, 1.0f);
  return x * x;
}

/* Film-like gamut compression: blend out-of-gamut channels toward white */
static inline float3 st_gamut_compress(float3 rgb)
{
  const float m = fmin(fmin(rgb.x, rgb.y), rgb.z);
  if(m >= 0.0f) return rgb;

  float t = 0.0f;
  if(rgb.x < 0.0f) { const float ti = -rgb.x / (1.0f - rgb.x); if(ti > t) t = ti; }
  if(rgb.y < 0.0f) { const float ti = -rgb.y / (1.0f - rgb.y); if(ti > t) t = ti; }
  if(rgb.z < 0.0f) { const float ti = -rgb.z / (1.0f - rgb.z); if(ti > t) t = ti; }
  t = fmin(t, 1.0f);

  return (1.0f - t) * rgb + t * (float3)(1.0f, 1.0f, 1.0f);
}

/* Spectral gamut: film-like chromaticity roll-off in CIE xy, preserving Y */
static inline void st_spectral_gamut(float *x_tm, float *z_tm, const float y_tm,
                                     const float white_x_ratio, const float white_z_ratio,
                                     const float knee, const float steepness)
{
  if(y_tm <= 0.0f) return;

  const float sum = *x_tm + y_tm + *z_tm;
  if(sum <= 0.0f) return;
  const float cie_x = *x_tm / sum;
  const float cie_z = *z_tm / sum;

  const float wy = 1.0f;
  const float wx = white_x_ratio;
  const float wz = white_z_ratio;
  const float wsum = wx + wy + wz;
  const float white_cie_x = wx / wsum;
  const float white_cie_z = wz / wsum;

  const float dx = cie_x - white_cie_x;
  const float dz = cie_z - white_cie_z;
  const float chroma_sq = dx * dx + dz * dz;

  if(chroma_sq > knee * knee)
  {
    const float chroma = sqrt(chroma_sq);
    const float excess = chroma - knee;
    const float compression = excess / (excess + steepness);
    const float scale = (chroma - compression * excess) / chroma;

    const float x_new = white_cie_x + scale * dx;
    const float z_new = white_cie_z + scale * dz;
    const float y_new = 1.0f - x_new - z_new;

    if(y_new > 0.0f)
    {
      const float S_new = y_tm / y_new;
      *x_tm = x_new * S_new;
      *z_tm = z_new * S_new;
    }
  }
}

/* Complete spectral tone mapping pipeline for one pixel (mirrors dt_st_pipeline_eval) */
static inline float3 st_pipeline_eval(float3 rgb_in, const dt_st_cl_params_t *p)
{
  /* Step 1: D50-adapted Rec.2020 RGB -> D50 XYZ */
  const float r = rgb_in.x, g = rgb_in.y, b = rgb_in.z;
  const float x_abs = p->input_matrix[0] * r + p->input_matrix[1] * g + p->input_matrix[2] * b;
  const float y_abs = p->input_matrix[3] * r + p->input_matrix[4] * g + p->input_matrix[5] * b;
  const float z_abs = p->input_matrix[6] * r + p->input_matrix[7] * g + p->input_matrix[8] * b;

  if(y_abs <= 0.0f) return (float3)(0.0f, 0.0f, 0.0f);

  const float x_ratio = x_abs / y_abs;
  const float z_ratio = z_abs / y_abs;

  /* Step 2-3: tone-mapped Y */
  float y_tm = st_compute_y_tm(y_abs, p);

  /* Step 4: mid-tone gamma */
  if(p->gray_point != 0.0f)
  {
    float y_lvl = fmin(fmax(y_tm, 0.0f), 1.0f);
    y_lvl = pow(y_lvl, p->gray_gamma);
    y_tm = y_lvl;
  }

  /* Step 5: scale chromaticity with tone-mapped luminance */
  float x_tm = x_ratio * y_tm;
  float z_tm = z_ratio * y_tm;

  /* Step 6: spectral gamut roll-off */
  st_spectral_gamut(&x_tm, &z_tm, y_tm,
                    p->white_chroma_x, p->white_chroma_z,
                    p->gamut_knee, p->gamut_steepness);

  /* Step 7: XYZ -> output RGB */
  float3 rgb;
  rgb.x = p->output_matrix[0] * x_tm + p->output_matrix[1] * y_tm + p->output_matrix[2] * z_tm;
  rgb.y = p->output_matrix[3] * x_tm + p->output_matrix[4] * y_tm + p->output_matrix[5] * z_tm;
  rgb.z = p->output_matrix[6] * x_tm + p->output_matrix[7] * y_tm + p->output_matrix[8] * z_tm;

  /* Step 8: highlight desaturation with Abney hue correction */
  {
    const float y_exposed = y_abs * p->exposure_factor;
    const float w = st_desat_weight(y_exposed, p->hl_desat);
    if(w > 0.0f)
    {
      const float lc0 = p->luma_coeff[0], lc1 = p->luma_coeff[1], lc2 = p->luma_coeff[2];
      const float y = lc0 * rgb.x + lc1 * rgb.y + lc2 * rgb.z;
      float u = rgb.x - y;
      float v = rgb.z - y;

      if(p->hl_rotation != 0.0f)
      {
        const float ca = p->abney_cos;
        const float sa = p->abney_sin;
        const float ur = u * ca - v * sa;
        const float vr = u * sa + v * ca;
        u = ur;
        v = vr;
      }

      u *= (1.0f - w);
      v *= (1.0f - w);

      rgb.x = y + u;
      rgb.z = y + v;
      if(lc1 > 0.0f)
        rgb.y = y - (lc0 / lc1) * u - (lc2 / lc1) * v;
      else
        rgb.y = y;
    }
  }

  /* Step 9: vibrance with high-sat protection */
  if(p->vibrance != 1.0f)
  {
    const float vib = fmax(p->vibrance, 0.0f);
    const float lc0 = p->luma_coeff[0], lc1 = p->luma_coeff[1], lc2 = p->luma_coeff[2];
    const float luma = lc0 * rgb.x + lc1 * rgb.y + lc2 * rgb.z;
    const float maxc = fmax(fmax(rgb.x, rgb.y), rgb.z);
    const float minc = fmin(fmin(rgb.x, rgb.y), rgb.z);
    const float sat_m = maxc - minc;
    const float level = fmax(maxc, fmax(fabs(minc), fabs(luma)));
    const float sat_norm = (level > 0.0f) ? sat_m / level : 0.0f;

    if(vib > 1.0f)
    {
      const float pp = 1.0f - fmin(sat_norm, 1.0f);
      const float vib_gain = 1.0f + (vib - 1.0f) * (pp * pp);
      rgb.x = luma + vib_gain * (rgb.x - luma);
      rgb.y = luma + vib_gain * (rgb.y - luma);
      rgb.z = luma + vib_gain * (rgb.z - luma);
    }
    else
    {
      rgb.x = luma + vib * (rgb.x - luma);
      rgb.y = luma + vib * (rgb.y - luma);
      rgb.z = luma + vib * (rgb.z - luma);
    }
  }

  /* Step 10: gamut compression */
  rgb = st_gamut_compress(rgb);

  /* Step 11: clamp negatives */
  return fmax(rgb, (float3)(0.0f, 0.0f, 0.0f));
}

__kernel void spectral_tone(
    read_only image2d_t input,
    write_only image2d_t output,
    const int width,
    const int height,
    const dt_st_cl_params_t p)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int2 pos = (int2)(x, y);
  float4 pixel = read_imagef(input, sampleri, pos);

  /* sanitize input range and drop NaNs (matches agx sanitisation) */
  pixel = select(clamp(pixel, -1e6f, 1e6f), (float4)(0.0f), isnan(pixel));

  float3 rgb_in = (float3)(pixel.x, pixel.y, pixel.z);

  /* Luma-clipping desaturation safety net (mirrors process() in spectral_tone.c) */
  const float luma_in = fmax(fmax(rgb_in.x, rgb_in.y), rgb_in.z);
  const float safety_threshold = 0.9f;
  const float hard_clip = 1.1f;
  if(luma_in > safety_threshold)
  {
    float amount = (luma_in - safety_threshold) / (hard_clip - safety_threshold);
    amount = fmin(fmax(amount, 0.0f), 1.0f);
    const float weight = amount * p.hl_desat;
    rgb_in = rgb_in * (1.0f - weight) + (float3)(luma_in) * weight;
  }

  float3 rgb_out = st_pipeline_eval(rgb_in, &p);

  /* Color look matrix + opacity blend (mirrors process() in spectral_tone.c) */
  if(p.look_idx > 0)
  {
    const float r = rgb_out.x, g = rgb_out.y, b = rgb_out.z;
    const float tr = r * p.color_look_mat[0] + g * p.color_look_mat[1] + b * p.color_look_mat[2];
    const float tg = r * p.color_look_mat[3] + g * p.color_look_mat[4] + b * p.color_look_mat[5];
    const float tb = r * p.color_look_mat[6] + g * p.color_look_mat[7] + b * p.color_look_mat[8];

    rgb_out.x = r * (1.0f - p.look_opacity) + tr * p.look_opacity;
    rgb_out.y = g * (1.0f - p.look_opacity) + tg * p.look_opacity;
    rgb_out.z = b * (1.0f - p.look_opacity) + tb * p.look_opacity;
    rgb_out = fmax(rgb_out, (float3)(0.0f, 0.0f, 0.0f));
  }

  float4 out_pixel = (float4)(rgb_out.x, rgb_out.y, rgb_out.z, pixel.w);
  write_imagef(output, pos, out_pixel);
}

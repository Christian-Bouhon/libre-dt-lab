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
    (src/iop/3dcf.c, dt_st_pipeline_eval) uses single precision for
    per-pixel maths; this kernel uses the same precision throughout. Both pre-
    compute the context (matrices, SSTS parameters) on the host — the kernel
    receives them as floats. Every code path mirrors the CPU version 1:1 so the
    GPU and CPU previews match.
*/

#include "common.h"
#include "colorspace.h"

/* Must match dt_st_cl_params_t in src/iop/3dcf.c byte for byte.
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
  float hl_desat_threshold;
  float hl_rotation;
  float white_chroma_x;
  float white_chroma_z;
  float gray_point;
  float gray_gamma;
  float vibrance;
  float gamut_knee;
  float gamut_steepness;
  float toe_power;
  float shoulder_power;
  float gamut_fwd[9];
  float gamut_inv[9];
  float ssts_s_2;
  float ssts_m_2;
  float ssts_g;
  float ssts_t_1;
  float ssts_n_r;
  float ssts_n;
  float look_opacity;
  float hl_detail_recovery;
  float spectral_boundary[360];
  int   look_idx;            // 0 = no look, 1..10 = color_look_mat is active
  int   gamut_enable;
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

  /* Clamp to [0, 1] before contrast curve: the shoulder formula assumes
   * y_tm ∈ [0,1] and produces NaN/Inf when y_tm > 1 (pow(0, negative)). */
  y_tm = fmin(y_tm, 1.0f);

  /* Post-SSTS contrast S-curve pivoted at contrast_pivot */
  if(p->contrast != 1.0f || p->toe_power != 1.0f || p->shoulder_power != 1.0f)
  {
    const float c  = p->contrast;
    const float pv = p->contrast_pivot;
    const float ct = p->toe_power;
    const float cs = p->shoulder_power;
    if(y_tm <= pv)
    {
      const float t = (pv > 0.0f) ? y_tm / pv : 0.0f;
      const float exp_eff = c * (ct + (1.0f - ct) * t);
      y_tm = pv * pow(fmax(y_tm / pv, 0.0f), exp_eff);
    }
    else
    {
      const float rp = 1.0f - pv;
      const float t = (rp > 0.0f) ? (1.0f - y_tm) / rp : 0.0f;
      const float exp_eff = c * (cs + (1.0f - cs) * t);
      y_tm = 1.0f - rp * pow(fmax((1.0f - y_tm) / rp, 0.0f), exp_eff);
    }
  }
  return y_tm;
}

/* Highlight desaturation weight */
static inline float st_desat_weight(const float y_norm, const float hl_desat, const float threshold)
{
  if(hl_desat <= 0.0f || y_norm <= threshold) return 0.0f;
  if(!isfinite(y_norm)) return 0.0f;
  const float t = fmax(y_norm - threshold, 0.0f) / y_norm;
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

/* Output gamut protection: convert to target color space, clamp negatives, revert.
 * Mirrors st_output_gamut_protect() in 3dcf.c. */
static inline float3 st_output_gamut_protect(float3 rgb, const float fwd[9], const float inv[9])
{
  float3 t;
  t.x = fwd[0]*rgb.x + fwd[1]*rgb.y + fwd[2]*rgb.z;
  t.y = fwd[3]*rgb.x + fwd[4]*rgb.y + fwd[5]*rgb.z;
  t.z = fwd[6]*rgb.x + fwd[7]*rgb.y + fwd[8]*rgb.z;

  t = fmax(t, (float3)(0.0f, 0.0f, 0.0f));

  float3 r;
  r.x = inv[0]*t.x + inv[1]*t.y + inv[2]*t.z;
  r.y = inv[3]*t.x + inv[4]*t.y + inv[5]*t.z;
  r.z = inv[6]*t.x + inv[7]*t.y + inv[8]*t.z;
  return r;
}

/* Spectral gamut: film-like chromaticity roll-off in CIE xy, preserving Y */
static inline void st_spectral_gamut(float *x_tm, float *z_tm, const float y_tm,
                                     const float white_x_ratio, const float white_z_ratio,
                                     const float knee, const float steepness,
                                     const float *spectral_boundary)
{
  if(y_tm <= 0.0f) return;
  if(!isfinite(*x_tm) || !isfinite(*z_tm)) return;

  const float sum = *x_tm + y_tm + *z_tm;
  if(sum <= 0.0f) return;
  float cie_x = *x_tm / sum;
  float cie_z = *z_tm / sum;

  const float wy = 1.0f;
  const float wx = white_x_ratio;
  const float wz = white_z_ratio;
  const float wsum = wx + wy + wz;
  const float white_cie_x = wx / wsum;
  const float white_cie_z = wz / wsum;

  float dx = cie_x - white_cie_x;
  float dz = cie_z - white_cie_z;
  float chroma_sq = dx * dx + dz * dz;

  /* Spectral locus boundary check (before knee) */
  if(spectral_boundary)
  {
    const float angle = atan2(dz, dx);
    float angle_deg = angle * (180.0f / M_PI_F);
    if(angle_deg < 0.0f) angle_deg += 360.0f;
    int bin = (int)angle_deg;
    if(bin < 0) bin = 0;
    if(bin >= 360) bin = 359;
    const float max_dist = spectral_boundary[bin];
    const float target_dist = max_dist * 0.95f;

    if(max_dist > 0.0f && chroma_sq > target_dist * target_dist)
    {
      const float chroma = sqrt(chroma_sq);
      const float excess = chroma - target_dist;
      const float bsteep = fmax(target_dist * 0.05f, 0.001f);
      const float compression = excess / (excess + bsteep);
      const float scale = (chroma - compression * excess) / chroma;

      cie_x = white_cie_x + scale * dx;
      cie_z = white_cie_z + scale * dz;
      const float y_new = 1.0f - cie_x - cie_z;
      if(y_new > 0.0f)
      {
        const float S_new = y_tm / y_new;
        *x_tm = cie_x * S_new;
        *z_tm = cie_z * S_new;
        /* Recompute for knee */
        dx = cie_x - white_cie_x;
        dz = cie_z - white_cie_z;
        chroma_sq = dx * dx + dz * dz;
      }
    }
  }

  /* Existing knee */
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

  if(!(y_abs > 1e-10f) || !isfinite(y_abs)) return (float3)(0.0f, 0.0f, 0.0f);

  const float x_ratio = clamp(x_abs / y_abs, -100.0f, 100.0f);
  const float z_ratio = clamp(z_abs / y_abs, -100.0f, 100.0f);

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
                    p->gamut_knee, p->gamut_steepness,
                    p->spectral_boundary);

  /* Step 7: XYZ -> output RGB */
  float3 rgb;
  rgb.x = p->output_matrix[0] * x_tm + p->output_matrix[1] * y_tm + p->output_matrix[2] * z_tm;
  rgb.y = p->output_matrix[3] * x_tm + p->output_matrix[4] * y_tm + p->output_matrix[5] * z_tm;
  rgb.z = p->output_matrix[6] * x_tm + p->output_matrix[7] * y_tm + p->output_matrix[8] * z_tm;

  /* Step 8: Film-print highlight desaturation toward white */
  {
    const float y_exposed = y_abs * p->exposure_factor;
    const float w = st_desat_weight(y_exposed, p->hl_desat, p->hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float maxc_pre = fmax(fmax(rgb.x, rgb.y), rgb.z);
      const float minc_pre = fmin(fmin(rgb.x, rgb.y), rgb.z);
      const float sat_pre = (maxc_pre > 0.0f) ? (maxc_pre - minc_pre) / maxc_pre : 0.0f;
      const float ss_pre = (sat_pre * sat_pre) / (sat_pre * sat_pre + (1.0f - sat_pre) * (1.0f - sat_pre) + 1e-6f);

      /* Progressive Abney hue rotation — weight independent of hl_desat */
      if(p->hl_rotation != 0.0f)
      {
        const float wr = fmin(st_desat_weight(y_exposed, 1.0f, p->hl_desat_threshold), 1.0f);
        const float angle = p->hl_rotation * 0.25f * wr; // CB
        const float ca = cos(angle);
        const float sa = sin(angle);
        if(isfinite(ca) && isfinite(sa))
        {
          const float lc0 = p->luma_coeff[0], lc1 = p->luma_coeff[1], lc2 = p->luma_coeff[2];
          const float y = lc0 * rgb.x + lc1 * rgb.y + lc2 * rgb.z;
          const float u = rgb.x - y;
          const float v = rgb.z - y;
          const float ur = u * ca - v * sa;
          const float vr = u * sa + v * ca;
          rgb.x = y + ur;
          rgb.z = y + vr;
          if(lc1 > 0.0f)
            rgb.y = y - (lc0 / lc1) * ur - (lc2 / lc1) * vr;
        }
      }

      /* Vibrance négative : désature plus les pixels saturés, activée par hl_desat et hl_hue_shift */
      float w_final = w;
      if(p->hl_desat > 0.0f || p->hl_rotation != 0.0f)
      {
        if(p->hl_desat > 0.0f)
        {
          const float vib_neg = p->hl_desat * ss_pre * 0.5f;
          const float w_vib = st_desat_weight(y_exposed, 1.0f, p->hl_desat_threshold) * vib_neg;
          w_final = fmin(w_final + w_vib, 1.0f);
        }
        if(p->hl_rotation != 0.0f)
        {
          const float vib_neg = fabs(p->hl_rotation) * ss_pre * 1.0f;
          const float w_rot = st_desat_weight(y_exposed, 1.0f, p->hl_desat_threshold) * vib_neg;
          w_final = fmax(w_final, w_rot);
        }
      }

      /* Blend toward white with sigmoidal curve — film-like */
      const float t = fmin(w_final, 1.0f);
      const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
      if(isfinite(ts))
        rgb = rgb * (1.0f - ts) + (float3)(1.0f, 1.0f, 1.0f) * ts;
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

  /* Output gamut protection: clamp to selected primary space */
  if(p->gamut_enable)
  {
    const float fwd[] = { p->gamut_fwd[0], p->gamut_fwd[1], p->gamut_fwd[2],
                          p->gamut_fwd[3], p->gamut_fwd[4], p->gamut_fwd[5],
                          p->gamut_fwd[6], p->gamut_fwd[7], p->gamut_fwd[8] };
    const float inv[] = { p->gamut_inv[0], p->gamut_inv[1], p->gamut_inv[2],
                          p->gamut_inv[3], p->gamut_inv[4], p->gamut_inv[5],
                          p->gamut_inv[6], p->gamut_inv[7], p->gamut_inv[8] };
    rgb = st_output_gamut_protect(rgb, fwd, inv);
  }

  /* Step 11: clamp negatives */
  rgb.x = isfinite(rgb.x) ? fmax(rgb.x, 0.0f) : 0.0f;
  rgb.y = isfinite(rgb.y) ? fmax(rgb.y, 0.0f) : 0.0f;
  rgb.z = isfinite(rgb.z) ? fmax(rgb.z, 0.0f) : 0.0f;
  return rgb;
}

/* Extract luminance from RGBA image to a float buffer, and simultaneously
 * produce a sanitized RGBA copy of the input to be used as the GUIDE for
 * guided_filter_cl. Negative/NaN channels (CA fringing, sharpening halos)
 * must be cleared on this guide too: it feeds the filter's local mean/
 * variance regression, and a single stray NaN corrupts an entire window,
 * not just one pixel. Mirrors the CPU's guide_sanitized buffer exactly. */
__kernel void kernel_3dcf_extract_lum(
    read_only image2d_t input,
    write_only image2d_t output_sanitized,
    __global float *dev_lum,
    const int width,
    const int height,
    const dt_st_cl_params_t p)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int2 pos = (int2)(x, y);
  const float4 pixel = read_imagef(input, sampleri, pos);

  const float3 rgb = select(fmax(pixel.xyz, 0.0f), (float3)(0.0f), !isfinite(pixel.xyz));

  const float lum = p.luma_coeff[0] * rgb.x + p.luma_coeff[1] * rgb.y + p.luma_coeff[2] * rgb.z;
  dev_lum[y * width + x] = lum;

  write_imagef(output_sanitized, pos, (float4)(rgb.x, rgb.y, rgb.z, pixel.w));
}

__kernel void kernel_3dcf(
    read_only image2d_t input,
    write_only image2d_t output,
    const int width,
    const int height,
    const dt_st_cl_params_t p,
    __global float *dev_base)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int2 pos = (int2)(x, y);
  float4 pixel = read_imagef(input, sampleri, pos);

  /* sanitize input: clamp negative (from CA / sharpening halos) to 0 and drop NaNs */
  pixel = (float4)(select(fmax(pixel.xyz, 0.0f), (float3)(0.0f), !isfinite(pixel.xyz)), pixel.w);

  float3 rgb_in = (float3)(pixel.x, pixel.y, pixel.z);

  /* Save original luminance for detail recovery (before safety net modifies rgb_in) */
  float lum_orig = 0.0f;
  if(p.hl_detail_recovery > 0.0f && dev_base != NULL)
    lum_orig = p.luma_coeff[0] * rgb_in.x + p.luma_coeff[1] * rgb_in.y + p.luma_coeff[2] * rgb_in.z;

  /* Pre-pipeline safety net: sigmoid rolloff toward luma-gray */
  {
    const float lum = fmax(fmax(rgb_in.x, rgb_in.y), rgb_in.z);
    const float w = st_desat_weight(lum * p.exposure_factor, p.hl_desat, p.hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float t = fmin(w, 1.0f);
      const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
      if(isfinite(ts) && ts > 0.0f)
        rgb_in = rgb_in * (1.0f - ts) + (float3)(lum, lum, lum) * ts;
    }
  }

  float3 rgb_out = st_pipeline_eval(rgb_in, &p);

  /* Color look matrix + opacity blend (mirrors process() in 3dcf.c) */
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

  /* HL detail recovery: re-inject guided-filter detail with gain compensation */
  if(p.hl_detail_recovery > 0.0f && dev_base != NULL)
  {
    const int pos_lin = y * width + x;
    const float base = dev_base[pos_lin];
    const float detail = lum_orig - base;
    const float lum_tm = p.luma_coeff[0] * rgb_out.x + p.luma_coeff[1] * rgb_out.y + p.luma_coeff[2] * rgb_out.z;
    if(lum_tm > 1e-6f && lum_orig > 1e-6f)
    {
      const float gain = lum_tm / lum_orig;
      const float lum_final = lum_tm + detail * p.hl_detail_recovery * gain;
      if(lum_final > 1e-6f)
      {
        const float scale = fmax(lum_final / lum_tm, 0.25f);
        rgb_out *= scale;
      }
    }
  }

  float4 out_pixel = (float4)(rgb_out.x, rgb_out.y, rgb_out.z, pixel.w);
  write_imagef(output, pos, out_pixel);
}

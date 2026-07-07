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
  float gamma;
  float gamma_power;
  float vibrance;
  float chromatic_boost;
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
  float hl_power;
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

  /* Highlight boost intégré à la courbe */
  if(p->hl_power != 1.0f)
  {
    const float hl_w = y_tm * y_tm;
    y_tm = y_tm + (1.0f - y_tm) * (p->hl_power - 1.0f) * hl_w;
    y_tm = fmin(y_tm, 1.0f);
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

/* Hue-preserving gamut compression: blend out-of-[0,1] channels toward the
 * pixel's own luma instead of a fixed white point. Mirrors st_gamut_compress()
 * in 3dcf.c 1:1. */
static inline float3 st_gamut_compress(float3 rgb, const float luma_coeff[3])
{
  const float luma = luma_coeff[0] * rgb.x + luma_coeff[1] * rgb.y + luma_coeff[2] * rgb.z;
  const float anchor = (luma > 1e-4f) ? luma : 1.0f;

  float t = 0.0f;
  if(rgb.x < 0.0f)     { const float d = anchor - rgb.x;     const float ti = (d > 1e-6f) ? (-rgb.x) / d : 1.0f; if(ti > t) t = ti; }
  else if(rgb.x > 1.0f){ const float d = rgb.x - anchor;     const float ti = (d > 1e-6f) ? (rgb.x - 1.0f) / d : 1.0f; if(ti > t) t = ti; }
  if(rgb.y < 0.0f)     { const float d = anchor - rgb.y;     const float ti = (d > 1e-6f) ? (-rgb.y) / d : 1.0f; if(ti > t) t = ti; }
  else if(rgb.y > 1.0f){ const float d = rgb.y - anchor;     const float ti = (d > 1e-6f) ? (rgb.y - 1.0f) / d : 1.0f; if(ti > t) t = ti; }
  if(rgb.z < 0.0f)     { const float d = anchor - rgb.z;     const float ti = (d > 1e-6f) ? (-rgb.z) / d : 1.0f; if(ti > t) t = ti; }
  else if(rgb.z > 1.0f){ const float d = rgb.z - anchor;     const float ti = (d > 1e-6f) ? (rgb.z - 1.0f) / d : 1.0f; if(ti > t) t = ti; }
  if(t <= 0.0f) return rgb;
  const float blend = fmin(t, 1.0f);
  return (1.0f - blend) * rgb + blend * (float3)(anchor, anchor, anchor);
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

/* ====================================================================
 * ACES 2.0-derived per-hue chroma shape (RC1-style static tables)
 * Mirrors the CPU definitions in 3dcf.c 1:1.
 * ==================================================================== */

__constant float st_gamut_reach[360] =
{
   215.33465f,    216.65399f,    217.88428f,    219.05153f,    220.14357f,    221.16588f,
   222.12170f,    223.01161f,    223.83662f,    224.59834f,    225.29921f,    225.93167f,
   226.51031f,    227.04002f,    227.52505f,    227.96997f,    228.37961f,    228.75839f,
   229.10887f,    229.44225f,    223.68723f,    214.16800f,    205.47426f,    197.50737f,
   190.18221f,    183.42755f,    177.40927f,    171.71423f,    166.01921f,    161.26114f,
   156.51205f,    152.09988f,    148.09119f,    144.08249f,    140.61681f,    137.19861f,
   133.96431f,    131.02483f,    128.08536f,    125.48020f,    122.93465f,    120.49192f,
   118.27464f,    116.05737f,    114.05550f,    112.11517f,    110.23206f,    108.52786f,
   106.82366f,    105.26328f,    103.76254f,    102.29211f,    100.96827f,     99.64443f,
    98.41933f,     97.25076f,     96.09600f,     95.06489f,     94.03378f,     93.07201f,
    92.16363f,     91.25852f,     90.46065f,     89.66278f,     88.91450f,     88.21698f,
    87.51945f,     86.91000f,     86.30434f,     85.73459f,     85.21377f,     84.69295f,
    84.24227f,     83.80045f,     83.38480f,     83.01721f,     82.64962f,     82.33952f,
    82.04231f,     81.76405f,     81.53416f,     81.30428f,     81.12289f,     80.95801f,
    80.83521f,     80.72168f,     80.65045f,     80.59034f,     80.57131f,     80.56710f,
    80.60517f,     80.65601f,     80.75144f,     80.85907f,     81.01254f,     81.17904f,
    81.39567f,     81.62642f,     81.90923f,     82.20887f,     82.55980f,     82.93072f,
    83.35113f,     83.79470f,     84.29233f,     84.81635f,     85.39422f,     86.00119f,
    86.66454f,     87.35604f,     88.10696f,     88.88807f,     89.72918f,     90.60318f,
    91.53980f,     92.51119f,     93.54398f,     94.61283f,     95.74192f,     96.90921f,
    98.13804f,     99.40525f,    100.73635f,    102.10578f,    103.54021f,    105.01234f,
   106.55258f,    108.13228f,    109.78187f,    111.47143f,    113.23616f,    115.04231f,
   116.92601f,    118.85246f,    120.86163f,    122.91274f,    125.04886f,    127.22766f,
   129.49422f,    131.80248f,    134.20437f,    136.64542f,    139.18477f,    141.75740f,
   144.42843f,    147.11901f,    149.91091f,    152.71301f,    155.62236f,    158.53839f,
   161.56854f,    164.59702f,    167.74177f,    170.87844f,    174.13757f,    177.38217f,
   180.75291f,    184.09992f,    187.58020f,    191.02417f,    194.61319f,    198.14956f,
   201.86339f,    205.49933f,    209.35075f,    213.10937f,    217.10868f,    220.98853f,
   225.15117f,    229.14913f,    233.51938f,    237.72111f,    242.30969f,    246.71441f,
   251.55396f,    256.19424f,    261.32180f,    266.20905f,    271.65866f,    276.83450f,
   282.62812f,    288.11413f,    294.30017f,    300.15022f,    306.75516f,    313.00060f,
   320.03412f,    326.65935f,    334.14812f,    341.20554f,    349.15821f,    356.62305f,
   365.05924f,    373.00500f,    381.98756f,    390.45261f,    400.03007f,    409.09923f,
   419.36868f,    429.09772f,    440.10982f,    450.60016f,    462.44890f,    473.75378f,
   486.51340f,    498.72287f,    512.51868f,    525.71943f,    540.63986f,    554.90985f,
   570.99998f,    586.35636f,    603.68996f,    620.20696f,    638.99757f,    656.90554f,
   677.15919f,    696.34557f,    718.06863f,    738.60920f,    761.87958f,    783.85566f,
   808.71603f,    832.03886f,    858.63238f,    883.55051f,    911.94378f,    938.53107f,
   969.00048f,    997.36899f,   1029.94947f,   1060.28584f,   1094.99923f,   1127.44384f,
  1164.36269f,   1198.98879f,   1237.19312f,   1273.05128f,   1313.63076f,   1351.65816f,
  1394.65221f,   1434.90942f,   1479.36892f,   1521.29923f,   1568.31051f,   1611.98419f,
  1660.62312f,   1705.85267f,   1756.34372f,   1803.03751f,   1855.55009f,   1903.87186f,
  1958.61575f,   2008.99678f,   2065.34081f,   2116.53174f,   2174.75144f,   2227.24046f,
  2286.71107f,   2334.72467f,   2382.30107f,   2412.86075f,   2440.19453f,   2454.06478f,
  2459.10001f,   2451.71571f,   2434.68372f,   2404.68880f,   2367.15629f,   2312.62957f,
  2255.38400f,   2181.09339f,   2107.34164f,   2016.32946f,   1927.02893f,   1817.55349f,
  1712.68006f,   1587.07577f,   1470.86887f,   1330.42737f,   1206.14536f,   1053.68427f,
   932.34317f,    788.29193f,    696.24225f,    611.53390f,    544.99676f,    492.60999f,
   450.65358f,    416.32756f,    387.83403f,    363.87664f,    343.46013f,    325.87004f,
   310.59306f,    297.22935f,    285.46049f,    275.03443f,    265.75110f,    257.45036f,
   250.00035f,    243.29181f,    237.23430f,    231.75173f,    226.77956f,    222.26262f,
   218.15373f,    214.41231f,    210.99364f,    207.86736f,    204.98974f,    202.32907f,
   199.85763f,    197.55163f,    195.39062f,    193.35700f,    191.43587f,    189.61455f,
   187.88248f,    186.23068f,    184.65166f,    183.13911f,    181.68774f,    180.29310f,
   178.95149f,    177.65981f,    176.41537f,    175.21587f,    174.05930f,    172.94393f,
   171.86825f,    170.83094f,    169.83086f,    168.86699f,    167.93843f,    167.04441f,
   166.18422f,    165.35726f,    164.56299f,    163.80093f,    163.07064f,    162.37175f,
};

static inline float st_reach_from_table(float h)
{
  if(!isfinite(h)) return 0.0f;
  float hw = fmod(h, 360.0f);
  if(hw < 0.0f) hw += 360.0f;
  int i0 = (int)hw;
  int i1 = (i0 + 1) % 360;
  const float t = hw - (float)i0;
  return st_gamut_reach[i0] + t * (st_gamut_reach[i1] - st_gamut_reach[i0]);
}

static inline float st_chroma_norm(float h)
{
  const float hr = h * (M_PI_F / 180.0f);
  const float a = cos(hr);
  const float b = sin(hr);
  const float a2 = a * a - b * b;
  const float b2 = 2.0f * a * b;
  const float a3 = 4.0f * a * a * a - 3.0f * a;
  const float b3 = 3.0f * b - 4.0f * b * b * b;
  const float m = 11.34072f * a + 16.46899f * a2 + 7.88380f * a3
                + 14.66441f * b - 6.37224f * b2 + 9.19364f * b3
                + 77.12896f;
  return m;
}

#define ST_GAMUT_SHAPE_REF  2.090563f

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
    if(angle_deg >= 360.0f) angle_deg -= 360.0f;
    int bin = (int)angle_deg;
    int next = (bin + 1) % 360;
    float frac = angle_deg - (float)bin;
    float max_dist = spectral_boundary[bin] + frac * (spectral_boundary[next] - spectral_boundary[bin]);
    const float target_dist = max_dist * 0.92f;

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

  /* Existing knee — modulated by hue-dependent shape factor */
  {
    const float angle = atan2(dz, dx);
    float angle_deg = angle * (180.0f / M_PI_F);
    if(angle_deg < 0.0f) angle_deg += 360.0f;
    if(angle_deg >= 360.0f) angle_deg -= 360.0f;
    const float shape = st_reach_from_table(angle_deg) / fmax(st_chroma_norm(angle_deg), 1e-6f);
    const float shape_norm = fmax(shape / ST_GAMUT_SHAPE_REF, 0.0f);
    const float knee_mod = knee * sqrt(shape_norm);

  if(chroma_sq > knee_mod * knee_mod)
  {
    const float chroma = sqrt(chroma_sq);
    const float excess = chroma - knee_mod;
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

  /* Step 4: gamma */
  if(p->gamma != 0.0f)
  {
    float y_lvl = fmin(fmax(y_tm, 0.0f), 1.0f);
    y_lvl = pow(y_lvl, p->gamma_power);
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

    /* Progressive Abney hue rotation — independent of hl_desat, only on
     * highlight luminance exceeding the threshold. */
    if(p->hl_rotation != 0.0f && isfinite(y_exposed))
    {
      const float wr = fmin(st_desat_weight(y_exposed, 1.0f, p->hl_desat_threshold), 1.0f);
      if(wr > 0.0f)
      {
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
    }

    const float w = st_desat_weight(y_exposed, p->hl_desat, p->hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float maxc_pre = fmax(fmax(rgb.x, rgb.y), rgb.z);
      const float minc_pre = fmin(fmin(rgb.x, rgb.y), rgb.z);
      const float sat_pre = (maxc_pre > 0.0f) ? (maxc_pre - minc_pre) / maxc_pre : 0.0f;
      const float ss_pre = (sat_pre * sat_pre) / (sat_pre * sat_pre + (1.0f - sat_pre) * (1.0f - sat_pre) + 1e-6f);

      /* Vibrance négative : désature plus les pixels saturés, activée par hl_desat et hl_hue_shift */
      float w_final = w;
      if(p->hl_desat > 0.0f || p->hl_rotation != 0.0f)
      {
        if(p->hl_desat > 0.0f)
        {
          const float vib_neg = p->hl_desat * ss_pre * 0.35f;
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

      /* Blend toward white with linear ramp */
      const float ts = fmin(w_final, 1.0f);
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

  /* Step 9b: chromatic contrast — luminance-adaptive mid-tone saturation boost */
  if(p->chromatic_boost > 0.0f)
  {
    const float lc0 = p->luma_coeff[0], lc1 = p->luma_coeff[1], lc2 = p->luma_coeff[2];
    const float luma = lc0 * rgb.x + lc1 * rgb.y + lc2 * rgb.z;
    const float maxc = fmax(fmax(rgb.x, rgb.y), rgb.z);
    const float minc = fmin(fmin(rgb.x, rgb.y), rgb.z);
    const float sat_m = maxc - minc;
    const float level = fmax(maxc, fmax(fabs(minc), fabs(luma)));
    const float sat_norm = (level > 0.0f) ? sat_m / level : 0.0f;
    const float y_mid = 0.18f, sigma = 1.85f;
    const float log_rel = log2(fmax(y_abs / y_mid, 1e-10f));
    const float w_gauss = exp(-(log_rel * log_rel) / (2.0f * sigma * sigma));
    const float w_mid = (log_rel <= 0.0f) ? 1.0f
                       : 1.328f * w_gauss - 0.328f;
    const float pp = 1.0f - fmin(sat_norm, 1.0f);
    /* Hue modulation: −10% jaunes (60°), +10% bleus (240°) */
    float hue_deg = 0.0f;
    if(sat_m > 0.0f)
    {
      if(maxc == rgb.x)
        hue_deg = 60.0f * fmod((rgb.y - rgb.z) / sat_m, 6.0f);
      else if(maxc == rgb.y)
        hue_deg = 60.0f * ((rgb.z - rgb.x) / sat_m + 2.0f);
      else
        hue_deg = 60.0f * ((rgb.x - rgb.y) / sat_m + 4.0f);
      if(hue_deg < 0.0f) hue_deg += 360.0f;
    }
    const float hue_mod = 1.0f + 0.1f * cos((hue_deg - 60.0f) * (M_PI_F / 180.0f));
    const float gain = 1.0f + p->chromatic_boost * w_mid * (pp * pp) * hue_mod;
    rgb.x = luma + gain * (rgb.x - luma);
    rgb.y = luma + gain * (rgb.y - luma);
    rgb.z = luma + gain * (rgb.z - luma);
  }

  /* Step 10: gamut compression */
  rgb = st_gamut_compress(rgb, p->luma_coeff);

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

  const float lum = (p.luma_coeff[0] * rgb.x + p.luma_coeff[1] * rgb.y + p.luma_coeff[2] * rgb.z) * p.exposure_factor;
  dev_lum[y * width + x] = lum;

  write_imagef(output_sanitized, pos, (float4)(rgb.x * p.exposure_factor, rgb.y * p.exposure_factor, rgb.z * p.exposure_factor, pixel.w));
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
    lum_orig = (p.luma_coeff[0] * rgb_in.x + p.luma_coeff[1] * rgb_in.y + p.luma_coeff[2] * rgb_in.z) * p.exposure_factor;

  /* Pre-pipeline safety net: sigmoid rolloff toward luma-gray.
   * Track the desaturation factor so the detail recovery below can
   * attenuate proportionally — a pixel that was heavily desaturated
   * (e.g. out-of-gamut blue → white) would otherwise receive a huge
   * gain boost from the pre-desat luminance vs post-tm luminance
   * mismatch, creating a halo outside the object. */
  float hl_desat_factor = 0.0f;
  {
    const float lum = fmax(fmax(rgb_in.x, rgb_in.y), rgb_in.z);
    const float w = st_desat_weight(lum * p.exposure_factor, p.hl_desat, p.hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float t = fmin(w, 1.0f);
      const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
      hl_desat_factor = ts;
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
    float detail = lum_orig - base;
    {
      const float t = fmin(p.hl_detail_recovery, 1.0f);
      const float detail_frac = 0.10f * (1.0f - t * 0.50f);
      const float bsteep_frac = 0.25f * (1.0f - t * 0.40f);
      const float dl = fmax(lum_orig * detail_frac, 1e-6f);
      const float da = fabs(detail);
      if(da > dl)
      {
        const float excess = da - dl;
        const float bsteep = fmax(dl * bsteep_frac, 0.001f);
        const float compression = excess / (excess + bsteep);
        const float clamped_abs = dl + compression * bsteep;
        detail = copysign(clamped_abs, detail);
      }
    }
    const float lum_tm = p.luma_coeff[0] * rgb_out.x + p.luma_coeff[1] * rgb_out.y + p.luma_coeff[2] * rgb_out.z;
    if(lum_tm > 1e-6f && lum_orig > 1e-6f)
    {
      const float gain = lum_tm / lum_orig;
      const float lum_final = lum_tm + detail * p.hl_detail_recovery * (1.0f - hl_desat_factor) * gain;
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

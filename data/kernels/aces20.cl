/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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
    ACES 2.0 CAM DRT Reference Rendering — OpenCL Kernel

    Full ACES 2.0 CAM DRT pipeline:
      pipe_RGB -> AP1 -> XYZ (D60)
      -> x100 to absolute nits -> CAT16 JMh
      -> Tonemap & compress in JMh:
          J -> Y -> tonescale(Y) -> J' (display J)
          M <- chroma_compress(M, J', orig_J)
      -> Gamut compression (JMh space)
      -> XYZ(D60) -> /100 -> AP1 (D60)
      -> pipe_RGB

    Must match src/iop/aces20.c exactly (byte-for-byte CL params struct,
    and numerically identical per-pixel math).
    ---------------------------------------------------------------------------
*/

/* ====================================================================
 * CAT16 CAM — Constant Matrices
 * ==================================================================== */

__constant float dt_ac_ap1_to_xyz[9] =
{
  0.6624541811f, 0.1340042065f, 0.1561876870f,
  0.2722287168f, 0.6740817658f, 0.0536895174f,
 -0.0055746495f, 0.0040607335f, 1.0103391003f
};

__constant float dt_ac_xyz_to_ap1[9] =
{
  1.6410233797f, -0.3248032942f, -0.2364246952f,
 -0.6636628587f,  1.6153315917f,  0.0167563477f,
  0.0117218943f, -0.0082844420f,  0.9883948585f
};

__constant float dt_ac_m16[9] =
{
  0.3640744836f,  0.5947008157f,  0.0411012735f,
 -0.2222450987f,  1.0738554823f,  0.1479453361f,
 -0.0020676190f,  0.0488260454f,  0.9503875570f
};

__constant float dt_ac_m16_inv[9] =
{
  2.0512756811f, -1.1400313440f,  0.0887556628f,
  0.4269389763f,  0.7005835278f, -0.1275225041f,
 -0.0174712780f, -0.0384725929f,  1.0589468739f
};

__constant float dt_ac_panlrcm[9] =
{
  460.0f,  451.0f,  288.0f,
  460.0f, -891.0f, -261.0f,
  460.0f, -220.0f, -6300.0f
};

/* CAT16 CAM viewing-condition parameters */
#define DT_AC_LA        100.0f
#define DT_AC_YB         20.0f
#define DT_AC_RA          2.0f
#define DT_AC_BA          0.05f
#define DT_AC_SURR_F      0.9f
#define DT_AC_SURR_C      0.59f
#define DT_AC_SURR_NC     0.9f

#define DT_AC_REF_LUM    100.0f
#define DT_AC_CAM_NL_OFFSET  27.13f
#define DT_AC_CAM_NL_SCALE   400.0f

/* M (colorfulness) scale factor */
#define DT_AC_M_SCALE  (DT_AC_CAM_NL_SCALE * 43.0f * DT_AC_SURR_NC)

/* Gamut compression constants */
#define DT_AC_GAMUT_SMOOTH_CUSPS       0.12f
#define DT_AC_GAMUT_SMOOTH_M           0.27f
#define DT_AC_GAMUT_CUSP_MID_BLEND     1.3f
#define DT_AC_GAMUT_FOCUS_GAIN_BLEND   0.3f
#define DT_AC_GAMUT_FOCUS_DISTANCE     1.35f
#define DT_AC_GAMUT_FOCUS_DIST_SCALING 1.75f
#define DT_AC_GAMUT_COMPRESSION_THR    0.75f
#define DT_AC_GAMUT_TABLE_SIZE         362
#define DT_AC_TABLE_SIZE                360
#define DT_AC_BASE_INDEX                  1

/* ====================================================================
 * CL Params Struct
 * ==================================================================== */

typedef struct
{
  float fwd_matrix[9];
  float inv_matrix[9];
  float exposure_factor;
  float _pad_gamut_strength;
  float _pad_gamut_knee;
  float f_l_n;
  float a_w;
  float z;
  float cz;
  float inv_cz;
  float d_rgb[3];
  float a_w_j;
  float ssts_s_2;
  float ssts_m_2;
  float ssts_g;
  float ssts_t_1;
  float ssts_n_r;
  float ssts_n;
  float model_gamma_inv;
  float chroma_compress_scale;
  float cc_sat;
  float cc_sat_thr;
  float cc_compr;
  float limit_j_max;

  /* Gamut compression */
  float mid_J;
  float focus_dist;
  float lower_hull_gamma_inv;
  float table_reach_m[DT_AC_GAMUT_TABLE_SIZE];
  float table_hues[DT_AC_GAMUT_TABLE_SIZE];
  float table_cusp_j[DT_AC_GAMUT_TABLE_SIZE];
  float table_cusp_m[DT_AC_GAMUT_TABLE_SIZE];
  float table_upper_hull_gamma[DT_AC_GAMUT_TABLE_SIZE];
  int   hue_search_min;
  int   hue_search_max;
  int   sdr_output_clip;          /* repurposed from _pad[2] — same struct size */
  int   _pad;
} dt_ac_cl_params_t;

/* ====================================================================
 * Helpers
 * ==================================================================== */

#define _mat_apply(out, M, in) do { \
  (out)[0] = (M)[0] * (in)[0] + (M)[1] * (in)[1] + (M)[2] * (in)[2]; \
  (out)[1] = (M)[3] * (in)[0] + (M)[4] * (in)[1] + (M)[5] * (in)[2]; \
  (out)[2] = (M)[6] * (in)[0] + (M)[7] * (in)[1] + (M)[8] * (in)[2]; \
} while(0)

static inline float ssts_fwd(float x, float s_2, float m_2,
                              float g, float t_1, float n_r)
{
  if(x <= 0.0f) return 0.0f;
  const float f = m_2 * pow(x / (x + s_2), g);
  const float h = (f * f) / (f + t_1);
  return h * n_r;
}

static inline float gamma_toe(float x, float limit, float k1, float k2,
                               int inverse)
{
  if(x > limit) return x;
  k2 = fmax(k2, 0.001f);
  k1 = sqrt(k1 * k1 + k2 * k2);
  const float k3 = (limit + k1) / (limit + k2);
  if(!inverse)
    return 0.5f * (k3 * x - k1 + sqrt((k3 * x - k1) * (k3 * x - k1)
                                       + 4.0f * k2 * k3 * x));
  else
    return (x * x + k1 * x) / (k3 * (x + k2));
}

static inline float chroma_norm(float h)
{
  const float hr = h * (float)(M_PI_F / 180.0f);
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

/* ====================================================================
 * Post-Adaptation Cone Response Compression (NLC)
 * ==================================================================== */

static inline float nlc_fwd_single(float v)
{
  const float abs_v = fabs(v);
  if(abs_v < 1e-12f) return 0.0f;
  const float fl = pow(abs_v, 0.42f);
  return (v >= 0.0f ? 1.0f : -1.0f) * fl / (DT_AC_CAM_NL_OFFSET + fl);
}

static inline float nlc_inv_single(float v)
{
  const float abs_v = min(fabs(v), 0.99f);
  if(abs_v < 1e-12f) return 0.0f;
  const float fl = (DT_AC_CAM_NL_OFFSET * abs_v) / (1.0f - abs_v);
  return (v >= 0.0f ? 1.0f : -1.0f) * pow(fl, 1.0f / 0.42f);
}

static inline void nlc_fwd(__private const float rgb[3],
                            __private float rgb_a[3])
{
  for(int c = 0; c < 3; c++)
    rgb_a[c] = nlc_fwd_single(rgb[c]);
}

static inline void nlc_inv(__private const float rgb_a[3],
                            __private float rgb[3])
{
  for(int c = 0; c < 3; c++)
    rgb[c] = nlc_inv_single(rgb_a[c]);
}

/* ====================================================================
 * CAT16 CAM — XYZ -> JMh
 * ==================================================================== */

static inline void xyz_to_jmh(__private const float xyz[3],
                               __constant const float d_rgb[3],
                               float a_w, float z,
                               __private float jmh[3])
{
  float rgb[3], rgb_c[3], rgb_a[3];
  _mat_apply(rgb, dt_ac_m16, xyz);

  for(int c = 0; c < 3; c++)
    rgb_c[c] = d_rgb[c] * rgb[c];

  nlc_fwd(rgb_c, rgb_a);

  const float a_op = rgb_a[0] - 12.0f * rgb_a[1] / 11.0f + rgb_a[2] / 11.0f;
  const float b_op = (rgb_a[0] + rgb_a[1] - 2.0f * rgb_a[2]) / 9.0f;

  float h = atan2(b_op, a_op) * (180.0f / M_PI_F);
  if(h < 0.0f) h += 360.0f;

  const float A = DT_AC_RA * rgb_a[0] + rgb_a[1] + DT_AC_BA * rgb_a[2];
  const float j = 100.0f * pow(fmax(A, 0.0f) / a_w, DT_AC_SURR_C * z);
  const float m = DT_AC_M_SCALE * sqrt(a_op * a_op + b_op * b_op);

  jmh[0] = (j > 0.0f) ? j : 0.0f;
  jmh[1] = m;
  jmh[2] = h;
}

/* ====================================================================
 * CAT16 CAM — JMh -> XYZ
 * ==================================================================== */

static inline void jmh_to_xyz(__private const float jmh[3],
                               __constant const float d_rgb[3],
                               float a_w, float z,
                               __private float xyz[3])
{
  const float j = fmax(jmh[0], 0.0f);
  const float m = jmh[1];
  const float hr = jmh[2] * (M_PI_F / 180.0f);

  const float A = a_w * pow(fmax(j, 1e-12f) / 100.0f,
                             1.0f / (DT_AC_SURR_C * z));
  const float gamma_v = m / DT_AC_M_SCALE;
  const float a_op = gamma_v * cos(hr);
  const float b_op = gamma_v * sin(hr);

  float p_in[3] = { A, a_op, b_op };
  float rgb_a[3];
  _mat_apply(rgb_a, dt_ac_panlrcm, p_in);
  for(int c = 0; c < 3; c++)
    rgb_a[c] /= 1403.0f;

  float rgb_c[3];
  nlc_inv(rgb_a, rgb_c);

  float rgb[3];
  for(int c = 0; c < 3; c++)
    rgb[c] = rgb_c[c] / fmax(d_rgb[c], 1e-12f);

  _mat_apply(xyz, dt_ac_m16_inv, rgb);
}

/* ====================================================================
 * CAT16 CAM — Lightness J <-> Luminance Y
 * ==================================================================== */

static inline float y_to_j(float y, float f_l_n, float a_w_j, float cz)
{
  if(y <= 0.0f) return 0.0f;
  const float ra = nlc_fwd_single(fabs(y) * f_l_n);
  const float a = ra / a_w_j;
  return 100.0f * pow(fmax(a, 0.0f), cz);
}

static inline float j_to_y(float j, float f_l_n, float a_w_j, float inv_cz)
{
  if(j <= 0.0f) return 0.0f;
  const float a = pow(j / 100.0f, inv_cz);
  const float ra = a_w_j * a;
  return nlc_inv_single(min(ra, 0.99f)) / f_l_n;
}

/* Forward declaration (defined in gamut compression section) */
static inline float reach_m_from_table(float h,
    __constant const dt_ac_cl_params_t * restrict p);

/* ====================================================================
 * Chroma Compression
 * ==================================================================== */

static inline void chroma_compress(__private float jmh[3], float orig_j,
                                    __constant const dt_ac_cl_params_t * restrict p)
{
  const float j = fmax(jmh[0], 1e-12f);
  float m = jmh[1];
  const float h = jmh[2];

  if(m <= 0.0f) return;

  m *= pow(j / fmax(orig_j, 1e-12f), p->model_gamma_inv);

  const float m_norm = chroma_norm(h) * p->chroma_compress_scale;
  m /= m_norm;

  const float n_j = j / p->limit_j_max;
  const float sn_j = fmax(0.0f, 1.0f - n_j);
  const float limit = pow(n_j, p->model_gamma_inv)
                      * reach_m_from_table(h, p) / m_norm;

  if(limit <= 0.0f) return;

  m = limit - gamma_toe(limit - m, limit - 0.001f,
                         sn_j * p->cc_sat,
                         sqrt(n_j * n_j + p->cc_sat_thr), 0);
  m = gamma_toe(m, limit, n_j * p->cc_compr, sn_j, 0);

  jmh[1] = fmax(m * m_norm, 0.0f);
}

/* ====================================================================
 * Forward Tonemap & Compress (inside JMh space)
 * ==================================================================== */

static inline void tonemap_and_compress_fwd(__private float jmh[3],
    __constant const dt_ac_cl_params_t * restrict p)
{
  const float orig_j = jmh[0];

  const float linear = j_to_y(orig_j, p->f_l_n, p->a_w_j, p->inv_cz)
                       / DT_AC_REF_LUM;

  const float tonemapped_y = ssts_fwd(linear, p->ssts_s_2, p->ssts_m_2,
                                       p->ssts_g, p->ssts_t_1, p->ssts_n_r);

  jmh[0] = y_to_j(tonemapped_y, p->f_l_n, p->a_w_j, p->cz);

  chroma_compress(jmh, orig_j, p);
}

/* ====================================================================
 * GAMUT COMPRESSION — Per-Pixel Functions (CL)
 *
 * Mirror of src/iop/aces20.c gamut compression functions.
 * ==================================================================== */

static inline float wrap_hue(float h)
{
  const float hw = h - 360.0f * floor(h / 360.0f);
  return (hw < 0.0f) ? hw + 360.0f : hw;
}

static inline float smin(float a, float b, float k)
{
  if(k <= 0.0f) return fmin(a, b);
  const float h = fmax(k - fabs(a - b), 0.0f) / k;
  return fmin(a, b) - h * h * h * k * (1.0f / 6.0f);
}

static inline float reach_m_from_table(float h,
    __constant const dt_ac_cl_params_t * restrict p)
{
  const float hw = wrap_hue(h);
  const int base = (int)hw;
  const int lo = DT_AC_BASE_INDEX + base;
  const int hi = lo + 1;
  const float t = hw - (float)base;
  return p->table_reach_m[lo] + t * (p->table_reach_m[hi] - p->table_reach_m[lo]);
}

static inline void cusp_from_table(__private float cusp_out[2], float h,
    __constant const dt_ac_cl_params_t * restrict p)
{
  const float hw = wrap_hue(h);
  const int base = (int)hw;
  const int lo = DT_AC_BASE_INDEX + base;
  const int hi = lo + 1;
  const float t = hw - (float)base;
  cusp_out[0] = p->table_cusp_j[lo] + t * (p->table_cusp_j[hi] - p->table_cusp_j[lo]);
  cusp_out[1] = p->table_cusp_m[lo] + t * (p->table_cusp_m[hi] - p->table_cusp_m[lo]);
}

static inline float hue_upper_hull_gamma(float h,
    __constant const dt_ac_cl_params_t * restrict p)
{
  const float hw = wrap_hue(h);
  const int base = (int)hw;
  const int lo = DT_AC_BASE_INDEX + base;
  const int hi = lo + 1;
  const float t = hw - (float)base;
  return p->table_upper_hull_gamma[lo] + t * (p->table_upper_hull_gamma[hi] - p->table_upper_hull_gamma[lo]);
}

static inline float get_focus_gain(float j, float analytical_threshold,
    float limit_j_max, float focus_dist)
{
  float gain = limit_j_max * focus_dist;

  if(j > analytical_threshold)
  {
    const float denom = fmax(limit_j_max - j, 1e-4f);
    const float ratio = (limit_j_max - analytical_threshold) / denom;
    const float gain_adj = log(ratio) * log(ratio) * (0.43429448190325182765f * 0.43429448190325182765f) + 1.0f;
    gain *= gain_adj;
  }

  return gain;
}

static inline float solve_J_intersect(float j, float m,
    float focus_j, float limit_j_max, float slope_gain)
{
  if(m <= 0.0f) return j;

  const float m_scaled = m / slope_gain;
  const float a = m_scaled / focus_j;

  if(j < focus_j)
  {
    const float b = 1.0f - m_scaled;
    const float c = -j;
    const float disc = fmax(b * b - 4.0f * a * c, 0.0f);
    const float root = sqrt(disc);
    return -2.0f * c / (b + root);
  }
  else
  {
    const float b = -(1.0f + m_scaled + limit_j_max * a);
    const float c = limit_j_max * m_scaled + j;
    const float disc = fmax(b * b - 4.0f * a * c, 0.0f);
    const float root = sqrt(disc);
    return -2.0f * c / (b - root);
  }
}

static inline float compression_slope(float intersect_j, float focus_j,
    float limit_j_max, float slope_gain)
{
  float direction_scalar;
  if(intersect_j < focus_j)
    direction_scalar = intersect_j;
  else
    direction_scalar = limit_j_max - intersect_j;
  return direction_scalar * (intersect_j - focus_j) / (focus_j * slope_gain);
}

static inline float smin_scaled(float a, float b, float scale_ref)
{
  return smin(a, b, DT_AC_GAMUT_SMOOTH_CUSPS * scale_ref);
}

static inline float estimate_intersect_M(float j_axis_intersect, float slope,
    float inv_gamma, float j_max, float m_max, float j_intersection_ref)
{
  if(slope >= 0.0f || m_max <= 0.0f) return m_max;
  const float j_ref = fmax(j_intersection_ref, 1e-12f);
  const float normalised_j = j_axis_intersect / j_ref;
  const float inv_gamma_safe = fmax(inv_gamma, 1e-6f);
  const float shifted_intersection = j_ref * pow(fmax(normalised_j, 1e-12f), inv_gamma_safe);
  const float denom = j_max - slope * m_max;
  if(denom <= 0.0f) return m_max;
  const float m_est = shifted_intersection * m_max / denom;
  return fmin(fmax(m_est, 0.0f), m_max);
}

static inline float find_boundary_M(float cusp_j, float cusp_m,
    float limit_j_max, float gamma_top_inv, float gamma_bottom_inv,
    float j_intersect_source, float slope, float j_intersect_cusp)
{
  if(cusp_m <= 0.0f || cusp_j <= 0.0f) return 0.0f;

  const float lower_m = estimate_intersect_M(j_intersect_source, slope,
      gamma_bottom_inv, cusp_j, cusp_m, j_intersect_cusp);

  const float f_intersect_cusp = limit_j_max - j_intersect_cusp;
  const float f_intersect_source = limit_j_max - j_intersect_source;
  const float f_cusp_j = limit_j_max - cusp_j;
  const float upper_m = estimate_intersect_M(f_intersect_source, -slope,
      gamma_top_inv, f_cusp_j, cusp_m, f_intersect_cusp);

  const float m_blend = smin(lower_m, upper_m, DT_AC_GAMUT_SMOOTH_CUSPS * cusp_m);
  return m_blend;
}

static inline float remap_M(float m, float gamut_boundary_m,
    float reach_boundary_m)
{
  if(m <= 0.0f || gamut_boundary_m <= 0.0f || reach_boundary_m <= gamut_boundary_m) return m;

  const float boundary_ratio = gamut_boundary_m / reach_boundary_m;
  const float proportion = fmax(boundary_ratio, DT_AC_GAMUT_COMPRESSION_THR);
  const float threshold = proportion * gamut_boundary_m;

  if(m <= threshold || proportion >= 1.0f) return m;

  const float m_offset = m - threshold;
  const float gamut_offset = gamut_boundary_m - threshold;
  const float reach_offset = reach_boundary_m - threshold;
  const float scale = reach_offset / ((reach_offset / gamut_offset) - 1.0f);
  const float nd = m_offset / scale;

  return threshold + scale * nd / (1.0f + nd);
}

static inline void compress_gamut(__private float jmh[3], float jx,
    __constant const dt_ac_cl_params_t * restrict p)
{
  float j = jmh[0], m = jmh[1], h = jmh[2];
  if(!isfinite(j) || !isfinite(m)) return;
  if(m <= 0.0f || j <= 0.0f) return;

  const float limit_j_max = p->limit_j_max;
  if(limit_j_max <= 0.0f) return;

  float cusp[2];
  cusp_from_table(cusp, h, p);
  const float cusp_j = cusp[0], cusp_m = cusp[1];
  if(!isfinite(cusp_j) || !isfinite(cusp_m) || cusp_m <= 0.0f || cusp_j <= 0.0f) return;

  const float blend_weight = fmin(1.0f, DT_AC_GAMUT_CUSP_MID_BLEND - cusp_j / limit_j_max);
  const float focus_j = cusp_j + blend_weight * (p->mid_J - cusp_j);
  if(!isfinite(focus_j) || focus_j <= 0.0f) return;

  const float analytical_threshold = cusp_j + DT_AC_GAMUT_FOCUS_GAIN_BLEND
                                     * (limit_j_max - cusp_j);

  const float gamma_top_inv = 1.0f / fmax(hue_upper_hull_gamma(h, p), 1e-6f);
  const float gamma_bottom_inv = p->lower_hull_gamma_inv;
  const float slope_gain = get_focus_gain(jx, analytical_threshold, limit_j_max, p->focus_dist);
  if(!isfinite(slope_gain) || slope_gain <= 0.0f) return;

  const float j_intersect_source = solve_J_intersect(j, m, focus_j, limit_j_max, slope_gain);
  const float j_intersect_cusp = solve_J_intersect(cusp_j, cusp_m, focus_j, limit_j_max, slope_gain);
  if(!isfinite(j_intersect_source) || !isfinite(j_intersect_cusp)) return;

  const float slope = compression_slope(j_intersect_source, focus_j, limit_j_max, slope_gain);
  if(!isfinite(slope) || slope >= 0.0f) return;

  const float gamut_boundary_m = find_boundary_M(cusp_j, cusp_m,
      limit_j_max, gamma_top_inv, gamma_bottom_inv,
      j_intersect_source, slope, j_intersect_cusp);
  if(!isfinite(gamut_boundary_m) || gamut_boundary_m <= 0.0f) return;

  const float reach_max_m = reach_m_from_table(h, p);
  if(!isfinite(reach_max_m) || reach_max_m <= 0.0f) return;

  const float reach_boundary_m = estimate_intersect_M(
      j_intersect_source, slope, p->model_gamma_inv,
      limit_j_max, reach_max_m, limit_j_max);
  if(!isfinite(reach_boundary_m)) return;

  const float new_m = remap_M(m, gamut_boundary_m, reach_boundary_m);
  if(!isfinite(new_m)) return;

  jmh[0] = j_intersect_source + new_m * slope;
  jmh[1] = fmax(new_m, 0.0f);
}

static inline void gamut_compress_fwd(__private float jmh[3],
    __constant const dt_ac_cl_params_t * restrict p)
{
  float j = jmh[0], m = jmh[1];
  if(!isfinite(j + m)) return;
  if(j <= 0.0f || m <= 0.0f) return;
  compress_gamut(jmh, j, p);
}

static inline void gamut_compress_inv(__private float jmh[3],
    __constant const dt_ac_cl_params_t * restrict p)
{
  float j = jmh[0], m = jmh[1];
  const float h = jmh[2];
  if(!isfinite(j + m)) return;
  if(j <= 0.0f || m <= 0.0f) return;

  const float limit_j_max = p->limit_j_max;
  float cusp[2];
  cusp_from_table(cusp, h, p);
  const float analytical_threshold = cusp[0]
    + DT_AC_GAMUT_FOCUS_GAIN_BLEND * (limit_j_max - cusp[0]);

  if(j < analytical_threshold)
  {
    compress_gamut(jmh, j, p);
  }
  else
  {
    float jmh_tmp[3] = { j, m, h };
    float jx = j;
    compress_gamut(jmh_tmp, jx, p);
    jx = jmh_tmp[0];
    m = jmh_tmp[1];
    jmh_tmp[0] = jx;
    jmh_tmp[1] = m;
    compress_gamut(jmh_tmp, jx, p);
    jmh[0] = jmh_tmp[0];
    jmh[1] = jmh_tmp[1];
    jmh[2] = h;
  }
}

/* ====================================================================
 * Full ACES 2.0 CAM DRT Pipeline
 * ==================================================================== */

static inline void pipeline_eval(__private const float rgb_in[3],
                                  __private float rgb_out[3],
                                  __constant const dt_ac_cl_params_t * restrict p)
{
  /* Step 1: pipe RGB -> AP1 (D60) */
  float ap1[3];
  _mat_apply(ap1, p->fwd_matrix, rgb_in);

  if(!isfinite(ap1[0] + ap1[1] + ap1[2]) || (ap1[0] + ap1[1] + ap1[2]) <= 0.0f)
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  for(int c = 0; c < 3; c++) ap1[c] = fmax(ap1[c], 0.0f);
  for(int c = 0; c < 3; c++) ap1[c] *= p->exposure_factor;

  /* Step 2: AP1 -> XYZ (scene linear) */
  float xyz[3];
  _mat_apply(xyz, dt_ac_ap1_to_xyz, ap1);

  /* Step 3: x100 to absolute nits for CAM */
  for(int c = 0; c < 3; c++) xyz[c] *= DT_AC_REF_LUM;

  /* Step 4: XYZ -> JMh */
  float jmh[3];
  xyz_to_jmh(xyz, p->d_rgb, p->a_w, p->z, jmh);

  /* Step 5: Tonemap & compress inside JMh */
  tonemap_and_compress_fwd(jmh, p);

  /* Step 6: Gamut compression */
  gamut_compress_fwd(jmh, p);

  /* Step 7: JMh -> XYZ (in nits) */
  float xyz_out[3];
  jmh_to_xyz(jmh, p->d_rgb, p->a_w, p->z, xyz_out);

  /* Step 8: XYZ -> AP1 (back to display-referred) */
  float ap1_out[3];
  _mat_apply(ap1_out, dt_ac_xyz_to_ap1, xyz_out);
  for(int c = 0; c < 3; c++)
    ap1_out[c] = fmax(ap1_out[c], 0.0f) / DT_AC_REF_LUM;

  /* Step 9: AP1 -> pipe RGB */
  float rgb[3];
  _mat_apply(rgb, p->inv_matrix, ap1_out);

  /* Step 10: Hard floor (reference: hardClip = fmax(v, 0.0f)) */
  for(int c = 0; c < 3; c++)
    rgb_out[c] = max(rgb[c], 0.0f);

  /* Optional Step 11 (NOT part of the official reference): clip to display
   * white (code-value 1.0). Off by default — see aces20.c for rationale. */
  if(p->sdr_output_clip)
    for(int c = 0; c < 3; c++)
      rgb_out[c] = min(rgb_out[c], 1.0f);
}

/* ====================================================================
 * Kernel Entry Point
 * ==================================================================== */

__kernel void kernel_aces20(__global const float *restrict in,
                             __global float *restrict out,
                             const int width, const int height,
                              __constant const dt_ac_cl_params_t * restrict params)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  const size_t idx = (size_t)y * width + x;
  const size_t pix = idx * 4;

  float rgb_in[3] = { in[pix], in[pix + 1], in[pix + 2] };

  for(int c = 0; c < 3; c++)
    rgb_in[c] = isfinite(rgb_in[c]) ? fmax(rgb_in[c], 0.0f) : 0.0f;

  float rgb_out[3];
  pipeline_eval(rgb_in, rgb_out, params);

  out[pix]     = rgb_out[0];
  out[pix + 1] = rgb_out[1];
  out[pix + 2] = rgb_out[2];
  out[pix + 3] = in[pix + 3];
}

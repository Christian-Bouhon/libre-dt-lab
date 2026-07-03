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

    Full ACES 2.0 CAM DRT pipeline (Hellwig 2022):
      pipe_RGB → AP1
      → SSTS per-channel on AP1 → display-referred AP1
      → XYZ abs nits (×100) → Hellwig JMh
      → Chroma compression (J unchanged by tone mapping)
      → XYZ(D60) → AP1 (÷100 to display-referred)
      → Gamut compression → pipe_RGB
    ---------------------------------------------------------------------------
*/

/* ====================================================================
 * Hellwig 2022 CAM — Constant Matrices
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

/* Hellwig CAM viewing-condition parameters */
#define DT_AC_LA        100.0f
#define DT_AC_YB         20.0f
#define DT_AC_RA          2.0f
#define DT_AC_BA          0.05f
#define DT_AC_SURR_F      0.9f
#define DT_AC_SURR_C      0.59f
#define DT_AC_SURR_NC     0.9f

#define DT_AC_LIMIT_JMAX 100.0f
#define DT_AC_MODEL_GAMMA 0.8794641436f
#define DT_AC_COMPR        2.4f
#define DT_AC_SAT          1.3f
#define DT_AC_SAT_THR      0.005f
#define DT_AC_CC_SCALE     1.000053f
#define DT_AC_REF_LUM    100.0f

/* AP1 reach table (360 entries) */
__constant float dt_ac_gamut_reach[360] =
{
  166.785f, 168.475f, 170.129f, 171.753f, 173.340f, 174.878f, 176.367f,
  177.802f, 179.181f, 180.505f, 181.763f, 182.965f, 184.106f, 185.187f,
  186.212f, 187.177f, 188.086f, 188.947f, 189.758f, 190.527f, 191.254f,
  191.949f, 192.615f, 193.250f, 193.872f, 194.360f, 187.091f, 180.402f,
  174.237f, 168.524f, 163.226f, 158.301f, 153.711f, 149.426f, 145.416f,
  141.663f, 138.135f, 134.821f, 131.702f, 128.766f, 125.995f, 123.376f,
  120.898f, 118.555f, 116.339f, 114.233f, 112.244f, 110.345f, 108.551f,
  106.842f, 105.219f, 103.680f, 102.209f, 100.818f,  99.487f,  98.224f,
   97.021f,  95.874f,  94.781f,  93.744f,  92.761f,  91.821f,  90.930f,
   90.082f,  89.276f,  88.513f,  87.787f,  87.103f,  86.450f,  85.840f,
   85.260f,  84.711f,  84.198f,  83.716f,  83.264f,  82.843f,  82.452f,
   82.086f,  81.750f,  81.445f,  81.165f,  80.908f,  80.682f,  80.481f,
   80.304f,  80.151f,  80.023f,  79.919f,  79.840f,  79.791f,  79.761f,
   79.755f,  79.773f,  79.816f,  79.889f,  79.980f,  80.096f,  80.243f,
   80.408f,  80.603f,  80.817f,  81.061f,  81.335f,  81.635f,  81.958f,
   82.312f,  82.690f,  83.099f,  83.539f,  84.009f,  84.509f,  85.046f,
   85.614f,  86.212f,  86.847f,  87.518f,  88.226f,  88.977f,  89.764f,
   90.601f,  91.473f,  92.395f,  93.359f,  94.379f,  95.447f,  96.570f,
   97.748f,  98.993f, 100.293f, 101.660f, 103.101f, 104.614f, 106.201f,
  107.880f, 109.637f, 111.493f, 113.446f, 115.503f, 117.670f, 119.965f,
  122.382f, 124.939f, 127.649f, 130.518f, 133.563f, 136.792f, 140.222f,
  143.878f, 141.687f, 138.110f, 134.729f, 131.525f, 128.491f, 125.616f,
  122.888f, 120.300f, 117.841f, 115.497f, 113.269f, 111.151f, 109.131f,
  107.202f, 105.371f, 103.619f, 101.947f, 100.354f,  98.828f,  97.375f,
   95.984f,  94.659f,  93.390f,  92.175f,  91.016f,  89.911f,  88.849f,
   87.836f,  86.871f,  85.950f,  85.065f,  84.222f,  83.417f,  82.654f,
   81.921f,  81.226f,  80.560f,  79.926f,  79.327f,  78.754f,  78.217f,
   77.698f,  77.216f,  76.758f,  76.324f,  75.916f,  75.537f,  75.177f,
   74.847f,  74.536f,  74.249f,  73.987f,  73.743f,  73.523f,  73.328f,
   73.151f,  72.992f,  72.858f,  72.742f,  72.650f,  72.577f,  72.522f,
   72.491f,  72.479f,  72.485f,  72.516f,  72.565f,  72.632f,  72.717f,
   72.827f,  72.961f,  73.108f,  73.279f,  73.474f,  73.688f,  73.926f,
   74.182f,  74.463f,  74.768f,  75.092f,  75.446f,  75.818f,  76.215f,
   76.642f,  77.094f,  77.570f,  78.070f,  78.601f,  79.163f,  79.749f,
   80.371f,  81.024f,  81.708f,  82.422f,  83.173f,  83.960f,  84.784f,
   85.645f,  86.548f,  87.488f,  88.477f,  89.508f,  90.588f,  91.711f,
   92.889f,  94.122f,  95.404f,  96.747f,  98.157f,  99.622f, 101.154f,
  102.759f, 104.437f, 106.195f, 108.032f, 109.949f, 111.963f, 114.069f,
  116.272f, 118.585f, 121.002f, 120.929f, 119.934f, 118.988f, 118.085f,
  117.230f, 116.418f, 115.643f, 114.911f, 114.221f, 113.568f, 112.952f,
  112.372f, 111.823f, 111.316f, 110.840f, 110.394f, 109.985f, 109.607f,
  109.265f, 108.948f, 108.661f, 108.411f, 108.185f, 107.990f, 107.825f,
  107.684f, 107.581f, 107.501f, 107.446f, 107.422f, 107.428f, 107.458f,
  107.520f, 107.611f, 107.727f, 107.874f, 108.044f, 108.246f, 108.472f,
  108.728f, 109.015f, 109.332f, 109.674f, 110.046f, 110.449f, 110.883f,
  111.346f, 111.841f, 112.366f, 112.921f, 113.507f, 114.124f, 114.777f,
  115.460f, 116.180f, 116.931f, 117.719f, 118.536f, 119.397f, 120.288f,
  121.216f, 122.186f, 123.187f, 124.225f, 125.305f, 126.422f, 127.582f,
  128.772f, 130.005f, 131.281f, 132.593f, 133.942f, 135.327f, 136.755f,
  138.220f, 139.722f, 141.254f, 142.822f, 144.421f, 146.057f, 147.711f,
  149.396f, 151.099f, 152.826f, 154.565f, 156.317f, 158.075f, 159.833f,
  161.584f, 163.336f, 165.070f
};

/* ====================================================================
 * CL Params Struct
 * ==================================================================== */

typedef struct
{
  float fwd_matrix[9];
  float inv_matrix[9];
  float exposure_factor;
  float gamut_strength;
  float gamut_knee;
  float f_l;
  float a_w;
  float z;
  float d_rgb[3];
  float ssts_s_2;
  float ssts_m_2;
  float ssts_g;
  float ssts_t_1;
  float ssts_n_r;
  float ssts_n;
  int   _pad[8];
} dt_ac_cl_params_t;

/* ====================================================================
 * Helpers
 * ==================================================================== */

static inline void apply_mat(__private float out[3],
                              __private const float M[9],
                              __private const float in[3])
{
  out[0] = M[0] * in[0] + M[1] * in[1] + M[2] * in[2];
  out[1] = M[3] * in[0] + M[4] * in[1] + M[5] * in[2];
  out[2] = M[6] * in[0] + M[7] * in[1] + M[8] * in[2];
}

static inline float ssts_fwd(float x, float s_2, float m_2,
                              float g, float t_1, float n_r)
{
  if(x <= 0.0f) return 0.0f;
  const float f = m_2 * pow(x / (x + s_2), g);
  const float h = (f * f) / (f + t_1);
  return h * n_r;
}

static inline float gamut_compress_max(float maxc, float strength, float knee)
{
  if(maxc <= 0.0f) return maxc;
  const float excess = fmax(maxc - 1.0f, 0.0f);
  if(excess <= 0.0f) return maxc;
  const float k = fmax(knee, 0.01f);
  const float compressed = (excess * excess) / (excess + k * strength);
  return maxc - excess + compressed;
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

static inline float reach_from_table(float h)
{
  const float hw = h - 360.0f * floor(h / 360.0f);
  const int lo = (int)hw;
  const int hi = (lo < 359) ? lo + 1 : 0;
  const float t = hw - (float)lo;
  return dt_ac_gamut_reach[lo] + t * (dt_ac_gamut_reach[hi] - dt_ac_gamut_reach[lo]);
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
  return m * DT_AC_CC_SCALE;
}

/* ====================================================================
 * Hellwig 2022 CAM — Forward NLC
 * ==================================================================== */

static inline void nlc_fwd(__private const float rgb[3],
                            __private float rgb_a[3], float f_l)
{
  for(int c = 0; c < 3; c++)
  {
    const float abs_rgb = fabs(rgb[c]);
    if(abs_rgb < 1e-12f)
    {
      rgb_a[c] = 0.0f;
      continue;
    }
    const float fl_lms = pow(f_l * abs_rgb / 100.0f, 0.42f);
    const float sign = (rgb[c] >= 0.0f) ? 1.0f : -1.0f;
    rgb_a[c] = sign * (400.0f * fl_lms) / (27.13f + fl_lms);
  }
}

static inline void nlc_inv(__private const float rgb_a[3],
                            __private float rgb[3], float f_l)
{
  for(int c = 0; c < 3; c++)
  {
    const float abs_a = fabs(rgb_a[c]);
    if(abs_a < 1e-12f)
    {
      rgb[c] = 0.0f;
      continue;
    }
    const float sign = (rgb_a[c] >= 0.0f) ? 1.0f : -1.0f;
    rgb[c] = sign * (100.0f / f_l)
             * pow((27.13f * abs_a) / (400.0f - abs_a), 1.0f / 0.42f);
  }
}

/* ====================================================================
 * Hellwig 2022 CAM — XYZ → JMh
 * ==================================================================== */

static inline void xyz_to_jmh(__private const float xyz[3],
                               __private const float d_rgb[3],
                               float f_l, float a_w, float z,
                               __private float jmh[3])
{
  float rgb[3], rgb_c[3], rgb_a[3];
  apply_mat(rgb, dt_ac_m16, xyz);

  for(int c = 0; c < 3; c++)
    rgb_c[c] = d_rgb[c] * rgb[c];

  nlc_fwd(rgb_c, rgb_a, f_l);

  const float a_op = rgb_a[0] - 12.0f * rgb_a[1] / 11.0f + rgb_a[2] / 11.0f;
  const float b_op = (rgb_a[0] + rgb_a[1] - 2.0f * rgb_a[2]) / 9.0f;

  float h = atan2(b_op, a_op) * (180.0f / M_PI_F);
  if(h < 0.0f) h += 360.0f;

  const float A = DT_AC_RA * rgb_a[0] + rgb_a[1] + DT_AC_BA * rgb_a[2];
  const float j = 100.0f * pow(fmax(A, 0.0f) / a_w, DT_AC_SURR_C * z);
  const float m = 43.0f * DT_AC_SURR_NC * sqrt(a_op * a_op + b_op * b_op);

  jmh[0] = (j > 0.0f) ? j : 0.0f;
  jmh[1] = m;
  jmh[2] = h;
}

/* ====================================================================
 * Hellwig 2022 CAM — JMh → XYZ
 * ==================================================================== */

static inline void jmh_to_xyz(__private const float jmh[3],
                               __private const float d_rgb[3],
                               float f_l, float a_w, float z,
                               __private float xyz[3])
{
  const float j = fmax(jmh[0], 0.0f);
  const float m = jmh[1];
  const float hr = jmh[2] * (M_PI_F / 180.0f);

  const float A = a_w * pow(fmax(j, 1e-12f) / 100.0f,
                             1.0f / (DT_AC_SURR_C * z));
  const float gamma_v = m / (43.0f * DT_AC_SURR_NC);
  const float a_op = gamma_v * cos(hr);
  const float b_op = gamma_v * sin(hr);

  float p_in[3] = { A, a_op, b_op };
  float rgb_a[3];
  apply_mat(rgb_a, dt_ac_panlrcm, p_in);
  for(int c = 0; c < 3; c++)
    rgb_a[c] /= 1403.0f;

  float rgb_c[3];
  nlc_inv(rgb_a, rgb_c, f_l);

  float rgb[3];
  for(int c = 0; c < 3; c++)
    rgb[c] = rgb_c[c] / fmax(d_rgb[c], 1e-12f);

  apply_mat(xyz, dt_ac_m16_inv, rgb);
}

/* ====================================================================
 * Chroma Compression
 * ==================================================================== */

static inline void chroma_compress(__private float jmh[3], float orig_j)
{
  const float j = fmax(jmh[0], 1e-12f);
  float m = jmh[1];
  const float h = jmh[2];

  if(m <= 0.0f) return;

  m *= pow(j / fmax(orig_j, 1e-12f), DT_AC_MODEL_GAMMA);

  const float m_norm = chroma_norm(h);
  m /= m_norm;

  const float n_j = j / DT_AC_LIMIT_JMAX;
  const float sn_j = fmax(0.0f, 1.0f - n_j);
  const float limit = pow(n_j, DT_AC_MODEL_GAMMA)
                      * reach_from_table(h) / m_norm;

  if(limit <= 0.0f) return;

  m = limit - gamma_toe(limit - m, limit - 0.001f,
                         sn_j * DT_AC_SAT,
                         sqrt(n_j * n_j + DT_AC_SAT_THR), 0);
  m = gamma_toe(m, limit, n_j * DT_AC_COMPR, sn_j, 0);

  jmh[1] = fmax(m * m_norm, 0.0f);
}

/* ====================================================================
 * Full ACES 2.0 CAM DRT Pipeline
 * ==================================================================== */

static inline void pipeline_eval(__private const float rgb_in[3],
                                  __private float rgb_out[3],
                                  __private const dt_ac_cl_params_t *p)
{
  /* Step 1: pipe RGB → AP1 (D60) */
  float ap1[3];
  apply_mat(ap1, p->fwd_matrix, rgb_in);

  if(!isfinite(ap1[0] + ap1[1] + ap1[2]) || (ap1[0] + ap1[1] + ap1[2]) <= 0.0f)
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  for(int c = 0; c < 3; c++) ap1[c] = fmax(ap1[c], 0.0f);

  /* Step 2: Per-channel SSTS tone mapping on AP1 (ACES 2.0 reference) */
  float ap1_disp[3];
  for(int c = 0; c < 3; c++)
    ap1_disp[c] = ssts_fwd(fmax(ap1[c], 0.0f) * p->exposure_factor,
                            p->ssts_s_2, p->ssts_m_2, p->ssts_g,
                            p->ssts_t_1, p->ssts_n_r)
                  / DT_AC_REF_LUM;

  /* Step 3: Display AP1 → XYZ (absolute nits for CAM) */
  float xyz_abs[3];
  apply_mat(xyz_abs, dt_ac_ap1_to_xyz, ap1_disp);
  for(int c = 0; c < 3; c++)
    xyz_abs[c] *= DT_AC_REF_LUM;

  /* Step 4: XYZ → JMh */
  float jmh[3];
  xyz_to_jmh(xyz_abs, p->d_rgb, p->f_l, p->a_w, p->z, jmh);

  /* Step 5: Chroma compression (J unchanged by tone mapping) */
  chroma_compress(jmh, jmh[0]);

  /* Step 6: JMh → XYZ (in nits) */
  float xyz_out[3];
  jmh_to_xyz(jmh, p->d_rgb, p->f_l, p->a_w, p->z, xyz_out);

  /* Step 7: XYZ → AP1 (back to display-referred) */
  float ap1_out[3];
  apply_mat(ap1_out, dt_ac_xyz_to_ap1, xyz_out);
  for(int c = 0; c < 3; c++)
    ap1_out[c] = fmax(ap1_out[c], 0.0f) / DT_AC_REF_LUM;

  /* Step 10: Gamut compression */
  const float maxc = fmax(ap1_out[0], fmax(ap1_out[1], ap1_out[2]));
  const float new_max = gamut_compress_max(maxc, p->gamut_strength, p->gamut_knee);
  if(maxc > 0.0f && new_max != maxc)
  {
    const float s = new_max / maxc;
    ap1_out[0] *= s;
    ap1_out[1] *= s;
    ap1_out[2] *= s;
  }

  /* Step 11: AP1 → pipe RGB */
  float rgb[3];
  apply_mat(rgb, p->inv_matrix, ap1_out);

  for(int c = 0; c < 3; c++)
    rgb_out[c] = max(rgb[c], 0.0f);
}

/* ====================================================================
 * Kernel Entry Point
 * ==================================================================== */

__kernel void kernel_aces20(__global const float *restrict in,
                             __global float *restrict out,
                             const int width, const int height,
                             __private const dt_ac_cl_params_t params)
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
  pipeline_eval(rgb_in, rgb_out, &params);

  out[pix]     = rgb_out[0];
  out[pix + 1] = rgb_out[1];
  out[pix + 2] = rgb_out[2];
  out[pix + 3] = in[pix + 3];
}

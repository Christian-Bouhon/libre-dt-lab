/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers
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
    ACES 2.0 CAM DRT Reference Rendering Module

    Implements the Academy Color Encoding System (ACES) 2.0 reference rendering
    pipeline using the Hellwig 2022 CAM DRT (colour appearance model display
    rendering transform):

      pipe_RGB → AP1 (D60) → XYZ (D60) → Hellwig JMh
        → SSTS on J (via Y) → Chroma compression (JMh)
        → XYZ (D60) → AP1 (D60)
        → Gamut Compression (AP1) → pipe_RGB

    The input/output colour spaces are determined by the pipe's working profile,
    not hardcoded.  The module always stays in IOP_CS_RGB and converts to/from
    ACES AP0 D60 internally.

    References:
      - ACES 2.0 CAM DRT (Hellwig 2022):
        github.com/nick-shaw/aces-ot-vwg-experiments (hellwig_lib_rc1.h)
      - ACES 2.0 SSTS: aces-core/lib/Lib.Academy.Tonescale.ctl
      - AP0/AP1 matrices: SMPTE ST 2065-1, ACES Specification
    ---------------------------------------------------------------------------
*/

#include "bauhaus/bauhaus.h"
#include "common/colorspaces.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "develop/imageop.h"
#include "develop/tiling.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_aces20_params_t)

/* ====================================================================
 * ACES 2.0 Constants
 * ==================================================================== */

/* ACES AP0 → CIE XYZ (D60) — SMPTE ST 2065-1
 * Row-major: X = row0·RGB, Y = row1·RGB, Z = row2·RGB */
static const float dt_ac_ap0_to_xyz[9] =
{
  0.9525523959f, 0.0000000000f, 0.0000936786f,
  0.3439664498f, 0.7281660966f, -0.0721325464f,
  0.0000000000f, 0.0000000000f, 1.0088251844f
};

static const float dt_ac_xyz_to_ap0[9] =
{
  1.0498110175f,  0.0000000000f, -0.0000974845f,
  -0.4959030231f, 1.3733130458f,  0.0982400361f,
  0.0000000000f,  0.0000000000f,  0.9912520182f
};

/* AP0 → AP1 (D60) */
static const float dt_ac_ap0_to_ap1[9] =
{
  0.8566271538f,  0.0952182420f,  0.0481545656f,
  0.1373189727f,  0.7992412413f,  0.0634398421f,
  -0.0045742169f, 0.0008962094f,  1.0846780501f
};

/* AP1 → AP0 (D60) — exact inverse of AP0→AP1 */
static const float dt_ac_ap1_to_ap0[9] =
{
   1.1898465294f, -0.1417033752f, -0.0445357062f,
  -0.2048412362f,  1.2756640747f, -0.0655161103f,
   0.0051869739f, -0.0016515906f,  0.9217988693f
};

/* CAT D50 → D60 (Bradford) — computed from D50 (0.34567,0.35850) → D60 (0.32168,0.33767) */
static const float dt_ac_cat_d50_to_d60[9] =
{
   0.9676376211f, -0.0168859733f,  0.0442617135f,
  -0.0209659211f,  1.0080178131f,  0.0147818137f,
   0.0085332853f, -0.0141582518f,  1.2297260627f
};

/* CAT D60 → D50 (Bradford) — inverse of D50→D60 */
static const float dt_ac_cat_d60_to_d50[9] =
{
   1.0341387208f,  0.0167979005f, -0.0374238268f,
   0.0216107969f,  0.9922295300f, -0.0127048482f,
  -0.0069272579f,  0.0113073104f,  0.8133026534f
};

/* Luma coefficients for ACES AP1 */
static const float dt_ac_luma_ap1[3] = { 0.272228f, 0.674082f, 0.053690f };

/* ====================================================================
 * Hellwig 2022 CAM (ACES 2.0) Constants
 * ==================================================================== */

/* AP1 D60 → XYZ D60 — from ACES 2.0 reference */
static const float dt_ac_ap1_to_xyz[9] =
{
  0.6624541811f, 0.1340042065f, 0.1561876870f,
  0.2722287168f, 0.6740817658f, 0.0536895174f,
 -0.0055746495f, 0.0040607335f, 1.0103391003f
};

/* XYZ D60 → AP1 D60 */
static const float dt_ac_xyz_to_ap1[9] =
{
  1.6410233797f, -0.3248032942f, -0.2364246952f,
 -0.6636628587f,  1.6153315917f,  0.0167563477f,
  0.0117218943f, -0.0082844420f,  0.9883948585f
};

/* Hellwig CAM: XYZ → sharpened LMS (MATRIX_16) */
static const float dt_ac_m16[9] =
{
  0.3640744836f,  0.5947008157f,  0.0411012735f,
 -0.2222450987f,  1.0738554823f,  0.1479453361f,
 -0.0020676190f,  0.0488260454f,  0.9503875570f
};

/* Hellwig CAM: sharpened LMS → XYZ (MATRIX_16 inverse) */
static const float dt_ac_m16_inv[9] =
{
  2.0512756811f, -1.1400313440f,  0.0887556628f,
  0.4269389763f,  0.7005835278f, -0.1275225041f,
 -0.0174712780f, -0.0384725929f,  1.0589468739f
};

/* Inverse CAM: opponent (P_p_2, a, b) → RGB_a — divided by 1403 after multiply */
static const float dt_ac_panlrcm[9] =
{
  460.0f,  451.0f,  288.0f,
  460.0f, -891.0f, -261.0f,
  460.0f, -220.0f, -6300.0f
};

/* Hellwig CAM viewing-condition parameters */
#define DT_AC_LA     100.0f
#define DT_AC_YB      20.0f
#define DT_AC_RA       2.0f
#define DT_AC_BA       0.05f
#define DT_AC_SURR_F   0.9f
#define DT_AC_SURR_C   0.59f
#define DT_AC_SURR_NC  0.9f

/* ACES D60 white point in XYZ (Y=100) */
static const float dt_ac_aces_white_xyz[3] = { 95.2646074570f, 100.0f, 100.8825184352f };

/* Chroma compression constants */
#define DT_AC_LIMIT_JMAX  100.0f
#define DT_AC_MODEL_GAMMA  0.8794641436f
#define DT_AC_COMPR         2.4f
#define DT_AC_SAT           1.3f
#define DT_AC_SAT_THR       0.005f
#define DT_AC_CC_SCALE      1.000053f
#define DT_AC_REF_LUM     100.0f

/* ====================================================================
 * Type Definitions
 * ==================================================================== */

typedef enum dt_iop_aces20_surround_t
{
  DT_AC_SURROUND_DARK = 0,   // $DESCRIPTION: "dark (cinema)"
  DT_AC_SURROUND_DIM,        // $DESCRIPTION: "dim (home theater)"
  DT_AC_SURROUND_AVERAGE,    // $DESCRIPTION: "average (monitor)"
} dt_iop_aces20_surround_t;

/* SSTS (ACES 2.0 Single-Stage Tone Scale) parameters */
typedef struct
{
  double s_2, m_2, g, t_1, n_r, n;
} dt_ac_ssts_params_t;

/* Precomputed pipeline context */
typedef struct
{
  dt_ac_ssts_params_t ssts;

  /* Combined forward matrix: working_RGB → AP1 (D60)
   *   forward = AP0→AP1 × XYZ→AP0 × CAT(D50→D60) × matrix_in(working→XYZ(D50)) */
  float fwd_matrix[9];

  /* Combined inverse matrix: AP1(D60) → working_RGB
   *   inverse = matrix_out(XYZ(D50)→working) × CAT(D60→D50) × AP0→XYZ × AP1→AP0 */
  float inv_matrix[9];

  float luma_coeff[3];      /* AP1 luma */
  float exposure_factor;
  float gamut_strength;
  float gamut_knee;
  int   surround_idx;

  /* Hellwig CAM precomputed values (constant for ACES D60 white) */
  float f_l;
  float a_w;
  float z;
  float d_rgb[3];           /* Chromatic adaptation D_RGB for D60 white */
} dt_ac_context_t;

/* Module parameters */
typedef struct dt_iop_aces20_params_t
{
  float peak_luminance;                // $MIN: 100 $MAX: 4000 $DEFAULT: 200 $STEP: 10 $DESCRIPTION: "peak luminance (nits)"
  dt_iop_aces20_surround_t surround;   // $DEFAULT: DT_AC_SURROUND_DIM $DESCRIPTION: "surround"
  float exposure_ev;                   // $MIN: -5 $MAX: 5 $DEFAULT: 0 $STEP: 0.05 $DESCRIPTION: "exposure (EV)"
  float gamut_strength;                // $MIN: 0 $MAX: 1 $DEFAULT: 0.85 $STEP: 0.01 $DESCRIPTION: "gamut compression"
  float gamut_knee;                    // $MIN: 0 $MAX: 1 $DEFAULT: 0.30 $STEP: 0.01 $DESCRIPTION: "gamut knee"
} dt_iop_aces20_params_t;

/* Per-pipepiece data */
typedef struct dt_iop_aces20_data_t
{
  dt_iop_aces20_params_t params;
  dt_ac_context_t ctx;
} dt_iop_aces20_data_t;

/* GPU-side context — byte-for-byte match with aces20.cl */
typedef struct dt_ac_cl_params_t
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

/* OpenCL global data */
typedef struct dt_iop_aces20_global_data_t
{
  int kernel_aces20;
} dt_iop_aces20_global_data_t;

/* GUI data */
typedef struct dt_iop_aces20_gui_data_t
{
  GtkWidget *peak_luminance;
  GtkWidget *surround;
  GtkWidget *exposure;
  GtkWidget *gamut_strength;
  GtkWidget *gamut_knee;
} dt_iop_aces20_gui_data_t;

/* ====================================================================
 * Matrix helpers (3×3 row-major multiply)
 * ==================================================================== */

static inline void _mat_mul(float out[9], const float a[9], const float b[9])
{
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      out[i * 3 + j] = 0.0f;
      for(int k = 0; k < 3; k++)
        out[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    }
}

static inline void _mat_apply(float out[3], const float M[9], const float in[3])
{
  out[0] = M[0] * in[0] + M[1] * in[1] + M[2] * in[2];
  out[1] = M[3] * in[0] + M[4] * in[1] + M[5] * in[2];
  out[2] = M[6] * in[0] + M[7] * in[1] + M[8] * in[2];
}

/* ====================================================================
 * ACES 2.0 SSTS (Single-Stage Tone Scale)
 *
 * Michaelis-Menten parametric curve with flare compensation.
 * Reference: aces-core/lib/Lib.Academy.Tonescale.ctl
 *
 *   f = m_2 * (x / (x + s_2))^g
 *   h = f^2 / (f + t_1)
 *   Y = h * n_r
 * ==================================================================== */

static void dt_ac_ssts_init(dt_ac_ssts_params_t *p, double peak_luminance,
                             int surround_idx)
{
  const double n_r     = 100.0;
  const double c       = 0.18;
  const double c_d     = 10.013;
  const double w_g     = 0.14;
  const double t_1     = 0.04;
  const double r_hit_min = 128.0;
  const double r_hit_max = 896.0;

  double g;
  switch(surround_idx)
  {
    case 0:  g = 1.25; break;
    case 1:  g = 1.15; break;
    default: g = 1.00; break;
  }

  const double n = fmax(peak_luminance, 1.0);
  const double r_hit = r_hit_min
    + (r_hit_max - r_hit_min) * (log(n / n_r) / log(10000.0 / 100.0));
  const double m_0 = n / n_r;
  const double m_1 = 0.5 * (m_0 + sqrt(m_0 * (m_0 + 4.0 * t_1)));
  const double u_u = pow((r_hit / m_1) / ((r_hit / m_1) + 1.0), g);
  const double m = m_1 / u_u;
  const double w_i = log(n / 100.0) / log(2.0);
  const double c_t = (c_d / n_r) * (1.0 + w_i * w_g);
  const double g_ip = 0.5 * (c_t + sqrt(c_t * (c_t + 4.0 * t_1)));
  const double g_ipp2 = -(m_1 * pow(g_ip / m, 1.0 / g))
                        / (pow(g_ip / m, 1.0 / g) - 1.0);
  const double w_2 = c / g_ipp2;
  const double s_2 = w_2 * m_1;
  const double u_2 = pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
  const double m_2 = m_1 / u_2;

  p->s_2 = s_2;
  p->m_2 = m_2;
  p->g   = g;
  p->t_1 = t_1;
  p->n_r = n_r;
  p->n   = n;
}

static inline float dt_ac_ssts_fwd(const dt_ac_ssts_params_t *p, float x)
{
  if(x <= 0.0f) return 0.0f;
  const float f = (float)p->m_2 * powf(x / (x + (float)p->s_2), (float)p->g);
  const float h = (f * f) / (f + (float)p->t_1);
  return h * (float)p->n_r;
}

/* ====================================================================
 * Helwig CAM Helper: "Toe" Function
 *
 * Smooth compression/expansion with adjustable knee shape.
 *   forward: remap x in [0, limit], toe compresses near 0
 *   inverse: reverses the forward mapping
 * ==================================================================== */

static inline float _ac_toe(float x, float limit, float k1, float k2,
                             int inverse)
{
  if(x > limit) return x;
  k2 = fmaxf(k2, 0.001f);
  k1 = sqrtf(k1 * k1 + k2 * k2);
  const float k3 = (limit + k1) / (limit + k2);
  if(!inverse)
    return 0.5f * (k3 * x - k1 + sqrtf((k3 * x - k1) * (k3 * x - k1)
                                       + 4.0f * k2 * k3 * x));
  else
    return (x * x + k1 * x) / (k3 * (x + k2));
}

/* ====================================================================
 * AP1 Reach Table (360-entry, hue in degrees)
 *
 * Maximum M (colorfulness) that AP1 can hold at each hue in Hellwig JMh.
 * Used by chroma compression to normalise per-hue gamut extent.
 * ==================================================================== */

static const float dt_ac_gamut_reach[360] =
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

static inline float _ac_reach_from_table(float h)
{
  /* wrap hue to [0, 360) */
  const float hw = h - 360.0f * floorf(h / 360.0f);
  const int lo = (int)hw;
  const int hi = (lo < 359) ? lo + 1 : 0;
  const float t = hw - (float)lo;
  return dt_ac_gamut_reach[lo] + t * (dt_ac_gamut_reach[hi] - dt_ac_gamut_reach[lo]);
}

/* ====================================================================
 * Chroma Compression Normalisation
 *
 * Trigonometric approximation of the AP1 gamut's M (colorfulness) cusp
 * as a function of hue.  Returns the normalisation factor.
 * ==================================================================== */

static inline float _ac_chroma_norm(float h)
{
  const float hr = h * (float)(M_PI / 180.0);
  const float a = cosf(hr);
  const float b = sinf(hr);
  const float a2 = a * a - b * b;   /* cos(2h) */
  const float b2 = 2.0f * a * b;    /* sin(2h) */
  const float a3 = 4.0f * a * a * a - 3.0f * a;  /* cos(3h) */
  const float b3 = 3.0f * b - 4.0f * b * b * b;  /* sin(3h) */
  const float m = 11.34072f * a + 16.46899f * a2 + 7.88380f * a3
                + 14.66441f * b - 6.37224f * b2 + 9.19364f * b3
                + 77.12896f;
  return m * DT_AC_CC_SCALE;
}

/* ====================================================================
 * Hellwig 2022 CAM — Post-Adaptation Non-Linear Compression
 *
 * Forward:  LMS → non-linear response RGB_a
 * Inverse:  RGB_a → linear LMS
 * ==================================================================== */

static inline void _ac_nlc_fwd(const float rgb[3], float rgb_a[3], float f_l)
{
  for(int c = 0; c < 3; c++)
  {
    const float abs_rgb = fabsf(rgb[c]);
    if(abs_rgb < 1e-12f)
    {
      rgb_a[c] = 0.0f;
      continue;
    }
    const float fl_lms = powf(f_l * abs_rgb / 100.0f, 0.42f);
    const float sign = (rgb[c] >= 0.0f) ? 1.0f : -1.0f;
    rgb_a[c] = sign * (400.0f * fl_lms) / (27.13f + fl_lms);
  }
}

static inline void _ac_nlc_inv(const float rgb_a[3], float rgb[3], float f_l)
{
  for(int c = 0; c < 3; c++)
  {
    const float abs_a = fabsf(rgb_a[c]);
    if(abs_a < 1e-12f)
    {
      rgb[c] = 0.0f;
      continue;
    }
    const float sign = (rgb_a[c] >= 0.0f) ? 1.0f : -1.0f;
    rgb[c] = sign * (100.0f / f_l)
             * powf((27.13f * abs_a) / (400.0f - abs_a), 1.0f / 0.42f);
  }
}

/* ====================================================================
 * Hellwig 2022 CAM — Viewing Condition Precomputation
 *
 * Precomputes F_L, n, z, and A_w from the (fixed) ACES viewing
 * conditions and the white point.
 * ==================================================================== */

static inline void _ac_hellwig_precompute(const float white_xyz[3],
    float *f_l, float *n, float *z, float *a_w)
{
  const float y_w = white_xyz[1];

  /* RGB_w = MATRIX_16 × white_xyz */
  float rgb_w[3];
  _mat_apply(rgb_w, dt_ac_m16, white_xyz);

  /* k, k4, F_L */
  const float k = 1.0f / (5.0f * DT_AC_LA + 1.0f);
  const float k4 = k * k * k * k;
  *f_l = 0.2f * k4 * (5.0f * DT_AC_LA)
       + 0.1f * (1.0f - k4) * (1.0f - k4) * powf(5.0f * DT_AC_LA, 1.0f / 3.0f);

  *n = DT_AC_YB / y_w;
  *z = 1.48f + sqrtf(*n);

  /* D = 1.0 (complete adaptation) */
  const float d = 1.0f;
  float d_rgb[3], rgb_wc[3];
  for(int c = 0; c < 3; c++)
    d_rgb[c] = d * y_w / fmaxf(rgb_w[c], 1e-12f) + 1.0f - d;
  for(int c = 0; c < 3; c++)
    rgb_wc[c] = d_rgb[c] * rgb_w[c];

  /* Non-linear compression of white */
  float rgb_aw[3];
  _ac_nlc_fwd(rgb_wc, rgb_aw, *f_l);

  /* A_w = 2*R_aw + G_aw + 0.05*B_aw */
  *a_w = DT_AC_RA * rgb_aw[0] + rgb_aw[1] + DT_AC_BA * rgb_aw[2];
}

/* ====================================================================
 * Hellwig 2022 CAM — XYZ D60 → JMh
 *
 * Converts CIE XYZ (D60) to Hellwig lightness J, colorfulness M, hue h.
 * ==================================================================== */

static inline void _ac_xyz_to_jmh(const float xyz[3],
    const float d_rgb[3], float f_l, float a_w, float z, float jmh[3])
{
  /* RGB = MATRIX_16 × XYZ */
  float rgb[3];
  _mat_apply(rgb, dt_ac_m16, xyz);

  /* Chromatic adaptation */
  float rgb_c[3];
  for(int c = 0; c < 3; c++)
    rgb_c[c] = d_rgb[c] * rgb[c];

  /* Non-linear response */
  float rgb_a[3];
  _ac_nlc_fwd(rgb_c, rgb_a, f_l);

  /* Opponent dimensions a, b */
  const float a = rgb_a[0] - 12.0f * rgb_a[1] / 11.0f + rgb_a[2] / 11.0f;
  const float b = (rgb_a[0] + rgb_a[1] - 2.0f * rgb_a[2]) / 9.0f;

  /* Hue h in degrees [0, 360) */
  float h = atan2f(b, a) * (180.0f / (float)M_PI);
  if(h < 0.0f) h += 360.0f;

  /* Achromatic response A */
  const float A = DT_AC_RA * rgb_a[0] + rgb_a[1] + DT_AC_BA * rgb_a[2];

  /* Lightness J */
  const float j = 100.0f * powf(fmaxf(A, 0.0f) / a_w, DT_AC_SURR_C * z);

  /* Colourfulness M */
  const float m = 43.0f * DT_AC_SURR_NC * sqrtf(a * a + b * b);

  jmh[0] = (j > 0.0f) ? j : 0.0f;
  jmh[1] = m;
  jmh[2] = h;
}

/* ====================================================================
 * Hellwig 2022 CAM — JMh → XYZ D60
 *
 * Converts Hellwig lightness J, colorfulness M, hue h back to XYZ (D60).
 * ==================================================================== */

static inline void _ac_jmh_to_xyz(const float jmh[3],
    const float d_rgb[3], float f_l, float a_w, float z, float xyz[3])
{
  const float j = fmaxf(jmh[0], 0.0f);
  const float m = jmh[1];
  const float hr = jmh[2] * ((float)M_PI / 180.0f);

  /* Achromatic response A */
  const float A = a_w * powf(fmaxf(j, 1e-12f) / 100.0f, 1.0f / (DT_AC_SURR_C * z));

  /* Opponent dimensions */
  const float gamma_v = m / (43.0f * DT_AC_SURR_NC);
  const float a_op = gamma_v * cosf(hr);
  const float b_op = gamma_v * sinf(hr);

  /* RGB_a from opponent */
  float p_in[3] = { A, a_op, b_op };
  float rgb_a[3];
  _mat_apply(rgb_a, dt_ac_panlrcm, p_in);
  for(int c = 0; c < 3; c++)
    rgb_a[c] /= 1403.0f;

  /* Inverse non-linear compression */
  float rgb_c[3];
  _ac_nlc_inv(rgb_a, rgb_c, f_l);

  /* Inverse chromatic adaptation */
  float rgb[3];
  for(int c = 0; c < 3; c++)
    rgb[c] = rgb_c[c] / fmaxf(d_rgb[c], 1e-12f);

  /* XYZ = MATRIX_16_INV × RGB */
  _mat_apply(xyz, dt_ac_m16_inv, rgb);
}

/* ====================================================================
 * Hellwig 2022 — Lightness J ↔ Luminance Y
 * ==================================================================== */

static inline float _ac_y_to_j(float y, float f_l, float a_w, float z)
{
  if(y <= 0.0f) return 0.0f;
  const float fl_y = powf(f_l * fabsf(y) / 100.0f, 0.42f);
  const float a_y = (400.0f * fl_y) / (27.13f + fl_y);
  return 100.0f * powf(fmaxf(a_y, 0.0f) / a_w, DT_AC_SURR_C * z);
}

static inline float _ac_j_to_y(float j, float f_l, float a_w, float z)
{
  if(j <= 0.0f) return 0.0f;
  const float A = a_w * powf(j / 100.0f, 1.0f / (DT_AC_SURR_C * z));
  const float abs_a = fabsf(A);
  if(abs_a < 1e-12f) return 0.0f;
  return (100.0f / f_l) * powf((27.13f * abs_a) / (400.0f - fminf(abs_a, 399.9f)),
                                1.0f / 0.42f);
}

/* ====================================================================
 * Chroma Compression (Hellwig JMh Space)
 *
 * Compresses colorfulness M as a function of tone-mapped J and hue.
 * Includes expansion of low-saturation colors and compression of
 * high-saturation colors with different per-hue limits.
 * ==================================================================== */

static inline void _ac_chroma_compress(float jmh[3], float orig_j)
{
  const float j = fmaxf(jmh[0], 1e-12f);
  float m = jmh[1];
  const float h = jmh[2];

  if(m <= 0.0f) return;

  /* Rescale M by tone-mapped J ratio (chromaticity preservation in JMh) */
  m *= powf(j / fmaxf(orig_j, 1e-12f), DT_AC_MODEL_GAMMA);

  /* Hue-dependent normalisation */
  const float m_norm = _ac_chroma_norm(h);
  m /= m_norm;

  /* Compute limit from AP1 reach */
  const float n_j = j / DT_AC_LIMIT_JMAX;
  const float sn_j = fmaxf(0.0f, 1.0f - n_j);
  const float limit = powf(n_j, DT_AC_MODEL_GAMMA)
                      * _ac_reach_from_table(h) / m_norm;

  if(limit <= 0.0f) return;

  /* Expand low-saturation (reverse toe) */
  m = limit - _ac_toe(limit - m, limit - 0.001f,
                       sn_j * DT_AC_SAT,
                       sqrtf(n_j * n_j + DT_AC_SAT_THR), 0);

  /* Compress high-saturation (forward toe) */
  m = _ac_toe(m, limit, n_j * DT_AC_COMPR, sn_j, 0);

  /* Denormalize */
  m *= m_norm;

  jmh[1] = fmaxf(m, 0.0f);
}

/* ====================================================================
 * ACES 2.0 Gamut Compression — soft knee toward AP1 gamut boundary
 * ==================================================================== */

static inline void dt_ac_gamut_compress(float rgb[3], float strength,
                                         float knee)
{
  float maxc = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
  if(maxc <= 0.0f) return;
  const float excess = fmaxf(maxc - 1.0f, 0.0f);
  if(excess <= 0.0f) return;
  const float k = fmaxf(knee, 0.01f);
  const float compressed = (excess * excess) / (excess + k * strength);
  const float scale = (maxc - excess + compressed) / maxc;
  rgb[0] *= scale;
  rgb[1] *= scale;
  rgb[2] *= scale;
}

/* ====================================================================
 * Context Precomputation
 *
 * Builds the conversion chain:
 *   forward:  working_RGB → XYZ(D50) → CAT→D60 → AP0 → AP1
 *   inverse:  AP1 → AP0 → XYZ(D60) → CAT→D50 → XYZ(D50) → working_RGB
 * ==================================================================== */

static void dt_ac_compute_context(const dt_iop_aces20_params_t *p,
                                   const dt_iop_order_iccprofile_info_t *wp,
                                   dt_ac_context_t *ctx)
{
  memset(ctx, 0, sizeof(*ctx));

  ctx->exposure_factor = exp2f(p->exposure_ev);
  ctx->gamut_strength = fmaxf(p->gamut_strength, 0.0f);
  ctx->gamut_knee = fmaxf(p->gamut_knee, 0.01f);
  ctx->surround_idx = p->surround;
  for(int i = 0; i < 3; i++) ctx->luma_coeff[i] = dt_ac_luma_ap1[i];

  /* Get RGB→XYZ(D50) and XYZ(D50)→RGB matrices from the pipe's working profile.
   * Fall back to Rec.2020 D50 if no profile is available. */
  float rgb_to_xyz[9], xyz_to_rgb[9];

  if(wp && dt_is_valid_colormatrix(wp->matrix_in[0][0])
       && dt_is_valid_colormatrix(wp->matrix_out[0][0]))
  {
    for(int r = 0; r < 3; r++)
      for(int c = 0; c < 3; c++)
      {
        rgb_to_xyz[r * 3 + c] = wp->matrix_in[r][c];
        xyz_to_rgb[r * 3 + c] = wp->matrix_out[r][c];
      }
  }
  else
  {
    /* Fallback: Rec.2020 D65 → D50 adapted (this is what dt pipeline commonly uses) */
    static const float rec2020_d50_to_xyz[9] =
    {
      0.636958f, 0.144617f, 0.168881f,
      0.262700f, 0.678009f, 0.059293f,
      0.000000f, 0.028073f, 1.060827f
    };
    for(int i = 0; i < 9; i++) rgb_to_xyz[i] = rec2020_d50_to_xyz[i];

    /* Invert: approx XYZ→Rec.2020 D50 */
    static const float xyz_to_rec2020_d50[9] =
    {
      1.716492f, -0.355290f, -0.252223f,
      -0.666821f,  1.616544f,  0.015768f,
      0.017642f, -0.042775f,  0.942308f
    };
    for(int i = 0; i < 9; i++) xyz_to_rgb[i] = xyz_to_rec2020_d50[i];
  }

  /* forward = ap0_to_ap1 × xyz_to_ap0 × cat_d50_to_d60 × rgb_to_xyz */
  {
    float t1[9], t2[9], t3[9];
    /* t1 = cat_d50_to_d60 × rgb_to_xyz */
    _mat_mul(t1, dt_ac_cat_d50_to_d60, rgb_to_xyz);
    /* t2 = xyz_to_ap0 × t1 */
    _mat_mul(t2, dt_ac_xyz_to_ap0, t1);
    /* t3 = ap0_to_ap1 × t2 */
    _mat_mul(t3, dt_ac_ap0_to_ap1, t2);
    for(int i = 0; i < 9; i++) ctx->fwd_matrix[i] = t3[i];
  }

  /* inverse = xyz_to_rgb × cat_d60_to_d50 × ap0_to_xyz × ap1_to_ap0 */
  {
    float t1[9], t2[9], t3[9];
    /* t1 = ap0_to_xyz × ap1_to_ap0 */
    _mat_mul(t1, dt_ac_ap0_to_xyz, dt_ac_ap1_to_ap0);
    /* t2 = cat_d60_to_d50 × t1 */
    _mat_mul(t2, dt_ac_cat_d60_to_d50, t1);
    /* t3 = xyz_to_rgb × t2 */
    _mat_mul(t3, xyz_to_rgb, t2);
    for(int i = 0; i < 9; i++) ctx->inv_matrix[i] = t3[i];
  }

  /* SSTS init */
  dt_ac_ssts_init(&ctx->ssts, (double)p->peak_luminance, p->surround);

  /* Hellwig CAM precomputation */
  {
    float n_unused;
    _ac_hellwig_precompute(dt_ac_aces_white_xyz,
                           &ctx->f_l, &n_unused, &ctx->z, &ctx->a_w);

    /* Precompute D_RGB for the ACES D60 white point (constant for D=1) */
    float rgb_w[3];
    _mat_apply(rgb_w, dt_ac_m16, dt_ac_aces_white_xyz);
    const float y_w = dt_ac_aces_white_xyz[1];
    for(int c = 0; c < 3; c++)
      ctx->d_rgb[c] = y_w / fmaxf(rgb_w[c], 1e-12f);
  }
}

/* ====================================================================
 * ACES 2.0 CAM DRT Pipeline Evaluation (per-pixel)
 *
 *   pipe_RGB  --[fwd_matrix]→  AP1 D60
 *     → SSTS per-channel on AP1 → display-referred AP1
 *     → XYZ abs nits (×100) → Hellwig JMh
 *     → Chroma compression (J unchanged by tone mapping)
 *     → XYZ(D60) → AP1 (÷100 to display-referred)
 *     → Gamut compression in AP1
 *     --[inv_matrix]→ pipe_RGB
 * ==================================================================== */

static void dt_ac_pipeline_eval(const float rgb_in[3], float rgb_out[3],
                                 const dt_ac_context_t *ctx)
{
  /* Step 1: pipe RGB → AP1 (D60) */
  float ap1[3];
  _mat_apply(ap1, ctx->fwd_matrix, rgb_in);

  if(!isfinite(ap1[0] + ap1[1] + ap1[2]) || (ap1[0] + ap1[1] + ap1[2]) <= 0.0f)
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  for(int c = 0; c < 3; c++) ap1[c] = fmaxf(ap1[c], 0.0f);

  /* Step 2: Per-channel SSTS tone mapping on AP1 (ACES 2.0 reference) */
  float ap1_disp[3];
  for(int c = 0; c < 3; c++)
    ap1_disp[c] = dt_ac_ssts_fwd(&ctx->ssts, fmaxf(ap1[c], 0.0f) * ctx->exposure_factor)
                  / DT_AC_REF_LUM;

  /* Step 3: Display AP1 → XYZ D60 (absolute nits for CAM) */
  float xyz_abs[3];
  _mat_apply(xyz_abs, dt_ac_ap1_to_xyz, ap1_disp);
  for(int c = 0; c < 3; c++)
    xyz_abs[c] *= DT_AC_REF_LUM;

  /* Step 4: XYZ D60 → Hellwig JMh */
  float jmh[3];
  _ac_xyz_to_jmh(xyz_abs, ctx->d_rgb, ctx->f_l, ctx->a_w, ctx->z, jmh);

  /* Step 5: Chroma compression in JMh (J unchanged by tone mapping) */
  _ac_chroma_compress(jmh, jmh[0]);

  /* Step 6: JMh → XYZ D60 (in nits) */
  float xyz_out[3];
  _ac_jmh_to_xyz(jmh, ctx->d_rgb, ctx->f_l, ctx->a_w, ctx->z, xyz_out);

  /* Step 7: XYZ D60 → AP1 (back to display-referred) */
  float ap1_out[3];
  _mat_apply(ap1_out, dt_ac_xyz_to_ap1, xyz_out);
  for(int c = 0; c < 3; c++)
    ap1_out[c] = fmaxf(ap1_out[c], 0.0f) / DT_AC_REF_LUM;

  /* Step 8: Gamut compression in AP1 */
  for(int c = 0; c < 3; c++)
    ap1_out[c] = fmaxf(ap1_out[c], 0.0f);
  dt_ac_gamut_compress(ap1_out, ctx->gamut_strength, ctx->gamut_knee);

  /* Step 9: AP1 → pipe RGB */
  float rgb[3];
  _mat_apply(rgb, ctx->inv_matrix, ap1_out);

  for(int c = 0; c < 3; c++)
    rgb_out[c] = isfinite(rgb[c]) ? fmaxf(rgb[c], 0.0f) : 0.0f;
}

/* ====================================================================
 * Framework Functions
 * ==================================================================== */

const char *name()
{
  return _("ACES 2.0 Reference Rendering");
}

const char *aliases()
{
  return _("aces|rrt|tone mapping");
}

int default_group()
{
  return IOP_GROUP_TONE | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);
  g_assert(self->default_params != NULL);
}

void cleanup(dt_iop_module_t *self)
{
  dt_iop_default_cleanup(self);
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_alloc1_align_type(dt_iop_aces20_data_t);
  if(!piece->data) return;
  dt_iop_aces20_data_t *d = piece->data;
  memset(d, 0, sizeof(dt_iop_aces20_data_t));
  memcpy(&d->params, self->default_params, sizeof(dt_iop_aces20_params_t));
  dt_ac_compute_context(&d->params, NULL, &d->ctx);
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  if(piece->data)
  {
    dt_free_align(piece->data);
    piece->data = NULL;
  }
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_aces20_params_t *p = (dt_iop_aces20_params_t *)p1;
  dt_iop_aces20_data_t *d = piece->data;
  if(!d) return;

  /* Get the pipe's working profile for correct colour space conversion */
  const dt_iop_order_iccprofile_info_t *wp = NULL;
  if(pipe) wp = dt_ioppr_get_pipe_current_profile_info(self, pipe);

  dt_ac_compute_context(p, wp, &d->ctx);
  memcpy(&d->params, p, sizeof(dt_iop_aces20_params_t));
}

void tiling_callback(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in, const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  tiling->factor = 2.0f;
  tiling->factor_cl = 2.0f;
  tiling->maxbuf = 1.0f;
  tiling->maxbuf_cl = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 0;
  tiling->align = 1;
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_aces20_data_t *d = piece->data;
  const int width = roi_in->width;
  const int height = roi_in->height;
  const int ch = piece->colors;
  const size_t npixels = (size_t)width * height;

  if(!d || d->ctx.ssts.n <= 0.0)
  {
    memcpy(ovoid, ivoid, sizeof(float) * npixels * ch);
    return;
  }

  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;
  const dt_ac_context_t *ctx = &d->ctx;

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    shared(in, out, width, height, ch, npixels, ctx)
  #endif
  for(size_t k = 0; k < npixels; k++)
  {
    const size_t idx = k * ch;
    float rgb_in[3] = { in[idx], in[idx + 1], in[idx + 2] };

    for(int c = 0; c < 3; c++)
      rgb_in[c] = isfinite(rgb_in[c]) ? fmaxf(rgb_in[c], 0.0f) : 0.0f;

    float rgb_out[3];
    dt_ac_pipeline_eval(rgb_in, rgb_out, ctx);

    out[idx]     = rgb_out[0];
    out[idx + 1] = rgb_out[1];
    out[idx + 2] = rgb_out[2];
    if(ch == 4) out[idx + 3] = in[idx + 3];
  }
}

#ifdef HAVE_OPENCL

static void dt_ac_fill_cl_params(const dt_iop_aces20_data_t *d,
                                  dt_ac_cl_params_t *clp)
{
  const dt_ac_context_t *ctx = &d->ctx;
  memset(clp, 0, sizeof(*clp));

  for(int i = 0; i < 9; i++) clp->fwd_matrix[i]  = ctx->fwd_matrix[i];
  for(int i = 0; i < 9; i++) clp->inv_matrix[i]  = ctx->inv_matrix[i];

  clp->exposure_factor  = ctx->exposure_factor;
  clp->gamut_strength   = ctx->gamut_strength;
  clp->gamut_knee       = ctx->gamut_knee;
  clp->f_l              = ctx->f_l;
  clp->a_w              = ctx->a_w;
  clp->z                = ctx->z;
  for(int i = 0; i < 3; i++) clp->d_rgb[i] = ctx->d_rgb[i];

  clp->ssts_s_2 = (float)ctx->ssts.s_2;
  clp->ssts_m_2 = (float)ctx->ssts.m_2;
  clp->ssts_g   = (float)ctx->ssts.g;
  clp->ssts_t_1 = (float)ctx->ssts.t_1;
  clp->ssts_n_r = (float)ctx->ssts.n_r;
  clp->ssts_n   = (float)ctx->ssts.n;
}

int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_aces20_global_data_t *gd = self->global_data;
  const dt_iop_aces20_data_t *d = piece->data;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  if(!d || d->ctx.ssts.n <= 0.0)
    return DT_OPENCL_PROCESS_CL;

  dt_ac_cl_params_t clp;
  dt_ac_fill_cl_params(d, &clp);

  cl_int err = CL_SUCCESS;
  err = dt_opencl_enqueue_kernel_2d_args(
    devid, gd->kernel_aces20, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(clp));

  return err;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 46;
  dt_iop_aces20_global_data_t *gd = malloc(sizeof(dt_iop_aces20_global_data_t));
  self->data = gd;
  gd->kernel_aces20 = dt_opencl_create_kernel(program, "kernel_aces20");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_aces20_global_data_t *gd = self->data;
  if(gd)
  {
    dt_opencl_free_kernel(gd->kernel_aces20);
    free(self->data);
    self->data = NULL;
  }
}
#endif

void init_presets(dt_iop_module_so_t *self)
{
  self->pref_based_presets = TRUE;

  const char *workflow = dt_conf_get_string_const("plugins/darkroom/workflow");
  const gboolean auto_apply_st = workflow
    && strcmp(workflow, "scene-referred (ACES 2.0)") == 0;

  dt_iop_aces20_params_t p;
  memset(&p, 0, sizeof(p));
  p.peak_luminance = 200.0f;
  p.surround = DT_AC_SURROUND_DIM;
  p.exposure_ev = 0.0f;
  p.gamut_strength = 0.85f;
  p.gamut_knee = 0.30f;

  if(auto_apply_st)
  {
    dt_gui_presets_add_generic(_("scene-referred default"), self->op,
                                self->version(),
                                &p, sizeof(p),
                                TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

    dt_gui_presets_update_format(BUILTIN_PRESET("scene-referred default"),
                                 self->op, self->version(), FOR_RAW | FOR_MATRIX);

    dt_gui_presets_update_autoapply(BUILTIN_PRESET("scene-referred default"),
                                    self->op, self->version(), TRUE);
  }
}

void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  (void)self;
  (void)picker;
  (void)pipe;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_aces20_gui_data_t *g = self->gui_data;
  dt_iop_aces20_params_t *p = self->params;

  dt_bauhaus_slider_set(g->peak_luminance, p->peak_luminance);
  dt_bauhaus_combobox_set(g->surround, p->surround);
  dt_bauhaus_slider_set(g->exposure, p->exposure_ev);
  dt_bauhaus_slider_set(g->gamut_strength, p->gamut_strength);
  dt_bauhaus_slider_set(g->gamut_knee, p->gamut_knee);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_aces20_gui_data_t *g = IOP_GUI_ALLOC(aces20);

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
  GtkWidget *main_vbox = self->widget;

  /* Peak luminance */
  g->peak_luminance = dt_bauhaus_slider_from_params(self, "peak_luminance");
  dt_bauhaus_slider_set_format(g->peak_luminance, _(" nits"));
  dt_bauhaus_slider_set_digits(g->peak_luminance, 0);
  gtk_widget_set_tooltip_text(g->peak_luminance,
    _("Peak luminance of the target display in cd/m² (nits). "
      "200 nits = typical SDR monitor, 1000+ nits = HDR."));

  /* Surround */
  g->surround = dt_bauhaus_combobox_from_params(self, "surround");
  gtk_widget_set_tooltip_text(g->surround,
    _("Viewing environment: dark (cinema), dim (home theatre), "
      "or average (monitor/office). Affects SSTS contrast."));

  /* Exposure */
  g->exposure = dt_bauhaus_slider_from_params(self, "exposure_ev");
  dt_bauhaus_slider_set_format(g->exposure, _(" EV"));
  dt_bauhaus_slider_set_digits(g->exposure, 2);
  gtk_widget_set_tooltip_text(g->exposure,
    _("Exposure compensation applied before tone mapping. "
      "Positive values brighten, negative values darken."));

  /* Gamut compression section */
  dt_gui_box_add(GTK_BOX(main_vbox),
    dt_ui_section_label_new(C_("section", "gamut compression")));

  g->gamut_strength = dt_bauhaus_slider_from_params(self, "gamut_strength");
  dt_bauhaus_slider_set_factor(g->gamut_strength, 100.0f);
  dt_bauhaus_slider_set_format(g->gamut_strength, " %");
  dt_bauhaus_slider_set_digits(g->gamut_strength, 0);
  gtk_widget_set_tooltip_text(g->gamut_strength,
    _("Strength of the ACES 2.0 gamut compression. "
      "Higher values compress out-of-gamut colours more aggressively "
      "toward the AP1 gamut boundary."));

  g->gamut_knee = dt_bauhaus_slider_from_params(self, "gamut_knee");
  dt_bauhaus_slider_set_factor(g->gamut_knee, 100.0f);
  dt_bauhaus_slider_set_format(g->gamut_knee, " %");
  dt_bauhaus_slider_set_digits(g->gamut_knee, 0);
  gtk_widget_set_tooltip_text(g->gamut_knee,
    _("Knee point for gamut compression. Lower values start compression "
      "sooner, preserving more detail at the gamut boundary. "
      "Higher values allow more saturation before compression begins."));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

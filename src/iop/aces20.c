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
    pipeline using the CAT16-based CAM DRT (colour appearance model display
    rendering transform) aligning with ACES 2.0 reference:

      pipe_RGB → AP1 (D60) → XYZ (D60)
        → ×100 to absolute nits → CAT16 JMh
        → Tonemap & compress in JMh:
            J → Y → tonescale(Y) → J' (display J)
            M ← chroma_compress(M, J', orig_J)
        → Gamut Compression (JMh)
        → XYZ (D60) → /100 → AP1 (D60)
        → Gamut Compression (AP1) → pipe_RGB

    The input/output colour spaces are determined by the pipe's working profile,
    not hardcoded.  The module always stays in IOP_CS_RGB and converts to/from
    ACES AP1 D60 internally.

    References:
      - ACES 2.0 (official, April 2025):
        github.com/aces-aswf/aces-core
        lib/Lib.Academy.OutputTransform.ctl, Lib.Academy.Tonescale.ctl
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

DT_MODULE_INTROSPECTION(3, dt_iop_aces20_params_t)

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
 * CAT16 / ACES 2.0 CAM Constants
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

/* CAT16: XYZ → sharpened LMS (MATRIX_16 — transposed of CTL's XYZtoRGB(CAM16_PRI))
 *
 * Matches the ACES 2.0 CTL (Lib.Academy.OutputTransform.ctl) which computes
 * this matrix from the CAM16_PRI chromaticities via XYZtoRGB_f33():
 *   {0.8336, 0.1735}, {2.3854, -1.4659}, {0.087, -0.125}, {0.333, 0.333}
 * This differs slightly from the published CAT16 standard matrix in the
 * blue primary, per the ACES 2.0 reference. */
static const float dt_ac_m16[9] =
{
  0.3640744836f,  0.5947008157f,  0.0411012735f,
 -0.2222450987f,  1.0738554823f,  0.1479453361f,
 -0.0020676190f,  0.0488260454f,  0.9503875570f
};

/* CAT16: sharpened LMS → XYZ (exact inverse of dt_ac_m16) */
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

/* CAT16 CAM viewing-condition parameters */
#define DT_AC_LA         100.0f
#define DT_AC_YB          20.0f
#define DT_AC_RA           2.0f
#define DT_AC_BA           0.05f    /* 1/20 — CAM16 standard, verified with panlrcm/1403 */
#define DT_AC_SURR_F       0.9f
#define DT_AC_SURR_C       0.59f
#define DT_AC_SURR_NC      0.9f

/* ACES 2.0 CAM non-linear compression constants */
#define DT_AC_CAM_NL_OFFSET  27.13f   /* = 0.2713 * ref_lum */
#define DT_AC_CAM_NL_SCALE  400.0f    /* = 4.0 * ref_lum — unused in NLC itself (matches
                                          official CTL, where it is also dead there), but
                                          required in the Aab chromatic (a,b) scaling below */

/* M (colorfulness) scale factor.
 *
 * Reference: Lib.Academy.OutputTransform.ctl / init_JMhParams():
 *   cone_response_to_Aab = cam_nl_scale * base_cone_response_to_Aab
 *   MATRIX_cone_response_to_Aab column 0 (A)   is further divided by A_w
 *     -> cam_nl_scale cancels out in the A/J ratio (hence Y<->J needs no scale)
 *   MATRIX_cone_response_to_Aab columns 1,2 (a,b) are NOT divided by A_w
 *     -> cam_nl_scale (400) does NOT cancel and must be applied to M
 *
 * M = |Aab.a, Aab.b| = cam_nl_scale * 43 * surround[2] * sqrt(raw_a^2 + raw_b^2)
 *                    = DT_AC_M_SCALE * sqrt(raw_a^2 + raw_b^2)
 */
#define DT_AC_M_SCALE  (DT_AC_CAM_NL_SCALE * 43.0f * DT_AC_SURR_NC)   /* = 400*43*0.9 = 15480 */

/* ACES D60 white point in XYZ (Y=100) */
static const float dt_ac_aces_white_xyz[3] = { 95.2646074570f, 100.0f, 100.8825184352f };

/* Reference luminance (nits) — ACES D60 white = 100 nits */
#define DT_AC_REF_LUM     100.0f

/* Chroma compression constants (base values, scaled by peak luminance) */
#define DT_AC_COMPR_BASE      2.4f
#define DT_AC_COMPR_FACT      3.3f
#define DT_AC_SAT_BASE        1.3f
#define DT_AC_SAT_FACT        0.69f
#define DT_AC_EXPAND_THR      0.5f

/* Gamut compression constants (ACES 2.0 reference) */
#define DT_AC_GAMUT_SMOOTH_CUSPS       0.12f
#define DT_AC_GAMUT_SMOOTH_M           0.27f
#define DT_AC_GAMUT_CUSP_MID_BLEND     1.3f
#define DT_AC_GAMUT_FOCUS_GAIN_BLEND   0.3f
#define DT_AC_GAMUT_FOCUS_DISTANCE     1.35f
#define DT_AC_GAMUT_FOCUS_DIST_SCALING 1.75f
#define DT_AC_GAMUT_COMPRESSION_THR    0.75f
#define DT_AC_GAMUT_TABLE_SIZE         362
#define DT_AC_TABLE_SIZE                360   /* uniform hue entries (0°..359°) */
#define DT_AC_BASE_INDEX                  1   /* first valid entry index */
#define DT_AC_CUSP_CORNER_COUNT         6
#define DT_AC_TOTAL_CORNER_COUNT        8   /* 6 + 2 wrap-around */
#define DT_AC_DISPLAY_CUSP_TOL          1e-7f

/* Reach primaries — ACES 2.0 reference uses AP1 (D60) as the reach gamut */
#define DT_AC_REACH_Y             1.0f
#define DT_AC_REACH_R_X           0.7130f
#define DT_AC_REACH_R_Y           0.2930f
#define DT_AC_REACH_G_X           0.1650f
#define DT_AC_REACH_G_Y           0.8300f
#define DT_AC_REACH_B_X           0.1280f
#define DT_AC_REACH_B_Y           0.0440f
#define DT_AC_REACH_W_X           0.32168f
#define DT_AC_REACH_W_Y           0.33767f

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
  float forward_limit;      /* upper clamp limit in AP1 (from SSTS: r_hit_max / n_r = 8.96) */
  int   surround_idx;
  int   sdr_clip_enable;    /* optional soft ceiling at code-value 1.0 */
  float sdr_clip_softness;  /* smooth transition width for the ceiling */

  /* CAT16 CAM precomputed values (constant for ACES D60 white) */
  float f_l;
  float a_w;                /* A_w — white achromatic signal for J = 100*(A/A_w)^cz */
  float z;
  float f_l_n;              /* F_L_n = F_L / ref_lum */
  float cz;                 /* model_gamma = surround_c * z */
  float inv_cz;             /* 1 / cz */
  float a_w_j;              /* A_w_J = NLC(F_L) for Y↔J conversions */
  float d_rgb[3];           /* D_RGB = F_L_n * Y_w / RGB_w[c] (includes F_L adaptation) */

  /* Dynamic chroma compression parameters (per ACES 2.0 official) */
  float model_gamma_inv;
  float chroma_compress_scale;
  float cc_sat;
  float cc_sat_thr;
  float cc_compr;
  float limit_j_max;

  /* Gamut compression parameters (ACES 2.0 reference, computed at init) */
  float mid_J;
  float focus_dist;
  float lower_hull_gamma_inv;
  float table_reach_m[DT_AC_GAMUT_TABLE_SIZE];
  float table_hues[DT_AC_GAMUT_TABLE_SIZE];
  float table_gamut_cusps[DT_AC_GAMUT_TABLE_SIZE][3];   /* J, M, h */
  float table_upper_hull_gamma[DT_AC_GAMUT_TABLE_SIZE];
  int   hue_search_range[2];
} dt_ac_context_t;

/* Module parameters */
typedef struct dt_iop_aces20_params_t
{
  float peak_luminance;                // $MIN: 100 $MAX: 4000 $DEFAULT: 200 $STEP: 10 $DESCRIPTION: "peak luminance (nits)"
  dt_iop_aces20_surround_t surround;   // $DEFAULT: DT_AC_SURROUND_DIM $DESCRIPTION: "surround"
  float exposure_ev;                   // $MIN: -5 $MAX: 5 $DEFAULT: 0 $STEP: 0.05 $DESCRIPTION: "exposure (EV)"
  gboolean sdr_clip_enable;            // $DEFAULT: FALSE $DESCRIPTION: "soft clip output to display white"
  float sdr_clip_softness;             // $MIN: 0 $MAX: 0.5 $DEFAULT: 0.15 $STEP: 0.01 $DESCRIPTION: "clip softness"
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
  float forward_limit;            /* upper clamp limit in AP1 (from SSTS) */
  float f_l_n;                    /* F_L_n = F_L / ref_lum */
  float a_w;                      /* A_w — white achromatic signal (JMh) */
  float z;                        /* 1.48 + sqrt(Y_b/Y_w) */
  float cz;                       /* model_gamma = surround_c * z */
  float inv_cz;                   /* 1 / cz */
  float d_rgb[3];                 /* D_RGB = F_L_n * Y_w / RGB_w[c] */
  float a_w_j;                    /* A_w_J = NLC(F_L) for Y↔J */
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

  /* Gamut compression (ACES 2.0 reference) */
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
  int   sdr_clip_enable;          /* soft ceiling enable */
  float sdr_clip_softness;       /* smooth transition width */
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
  GtkWidget *sdr_clip_enable;
  GtkWidget *sdr_clip_softness;
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
  return m;
}

/* ====================================================================
 * ACES 2.0 — Post-Adaptation Cone Response Compression
 *
 * Matches aces-core Lib.Academy.OutputTransform.ctl:
 *   fwd: Ra = copysign(pow(|v|, 0.42) / (27.13 + pow(|v|, 0.42)), v)
 *   inv: Rc = copysign(pow(27.13 * |Ra| / (1 - |Ra|), 1/0.42), Ra)
 *
 * F_L and ref_lum scalings are baked into the d_rgb factor.
 * ==================================================================== */

static inline float _ac_nlc_fwd_single(float v)
{
  const float abs_v = fabsf(v);
  if(abs_v < 1e-12f) return 0.0f;
  const float fl = powf(abs_v, 0.42f);
  return copysignf(fl / (DT_AC_CAM_NL_OFFSET + fl), v);
}

static inline float _ac_nlc_inv_single(float v)
{
  const float abs_v = fminf(fabsf(v), 0.99f);
  if(abs_v < 1e-12f) return 0.0f;
  const float fl = (DT_AC_CAM_NL_OFFSET * abs_v) / (1.0f - abs_v);
  return copysignf(powf(fl, 1.0f / 0.42f), v);
}

/* ====================================================================
 * 3×3 Matrix Inversion (general, for cofactor method)
 * ==================================================================== */

static inline void _mat_inv_3x3(float inv[9], const float m[9])
{
  const float a = m[0], b = m[1], c = m[2];
  const float d = m[3], e = m[4], f = m[5];
  const float g = m[6], h = m[7], i = m[8];
  const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if(fabsf(det) < 1e-18f) { for(int j = 0; j < 9; j++) inv[j] = 0.0f; return; }
  const float id = 1.0f / det;
  inv[0] =  (e * i - f * h) * id;
  inv[1] = -(b * i - c * h) * id;
  inv[2] =  (b * f - c * e) * id;
  inv[3] = -(d * i - f * g) * id;
  inv[4] =  (a * i - c * g) * id;
  inv[5] = -(a * f - c * d) * id;
  inv[6] =  (d * h - e * g) * id;
  inv[7] = -(a * h - b * g) * id;
  inv[8] =  (a * e - b * d) * id;
}

/* Vector wrappers for NLC (operate on all 3 channels) */
static inline void _ac_nlc_fwd(const float rgb[3], float rgb_a[3])
{
  for(int c = 0; c < 3; c++)
    rgb_a[c] = _ac_nlc_fwd_single(rgb[c]);
}

static inline void _ac_nlc_inv(const float rgb_a[3], float rgb[3])
{
  for(int c = 0; c < 3; c++)
    rgb[c] = _ac_nlc_inv_single(rgb_a[c]);
}

/* ====================================================================
 * CAT16 CAM — Viewing Condition Precomputation
 *
 * Precomputes F_L, n, z, and A_w from the (fixed) ACES viewing
 * conditions and the white point.
 * ==================================================================== */

static inline void _ac_hellwig_precompute(const float white_xyz[3],
    float *f_l, float *n, float *z, float *cz, float *inv_cz,
    float *f_l_n, float *a_w, float *a_w_j, float d_rgb[3])
{
  const float y_w = white_xyz[1];

  /* RGB_w = MATRIX_16 × white_xyz */
  float rgb_w[3];
  _mat_apply(rgb_w, dt_ac_m16, white_xyz);

  /* F_L (luminance-level adaptation factor) — same as before */
  const float k = 1.0f / (5.0f * DT_AC_LA + 1.0f);
  const float k4 = k * k * k * k;
  *f_l = 0.2f * k4 * (5.0f * DT_AC_LA)
       + 0.1f * (1.0f - k4) * (1.0f - k4) * powf(5.0f * DT_AC_LA, 1.0f / 3.0f);

  *n = DT_AC_YB / y_w;
  *z = 1.48f + sqrtf(*n);
  *cz = DT_AC_SURR_C * (*z);
  *inv_cz = 1.0f / (*cz);

  /* F_L_n = F_L / ref_lum (used in D_RGB and Y↔J) */
  *f_l_n = *f_l / DT_AC_REF_LUM;

  /* D_RGB = F_L_n * Y_w / RGB_w[c] (includes F_L adaptation) */
  for(int c = 0; c < 3; c++)
    d_rgb[c] = (*f_l_n) * y_w / fmaxf(rgb_w[c], 1e-12f);

  /* Adapted white: rgb_wc = D_RGB * rgb_w → each channel = F_L_n * Y_w = F_L */
  float rgb_aw[3];
  for(int c = 0; c < 3; c++)
    rgb_aw[c] = _ac_nlc_fwd_single(d_rgb[c] * rgb_w[c]);

  /* A_w = 2*R_aw + G_aw + (1/20)*B_aw  (standard CAM16 formula) */
  *a_w = DT_AC_RA * rgb_aw[0] + rgb_aw[1] + DT_AC_BA * rgb_aw[2];

  /* A_w_J = NLC(F_L) = NLC(f_l) for Y↔J conversions */
  *a_w_j = _ac_nlc_fwd_single(*f_l);
}

/* ====================================================================
 * CAT16 CAM — XYZ D60 → JMh
 *
 * Converts CIE XYZ (D60) to lightness J, colorfulness M, hue h.
 * ==================================================================== */

static inline void _ac_xyz_to_jmh(const float xyz[3],
    const float d_rgb[3], float a_w, float z, float jmh[3])
{
  float rgb[3];
  _mat_apply(rgb, dt_ac_m16, xyz);

  float rgb_c[3];
  for(int c = 0; c < 3; c++)
    rgb_c[c] = d_rgb[c] * rgb[c];

  float rgb_a[3];
  _ac_nlc_fwd(rgb_c, rgb_a);

  const float a = rgb_a[0] - 12.0f * rgb_a[1] / 11.0f + rgb_a[2] / 11.0f;
  const float b = (rgb_a[0] + rgb_a[1] - 2.0f * rgb_a[2]) / 9.0f;

  float h = atan2f(b, a) * (180.0f / (float)M_PI);
  if(h < 0.0f) h += 360.0f;

  const float A = DT_AC_RA * rgb_a[0] + rgb_a[1] + DT_AC_BA * rgb_a[2];

  const float j = 100.0f * powf(fmaxf(A, 0.0f) / a_w, DT_AC_SURR_C * z);

  const float m = DT_AC_M_SCALE * sqrtf(a * a + b * b);

  jmh[0] = (j > 0.0f) ? j : 0.0f;
  jmh[1] = m;
  jmh[2] = h;
}

/* ====================================================================
 * CAT16 CAM — JMh → XYZ D60
 *
 * Converts lightness J, colorfulness M, hue h back to XYZ (D60).
 * ==================================================================== */

static inline void _ac_jmh_to_xyz(const float jmh[3],
    const float d_rgb[3], float a_w, float z, float xyz[3])
{
  const float j = fmaxf(jmh[0], 0.0f);
  const float m = jmh[1];
  const float hr = jmh[2] * ((float)M_PI / 180.0f);

  const float A = a_w * powf(fmaxf(j, 1e-12f) / 100.0f, 1.0f / (DT_AC_SURR_C * z));

  const float gamma_v = m / DT_AC_M_SCALE;
  const float a_op = gamma_v * cosf(hr);
  const float b_op = gamma_v * sinf(hr);

  float p_in[3] = { A, a_op, b_op };
  float rgb_a[3];
  _mat_apply(rgb_a, dt_ac_panlrcm, p_in);
  for(int c = 0; c < 3; c++)
    rgb_a[c] /= 1403.0f;

  float rgb_c[3];
  _ac_nlc_inv(rgb_a, rgb_c);

  float rgb[3];
  for(int c = 0; c < 3; c++)
    rgb[c] = rgb_c[c] / fmaxf(d_rgb[c], 1e-12f);

  _mat_apply(xyz, dt_ac_m16_inv, rgb);
}

/* ====================================================================
 * CAT16 — Lightness J ↔ Luminance Y
 * ==================================================================== */

static inline float _ac_y_to_j(float y, float f_l_n, float a_w_j, float cz)
{
  if(y <= 0.0f) return 0.0f;
  const float ra = _ac_nlc_fwd_single(fabsf(y) * f_l_n);
  const float a = ra / a_w_j;
  return 100.0f * powf(fmaxf(a, 0.0f), cz);
}

static inline float _ac_j_to_y(float j, float f_l_n, float a_w_j, float inv_cz)
{
  if(j <= 0.0f) return 0.0f;
  const float a = powf(j / 100.0f, inv_cz);
  const float ra = a_w_j * a;
  return _ac_nlc_inv_single(fminf(ra, 0.99f)) / f_l_n;
}

/* ====================================================================
 * Chroma Compression (CAT16 JMh Space)
 *
 * Compresses colorfulness M as a function of tone-mapped J and hue.
 * Includes expansion of low-saturation colors and compression of
 * high-saturation colors with different per-hue limits.
 *
 * Parameters come from the precomputed context (ctx) which are
 * computed dynamically per peak_luminance following the official
 * ACES 2.0 specification (aces-core Lib.Academy.OutputTransform.ctl).
 * ==================================================================== */

/* Forward declaration (defined in gamut compression section) */
static inline float _ac_reach_m_from_table(float h,
    const float reach_m[DT_AC_GAMUT_TABLE_SIZE],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const int search_range[2]);

static inline void _ac_chroma_compress(float jmh[3], float orig_j,
                                        const dt_ac_context_t *ctx)
{
  const float j = fmaxf(jmh[0], 1e-12f);
  float m = jmh[1];
  const float h = jmh[2];

  if(m <= 0.0f) return;

  /* Rescale M by tone-mapped J ratio (chromaticity preservation in JMh) */
  m *= powf(j / fmaxf(orig_j, 1e-12f), ctx->model_gamma_inv);

  /* Hue-dependent normalisation with dynamic scale */
  const float m_norm = _ac_chroma_norm(h) * ctx->chroma_compress_scale;
  m /= m_norm;

  /* Compute limit from AP1 reach (dynamic table) */
  const float n_j = j / ctx->limit_j_max;
  const float sn_j = fmaxf(0.0f, 1.0f - n_j);
  const float limit = powf(n_j, ctx->model_gamma_inv)
                      * _ac_reach_m_from_table(h, ctx->table_reach_m, ctx->table_hues, ctx->hue_search_range) / m_norm;

  if(limit <= 0.0f) return;

  /* Expand low-saturation (reverse toe) */
  m = limit - _ac_toe(limit - m, limit - 0.001f,
                       sn_j * ctx->cc_sat,
                       sqrtf(n_j * n_j + ctx->cc_sat_thr), 0);

  /* Compress high-saturation (forward toe) */
  m = _ac_toe(m, limit, n_j * ctx->cc_compr, sn_j, 0);

  /* Denormalize */
  m *= m_norm;

  jmh[1] = fmaxf(m, 0.0f);
}

/* ====================================================================
 * Chromaticity → XYZ Matrix Construction
 *
 * Builds the 3×3 matrix that converts linear RGB to CIE XYZ for a given
 * set of primaries (specified as CIE 1931 xy chromaticities and white point).
 *
 *   RGB = [R, G, B] in [0,1]
 *   X = M[0]*R + M[1]*G + M[2]*B
 *   Y = M[3]*R + M[4]*G + M[5]*B
 *   Z = M[6]*R + M[7]*G + M[8]*B
 *
 * Primary sets:
 *   0 = AP0 (ACES AP0 primaries)
 *   1 = AP1 (ACES AP1 primaries)
 *   2 = Reach (AP0 + AP1 + BT.2020 merged)
 * ==================================================================== */

/* AP0 primaries in CIE 1931 xyY */
#define DT_AC_AP0_R_X  0.7347f
#define DT_AC_AP0_R_Y  0.2653f
#define DT_AC_AP0_G_X  0.0000f
#define DT_AC_AP0_G_Y  1.0000f
#define DT_AC_AP0_B_X  0.0001f
#define DT_AC_AP0_B_Y -0.0770f
#define DT_AC_AP0_W_X  0.32168f
#define DT_AC_AP0_W_Y  0.33767f

/* AP1 primaries in CIE 1931 xyY */
#define DT_AC_AP1_R_X  0.7130f
#define DT_AC_AP1_R_Y  0.2930f
#define DT_AC_AP1_G_X  0.1650f
#define DT_AC_AP1_G_Y  0.8300f
#define DT_AC_AP1_B_X  0.1280f
#define DT_AC_AP1_B_Y  0.0440f
#define DT_AC_AP1_W_X  0.32168f
#define DT_AC_AP1_W_Y  0.33767f

/* Build RGB→XYZ matrix from primaries xy + white xy (Y=1 for white) */
static inline void _ac_xyy_to_xyz_matrix(float m[9],
    float rx, float ry, float gx, float gy, float bx, float by,
    float wx, float wy)
{
  /* Build the primaries matrix (RGB of [1,0,0], [0,1,0], [0,0,1]) */
  float prim[9];
  prim[0] = rx / ry;  prim[1] = gx / gy;  prim[2] = bx / by;
  prim[3] = 1.0f;     prim[4] = 1.0f;     prim[5] = 1.0f;
  prim[6] = (1.0f - rx - ry) / ry;
  prim[7] = (1.0f - gx - gy) / gy;
  prim[8] = (1.0f - bx - by) / by;

  /* Invert to solve for scale factors S = prim⁻¹ × white_XYZ */
  float inv[9];
  _mat_inv_3x3(inv, prim);

  const float white_xyz[3] = { wx / wy, 1.0f, (1.0f - wx - wy) / wy };
  float s[3];
  _mat_apply(s, inv, white_xyz);

  /* Scale primaries columns by s */
  for(int c = 0; c < 3; c++)
  {
    m[c]     = prim[c]     * s[c];
    m[3 + c] = prim[3 + c] * s[c];
    m[6 + c] = prim[6 + c] * s[c];
  }
}

/* Return RGB→XYZ for the specified primary set:
 *   0 = AP0, 1 = AP1, 2 = Reach primaries */
static inline void _ac_get_rgb_to_xyz(float m[9], int prim_set)
{
  if(prim_set == 0)
    _ac_xyy_to_xyz_matrix(m,
      DT_AC_AP0_R_X, DT_AC_AP0_R_Y,
      DT_AC_AP0_G_X, DT_AC_AP0_G_Y,
      DT_AC_AP0_B_X, DT_AC_AP0_B_Y,
      DT_AC_AP0_W_X, DT_AC_AP0_W_Y);
  else if(prim_set == 1)
    _ac_xyy_to_xyz_matrix(m,
      DT_AC_AP1_R_X, DT_AC_AP1_R_Y,
      DT_AC_AP1_G_X, DT_AC_AP1_G_Y,
      DT_AC_AP1_B_X, DT_AC_AP1_B_Y,
      DT_AC_AP1_W_X, DT_AC_AP1_W_Y);
  else
    _ac_xyy_to_xyz_matrix(m,
      DT_AC_REACH_R_X, DT_AC_REACH_R_Y,
      DT_AC_REACH_G_X, DT_AC_REACH_G_Y,
      DT_AC_REACH_B_X, DT_AC_REACH_B_Y,
      DT_AC_REACH_W_X, DT_AC_REACH_W_Y);
}

/* ====================================================================
 * RGB → CAT16 JMh (via arbitrary primaries)
 *
 * Convert linear RGB in a given primary space to CAT16 JMh.
 *   prim_set: 0=AP0, 1=AP1, 2=Reach
 * ==================================================================== */
static inline void _ac_rgb_to_jmh_prim(float jmh[3], const float rgb[3],
    int prim_set, const dt_ac_context_t *ctx)
{
  float rgb_to_xyz[9];
  _ac_get_rgb_to_xyz(rgb_to_xyz, prim_set);
  float xyz[3];
  _mat_apply(xyz, rgb_to_xyz, rgb);
  for(int c = 0; c < 3; c++) xyz[c] *= DT_AC_REF_LUM;
  _ac_xyz_to_jmh(xyz, ctx->d_rgb, ctx->a_w, ctx->z, jmh);
}

/* JMh→RGB(prim) is not used — the reach cusp search fw-converts RGB→JMh
 * via _ac_rgb_to_jmh_prim in a binary search along RGB edges instead. */

/* ====================================================================
 * Utility: Hue wrapping
 * ==================================================================== */
static inline float _ac_wrap_hue(float h)
{
  const float hw = h - 360.0f * floorf(h / 360.0f);
  return (hw < 0.0f) ? hw + 360.0f : hw;
}

/* ====================================================================
 * Utility: smin (smooth minimum) — matches ACES 2.0 reference
 * ==================================================================== */
static inline float _ac_smin(float a, float b, float k)
{
  if(k <= 0.0f) return fminf(a, b);
  const float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
  return fminf(a, b) - h * h * h * k * (1.0f / 6.0f);
}

/* ====================================================================
 * GAMUT COMPRESSION — TABLE GENERATION
 *
 * These functions build the hue-dependent tables used by the per-pixel
 * gamut compression at runtime. They are called once per context init
 * from dt_ac_compute_context().
 *
 * Translation of ACES 2.0 reference:
 *   aces-core/lib/Lib.Academy.OutputTransform.ctl
 *   lines 999–1628
 * ==================================================================== */

/* ---- Struct: pre-built 8-element corner tables (limiting or reach) ---- */
/* Entries [1..6] are the 6 corners rotated so lowest hue is at [1];
 * entries [0] and [7] are wrap-around copies for monotonic hue interpolation. */
typedef struct
{
  float rgb[DT_AC_TOTAL_CORNER_COUNT][3];
  float jmh[DT_AC_TOTAL_CORNER_COUNT][3];
} _ac_corner_tables_t;

/* ---- Generate one unit-cube cusp corner (R,Y,G,C,B,M order) ---- */
/* Matches CTL generate_unit_cube_cusp_corners (lines 999-1018) */
static inline void _ac_gen_unit_cube_cusp_corner(float rgb[3], int corner)
{
  rgb[0] = ((corner + 1) % DT_AC_CUSP_CORNER_COUNT < 3) ? 1.0f : 0.0f;
  rgb[1] = ((corner + 5) % DT_AC_CUSP_CORNER_COUNT < 3) ? 1.0f : 0.0f;
  rgb[2] = ((corner + 3) % DT_AC_CUSP_CORNER_COUNT < 3) ? 1.0f : 0.0f;
}

/* ---- Build limiting cusp corner tables (rotated, 8-element) ---- */
/* Matches CTL build_limiting_cusp_corners_tables (lines 1020-1058) */
static inline void _ac_build_limiting_cusp_corners_tables(
    _ac_corner_tables_t *tbl,
    int prim_set, float peak_scale, const dt_ac_context_t *ctx)
{
  float tmp_rgb[DT_AC_CUSP_CORNER_COUNT][3];
  float tmp_jmh[DT_AC_CUSP_CORNER_COUNT][3];

  int min_idx = 0;
  for(int i = 0; i < DT_AC_CUSP_CORNER_COUNT; i++)
  {
    _ac_gen_unit_cube_cusp_corner(tmp_rgb[i], i);
    for(int c = 0; c < 3; c++) tmp_rgb[i][c] *= peak_scale;
    _ac_rgb_to_jmh_prim(tmp_jmh[i], tmp_rgb[i], prim_set, ctx);
    if(tmp_jmh[i][2] < tmp_jmh[min_idx][2])
      min_idx = i;
  }

  /* Rotate so lowest hue is at index 1 */
  for(int i = 0; i < DT_AC_CUSP_CORNER_COUNT; i++)
  {
    const int src = (i + min_idx) % DT_AC_CUSP_CORNER_COUNT;
    for(int c = 0; c < 3; c++)
    {
      tbl->rgb[i + 1][c] = tmp_rgb[src][c];
      tbl->jmh[i + 1][c] = tmp_jmh[src][c];
    }
  }

  /* Wrap-around copies for monotonic hue interpolation */
  for(int c = 0; c < 3; c++)
  {
    tbl->rgb[0][c] = tbl->rgb[DT_AC_CUSP_CORNER_COUNT][c];
    tbl->jmh[0][c] = tbl->jmh[DT_AC_CUSP_CORNER_COUNT][c];
    tbl->rgb[DT_AC_CUSP_CORNER_COUNT + 1][c] = tbl->rgb[1][c];
    tbl->jmh[DT_AC_CUSP_CORNER_COUNT + 1][c] = tbl->jmh[1][c];
  }

  /* Wrap hues to maintain monotonicity across 0/360 boundary */
  tbl->jmh[0][2] -= 360.0f;
  tbl->jmh[DT_AC_CUSP_CORNER_COUNT + 1][2] += 360.0f;
}

/* ---- Find reach corners table ---- */
/* Matches CTL find_reach_corners_table (lines 1060-1121) */
static inline void _ac_find_reach_corners_table(
    _ac_corner_tables_t *tbl,
    const dt_ac_context_t *ctx)
{
  float tmp_jmh[DT_AC_CUSP_CORNER_COUNT][3];

  int min_idx = 0;
  for(int i = 0; i < DT_AC_CUSP_CORNER_COUNT; i++)
  {
    float corner_rgb[3];
    _ac_gen_unit_cube_cusp_corner(corner_rgb, i);

    /* Binary search for scale factor where J == limit_J_max in reach */
    float lo = 0.0f, hi = 10.0f;
    for(int iter = 0; iter < 32; iter++)
    {
      const float t = 0.5f * (lo + hi);
      float rgb[3];
      for(int c = 0; c < 3; c++)
        rgb[c] = corner_rgb[c] * t;
      float jmh[3];
      _ac_rgb_to_jmh_prim(jmh, rgb, 2, ctx);
      if(jmh[0] >= ctx->limit_j_max)
        hi = t;
      else
        lo = t;
    }

    /* Final JMh at the found scale factor */
    float rgb[3];
    for(int c = 0; c < 3; c++)
      rgb[c] = corner_rgb[c] * hi;
    _ac_rgb_to_jmh_prim(tmp_jmh[i], rgb, 2, ctx);

    if(tmp_jmh[i][2] < tmp_jmh[min_idx][2])
      min_idx = i;
  }

  /* Rotate so lowest hue is at index 1 */
  for(int i = 0; i < DT_AC_CUSP_CORNER_COUNT; i++)
  {
    const int src = (i + min_idx) % DT_AC_CUSP_CORNER_COUNT;
    for(int c = 0; c < 3; c++)
      tbl->jmh[i + 1][c] = tmp_jmh[src][c];
  }

  /* Wrap-around copies */
  for(int c = 0; c < 3; c++)
  {
    tbl->jmh[0][c] = tbl->jmh[DT_AC_CUSP_CORNER_COUNT][c];
    tbl->jmh[DT_AC_CUSP_CORNER_COUNT + 1][c] = tbl->jmh[1][c];
  }

  /* Wrap hues */
  tbl->jmh[0][2] -= 360.0f;
  tbl->jmh[DT_AC_CUSP_CORNER_COUNT + 1][2] += 360.0f;
}

/* ---- Build non-uniform hue table from 12 cube-corner hues ---- */
/* Matches CTL reference: 6 reach cube-corner hues + 6 limiting cube-corner
 * hues are sorted, deduplicated, then 360 entries are distributed
 * proportionally across the resulting segments. */
static inline void _ac_build_hue_table(float hues[DT_AC_GAMUT_TABLE_SIZE],
    const _ac_corner_tables_t *limit,
    const _ac_corner_tables_t *reach)
{
  /* Collect the 12 raw key hues: reach corners [1..6], limit corners [1..6].
   * Deliberately NOT deduplicated here -- this matches the official
   * build_hue_table() exactly, including its own collision-handling logic
   * below for near-coincident hues (samples_count adjustment), rather than
   * silently dropping duplicates beforehand. */
  float keys[12];
  int n_keys = 0;

  for(int i = 0; i < DT_AC_CUSP_CORNER_COUNT; i++)
  {
    keys[n_keys++] = reach->jmh[i + 1][2];
    keys[n_keys++] = limit->jmh[i + 1][2];
  }

  /* Insertion sort on n_keys (= 12, negligible cost) */
  for(int i = 1; i < n_keys; i++)
  {
    const float tmp = keys[i];
    int j = i - 1;
    while(j >= 0 && keys[j] > tmp)
    {
      keys[j + 1] = keys[j];
      j--;
    }
    keys[j + 1] = tmp;
  }

  /* Per-segment sample count proportional to angular width (official
   * build_hue_table): nominal_idx = round(hue * ideal_spacing), ideal_spacing
   * = tableSize/hue_limit = 360/360 = 1. This replaces an earlier equal-share
   * scheme (fixed ~360/n_keys samples per segment regardless of its angular
   * width), which produced gaps up to ~2.8 deg in wide segments while wasting
   * ~30 samples on segments under 1 deg wide -- worse local resolution than
   * even a plain uniform 1 deg table in the wide segments. */
  const float ideal_spacing = (float)DT_AC_TABLE_SIZE / 360.0f; /* = 1.0 */
  int samples_count[12];
  int last_idx = -1;
  int min_index = (keys[0] == 0.0f) ? 0 : 1;

  for(int i = 0; i < n_keys; i++)
  {
    int nominal_idx = (int)roundf(keys[i] * ideal_spacing);
    if(nominal_idx < min_index) nominal_idx = min_index;
    if(nominal_idx > DT_AC_TABLE_SIZE - 1) nominal_idx = DT_AC_TABLE_SIZE - 1;

    if(last_idx == nominal_idx)
    {
      /* Two consecutive key hues land on the same index: shrink the
       * previous segment by one sample if possible, else bump this one. */
      if(i > 1 && samples_count[i - 2] != samples_count[i - 1] - 1)
        samples_count[i - 1] -= 1;
      else
        nominal_idx += 1;
    }
    samples_count[i] = (nominal_idx < DT_AC_TABLE_SIZE - 1) ? nominal_idx : DT_AC_TABLE_SIZE - 1;
    min_index = nominal_idx;
    last_idx = min_index;
  }

  int total_samples = 0;

  /* First interval: [0, keys[0]) */
  {
    const int samples = samples_count[0];
    if(samples > 0)
    {
      const float delta = (keys[0] - 0.0f) / (float)samples;
      for(int j = 0; j < samples; j++)
        hues[DT_AC_BASE_INDEX + total_samples + j] = (float)j * delta;
    }
    total_samples += samples;
  }

  /* Middle intervals: [keys[i-1], keys[i]) */
  int i;
  for(i = 1; i < n_keys; i++)
  {
    const int samples = samples_count[i] - samples_count[i - 1];
    if(samples > 0)
    {
      const float delta = (keys[i] - keys[i - 1]) / (float)samples;
      for(int j = 0; j < samples; j++)
        hues[DT_AC_BASE_INDEX + total_samples + j] = keys[i - 1] + (float)j * delta;
    }
    total_samples += samples;
  }

  /* Final interval: [keys[n_keys-1], 360) */
  {
    const int samples = DT_AC_TABLE_SIZE - total_samples;
    if(samples > 0)
    {
      const float delta = (360.0f - keys[n_keys - 1]) / (float)samples;
      for(int j = 0; j < samples; j++)
        hues[DT_AC_BASE_INDEX + total_samples + j] = keys[n_keys - 1] + (float)j * delta;
    }
  }

  /* Padding entries for wraparound interpolation */
  hues[0] = hues[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE - 1] - 360.0f;
  hues[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE] = hues[DT_AC_BASE_INDEX] + 360.0f;
}

/* ---- Find display cusp for a given hue (interval + binary search) ---- */
/* Matches CTL find_display_cusp_for_hue (lines 1239-1323) */
static inline void _ac_find_display_cusp_for_hue(float jmh_out[2], float hue,
    const _ac_corner_tables_t *limit,
    const dt_ac_context_t *ctx)
{
  /* Find the correct edge: first corner whose hue exceeds the target */
  int upper = 1;
  for(int i = upper; i < DT_AC_TOTAL_CORNER_COUNT; i++)
  {
    if(limit->jmh[i][2] > hue)
    {
      upper = i;
      break;
    }
  }
  const int lower = upper - 1;

  /* Exact match at a corner */
  if(limit->jmh[lower][2] == hue)
  {
    jmh_out[0] = limit->jmh[lower][0];
    jmh_out[1] = limit->jmh[lower][1];
    return;
  }

  const float *cusp_lower = limit->rgb[lower];
  const float *cusp_upper = limit->rgb[upper];

  float lo_t = 0.0f, hi_t = 1.0f;
  while(hi_t - lo_t > DT_AC_DISPLAY_CUSP_TOL)
  {
    const float t = 0.5f * (lo_t + hi_t);
    float rgb[3];
    for(int c = 0; c < 3; c++)
      rgb[c] = cusp_lower[c] + t * (cusp_upper[c] - cusp_lower[c]);
    float jmh[3];
    _ac_rgb_to_jmh_prim(jmh, rgb, 1, ctx);

    if(jmh[2] < limit->jmh[lower][2])
      hi_t = t;
    else if(jmh[2] >= limit->jmh[upper][2])
      lo_t = t;
    else if(jmh[2] > hue)
      hi_t = t;
    else
      lo_t = t;
  }

  const float t = 0.5f * (lo_t + hi_t);
  float rgb[3];
  for(int c = 0; c < 3; c++)
    rgb[c] = cusp_lower[c] + t * (cusp_upper[c] - cusp_lower[c]);
  float jmh[3];
  _ac_rgb_to_jmh_prim(jmh, rgb, 1, ctx);

  jmh_out[0] = jmh[0];
  jmh_out[1] = jmh[1];
}

/* ---- Build full cusp table ---- */
/* Matches CTL build_cusp_table (lines 1325-1353) */
static inline void _ac_build_cusp_table(float cusps[DT_AC_GAMUT_TABLE_SIZE][3],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const _ac_corner_tables_t *limit,
    const dt_ac_context_t *ctx)
{
  const float smooth_factor = 1.0f
    + DT_AC_GAMUT_SMOOTH_M * DT_AC_GAMUT_SMOOTH_CUSPS;

  for(int i = 0; i < DT_AC_TABLE_SIZE; i++)
  {
    const int idx = DT_AC_BASE_INDEX + i;
    const float hue = hues[idx];
    float jm[2];
    _ac_find_display_cusp_for_hue(jm, hue, limit, ctx);
    cusps[idx][0] = jm[0];
    cusps[idx][1] = jm[1] * smooth_factor;
    cusps[idx][2] = hue;
  }

  /* Padding entries for wraparound interpolation */
  cusps[0][0] = cusps[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE - 1][0];
  cusps[0][1] = cusps[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE - 1][1];
  cusps[0][2] = hues[0];
  cusps[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE][0] = cusps[DT_AC_BASE_INDEX][0];
  cusps[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE][1] = cusps[DT_AC_BASE_INDEX][1];
  cusps[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE][2] = hues[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE];
}

/* ---- Make non-uniform hue gamut table (orchestrator) ---- */
/* Matches CTL reference: builds hue table from 12 cube-corner hues,
 * then constructs the cusp table at those non-uniform positions. */
static inline void _ac_make_hue_table(float hues[DT_AC_GAMUT_TABLE_SIZE],
    float cusps[DT_AC_GAMUT_TABLE_SIZE][3],
    int prim_set, float peak_scale, const dt_ac_context_t *ctx)
{
  _ac_corner_tables_t limit_tbl;
  _ac_build_limiting_cusp_corners_tables(&limit_tbl, prim_set, peak_scale, ctx);

  _ac_corner_tables_t reach_tbl;
  _ac_find_reach_corners_table(&reach_tbl, ctx);

  /* Build non-uniform hue table from sorted cube-corner hues */
  _ac_build_hue_table(hues, &limit_tbl, &reach_tbl);

  _ac_build_cusp_table(cusps, hues, &limit_tbl, ctx);
}

/* ---- Reach M table (binary search reach M at limit_J_max) ---- */
static inline void _ac_make_reach_m_table(float reach_m[DT_AC_GAMUT_TABLE_SIZE],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const dt_ac_context_t *ctx)
{
  float xyz_to_reach[9];
  float prim_to_xyz[9];
  _ac_get_rgb_to_xyz(prim_to_xyz, 2);
  _mat_inv_3x3(xyz_to_reach, prim_to_xyz);

  for(int i = 0; i < DT_AC_TABLE_SIZE; i++)
  {
    const float h = hues[DT_AC_BASE_INDEX + i];
    const float hr = h * (float)(M_PI / 180.0);
    float best_m = 0.0f;
    float lo_m = 0.0f, hi_m = 500.0f;

    for(int iter = 0; iter < 32; iter++)
    {
      const float mid_m = 0.5f * (lo_m + hi_m);

      const float A = ctx->a_w * powf(fmaxf(ctx->limit_j_max / 100.0f, 1e-12f), ctx->inv_cz);
      const float gamma_v = mid_m / DT_AC_M_SCALE;
      const float a_op = gamma_v * cosf(hr);
      const float b_op = gamma_v * sinf(hr);
      float p_in[3] = { A, a_op, b_op };
      float rgb_a[3], rgb_c[3], xyz[3], rgb_reach[3];
      _mat_apply(rgb_a, dt_ac_panlrcm, p_in);
      for(int c = 0; c < 3; c++) rgb_a[c] /= 1403.0f;
      _ac_nlc_inv(rgb_a, rgb_c);
      for(int c = 0; c < 3; c++)
        rgb_c[c] /= fmaxf(ctx->d_rgb[c], 1e-12f);
      _mat_apply(xyz, dt_ac_m16_inv, rgb_c);
      for(int c = 0; c < 3; c++) xyz[c] /= DT_AC_REF_LUM;
      _mat_apply(rgb_reach, xyz_to_reach, xyz);

      int in_gamut = 1;
      for(int c = 0; c < 3; c++)
        if(rgb_reach[c] < -0.001f)
          in_gamut = 0;

      if(in_gamut)
      {
        best_m = mid_m;
        lo_m = mid_m;
      }
      else
      {
        hi_m = mid_m;
      }
    }
    reach_m[DT_AC_BASE_INDEX + i] = best_m;
  }

  /* Padding entries for wraparound interpolation */
  reach_m[0] = reach_m[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE - 1];
  reach_m[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE] = reach_m[DT_AC_BASE_INDEX];
}

/* ---- Determine hue linearity search range (non-uniform table) ---- */
/* For each entry whose stored hue is < 360, computes how far the actual
 * table index deviates from the O(1) uniform estimate `(int)hues[idx]`.
 * Entries ≥ 360 live in the wrap zone and are served by a binary-search
 * fallback in `_ac_lookup_hue_index`. */
static inline void _ac_determine_search_range(int range[2],
    const float hues[DT_AC_GAMUT_TABLE_SIZE])
{
  int max_back = 1;
  int max_fwd = 1;

  for(int i = 0; i < DT_AC_TABLE_SIZE; i++)
  {
    const int idx = DT_AC_BASE_INDEX + i;
    const float h = hues[idx];
    if(h >= 360.0f) continue; /* handled by wrap-zone binary search */
    const int uniform_idx = DT_AC_BASE_INDEX + (int)h;
    const int delta = idx - uniform_idx;

    if(delta > 0)
    {
      if(delta > max_fwd) max_fwd = delta;
    }
    else
    {
      const int back = -delta;
      if(back > max_back) max_back = back;
    }
  }

  range[0] = max_back;
  range[1] = max_fwd;
}

/* ====================================================================
 * GAMUT COMPRESSION — PER-PIXEL FUNCTIONS
 *
 * Translation of ACES 2.0 reference:
 *   aces-core/lib/Lib.Academy.OutputTransform.ctl
 *   lines 348–916
 *
 * These operate on a single pixel in JMh space.
 * ==================================================================== */

/* ---- Hue index lookup (bounded linear scan, wrap-safe) ---- */
/* Returns the index `lo` such that `hues[lo] <= hw < hues[lo+1]` where `hw` is
 * the wrapped hue.  Starts from the O(1) uniform estimate `(int)hw` and walks
 * within the precomputed `search_range` to find the correct interval.  When
 * `hw` falls below the first table entry (non-uniform wrap zone), performs a
 * binary search using `hw + 360` against the stored (monotonic, unwrapped)
 * table values. */
static inline int _ac_lookup_hue_index(float h, const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const int search_range[2])
{
  const float hw = _ac_wrap_hue(h);
  const float smin = hues[DT_AC_BASE_INDEX];

  /* Wrap zone: hues below the first table entry.  The corresponding table
   * entries sit at the end of the table as unwrapped values in
   * [keys[n-1], keys[0]+360).  Use a binary search on hw + 360. */
  if(hw < smin)
  {
    const float target = hw + 360.0f;
    int lo = DT_AC_BASE_INDEX;
    int hi = DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE;
    while(hi - lo > 1)
    {
      const int mid = (lo + hi) / 2;
      if(hues[mid] > target)
        hi = mid;
      else
        lo = mid;
    }
    return lo;
  }

  int base = DT_AC_BASE_INDEX + (int)hw;

  int lo = base - search_range[0];
  if(lo < DT_AC_BASE_INDEX) lo = DT_AC_BASE_INDEX;
  int hi = base + search_range[1];
  if(hi > DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE) hi = DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE;

  /* Walk backward to find lower bound */
  while(lo > DT_AC_BASE_INDEX && hw < hues[lo])
    lo--;

  /* Walk forward to find upper bound */
  while(hi < DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE && hw >= hues[hi])
    hi++;

  /* Walk forward from lo to reach insertion interval */
  while(lo < hi - 1 && hw >= hues[lo + 1])
    lo++;

  return lo;
}

/* ---- Reach M from table (bounded scan, non-uniform hue table) ---- */
static inline float _ac_reach_m_from_table(float h,
    const float reach_m[DT_AC_GAMUT_TABLE_SIZE],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const int search_range[2])
{
  const int lo = _ac_lookup_hue_index(h, hues, search_range);
  const int hi = lo + 1;
  const float hw = _ac_wrap_hue(h);
  const float t = (hw - hues[lo]) / fmaxf(hues[hi] - hues[lo], 1e-6f);
  return reach_m[lo] + t * (reach_m[hi] - reach_m[lo]);
}

/* ---- Cusp from table (bounded scan, non-uniform hue table) ---- */
static inline void _ac_cusp_from_table(float cusp_out[2], float h,
    const float cusps[DT_AC_GAMUT_TABLE_SIZE][3],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const int search_range[2])
{
  const int lo = _ac_lookup_hue_index(h, hues, search_range);
  const int hi = lo + 1;
  const float hw = _ac_wrap_hue(h);
  const float t = (hw - hues[lo]) / fmaxf(hues[hi] - hues[lo], 1e-6f);

  cusp_out[0] = cusps[lo][0] + t * (cusps[hi][0] - cusps[lo][0]);
  cusp_out[1] = cusps[lo][1] + t * (cusps[hi][1] - cusps[lo][1]);
}

/* ---- Upper hull gamma from table (bounded scan, non-uniform hue table) ---- */
static inline float _ac_hue_upper_hull_gamma(float h,
    const float gamma_table[DT_AC_GAMUT_TABLE_SIZE],
    const float hues[DT_AC_GAMUT_TABLE_SIZE],
    const int search_range[2])
{
  const int lo = _ac_lookup_hue_index(h, hues, search_range);
  const int hi = lo + 1;
  const float hw = _ac_wrap_hue(h);
  const float t = (hw - hues[lo]) / fmaxf(hues[hi] - hues[lo], 1e-6f);
  return gamma_table[lo] + t * (gamma_table[hi] - gamma_table[lo]);
}

/* ---- Compute focus gain (slope_gain, reference match) ---- */
static inline float _ac_get_focus_gain(float j, float analytical_threshold,
    float limit_j_max, float focus_dist)
{
  float gain = limit_j_max * focus_dist;

  if(j > analytical_threshold)
  {
    const float denom = fmaxf(limit_j_max - j, 1e-4f);
    const float ratio = (limit_j_max - analytical_threshold) / denom;
    const float gain_adj = log10f(ratio) * log10f(ratio) + 1.0f;
    gain *= gain_adj;
  }

  return gain;
}

/* ---- Solve J intersection (quadratic) ---- */
static inline float _ac_solve_J_intersect(float j, float m,
    float focus_j, float limit_j_max, float slope_gain)
{
  if(m <= 0.0f) return j;

  const float m_scaled = m / slope_gain;
  const float a = m_scaled / focus_j;

  if(j < focus_j)
  {
    /* J_intersect solves: focus_J * (J - intersect) = M * intersect / slope_gain
     * → a*intersect² + b*intersect + c = 0  with:
     */
    const float b = 1.0f - m_scaled;
    const float c = -j;
    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float root = sqrtf(disc);
    return -2.0f * c / (b + root);
  }
  else
  {
    /* For J >= focus_J, equation is reflected about limit_J_max */
    const float b = -(1.0f + m_scaled + limit_j_max * a);
    const float c = limit_j_max * m_scaled + j;
    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float root = sqrtf(disc);
    return -2.0f * c / (b - root);
  }
}

/* ---- Compute compression vector slope ---- */
static inline float _ac_compression_slope(float intersect_j, float focus_j,
    float limit_j_max, float slope_gain)
{
  float direction_scalar;
  if(intersect_j < focus_j)
    direction_scalar = intersect_j;
  else
    direction_scalar = limit_j_max - intersect_j;

  return direction_scalar * (intersect_j - focus_j) / (focus_j * slope_gain);
}

/* ---- Gamma test point (one per test position) ---- */
typedef struct
{
  float test_JMh[3];
  float J_intersect_source;
  float slope;
  float J_intersect_cusp;
} _ac_gamma_test_point_t;

/* ---- Generate gamma test data (5 test J positions per hue) ---- */
/* Matches reference generate_gamma_test_data (OutputTransform.ctl lines 1456-1485):
 *   test positions {0.01, 0.1, 0.5, 0.8, 0.99} between cusp_J and limit_J_max,
 *   M = cusp_M (constant),
 *   reuses get_focus_gain / solve_J_intersect / compute_compression_vector_slope
 *   (the same functions used by compress_gamut itself). */
static inline void _ac_gamma_test_data(_ac_gamma_test_point_t tests[5],
    float cusp_j, float cusp_m, float hue,
    float limit_j_max, float mid_J, float focus_dist)
{
  const float test_positions[5] = { 0.01f, 0.10f, 0.50f, 0.80f, 0.99f };

  const float blend_weight = fminf(1.0f, DT_AC_GAMUT_CUSP_MID_BLEND - cusp_j / limit_j_max);
  const float focus_j = cusp_j + blend_weight * (mid_J - cusp_j);
  const float analytical_threshold = cusp_j + DT_AC_GAMUT_FOCUS_GAIN_BLEND
                                     * (limit_j_max - cusp_j);

  for(int i = 0; i < 5; i++)
  {
    const float test_j = cusp_j + test_positions[i] * (limit_j_max - cusp_j);
    const float slope_gain = _ac_get_focus_gain(test_j, analytical_threshold,
                                                limit_j_max, focus_dist);
    const float j_intersect = _ac_solve_J_intersect(test_j, cusp_m, focus_j,
                                                    limit_j_max, slope_gain);
    const float slope = _ac_compression_slope(j_intersect, focus_j,
                                              limit_j_max, slope_gain);
    const float j_cusp = _ac_solve_J_intersect(cusp_j, cusp_m, focus_j,
                                               limit_j_max, slope_gain);

    tests[i].test_JMh[0] = test_j;
    tests[i].test_JMh[1] = cusp_m;
    tests[i].test_JMh[2] = hue;
    tests[i].J_intersect_source = j_intersect;
    tests[i].slope = slope;
    tests[i].J_intersect_cusp = j_cusp;
  }
}

/* ---- Estimate line and boundary intersection M (reference match) ---- */
static inline float _ac_estimate_intersect_M(float j_axis_intersect, float slope,
    float inv_gamma, float j_max, float m_max, float j_intersection_ref)
{
  if(slope >= 0.0f || m_max <= 0.0f) return m_max;

  /* Project the J-axis intercept through the boundary power law
   * using the reference point for scaling */
  const float j_ref = fmaxf(j_intersection_ref, 1e-12f);
  const float normalised_j = j_axis_intersect / j_ref;
  const float inv_gamma_safe = fmaxf(inv_gamma, 1e-6f);
  const float shifted_intersection = j_ref * powf(fmaxf(normalised_j, 1e-12f), inv_gamma_safe);

  /* Find intersection of two lines:
   *   line from origin to (J_max, M_max):   J = (J_max / M_max) * M
   *   line from (shifted, 0) with slope:    J = slope * M + shifted
   * Solve:  slope * M + shifted = (J_max / M_max) * M
   *         shifted = (J_max/M_max - slope) * M
   *         M = shifted * M_max / (J_max - slope * M_max)
   */
  const float denom = j_max - slope * m_max;
  if(denom <= 0.0f) return m_max;
  const float m_est = shifted_intersection * m_max / denom;
  return fminf(fmaxf(m_est, 0.0f), m_max);
}

/* ---- Find gamut boundary intersection M (reference match) ---- */
static inline float _ac_find_boundary_M(float cusp_j, float cusp_m,
    float limit_j_max, float gamma_top_inv, float gamma_bottom_inv,
    float j_intersect_source, float slope, float j_intersect_cusp)
{
  if(cusp_m <= 0.0f || cusp_j <= 0.0f) return 0.0f;

  /* Lower hull: boundary is power law from (0,0) up to (cusp_J, cusp_M) */
  const float M_boundary_lower = _ac_estimate_intersect_M(j_intersect_source, slope,
      gamma_bottom_inv, cusp_j, cusp_m, j_intersect_cusp);

  /* Upper hull: flip about limit_J_max, negate slope, reuse the same estimator */
  const float f_intersect_cusp = limit_j_max - j_intersect_cusp;
  const float f_intersect_source = limit_j_max - j_intersect_source;
  const float f_cusp_j = limit_j_max - cusp_j;
  const float M_boundary_upper = _ac_estimate_intersect_M(f_intersect_source, -slope,
      gamma_top_inv, f_cusp_j, cusp_m, f_intersect_cusp);

  /* Smooth min between the two boundary estimates */
  const float m_blend = _ac_smin(M_boundary_lower, M_boundary_upper,
      DT_AC_GAMUT_SMOOTH_CUSPS * cusp_m);
  return m_blend;
}

/* ---- Evaluate gamma fit (reference match) ---- */
/* Matches reference evaluate_gamma_fit (OutputTransform.ctl lines 1487-1524):
 *   projects each test point through find_gamut_boundary_intersection,
 *   converts to limiting RGB, and checks outside_hull(rgb, peak/ref_lum). */
static inline int _ac_eval_gamma_fit(
    const _ac_gamma_test_point_t tests[5],
    float cusp_j, float cusp_m,
    float top_gamma_inv, float peak_luminance,
    float limit_j_max, float lower_hull_gamma_inv,
    const dt_ac_context_t *ctx)
{
  const float luminance_limit = peak_luminance / DT_AC_REF_LUM;
  float xyz_to_limit[9];
  float prim_to_xyz[9];
  _ac_get_rgb_to_xyz(prim_to_xyz, 1);
  _mat_inv_3x3(xyz_to_limit, prim_to_xyz);

  for(int i = 0; i < 5; i++)
  {
    const float approx_limit_M = _ac_find_boundary_M(cusp_j, cusp_m,
        limit_j_max, top_gamma_inv, lower_hull_gamma_inv,
        tests[i].J_intersect_source, tests[i].slope, tests[i].J_intersect_cusp);

    if(approx_limit_M <= 0.0f) return 0;

    const float approx_limit_J = tests[i].J_intersect_source + tests[i].slope * approx_limit_M;

    float approx_jmh[3] = { approx_limit_J, approx_limit_M, tests[i].test_JMh[2] };
    float xyz[3];
    _ac_jmh_to_xyz(approx_jmh, ctx->d_rgb, ctx->a_w, ctx->z, xyz);

    for(int c = 0; c < 3; c++) xyz[c] /= DT_AC_REF_LUM;

    float rgb_limit[3];
    _mat_apply(rgb_limit, xyz_to_limit, xyz);

    int outside = 0;
    for(int c = 0; c < 3; c++)
      if(rgb_limit[c] > luminance_limit)
        outside = 1;

    if(!outside) return 0;
  }
  return 1;
}

/* ---- Make upper hull gamma table (reference match) ---- */
/* Matches reference make_upper_hull_gamma_table (OutputTransform.ctl lines 1526-1604):
 *   step search gamma 0→5 step 0.4, then binary search to 1e-5 accuracy,
 *   stores 1/hi (i.e. gamma_inv) in the table. */
static inline void _ac_make_upper_hull_gamma_table(
    float gamma_table[DT_AC_GAMUT_TABLE_SIZE],
    const float cusps[DT_AC_GAMUT_TABLE_SIZE][3],
    float limit_j_max, float lower_hull_gamma_inv,
    float peak_luminance, float mid_J, float focus_dist,
    const dt_ac_context_t *ctx)
{
  const float gamma_minimum = 0.0f;
  const float gamma_maximum = 5.0f;
  const float gamma_search_step = 0.4f;
  const float gamma_accuracy = 1e-5f;

  for(int i = 0; i < DT_AC_TABLE_SIZE; i++)
  {
    const int idx = DT_AC_BASE_INDEX + i;
    const float cusp_j = cusps[idx][0];
    const float cusp_m = cusps[idx][1];
    const float hue = cusps[idx][2];

    if(cusp_m <= 0.0f || cusp_j <= 0.0f)
    {
      gamma_table[idx] = 0.0f;
      continue;
    }

    _ac_gamma_test_point_t tests[5];
    _ac_gamma_test_data(tests, cusp_j, cusp_m, hue,
                        limit_j_max, mid_J, focus_dist);

    float lo = gamma_minimum;
    float hi = lo + gamma_search_step;
    int found = 0;
    while(!found && hi < gamma_maximum)
    {
      if(_ac_eval_gamma_fit(tests, cusp_j, cusp_m,
          1.0f / hi, peak_luminance, limit_j_max, lower_hull_gamma_inv, ctx))
      {
        found = 1;
      }
      else
      {
        lo = hi;
        hi = hi + gamma_search_step;
      }
    }

    while((hi - lo) > gamma_accuracy)
    {
      const float mid = 0.5f * (hi + lo);
      if(_ac_eval_gamma_fit(tests, cusp_j, cusp_m,
          1.0f / mid, peak_luminance, limit_j_max, lower_hull_gamma_inv, ctx))
        hi = mid;
      else
        lo = mid;
    }

    gamma_table[idx] = 1.0f / hi;
  }

  /* Padding entries for wraparound interpolation */
  gamma_table[0] = gamma_table[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE - 1];
  gamma_table[DT_AC_BASE_INDEX + DT_AC_TABLE_SIZE] = gamma_table[DT_AC_BASE_INDEX];
}

/* ---- Remap M (adaptive Reinhard, reference match) ---- */
static inline float _ac_remap_M(float m, float gamut_boundary_m,
    float reach_boundary_m)
{
  if(m <= 0.0f || gamut_boundary_m <= 0.0f || reach_boundary_m <= gamut_boundary_m) return m;

  const float boundary_ratio = gamut_boundary_m / reach_boundary_m;
  const float proportion = fmaxf(boundary_ratio, DT_AC_GAMUT_COMPRESSION_THR);
  const float threshold = proportion * gamut_boundary_m;

  if(m <= threshold || proportion >= 1.0f) return m;

  const float m_offset = m - threshold;
  const float gamut_offset = gamut_boundary_m - threshold;
  const float reach_offset = reach_boundary_m - threshold;
  const float scale = reach_offset / ((reach_offset / gamut_offset) - 1.0f);
  const float nd = m_offset / scale;

  /* Reinhard: scale * nd / (1 + nd) */
  return threshold + scale * nd / (1.0f + nd);
}

/* ---- Core compress_gamut function ---- */
static inline void _ac_compress_gamut(float jmh[3], float jx,
    const dt_ac_context_t *ctx)
{
  float j = jmh[0], m = jmh[1], h = jmh[2];
  if(!isfinite(j) || !isfinite(m)) return;
  if(m <= 0.0f || j <= 0.0f) return;

  const float limit_j_max = ctx->limit_j_max;
  if(limit_j_max <= 0.0f) return;

  /* Get hue-dependent params */
  float cusp[2];
  _ac_cusp_from_table(cusp, h, ctx->table_gamut_cusps, ctx->table_hues, ctx->hue_search_range);
  const float cusp_j = cusp[0], cusp_m = cusp[1];

  if(!isfinite(cusp_j) || !isfinite(cusp_m) || cusp_m <= 0.0f || cusp_j <= 0.0f)
    return;

  /* Focus J: lerp(cusp_J, mid_J, min(1, cusp_mid_blend - cusp_J/limit_J_max)) */
  const float blend_weight = fminf(1.0f, DT_AC_GAMUT_CUSP_MID_BLEND - cusp_j / limit_j_max);
  const float focus_j = cusp_j + blend_weight * (ctx->mid_J - cusp_j);
  if(!isfinite(focus_j) || focus_j <= 0.0f) return;

  /* Analytical threshold: lerp(cusp_J, limit_J_max, FOCUS_GAIN_BLEND) */
  const float analytical_threshold = cusp_j + DT_AC_GAMUT_FOCUS_GAIN_BLEND
                                     * (limit_j_max - cusp_j);

  const float gamma_top_inv = fmaxf(
      _ac_hue_upper_hull_gamma(h, ctx->table_upper_hull_gamma, ctx->table_hues, ctx->hue_search_range), 1e-6f);
  const float gamma_bottom_inv = ctx->lower_hull_gamma_inv;

  const float slope_gain = _ac_get_focus_gain(jx, analytical_threshold,
      limit_j_max, ctx->focus_dist);
  if(!isfinite(slope_gain) || slope_gain <= 0.0f) return;

  /* Solve intersections */
  const float j_intersect_source = _ac_solve_J_intersect(j, m, focus_j,
      limit_j_max, slope_gain);
  const float j_intersect_cusp = _ac_solve_J_intersect(cusp_j, cusp_m, focus_j,
      limit_j_max, slope_gain);

  if(!isfinite(j_intersect_source) || !isfinite(j_intersect_cusp)) return;

  /* Compute compression vector slope */
  const float slope = _ac_compression_slope(j_intersect_source, focus_j,
      limit_j_max, slope_gain);

  if(!isfinite(slope) || slope >= 0.0f) return;

  /* Gamut boundary M at the source intersection */
  const float gamut_boundary_m = _ac_find_boundary_M(cusp_j, cusp_m,
      limit_j_max, gamma_top_inv, gamma_bottom_inv,
      j_intersect_source, slope, j_intersect_cusp);

  if(!isfinite(gamut_boundary_m) || gamut_boundary_m <= 0.0f) return;

  /* Reach maximum M from table */
  const float reach_max_m = _ac_reach_m_from_table(h,
      ctx->table_reach_m, ctx->table_hues, ctx->hue_search_range);
  if(!isfinite(reach_max_m) || reach_max_m <= 0.0f) return;

  /* Reach boundary M — uses model_gamma_inv and limit_J_max as ref (reference match) */
  const float reach_boundary_m = _ac_estimate_intersect_M(
      j_intersect_source, slope, ctx->model_gamma_inv,
      limit_j_max, reach_max_m, limit_j_max);
  if(!isfinite(reach_boundary_m)) return;

  /* Remap M (adaptive threshold, reference match) */
  const float new_m = _ac_remap_M(m, gamut_boundary_m, reach_boundary_m);
  if(!isfinite(new_m)) return;

  jmh[0] = j_intersect_source + new_m * slope;
  jmh[1] = fmaxf(new_m, 0.0f);
}

/* ---- gamut_compress_fwd ---- */
static inline void _ac_gamut_compress_fwd(float jmh[3],
    const dt_ac_context_t *ctx)
{
  float j = jmh[0], m = jmh[1];

  if(!isfinite(j + m)) return;
  if(j <= 0.0f || m <= 0.0f) return;

  const float jx = j;

  /* Apply compression using J as the reference Jx */
  _ac_compress_gamut(jmh, jx, ctx);
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
  ctx->surround_idx = p->surround;
  ctx->sdr_clip_enable = p->sdr_clip_enable ? 1 : 0;
  ctx->sdr_clip_softness = p->sdr_clip_softness;
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

  /* SSTS init (ACES 2.0 official tonescale params) */
  dt_ac_ssts_init(&ctx->ssts, (double)p->peak_luminance, p->surround);

  /* Forward limit for AP1 clamp — matches CTL's r_hit / n_r in init_TSParams */
  {
    const float peak = fmaxf(p->peak_luminance, 1.0f);
    const float log_ratio = logf(peak / 100.0f) / logf(10000.0f / 100.0f);
    const float r_hit = 128.0f + (896.0f - 128.0f) * fminf(fmaxf(log_ratio, 0.0f), 1.0f);
    ctx->forward_limit = r_hit / 100.0f;
  }

  /* CAT16 CAM precomputation */
  {
    float n_unused;
    _ac_hellwig_precompute(dt_ac_aces_white_xyz,
                           &ctx->f_l, &n_unused, &ctx->z,
                           &ctx->cz, &ctx->inv_cz,
                           &ctx->f_l_n, &ctx->a_w, &ctx->a_w_j,
                           ctx->d_rgb);
  }

  /* Dynamic chroma compression parameters (ACES 2.0 official) */
  {
    const float peak = fmaxf(p->peak_luminance, 1.0f);

    /* model_gamma = surround.c * (1.48 + sqrt(Y_b / ref_lum));
     * model_gamma_inv = 1 / model_gamma */
    const float model_gamma = DT_AC_SURR_C * (1.48f + sqrtf(DT_AC_YB / DT_AC_REF_LUM));
    ctx->model_gamma_inv = 1.0f / model_gamma;

    /* chroma_compress_scale = pow(0.03379 * peak, 0.30596) - 0.45135 */
    ctx->chroma_compress_scale = powf(0.03379f * peak, 0.30596f) - 0.45135f;

    /* log_peak = log10(peak / 100) */
    const float log_peak = log10f(peak / DT_AC_REF_LUM);

    /* sat = max(0.2, SAT_BASE - SAT_BASE * SAT_FACT * log_peak) */
    ctx->cc_sat = fmaxf(0.2f, DT_AC_SAT_BASE - DT_AC_SAT_BASE * DT_AC_SAT_FACT * log_peak);

    /* sat_thr = EXPAND_THR / peak */
    ctx->cc_sat_thr = DT_AC_EXPAND_THR / peak;

    /* compr = COMPR_BASE + COMPR_BASE * COMPR_FACT * log_peak */
    ctx->cc_compr = DT_AC_COMPR_BASE + DT_AC_COMPR_BASE * DT_AC_COMPR_FACT * log_peak;

    /* limit_j_max = J(peak_luminance) via Y_to_J */
    ctx->limit_j_max = _ac_y_to_j(peak, ctx->f_l_n, ctx->a_w_j, ctx->cz);
  }

  /* Gamut compression parameters and tables (ACES 2.0 reference) */
  {
    const float peak = fmaxf(p->peak_luminance, 1.0f);
    const float log_peak = log10f(peak / DT_AC_REF_LUM);

    /* mid_J = Y_to_J(c_t * ref_lum, ...) from SSTS grey anchor (reference match) */
    {
      const double w_i = log((double)peak / (double)DT_AC_REF_LUM) / log(2.0);
      const double w_g = 0.14;
      const double c_d = 10.013;
      const double n_r = 100.0;
      const double c_t = (c_d / n_r) * (1.0 + w_i * w_g);
      ctx->mid_J = _ac_y_to_j((float)(c_t * (double)DT_AC_REF_LUM),
                               ctx->f_l_n, ctx->a_w_j, ctx->cz);
    }

    /* focus_dist = 1.35 + 1.35 * 1.75 * log10(peak / ref_lum) (reference match) */
    ctx->focus_dist = DT_AC_GAMUT_FOCUS_DISTANCE
                      + DT_AC_GAMUT_FOCUS_DISTANCE * DT_AC_GAMUT_FOCUS_DIST_SCALING
                        * log10f(peak / DT_AC_REF_LUM);

    /* lower_hull_gamma_inv = 1.0 / (1.14 + 0.07 * log10(peak / ref_lum)) */
    const float lower_hull_gamma = 1.14f + 0.07f * log_peak;
    ctx->lower_hull_gamma_inv = 1.0f / fmaxf(lower_hull_gamma, 0.5f);

    /* Build all gamut compression tables.
     * Limiting (display) primaries = AP1 (1) — matches the fixed AP1 output
     * gamut used throughout the rest of the pipeline (Step 1/9 of
     * dt_ac_pipeline_eval). AP0 must NOT be used here: it is the reach
     * primaries (already handled separately inside _ac_find_reach_corners_table /
     * _ac_make_reach_m_table). AP0's blue primary sits at the edge of the
     * spectral locus and yields a degenerate (J=0) CAM response, which
     * corrupted the whole blue-hue region of the cusp table. */

    /* Hue table + cusp table (non-uniform sampling from cube-corner hues) */
    const float peak_scale = peak / DT_AC_REF_LUM;
    _ac_make_hue_table(ctx->table_hues, ctx->table_gamut_cusps, 1, peak_scale, ctx);

    /* Reach M table at limit_J_max (iterates over non-uniform hues) */
    _ac_make_reach_m_table(ctx->table_reach_m, ctx->table_hues, ctx);

    /* Upper hull gamma table */
    _ac_make_upper_hull_gamma_table(ctx->table_upper_hull_gamma,
        ctx->table_gamut_cusps, ctx->limit_j_max,
        ctx->lower_hull_gamma_inv, peak, ctx->mid_J, ctx->focus_dist, ctx);

    /* Hue linearity search range */
    _ac_determine_search_range(ctx->hue_search_range, ctx->table_hues);
  }
}

/* ====================================================================
 * Forward Tonemap & Compress (inside JMh space)
 *
 * Matches ACES 2.0 official tonemap_and_compress_fwd:
 *   J → Y → tonescale(Y/ref_lum) → J_ts (display J)
 *   M ← chroma_compress(M, J_ts, orig_J) via ctx parameters
 *
 * The SSTS (tonescale) is applied on the achromatic J-derived Y,
 * NOT per-channel on AP1 before the CAM.
 * ==================================================================== */

static inline void _ac_tonemap_and_compress_fwd(float jmh[3],
    const dt_ac_context_t *ctx)
{
  const float orig_j = jmh[0];

  /* J → Y (nits) → scene-linear normalized */
  const float linear = _ac_j_to_y(orig_j, ctx->f_l_n, ctx->a_w_j, ctx->inv_cz)
                       / DT_AC_REF_LUM;

  /* Apply SSTS on luminance Y */
  const float tonemapped_y = dt_ac_ssts_fwd(&ctx->ssts, linear);

  /* Y → J' (display J after tone mapping) */
  jmh[0] = _ac_y_to_j(tonemapped_y, ctx->f_l_n, ctx->a_w_j, ctx->cz);

  /* Chroma compression with original J reference */
  _ac_chroma_compress(jmh, orig_j, ctx);
}

/* ====================================================================
 * ACES 2.0 CAM DRT Pipeline Evaluation (per-pixel)
 *
 *   pipe_RGB  →  AP1 (D60)  →  XYZ (D60)
 *     →  ×100 to absolute nits
 *     →  CAT16 JMh
 *     →  _ac_tonemap_and_compress_fwd  (SSTS on Y + chroma compress)
 *     →  gamut_compress_fwd  (gamut boundary compression)
 *     →  JMh  →  XYZ  →  /100
 *     →  AP1 (D60)
 *     →  pipe_RGB
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

  /* Apply exposure */
  for(int c = 0; c < 3; c++) ap1[c] *= ctx->exposure_factor;

  /* Step 2: Clamp AP1 to [0, forward_limit] — CTL clamp_AP0_to_AP1 equivalent */
  for(int c = 0; c < 3; c++)
    ap1[c] = fminf(fmaxf(ap1[c], 0.0f), ctx->forward_limit);

  /* Step 3: AP1 → XYZ D60 */
  float xyz[3];
  _mat_apply(xyz, dt_ac_ap1_to_xyz, ap1);

  /* Step 4: ×100 to absolute nits for CAM */
  for(int c = 0; c < 3; c++) xyz[c] *= DT_AC_REF_LUM;

  /* Step 5: XYZ D60 → CAT16 JMh */
  float jmh[3];
  _ac_xyz_to_jmh(xyz, ctx->d_rgb, ctx->a_w, ctx->z, jmh);

  /* Step 5: Tonemap & compress inside JMh */
  _ac_tonemap_and_compress_fwd(jmh, ctx);

  /* Step 6: Gamut compression (ACES 2.0 reference, after tonemap+chroma comp) */
  _ac_gamut_compress_fwd(jmh, ctx);

  /* Step 7: JMh → XYZ D60 (in nits) */
  float xyz_out[3];
  _ac_jmh_to_xyz(jmh, ctx->d_rgb, ctx->a_w, ctx->z, xyz_out);

  /* Step 8: XYZ D60 → AP1 (back to display-referred) */
  float ap1_out[3];
  _mat_apply(ap1_out, dt_ac_xyz_to_ap1, xyz_out);
  for(int c = 0; c < 3; c++)
    ap1_out[c] = fmaxf(ap1_out[c], 0.0f) / DT_AC_REF_LUM;

  /* Step 9: AP1 → pipe RGB */
  float rgb[3];
  _mat_apply(rgb, ctx->inv_matrix, ap1_out);

  /* Step 10: Hard floor (reference: hardClip = fmax(v, 0.0f)) */
  for(int c = 0; c < 3; c++)
    rgb_out[c] = isfinite(rgb[c]) ? fmaxf(rgb[c], 0.0f) : 0.0f;

  /* Optional Step 11 (NOT part of the official reference, which never
   * ceiling-clamps its output): soft clip to display white (code-value 1.0).
   * Uses _ac_smin for a smooth roll-off when the channel exceeds the ceiling. */
  if(ctx->sdr_clip_enable)
    for(int c = 0; c < 3; c++)
      rgb_out[c] = _ac_smin(rgb_out[c], 1.0f, ctx->sdr_clip_softness);
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

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 2)
  {
    typedef struct dt_iop_aces20_params_v2_t
    {
      float peak_luminance;
      int surround;
      float exposure_ev;
      int sdr_output_clip;
    } dt_iop_aces20_params_v2_t;

    dt_iop_aces20_params_v2_t *o = (dt_iop_aces20_params_v2_t *)old_params;
    dt_iop_aces20_params_t *n = malloc(sizeof(dt_iop_aces20_params_t));
    n->peak_luminance = o->peak_luminance;
    n->surround = o->surround;
    n->exposure_ev = o->exposure_ev;
    n->sdr_clip_enable = o->sdr_output_clip ? TRUE : FALSE;
    n->sdr_clip_softness = 0.15f;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_aces20_params_t);
    *new_version = 3;
    return 0;
  }

  return 1;
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);
  g_assert(self->default_params != NULL);
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_aces20_params_t *d = (dt_iop_aces20_params_t *)self->default_params;
  const gboolean raw = dt_image_is_rawprepare_supported(&self->dev->image_storage);
  if(raw && dt_is_scene_referred())
    d->exposure_ev = 1.0f;
  else
    d->exposure_ev = 0.0f;
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
  clp->forward_limit    = ctx->forward_limit;
  clp->f_l_n            = ctx->f_l_n;
  clp->a_w              = ctx->a_w;
  clp->z                = ctx->z;
  clp->cz               = ctx->cz;
  clp->inv_cz           = ctx->inv_cz;
  for(int i = 0; i < 3; i++) clp->d_rgb[i] = ctx->d_rgb[i];
  clp->a_w_j            = ctx->a_w_j;

  clp->ssts_s_2 = (float)ctx->ssts.s_2;
  clp->ssts_m_2 = (float)ctx->ssts.m_2;
  clp->ssts_g   = (float)ctx->ssts.g;
  clp->ssts_t_1 = (float)ctx->ssts.t_1;
  clp->ssts_n_r = (float)ctx->ssts.n_r;
  clp->ssts_n   = (float)ctx->ssts.n;

  clp->model_gamma_inv       = ctx->model_gamma_inv;
  clp->chroma_compress_scale = ctx->chroma_compress_scale;
  clp->cc_sat                = ctx->cc_sat;
  clp->cc_sat_thr            = ctx->cc_sat_thr;
  clp->cc_compr              = ctx->cc_compr;
  clp->limit_j_max           = ctx->limit_j_max;

  /* Gamut compression fields */
  clp->mid_J                  = ctx->mid_J;
  clp->focus_dist             = ctx->focus_dist;
  clp->lower_hull_gamma_inv   = ctx->lower_hull_gamma_inv;
  for(int i = 0; i < DT_AC_GAMUT_TABLE_SIZE; i++)
  {
    clp->table_reach_m[i]     = ctx->table_reach_m[i];
    clp->table_hues[i]        = ctx->table_hues[i];
    clp->table_cusp_j[i]      = ctx->table_gamut_cusps[i][0];
    clp->table_cusp_m[i]      = ctx->table_gamut_cusps[i][1];
    clp->table_upper_hull_gamma[i] = ctx->table_upper_hull_gamma[i];
  }
  clp->hue_search_min         = ctx->hue_search_range[0];
  clp->hue_search_max         = ctx->hue_search_range[1];
  clp->sdr_clip_enable        = ctx->sdr_clip_enable;
  clp->sdr_clip_softness      = ctx->sdr_clip_softness;
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

  cl_mem dev_clp = dt_opencl_copy_host_to_device_constant(devid, sizeof(clp), &clp);
  if(!dev_clp) return DT_OPENCL_PROCESS_CL;

  cl_int err = CL_SUCCESS;
  err = dt_opencl_enqueue_kernel_2d_args(
    devid, gd->kernel_aces20, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(dev_clp));

  dt_opencl_release_mem_object(dev_clp);

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
  p.exposure_ev = 1.0f;

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
  gtk_widget_set_visible(g->sdr_clip_softness, p->sdr_clip_enable);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_aces20_gui_data_t *g = IOP_GUI_ALLOC(aces20);

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  /* Exposure */
  g->exposure = dt_bauhaus_slider_from_params(self, "exposure_ev");
  dt_bauhaus_slider_set_format(g->exposure, _(" EV"));
  dt_bauhaus_slider_set_digits(g->exposure, 2);
  gtk_widget_set_tooltip_text(g->exposure,
    _("Exposure compensation applied before tone mapping. "
      "Positive values brighten, negative values darken."));

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

  /* Optional SDR soft clip */
  g->sdr_clip_enable = dt_bauhaus_toggle_from_params(self, "sdr_clip_enable");
  gtk_widget_set_tooltip_text(g->sdr_clip_enable,
    _("Enable soft ceiling output at display white (1.0). "
      "When enabled, values exceeding 1.0 are smoothly rolled off "
      "using a cubic soft clip instead of a hard clamp."));

  g->sdr_clip_softness = dt_bauhaus_slider_from_params(self, "sdr_clip_softness");
  dt_bauhaus_slider_set_factor(g->sdr_clip_softness, 100.0f);
  dt_bauhaus_slider_set_format(g->sdr_clip_softness, _(" %%"));
  dt_bauhaus_slider_set_digits(g->sdr_clip_softness, 1);
  gtk_widget_set_tooltip_text(g->sdr_clip_softness,
    _("Width of the smooth roll-off transition at the ceiling. "
      "Higher values produce a softer knee; 0 gives a hard clip."));
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_aces20_gui_data_t *g = self->gui_data;
  dt_iop_aces20_params_t *p = self->params;

  if(!w || w == g->sdr_clip_enable)
    gtk_widget_set_visible(g->sdr_clip_softness, p->sdr_clip_enable);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

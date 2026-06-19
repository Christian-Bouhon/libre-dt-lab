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
    Acknowledgments and Technical References (Libre DT-lab):

    - ACES 2.0 Single-Stage Tone Scale (SSTS) : Academy Color Encoding System,
      Michaelis-Menten parametric curve with flare compensation. The SSTS defines
      the "texture of light" — the character of tone reproduction from scene-linear
      to display luminance.
      Reference: aces-core / lib / Lib.Academy.Tonescale.ctl

    - Spektrafilm spectral film simulation : Andrea Volpato (2024). Inspiration
      for the spectral gamut management approach — film dye absorption naturally
      limits chroma via smooth asymptotic roll-off in CIE xy chromaticity space,
      preserving perceived hue while compressing out-of-gamut colors.
      https://github.com/andreavolpato/spektrafilm

    - ACES 1.0 Filmic Tone Mapping Curve : Krzysztof Narkowicz (2016). Reference
      for early tone scale work, prior to adopting ACES 2.0 SSTS.
      https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/

    - Bradford chromatic adaptation transform (D50 ↔ D65) : standard CAT used
      for white point adaptation between Rec.2020 D65 and the D50 working space.

    - BT.1886 electro-optical transfer function : ITU-R BT.1886 reference EOTF
      for gamma correction in the display pipeline.

    - CIE 1931 standard observer (XYZ colour matching functions) : foundation
      for all spectral and colorimetric computations.

    - Rec.2020 colour space : ITU-R BT.2020 ultra-high definition television
      standard, used as the working space for wide-gamut spectral processing.

    ---------------------------------------------------------------------------

    */

#include "bauhaus/bauhaus.h"
#include "common/colorspaces.h"
#include "common/guided_filter.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/imagebuf.h"
#include "common/math.h"
#include "common/matrices.h"
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
#include <string.h>

DT_MODULE_INTROSPECTION(5, dt_iop_3dcf_params_t)

/* Type definitions for the ACES 2.0 SSTS pipeline context.
 * spectral_tone_data.c and spectral_tone_pipeline.c have been merged into 3dcf.c
 * into this single translation unit for Windows/MinGW compatibility. */

typedef enum dt_iop_st_colorspace_t
{
  DT_ST_CS_REC709 = 0,    // $DESCRIPTION: "sRGB"
  DT_ST_CS_REC2020,       // $DESCRIPTION: "Rec. 2020"
  DT_ST_CS_DISPLAYP3,     // $DESCRIPTION: "Display P3"
  DT_ST_CS_PROPHOTO,      // $DESCRIPTION: "ProPhoto RGB"
  DT_ST_CS_ADOBERGB,      // $DESCRIPTION: "Adobe RGB"
} dt_iop_st_colorspace_t;

/* MUST be a proper enum, NOT int.
 * dt_bauhaus_combobox_from_params uses introspection: for a plain int it creates
 * an empty combobox and immediately tries to set the default index on it.
 * Setting index 0 on an empty combobox is a silent no-op on Linux/GTK but
 * an assertion failure or out-of-bounds access on Windows/GTK → crash at
 * module mount, before any image is loaded. */
typedef enum dt_iop_st_look_t
{
  DT_ST_LOOK_NEUTRAL = 0,    // $DESCRIPTION: "neutral"
  DT_ST_LOOK_NATURAL,        // $DESCRIPTION: "natural look"
  DT_ST_LOOK_PORTRAIT,       // $DESCRIPTION: "portrait"
  DT_ST_LOOK_VIBRANT,        // $DESCRIPTION: "vibrant"
  DT_ST_LOOK_NATURE,         // $DESCRIPTION: "nature"
  DT_ST_LOOK_BLUESKY,        // $DESCRIPTION: "blue sky"
  DT_ST_LOOK_SOFTWARM,       // $DESCRIPTION: "soft warm"
  DT_ST_LOOK_SOFT,           // $DESCRIPTION: "soft"
  DT_ST_LOOK_DEEPCOOL,       // $DESCRIPTION: "deep cool"
  DT_ST_LOOK_CINEMA,         // $DESCRIPTION: "authentic cinema"
  DT_ST_LOOK_BRIGHT,         // $DESCRIPTION: "bright atmosphere"
} dt_iop_st_look_t;

typedef struct dt_iop_3dcf_params_t
{
  float contrast;              // $MIN: 0.25 $MAX: 4.25 $DEFAULT: 2.25 $DESCRIPTION: "contrast"
  float gray_point;            // $MIN: -1 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "mid-tones"
  float vibrance;              // $MIN: 0 $MAX: 2 $DEFAULT: 1.0 $DESCRIPTION: "vibrance"
  float spectral_brilliance;   // $MIN: 0 $MAX: 100 $DEFAULT: 5 $DESCRIPTION: "perceptual brightness"
  float hl_hue_shift;          // $MIN: -1 $MAX: 1 $DEFAULT: 0 $STEP: 0.01 $DESCRIPTION: "Abney rotation"
  float hl_desaturation;       // $MIN: 0 $MAX: 1 $DEFAULT: 0.60 $DESCRIPTION: "highlight roll-off"
  float hl_desat_threshold;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.45 $DESCRIPTION: "desaturation threshold"
  float gamut_knee;            // $MIN: 0 $MAX: 1 $DEFAULT: 0.20 $DESCRIPTION: "gamut knee"
  float gamut_steepness;       // $MIN: 0 $MAX: 1 $DEFAULT: 0.50 $DESCRIPTION: "gamut steepness"
  dt_iop_st_colorspace_t output_cs;  // $DEFAULT: DT_ST_CS_REC2020 $DESCRIPTION: "color space"
  dt_iop_st_look_t color_look;       // $DEFAULT: DT_ST_LOOK_NEUTRAL $DESCRIPTION: "color look"
  float look_opacity;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "look opacity"
  float contrast_pivot;        // $MIN: 0.01 $MAX: 0.99 $DEFAULT: 0.5 $DESCRIPTION: "contrast pivot"
  float toe_power;             // $MIN: 0.25 $MAX: 3.0 $DEFAULT: 1.0 $DESCRIPTION: "toe power"
  float shoulder_power;        // $MIN: 0.25 $MAX: 3.0 $DEFAULT: 1.0 $DESCRIPTION: "shoulder power"
  float hl_detail_recovery;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.35 $DESCRIPTION: "detail recovery"
} dt_iop_3dcf_params_t;

/* SSTS (ACES 2.0 Single-Stage Tone Scale) precomputed parameters */
typedef struct
{
  double s_2;
  double m_2;
  double g;
  double t_1;
  double n_r;
  double n;
} dt_st_ssts_params_t;

/* Pipeline context — precomputed at commit time */
typedef struct
{
  float exposure_factor;
  float contrast;
  float contrast_pivot;
  float input_matrix[9];
  float output_matrix[9];
  float luma_coeff[3];
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
  float hl_detail_recovery;
  float spectral_boundary[360];  /* max CIE xz distance from D50 per angle degree */
  float gamut_fwd[9];
  float gamut_inv[9];
  int   gamut_enable;
  dt_st_ssts_params_t ssts;
} dt_st_context_t;

typedef struct dt_iop_3dcf_data_t
{
  dt_iop_3dcf_params_t params;
  dt_st_context_t ctx;
} dt_iop_3dcf_data_t;

/* GPU-side context — MUST match dt_st_cl_params_t in 3dcf.cl byte for
 * byte. Only 4-byte members (float / int) are used so host and device layouts
 * are identical (no padding); the struct is passed to the kernel by value. The
 * SSTS parameters are double in dt_st_context_t but narrowed to float here. */
typedef struct dt_st_cl_params_t
{
  float input_matrix[9];
  float output_matrix[9];
  float luma_coeff[3];
  float color_look_mat[9];
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
  int   look_idx;
  int   gamut_enable;
} dt_st_cl_params_t;

typedef struct dt_iop_3dcf_global_data_t
{
  int kernel_3dcf;
  int kernel_3dcf_extract_lum;
} dt_iop_3dcf_global_data_t;

typedef struct dt_iop_3dcf_gui_data_t
{
  GtkWidget *contrast;
  GtkWidget *contrast_pivot;
  GtkWidget *toe_power;
  GtkWidget *shoulder_power;
  GtkWidget *spectral_brilliance;
  GtkWidget *mid_tone;
  GtkWidget *vibrance;
  GtkWidget *hl_desaturation;
  GtkWidget *hl_desat_threshold;
  GtkWidget *hl_hue_shift;
  GtkWidget *hl_detail_recovery;
  GtkWidget *gamut_knee;
  GtkWidget *gamut_steepness;
  dt_gui_collapsible_section_t advanced_section;
  GtkWidget *color_space;
  GtkWidget *color_look;
  GtkWidget *look_opacity;
  GtkDrawingArea *graph;
  GtkAllocation allocation;
  PangoRectangle ink;
  GtkStyleContext *context;
} dt_iop_3dcf_gui_data_t;

/* Conversion matrices */
static const float st_luma_rec709[3]    = { 0.2126f,  0.7152f,  0.0722f };
static const float st_luma_rec2020[3]   = { 0.2627f,  0.6780f,  0.0593f };
static const float st_luma_displayp3[3] = { 0.2289f,  0.6918f,  0.0793f };
static const float st_luma_prophoto[3]  = { 0.2880f,  0.7119f,  0.0001f };
static const float st_luma_adobergb[3]  = { 0.2973f,  0.6274f,  0.0753f };

static const float color_looks[11][9] = {
  {1.000f, 0.000f, 0.000f,  0.000f, 1.000f, 0.000f,  0.000f, 0.000f, 1.000f}, // 1. neutral
  {1.076f, -0.047f, -0.058f, -0.014f, 1.044f, -0.052f, -0.105f, 0.049f, 1.076f}, // 2. natural look
  {1.029f, -0.023f, -0.002f, -0.008f, 1.008f, 0.007f, -0.074f, 0.046f, 1.010f}, // 3. portrait
  {1.074f, -0.054f, -0.071f, 0.006f, 1.009f, -0.059f, -0.103f, 0.060f, 1.086f}, // 4. vibrant
  {1.084f, -0.006f, -0.093f, -0.074f, 1.008f, 0.060f, -0.011f, 0.005f, 1.024f}, // 5. nature
  {1.218f, -0.119f, -0.099f, 0.007f, 1.076f, -0.069f, -0.192f, 0.048f, 1.154f}, // 6. blue sky
  {1.050f, 0.020f, -0.010f, -0.020f, 1.020f, 0.000f, -0.010f, -0.020f, 1.030f}, // 7. soft warm
  {1.082f, -0.051f, -0.047f, -0.020f, 1.052f, -0.045f, 0.103f, 0.042f, 1.073f}, // 8. soft
  {0.980f, -0.010f, -0.010f,  0.000f, 1.050f, -0.020f,  0.020f, 0.010f, 1.100f}, // 9. deep cool
  {1.020f, -0.010f, -0.010f, -0.030f, 1.040f, -0.010f, 0.000f, -0.030f, 1.030f}, // 10. authentic cinema
  {1.067f, -0.049f, -0.031f, -0.017f, 1.033f, -0.026f, -0.088f, 0.042f, 1.055f}  // 11. bright atmosphere
};

/* Rec.2020 RGB -> CIE XYZ (D50) — ITU-R BT.2020-2 Table 4 */
const double dt_st_rec2020_to_xyz[9] = {
  6.369580467986713e-01, 1.446169121462670e-01, 1.688809980493937e-01,
  2.627002366306831e-01, 6.779980956614056e-01, 5.930176218802780e-02,
 -3.132235483284409e-08, 2.807267854412832e-02, 1.060985074810122e+00
};

/* Chromatic adaptation matrix CIE CAT16: D65 -> D50 */
const double dt_st_cat_d65_to_d50[9] = {
  1.047839954303051e+00,  2.289791610380174e-02, -5.018079725046408e-02,
  2.955368681442254e-02,  9.904924221623178e-01, -1.706631418019539e-02,
  -9.245918452778928e-03,  1.506326034916465e-02,  7.518388616796452e-01
};

/* Pre-computed combined matrices for output gamut protection:
 *   gamut_fwd[cs]  : Rec.2020 D65 RGB -> Target D65 RGB
 *   gamut_inv[cs]  : Target D65 RGB -> Rec.2020 D65 RGB
 * Indexed by dt_iop_st_colorspace_t. Slot REC2020 is identity (unused).
 * Round-trip verified: fwd @ inv = I to within ~3e-16. */
static const float st_gamut_fwd[5][9] = {
  /* DT_ST_CS_REC709 — sRGB (same primaries) */
  { 1.6604910021084340e+00f, -5.8764113878854918e-01f, -7.2849863319884856e-02f,
   -1.2455047452159060e-01f,  1.1328998971259596e+00f, -8.3494226043694768e-03f,
   -1.8150763354905213e-02f, -1.0057889800800739e-01f,  1.1187296613629127e+00f },
  /* DT_ST_CS_REC2020 (identity, skipped) */
  { 1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f },
  /* DT_ST_CS_DISPLAYP3 */
  { 1.3435782525843321e+00f, -2.8217967052613580e-01f, -6.1398582058196219e-02f,
   -6.5297452789119470e-02f,  1.0757879158485739e+00f, -1.0490463059454964e-02f,
    2.8217872617010515e-03f, -1.9598494524494175e-02f,  1.0167767072627931e+00f },
  /* DT_ST_CS_PROPHOTO (via D50 adaptation) */
  { 8.3510708914902376e-01f,  4.8796017329827877e-02f,  1.1593570461330915e-01f,
    5.4024518630719040e-02f,  9.2897841074060328e-01f,  1.7056261398002901e-02f,
   -2.3416915615907760e-03f,  3.6337073948144318e-02f,  9.6596433118070490e-01f },
  /* DT_ST_CS_ADOBERGB */
  { 1.1519783947159161e+00f, -9.7503055302408478e-02f, -5.4475339413507635e-02f,
   -1.2455047452159049e-01f,  1.1328998971259596e+00f, -8.3494226043695028e-03f,
   -2.2530382781055808e-02f, -4.9806507428388894e-02f,  1.0723368902094448e+00f },
};

static const float st_gamut_inv[5][9] = {
  /* DT_ST_CS_REC709 — sRGB (same primaries) */
  { 6.2740389593469914e-01f,  3.2928303837788381e-01f,  4.3313065687417218e-02f,
    6.9097289358232006e-02f,  9.1954039507545904e-01f,  1.1362315566309176e-02f,
    1.6391438875150231e-02f,  8.8013307877225763e-02f,  8.9559525324762390e-01f },
  /* DT_ST_CS_REC2020 (identity, skipped) */
  { 1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f },
  /* DT_ST_CS_DISPLAYP3 */
  { 7.5383303436172178e-01f,  1.9859736905261646e-01f,  4.7569596585661844e-02f,
    4.5743848965358248e-02f,  9.4177721981169393e-01f,  1.2478931222948127e-02f,
   -1.2103403545183937e-03f,  1.7601717301089989e-02f,  9.8360862305342833e-01f },
  /* DT_ST_CS_PROPHOTO (via D50 adaptation) */
  { 1.2007680891347712e+00f, -5.7474733312325541e-02f, -1.4310216842779236e-01f,
   -6.9932127259518431e-02f,  1.0805425606086707e+00f, -1.0686094282437804e-02f,
    5.5415683670136931e-03f, -4.0786540201541162e-02f,  1.0352899874010184e+00f },
  /* DT_ST_CS_ADOBERGB */
  { 8.7733384166365691e-01f,  7.7493706515719976e-02f,  4.5172451820623120e-02f,
    9.6622591466203681e-02f,  8.9152732024418091e-01f,  1.1850088289615686e-02f,
    2.2921062702848320e-02f,  4.3036685010679435e-02f,  9.3404225228647197e-01f },
};

static const float *st_get_luma_coeff(dt_iop_st_colorspace_t cs)
{
  switch(cs)
  {
    case DT_ST_CS_REC709:    return st_luma_rec709;
    case DT_ST_CS_REC2020:   return st_luma_rec2020;
    case DT_ST_CS_DISPLAYP3: return st_luma_displayp3;
    case DT_ST_CS_PROPHOTO:  return st_luma_prophoto;
    case DT_ST_CS_ADOBERGB:  return st_luma_adobergb;
    default:                 return st_luma_rec2020;
  }
}

/* ========================================================================
 * ACES 2.0 SSTS (Single-Stage Tone Scale) and spectral pipeline functions
 * ======================================================================== */

/* ACES 2.0 SSTS (Single-Stage Tone Scale) — official ACES 2.0 RRT tone scale
 *
 * Parametric MM (Michaelis-Menten) curve with flare compensation.
 * Reference: aces-core/lib/Lib.Academy.Tonescale.ctl
 *
 *   f = m_2 * (x / (x + s_2))^g
 *   h = f^2 / (f + t_1)
 *   Y = h * n_r          (output: display luminance in cd/m^2)
 *
 * The SSTS defines the "texture of light" — the character of the tone
 * reproduction from scene-linear to display. Its shape is determined by
 * the peak luminance of the target display.
 */
void dt_st_ssts_init(dt_st_ssts_params_t *p, double peak_luminance)
{
  /* --- Fixed SSTS design constants --- */
  const double n_r     = 100.0;    /* normalized white in nits */
  const double g       = 1.15;     /* surround / contrast */
  const double c       = 0.18;     /* anchor: 18% grey */
  const double c_d     = 10.013;   /* output luminance of 18% grey (nits) */
  const double w_g     = 0.14;     /* grey change vs peak luminance */
  const double t_1     = 0.04;     /* shadow toe / flare compensation */
  const double r_hit_min = 128.0;  /* scene value hitting roof for SDR */
  const double r_hit_max = 896.0;  /* scene value hitting roof for 10000 nits */

  const double n = fmax(peak_luminance, 1.0);

  /* --- Precomputation --- */
  const double r_hit = r_hit_min
    + (r_hit_max - r_hit_min) * (log(n / n_r) / log(10000.0 / 100.0));

  const double m_0 = n / n_r;
  const double m_1 = 0.5 * (m_0 + sqrt(m_0 * (m_0 + 4.0 * t_1)));

  const double u = pow((r_hit / m_1) / ((r_hit / m_1) + 1.0), g);
  const double m = m_1 / u;

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

float dt_st_ssts_fwd(const dt_st_ssts_params_t *p, float x)
{
  if(x <= 0.0f) return 0.0f;

  const float s_2 = (float)p->s_2;
  const float m_2 = (float)p->m_2;
  const float g   = (float)p->g;
  const float t_1 = (float)p->t_1;
  const float n_r = (float)p->n_r;

  const float f = m_2 * powf(x / (x + s_2), g);
  const float h = (f * f) / (f + t_1);
  return h * n_r;
}

float dt_st_compute_y_tm(float y_scene, const dt_st_context_t *ctx)
{
  if(ctx->ssts.n <= 0.0) return 0.0f;

  float y_tm = dt_st_ssts_fwd(&ctx->ssts, y_scene * ctx->exposure_factor)
             / (float)ctx->ssts.n;

  y_tm = powf(fmaxf(y_tm, 0.0f), 1.0f / 2.4f);

  /* Clamp to [0, 1] before contrast curve: the shoulder formula assumes
   * y_tm ∈ [0,1] and produces NaN/Inf when y_tm > 1 (powf(0, negative)). */
  y_tm = fminf(y_tm, 1.0f);

  if(ctx->contrast != 1.0f || ctx->toe_power != 1.0f || ctx->shoulder_power != 1.0f)
  {
    const float c  = ctx->contrast;
    const float p  = ctx->contrast_pivot;
    const float ct = ctx->toe_power;
    const float cs = ctx->shoulder_power;
    if(y_tm <= p)
    {
      const float t = (p > 0.0f) ? y_tm / p : 0.0f;
      const float exp_eff = c * (ct + (1.0f - ct) * t);
      y_tm = p * powf(fmaxf(y_tm / p, 0.0f), exp_eff);
    }
    else
    {
      const float rp = 1.0f - p;
      const float t = (rp > 0.0f) ? (1.0f - y_tm) / rp : 0.0f;
      const float exp_eff = c * (cs + (1.0f - cs) * t);
      y_tm = 1.0f - rp * powf(fmaxf((1.0f - y_tm) / rp, 0.0f), exp_eff);
    }
  }

  return y_tm;
}

/* Desaturation weight for highlights
 *
 * Desaturation activates above a scene luminance threshold (in SSTS-exposed space)
 * and ramps up quadratically as luminance increases.
 * hl_desat controls the strength: 0 = off, 1 = full desat well above threshold,
 * > 1 = faster desaturation.
 */
static inline float st_desat_weight(float y_norm, float hl_desat, float threshold)
{
  if(hl_desat <= 0.0f || y_norm <= threshold) return 0.0f;
  if(!isfinite(y_norm)) return 0.0f;
  const float t = fmaxf(y_norm - threshold, 0.0f) / y_norm;
  const float x = fminf(t * hl_desat, 1.0f);
  return x * x;
}

/* Film-like gamut compression: smoothly desaturate out-of-gamut colors toward
 * white (1,1,1) with linear blend. Blending toward white preserves hue better
 * than luma-gray. The linear blend guarantees that the most-negative channel
 * reaches exactly zero (in-gamut), avoiding residual negativity that shifts hue.
 * Only activates when a channel falls below zero.
 */
static inline void st_gamut_compress(float rgb[3])
{
  const float m = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
  if(m >= 0.0f) return;

  /* Linear blend toward white — guarantees full correction */
  float t = 0.0f;
  if(rgb[0] < 0.0f) { const float ti = -rgb[0] / (1.0f - rgb[0]); if(ti > t) t = ti; }
  if(rgb[1] < 0.0f) { const float ti = -rgb[1] / (1.0f - rgb[1]); if(ti > t) t = ti; }
  if(rgb[2] < 0.0f) { const float ti = -rgb[2] / (1.0f - rgb[2]); if(ti > t) t = ti; }
  t = fminf(t, 1.0f);

  rgb[0] = (1.0f - t) * rgb[0] + t * 1.0f;
  rgb[1] = (1.0f - t) * rgb[1] + t * 1.0f;
  rgb[2] = (1.0f - t) * rgb[2] + t * 1.0f;
}

/* Output gamut protection: convert from Rec.2020 D65 to the target color space,
 * hard-clamp negative (out-of-gamut) channels to zero, then convert back.
 * Called after st_gamut_compress() as an additional safety net when the user
 * selects a colour space narrower than Rec. 2020. */
static inline void st_output_gamut_protect(float rgb[3],
                                            const float fwd[9],
                                            const float inv[9])
{
  float t[3];
  t[0] = fwd[0]*rgb[0] + fwd[1]*rgb[1] + fwd[2]*rgb[2];
  t[1] = fwd[3]*rgb[0] + fwd[4]*rgb[1] + fwd[5]*rgb[2];
  t[2] = fwd[6]*rgb[0] + fwd[7]*rgb[1] + fwd[8]*rgb[2];

  t[0] = fmaxf(t[0], 0.0f);
  t[1] = fmaxf(t[1], 0.0f);
  t[2] = fmaxf(t[2], 0.0f);

  rgb[0] = inv[0]*t[0] + inv[1]*t[1] + inv[2]*t[2];
  rgb[1] = inv[3]*t[0] + inv[4]*t[1] + inv[5]*t[2];
  rgb[2] = inv[6]*t[0] + inv[7]*t[1] + inv[8]*t[2];
}

/* Spectral locus CIE 1931 2-degree xy at 5 nm (380–700 nm) + endpoint at 780 nm.
 * From 700 nm onward the coordinates plateau at (0.734690, 0.265310).
 * Used to build the angle→max-distance lookup table for automatic non-spectral
 * colour detection.  Purple-line interpolation is added at precompute time. */
#define SPECTRAL_LOCUS_N 66
static const float st_spectral_locus_xy[SPECTRAL_LOCUS_N][2] =
{
  { 0.174112f, 0.004964f },  /* 380 nm */
  { 0.174008f, 0.004981f },  /* 385 */
  { 0.173801f, 0.004915f },  /* 390 */
  { 0.173560f, 0.004923f },  /* 395 */
  { 0.173337f, 0.004797f },  /* 400 */
  { 0.173021f, 0.004775f },  /* 405 */
  { 0.172577f, 0.004799f },  /* 410 */
  { 0.172087f, 0.004833f },  /* 415 */
  { 0.171407f, 0.005102f },  /* 420 */
  { 0.170301f, 0.005789f },  /* 425 */
  { 0.168878f, 0.006900f },  /* 430 */
  { 0.166895f, 0.008556f },  /* 435 */
  { 0.164412f, 0.010858f },  /* 440 */
  { 0.161105f, 0.013793f },  /* 445 */
  { 0.156641f, 0.017705f },  /* 450 */
  { 0.150985f, 0.022740f },  /* 455 */
  { 0.143960f, 0.029703f },  /* 460 */
  { 0.135503f, 0.039879f },  /* 465 */
  { 0.124118f, 0.057803f },  /* 470 */
  { 0.109594f, 0.086843f },  /* 475 */
  { 0.091294f, 0.132702f },  /* 480 */
  { 0.068706f, 0.200723f },  /* 485 */
  { 0.045391f, 0.294976f },  /* 490 */
  { 0.023460f, 0.412703f },  /* 495 */
  { 0.008168f, 0.538423f },  /* 500 */
  { 0.003859f, 0.654823f },  /* 505 */
  { 0.013870f, 0.750186f },  /* 510 */
  { 0.038852f, 0.812016f },  /* 515 */
  { 0.074302f, 0.833803f },  /* 520 */
  { 0.114161f, 0.826207f },  /* 525 */
  { 0.154722f, 0.805864f },  /* 530 */
  { 0.192876f, 0.781629f },  /* 535 */
  { 0.229620f, 0.754329f },  /* 540 */
  { 0.265775f, 0.724324f },  /* 545 */
  { 0.301604f, 0.692308f },  /* 550 */
  { 0.337363f, 0.658848f },  /* 555 */
  { 0.373102f, 0.624451f },  /* 560 */
  { 0.408736f, 0.589607f },  /* 565 */
  { 0.444062f, 0.554714f },  /* 570 */
  { 0.478775f, 0.520202f },  /* 575 */
  { 0.512486f, 0.486591f },  /* 580 */
  { 0.544787f, 0.454434f },  /* 585 */
  { 0.575151f, 0.424232f },  /* 590 */
  { 0.602933f, 0.396497f },  /* 595 */
  { 0.627037f, 0.372491f },  /* 600 */
  { 0.648233f, 0.351395f },  /* 605 */
  { 0.665764f, 0.334011f },  /* 610 */
  { 0.680079f, 0.319747f },  /* 615 */
  { 0.691504f, 0.308342f },  /* 620 */
  { 0.700606f, 0.299301f },  /* 625 */
  { 0.707918f, 0.292027f },  /* 630 */
  { 0.714032f, 0.285929f },  /* 635 */
  { 0.719033f, 0.280935f },  /* 640 */
  { 0.723032f, 0.276948f },  /* 645 */
  { 0.725992f, 0.274008f },  /* 650 */
  { 0.728272f, 0.271728f },  /* 655 */
  { 0.729969f, 0.270031f },  /* 660 */
  { 0.731089f, 0.268911f },  /* 665 */
  { 0.731993f, 0.268007f },  /* 670 */
  { 0.732719f, 0.267281f },  /* 675 */
  { 0.733417f, 0.266583f },  /* 680 */
  { 0.734047f, 0.265953f },  /* 685 */
  { 0.734390f, 0.265610f },  /* 690 */
  { 0.734592f, 0.265408f },  /* 695 */
  { 0.734690f, 0.265310f },  /* 700 nm */
  { 0.734690f, 0.265310f },  /* 780 nm (plateau) */
};

/* Build 360-bin lookup table: for each integer degree angle (0–359) from the
 * D50 white point, store the maximum CIE xz distance of the spectral locus
 * (including the purple line).  The table is used by st_spectral_gamut() to
 * detect and smoothly roll off non-spectral chromaticities. */
static void st_compute_spectral_boundary(float boundary[360],
                                          float white_x_ratio,
                                          float white_z_ratio)
{
  /* Convert white ratios to CIE xz */
  const float wy = 1.0f;
  const float wsum = white_x_ratio + wy + white_z_ratio;
  const float white_cx = white_x_ratio / wsum;
  const float white_cz = white_z_ratio / wsum;

  /* Initialise to zero */
  for(int i = 0; i < 360; i++) boundary[i] = 0.0f;

  /* Process spectral locus points */
  for(int i = 0; i < SPECTRAL_LOCUS_N; i++)
  {
    const float x = st_spectral_locus_xy[i][0];
    const float y = st_spectral_locus_xy[i][1];
    const float z = 1.0f - x - y;  /* CIE z = 1 − x − y */

    const float dx = x - white_cx;
    const float dz = z - white_cz;
    const float dist = sqrtf(dx * dx + dz * dz);

    float angle = atan2f(dz, dx) * (180.0f / (float)M_PI);
    if(angle < 0.0f) angle += 360.0f;
    int bin = (int)angle;
    if(bin < 0) bin = 0;
    if(bin >= 360) bin = 359;

    if(dist > boundary[bin]) boundary[bin] = dist;
  }

  /* Purple line: interpolate N segments between endpoint 780 nm (idx N-1)
   * and 380 nm (idx 0), then back to 780 nm to close the locus. */
  {
    const float x0 = st_spectral_locus_xy[SPECTRAL_LOCUS_N - 1][0];
    const float y0 = st_spectral_locus_xy[SPECTRAL_LOCUS_N - 1][1];
    const float x1 = st_spectral_locus_xy[0][0];
    const float y1 = st_spectral_locus_xy[0][1];

    const int NSEG = 32;
    for(int i = 1; i <= NSEG; i++)
    {
      const float t = (float)i / (float)(NSEG + 1);
      const float x = x0 + t * (x1 - x0);
      const float y = y0 + t * (y1 - y0);
      const float z = 1.0f - x - y;

      const float dx = x - white_cx;
      const float dz = z - white_cz;
      const float dist = sqrtf(dx * dx + dz * dz);

      float angle = atan2f(dz, dx) * (180.0f / (float)M_PI);
      if(angle < 0.0f) angle += 360.0f;
      int bin = (int)angle;
      if(bin < 0) bin = 0;
      if(bin >= 360) bin = 359;

      if(dist > boundary[bin]) boundary[bin] = dist;
    }
  }

  /* Fill any empty bins by forward-fill then backward-fill */
  {
    float last = 0.0f;
    for(int i = 0; i < 360; i++)
    {
      if(boundary[i] > 0.0f) last = boundary[i];
      else if(last > 0.0f) boundary[i] = last;
    }
    last = 0.0f;
    for(int i = 359; i >= 0; i--)
    {
      if(boundary[i] > 0.0f) last = boundary[i];
      else if(last > 0.0f) boundary[i] = last;
    }
  }

  /* Ensure no bin is exactly zero (safety: fall back to minimum positive) */
  float min_nonzero = 1e6f;
  for(int i = 0; i < 360; i++)
    if(boundary[i] > 0.0f && boundary[i] < min_nonzero)
      min_nonzero = boundary[i];
  if(min_nonzero > 1e5f) min_nonzero = 1.0f;  /* guard against all-zero */
  for(int i = 0; i < 360; i++)
    if(boundary[i] <= 0.0f) boundary[i] = min_nonzero;
}

static inline void st_spectral_gamut(
  float *x_tm, float *z_tm, float y_tm,
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

  /* Spectral locus boundary check — automatic detection of non-spectral colours.
   * Runs before the user knee to catch colours whose chromaticity ratios lie
   * outside the visible spectrum (e.g. laser primaries, narrow-band LEDs). */
  if(spectral_boundary)
  {
    const float angle = atan2f(dz, dx);
    float angle_deg = angle * (180.0f / (float)M_PI);
    if(angle_deg < 0.0f) angle_deg += 360.0f;
    int bin = (int)angle_deg;
    if(bin < 0) bin = 0;
    if(bin >= 360) bin = 359;
    const float max_dist = spectral_boundary[bin];
    const float target_dist = max_dist * 0.95f;  // CB margin: roll-off before boundary

    if(max_dist > 0.0f && chroma_sq > target_dist * target_dist)
    {
      const float chroma = sqrtf(chroma_sq);
      const float excess = chroma - target_dist;
      const float bsteep = fmaxf(target_dist * 0.05f, 0.001f);
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
        /* Recompute for knee below */
        dx = cie_x - white_cie_x;
        dz = cie_z - white_cie_z;
        chroma_sq = dx * dx + dz * dz;
      }
    }
  }

  /* Existing knee — smooth circular roll-off controlled by user sliders */
  if(chroma_sq > knee * knee)
  {
    const float chroma = sqrtf(chroma_sq);
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

/* Complete spectral tone mapping pipeline for one pixel.
 *
 * Pipeline order:
 *   1. D50-adapted Rec.2020 RGB -> D50 XYZ (precise matrix)
 *   2. ACES 2.0 SSTS on luminance Y only (spectral tone scale)
 *   3. BT.1886 OETF + contrast S-curve
 *   4. Mid-tone gamma adjustment
 *   5. Chromaticity ratio scaling: x = ratio * Y
 *   6. Spectral gamut: film-like chromaticity roll-off in CIE xy
 *   7. XYZ -> output RGB via output matrix
 *   8. Highlight desaturation (blend toward achromatic luma)
 *   9. Vibrance (saturation with high-sat protection)
 *  10. Gamut compression safety net
 *  11. Clamp to [0, inf)
 */
void dt_st_pipeline_eval(const float rgb_in[3], float rgb_out[3],
                          const dt_st_context_t *ctx)
{
  /* Step 1: Accurate XYZ from D50-adapted Rec.2020 input */
  const float r = rgb_in[0], g = rgb_in[1], b = rgb_in[2];
  const float *M_in = ctx->input_matrix;
  float x_abs = M_in[0] * r + M_in[1] * g + M_in[2] * b;
  float y_abs = M_in[3] * r + M_in[4] * g + M_in[5] * b;
  float z_abs = M_in[6] * r + M_in[7] * g + M_in[8] * b;

  if(!(y_abs > 1e-10f) || !isfinite(y_abs))
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  /* Chromaticity ratios */
  const float x_ratio = fminf(fmaxf(x_abs / y_abs, -100.0f), 100.0f);
  const float z_ratio = fminf(fmaxf(z_abs / y_abs, -100.0f), 100.0f);

  /* Step 2-3: Tone-mapped Y (SSTS + BT.1886 + contrast) */
  float y_tm = dt_st_compute_y_tm(y_abs, ctx);

  /* Step 4: Mid-tone adjustment — gamma pivot */
  if(ctx->gray_point != 0.0f)
  {
    float y_lvl = fminf(fmaxf(y_tm, 0.0f), 1.0f);
    y_lvl = powf(y_lvl, ctx->gray_gamma);
    y_tm = y_lvl;
  }

  /* Step 5: Scale chromaticity with tone-mapped luminance */
  float x_tm = x_ratio * y_tm;
  float z_tm = z_ratio * y_tm;

  /* Step 6: Spectral gamut — film-like chromaticity roll-off in CIE xy */
  st_spectral_gamut(&x_tm, &z_tm, y_tm,
                    ctx->white_chroma_x,
                    ctx->white_chroma_z,
                    ctx->gamut_knee,
                    ctx->gamut_steepness,
                    ctx->spectral_boundary);

  /* Step 7: XYZ -> Output RGB via precomputed matrix */
  const float *M = ctx->output_matrix;
  float rgb[3];
  rgb[0] = M[0] * x_tm + M[1] * y_tm + M[2] * z_tm;
  rgb[1] = M[3] * x_tm + M[4] * y_tm + M[5] * z_tm;
  rgb[2] = M[6] * x_tm + M[7] * y_tm + M[8] * z_tm;

  /* Step 8: Film-print highlight desaturation toward white */
  {
    const float y_exposed = y_abs * ctx->exposure_factor;
    const float w = st_desat_weight(y_exposed, ctx->hl_desat, ctx->hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float maxc_pre = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
      const float minc_pre = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
      const float sat_pre = (maxc_pre > 0.0f) ? (maxc_pre - minc_pre) / maxc_pre : 0.0f;
      const float ss_pre = (sat_pre * sat_pre) / (sat_pre * sat_pre + (1.0f - sat_pre) * (1.0f - sat_pre) + 1e-6f);

      /* Progressive Abney hue rotation — independent of hl_desat */
      if(ctx->hl_rotation != 0.0f)
      {
        const float wr = fminf(st_desat_weight(y_exposed, 1.0f, ctx->hl_desat_threshold), 1.0f);
        const float angle = ctx->hl_rotation * 0.25f * wr; //CB
        const float ca = cosf(angle);
        const float sa = sinf(angle);
        if(isfinite(ca) && isfinite(sa))
        {
          const float *lc = ctx->luma_coeff;
          const float y = lc[0] * rgb[0] + lc[1] * rgb[1] + lc[2] * rgb[2];
          float u = rgb[0] - y;
          float v = rgb[2] - y;
          const float ur = u * ca - v * sa;
          const float vr = u * sa + v * ca;
          rgb[0] = y + ur;
          rgb[2] = y + vr;
          if(lc[1] > 0.0f)
            rgb[1] = y - (lc[0] / lc[1]) * ur - (lc[2] / lc[1]) * vr;
        }
      }

      /* Negative vibrance: desaturate saturated pixels more, driven by hl_desat and hl_hue_shift */
      float w_final = w;
      if(ctx->hl_desat > 0.0f || ctx->hl_rotation != 0.0f)
      {
        if(ctx->hl_desat > 0.0f)
        {
          const float vib_neg = ctx->hl_desat * ss_pre * 0.5f;
          const float w_vib = st_desat_weight(y_exposed, 1.0f, ctx->hl_desat_threshold) * vib_neg;
          w_final = fminf(w_final + w_vib, 1.0f);
        }
        if(ctx->hl_rotation != 0.0f)
        {
          const float vib_neg = fabsf(ctx->hl_rotation) * ss_pre * 1.0f;
          const float w_rot = st_desat_weight(y_exposed, 1.0f, ctx->hl_desat_threshold) * vib_neg;
          w_final = fmaxf(w_final, w_rot);
        }
      }

      /* Blend toward white with sigmoidal curve — film-like */
      const float t = fminf(w_final, 1.0f);
      const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
      if(isfinite(ts))
      {
        rgb[0] = rgb[0] * (1.0f - ts) + 1.0f * ts;
        rgb[1] = rgb[1] * (1.0f - ts) + 1.0f * ts;
        rgb[2] = rgb[2] * (1.0f - ts) + 1.0f * ts;
      }
    }
  }

  if(ctx->vibrance != 1.0f)
  {
    const float vib = fmaxf(ctx->vibrance, 0.0f);
    const float *lc = ctx->luma_coeff;
    const float luma = lc[0] * rgb[0] + lc[1] * rgb[1] + lc[2] * rgb[2];
    const float maxc = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
    const float minc = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
    const float sat_m = maxc - minc;
    const float level = fmaxf(maxc, fmaxf(fabsf(minc), fabsf(luma)));
    const float sat_norm = (level > 0.0f) ? sat_m / level : 0.0f;

    if(vib > 1.0f)
    {
      const float p = 1.0f - fminf(sat_norm, 1.0f);
      const float vib_gain = 1.0f + (vib - 1.0f) * (p * p);
      rgb[0] = luma + vib_gain * (rgb[0] - luma);
      rgb[1] = luma + vib_gain * (rgb[1] - luma);
      rgb[2] = luma + vib_gain * (rgb[2] - luma);
    }
    else
    {
      rgb[0] = luma + vib * (rgb[0] - luma);
      rgb[1] = luma + vib * (rgb[1] - luma);
      rgb[2] = luma + vib * (rgb[2] - luma);
    }
  }

  st_gamut_compress(rgb);

  /* Output gamut protection: clamp to selected primary space */
  if(ctx->gamut_enable)
    st_output_gamut_protect(rgb, ctx->gamut_fwd, ctx->gamut_inv);

  rgb_out[0] = isfinite(rgb[0]) ? fmaxf(rgb[0], 0.0f) : 0.0f;
  rgb_out[1] = isfinite(rgb[1]) ? fmaxf(rgb[1], 0.0f) : 0.0f;
  rgb_out[2] = isfinite(rgb[2]) ? fmaxf(rgb[2], 0.0f) : 0.0f;
}

/* Given input RGB in working space, compute the context */
static void st_compute_context(dt_iop_3dcf_params_t *p,
                                dt_st_context_t *ctx)
{
  /* Auto-exposure compensation: single slider controls both SSTS tone curve
   * character and brightness. Higher spectral brilliance makes SSTS less
   * compressive, requiring positive exposure to maintain consistent perceived
   * brightness. Formula auto_ev = 0.061 × spectral_brilliance ensures 18% gray
   * and diffuse white map to the same BT.1886 output at any brilliance value. */
  {
    const double auto_ev = 0.061 * (double)p->spectral_brilliance;
    ctx->exposure_factor = exp2f((float)auto_ev);
  }

  /* === Combined input matrix: D50-adapted Rec.2020 RGB → D50 XYZ === */
  {
    double M_in[9];
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
      {
        M_in[i * 3 + j] = 0.0;
        for(int k = 0; k < 3; k++)
          M_in[i * 3 + j] += dt_st_cat_d65_to_d50[i * 3 + k]
                           * dt_st_rec2020_to_xyz[k * 3 + j];
      }
    for(int i = 0; i < 9; i++) ctx->input_matrix[i] = (float)M_in[i];
  }

  /* === Output matrix: D50 XYZ → Rec.2020 D50 RGB (working space) === */
  {
    float M_in_ws[9];
    for(int i = 0; i < 9; i++) M_in_ws[i] = ctx->input_matrix[i];
    
    /* Guard: check determinant before inversion */
    float det = M_in_ws[0] * (M_in_ws[4] * M_in_ws[8] - M_in_ws[5] * M_in_ws[7])
              - M_in_ws[1] * (M_in_ws[3] * M_in_ws[8] - M_in_ws[5] * M_in_ws[6])
              + M_in_ws[2] * (M_in_ws[3] * M_in_ws[7] - M_in_ws[4] * M_in_ws[6]);
    
    if(fabsf(det) < 1e-10f)
    {
      /* Fallback identity matrix if inversion would fail */
      ctx->output_matrix[0] = ctx->output_matrix[4] = ctx->output_matrix[8] = 1.0f;
      ctx->output_matrix[1] = ctx->output_matrix[2] = ctx->output_matrix[3] = 0.0f;
      ctx->output_matrix[5] = ctx->output_matrix[6] = ctx->output_matrix[7] = 0.0f;
    }
    else
    {
      mat3inv(ctx->output_matrix, M_in_ws);
    }
  }

  /* Luma coefficients matching the output color space */
  const float *lc = st_get_luma_coeff(p->output_cs);
  ctx->luma_coeff[0] = lc[0];
  ctx->luma_coeff[1] = lc[1];
  ctx->luma_coeff[2] = lc[2];

  ctx->contrast = fmaxf(p->contrast, 0.001f);
  /* Inversion du pivot pour que 'droite' (valeur élevée) éclaircisse l'image */
  ctx->contrast_pivot = 1.0f - fmaxf(fminf(p->contrast_pivot, 0.99f), 0.01f);

  ctx->hl_desat = fmaxf(p->hl_desaturation, 0.0f);
  ctx->hl_desat_threshold = fmaxf(p->hl_desat_threshold, 0.0f);
  ctx->hl_rotation = p->hl_hue_shift;
  ctx->gamut_knee = p->gamut_knee;
  ctx->gamut_steepness = p->gamut_steepness;
  ctx->toe_power = fmaxf(p->toe_power, 0.0f);
  ctx->shoulder_power = fmaxf(p->shoulder_power, 0.0f);
  ctx->hl_detail_recovery = fmaxf(p->hl_detail_recovery, 0.0f);

  /* Output gamut protection matrices from pre-computed lookup table */
  {
    const int cs = p->output_cs;
    for(int i = 0; i < 9; i++)
    {
      ctx->gamut_fwd[i] = st_gamut_fwd[cs][i];
      ctx->gamut_inv[i] = st_gamut_inv[cs][i];
    }
    ctx->gamut_enable = (cs != DT_ST_CS_REC2020);
  }

  /* White point chromaticity ratios from input matrix */
  {
    const double wy = (double)ctx->input_matrix[3]
                    + (double)ctx->input_matrix[4]
                    + (double)ctx->input_matrix[5];
    if(wy > 0.0)
    {
      const double wx = (double)ctx->input_matrix[0]
                      + (double)ctx->input_matrix[1]
                      + (double)ctx->input_matrix[2];
      const double wz = (double)ctx->input_matrix[6]
                      + (double)ctx->input_matrix[7]
                      + (double)ctx->input_matrix[8];
      ctx->white_chroma_x = (float)(wx / wy);
      ctx->white_chroma_z = (float)(wz / wy);
    }
    else
    {
      ctx->white_chroma_x = 0.333f;
      ctx->white_chroma_z = 0.333f;
    }
  }

  /* Precompute spectral locus boundary relative to the pipeline's white point */
  st_compute_spectral_boundary(ctx->spectral_boundary,
                                ctx->white_chroma_x,
                                ctx->white_chroma_z);

  /* Mid-tone gamma adjustment - Inversion pour que 'droite' éclaircisse (Gamma < 1.0) */
  ctx->gray_point = -fmaxf(fminf(p->gray_point, 1.0f), -1.0f);
  ctx->gray_gamma = exp2f(ctx->gray_point);

  ctx->vibrance = fmaxf(p->vibrance, 0.0f);

  /* Initialize ACES 2.0 SSTS for the target roll-off character */
  const double rolloff_t = fmin(fmax((double)p->spectral_brilliance / 100.0, 0.0), 1.0);
  const double peak = 100.0 * pow(100.0, rolloff_t);
  dt_st_ssts_init(&ctx->ssts, peak);
}

/* Begin framework functions */

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params, const int old_version,
                  void **new_params, int32_t *new_params_size,
                  int *new_version)
{
  // v4 → v5: added hl_detail_recovery field
  if(old_version == 4)
  {
    typedef struct dt_iop_3dcf_params_v4_t
    {
      float contrast;
      float gray_point;
      float vibrance;
      float spectral_brilliance;
      float hl_hue_shift;
      float hl_desaturation;
      float hl_desat_threshold;
      float gamut_knee;
      float gamut_steepness;
      dt_iop_st_colorspace_t output_cs;
      dt_iop_st_look_t color_look;
      float look_opacity;
      float contrast_pivot;
      float toe_power;
      float shoulder_power;
    } dt_iop_3dcf_params_v4_t;

    const dt_iop_3dcf_params_v4_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    new_p->contrast            = old->contrast;
    new_p->gray_point          = old->gray_point;
    new_p->vibrance            = old->vibrance;
    new_p->spectral_brilliance = old->spectral_brilliance;
    new_p->hl_hue_shift        = old->hl_hue_shift;
    new_p->hl_desaturation     = old->hl_desaturation;
    new_p->hl_desat_threshold  = old->hl_desat_threshold;
    new_p->gamut_knee          = old->gamut_knee;
    new_p->gamut_steepness     = old->gamut_steepness;
    new_p->output_cs           = old->output_cs;
    new_p->color_look          = old->color_look;
    new_p->look_opacity        = old->look_opacity;
    new_p->contrast_pivot      = old->contrast_pivot;
    new_p->toe_power           = old->toe_power;
    new_p->shoulder_power      = old->shoulder_power;
    new_p->hl_detail_recovery  = 0.0f;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 5;
    return 0;
  }

  return 1;
}

const char *name()
{
  return _("3D Colorimetric Film");
}

const char *aliases()
{
  return _("tone mapping|spectral");
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
  piece->data = dt_alloc1_align_type(dt_iop_3dcf_data_t);
  if(!piece->data) return;
  dt_iop_3dcf_data_t *d = piece->data;
  memset(d, 0, sizeof(dt_iop_3dcf_data_t));
  memcpy(&d->params, self->default_params, sizeof(dt_iop_3dcf_params_t));
  
  /* Initialise le contexte avec les params par défaut MAINTENANT, 
     pas plus tard au commit. Évite les race conditions sous Windows. */
  st_compute_context(&d->params, &d->ctx);
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
  dt_iop_3dcf_params_t *p = (dt_iop_3dcf_params_t *)p1;
  dt_iop_3dcf_data_t *d = piece->data;

  st_compute_context(p, &d->ctx);
  memcpy(&d->params, p, sizeof(dt_iop_3dcf_params_t));
}

void tiling_callback(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in, const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_3dcf_data_t *d = piece->data;

  /* HL detail recovery allocates one extra full-size guide buffer plus two
   * single-channel buffers (luminance + smoothed base). Account for that
   * extra memory so tiling doesn't under-estimate on very large images. */
  const gboolean hl_active = d && d->ctx.hl_detail_recovery > 0.0f;

  tiling->factor = hl_active ? 3.5f : 2.0f;
  tiling->factor_cl = hl_active ? 4.0f : 2.0f;
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
  dt_iop_3dcf_data_t *d = piece->data;
  const int width = roi_in->width;
  const int height = roi_in->height;
  const int ch = piece->colors;
  const size_t npixels = (size_t)width * height;

  /* Safety guard: ctx.ssts.n = 0 means commit_params() has not yet fired
   * (init_pipe only zeroes the struct). Pass the image through unchanged
   * rather than dividing by zero or crashing. Hits only on the very first
   * preview render on startup; commit_params() corrects this immediately. */
  if(!d || d->ctx.ssts.n <= 0.0)
  {
    memcpy(ovoid, ivoid, sizeof(float) * npixels * ch);
    return;
  }

  /* Bounds check: color_looks has 11 entries (0–10).
   * Clamp defensively in case a stale preset carries an out-of-range value. */
  const int look_idx = (d->params.color_look > 0 && d->params.color_look <= 10)
                       ? d->params.color_look : 0;

  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;

  const float *const mat = (look_idx > 0) ? color_looks[look_idx] : NULL;
  const float look_opacity = d->params.look_opacity;

  /* === HL detail recovery: pre-compute original luminance base via guided filter === */
  gray_image lum_orig_g = {0}, base_orig_g = {0};
  float *guide_sanitized = NULL;
  const float hl_detail_recovery = d->ctx.hl_detail_recovery;
  if(hl_detail_recovery > 0.0f)
  {
    lum_orig_g = new_gray_image(width, height);
    base_orig_g = new_gray_image(width, height);

    /* Sanitized copy of the full image, used as the guided-filter GUIDE.
     * Negative/NaN channels (CA fringing, sharpening halos) MUST be cleared
     * here too: the guide feeds the local mean/variance regression, and a
     * single stray NaN corrupts the entire filter window around it, not
     * just that pixel. Allocated/freed with dt_alloc_align/dt_free_align —
     * the same matched pair already used for piece->data in init_pipe /
     * cleanup_pipe, to avoid any allocator-mismatch crash on Windows. */
    guide_sanitized = (float *)dt_alloc_aligned(sizeof(float) * (size_t)npixels * ch);

    const float *lc = d->ctx.luma_coeff;
    #ifdef _OPENMP
    #pragma omp parallel for default(none) \
      shared(in, lum_orig_g, guide_sanitized, npixels, ch, lc)
    #endif
    for(size_t k = 0; k < npixels; k++)
    {
      const size_t idx = k * ch;
      float r = in[idx], g = in[idx + 1], b = in[idx + 2];
      r = isfinite(r) ? fmaxf(r, 0.0f) : 0.0f;
      g = isfinite(g) ? fmaxf(g, 0.0f) : 0.0f;
      b = isfinite(b) ? fmaxf(b, 0.0f) : 0.0f;

      if(guide_sanitized)
      {
        guide_sanitized[idx]     = r;
        guide_sanitized[idx + 1] = g;
        guide_sanitized[idx + 2] = b;
        if(ch == 4) guide_sanitized[idx + 3] = in[idx + 3];
      }

      lum_orig_g.data[k] = lc[0] * r + lc[1] * g + lc[2] * b;
    }

    /* Normalize guided filter radius to sensor resolution (ref: 36 MP 3:2 K1) */
    const float diag = sqrtf((float)piece->iwidth * piece->iwidth
                            + (float)piece->iheight * piece->iheight);
    const int gf_radius = fmaxf(4.0f, 8.0f * diag / 8848.0f);

    if(guide_sanitized)
      guided_filter(guide_sanitized, lum_orig_g.data, base_orig_g.data,
                    width, height, ch, gf_radius, 0.05f, 1.0f, 0.0f, FLT_MAX);
  }

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    shared(in, out, width, height, ch, d, npixels, mat, look_opacity, \
           hl_detail_recovery, lum_orig_g, base_orig_g)
  #endif
  for(size_t k = 0; k < npixels; k++)
  {
    const size_t idx = k * ch;
    float rgb_in[3] = { in[idx], in[idx + 1], in[idx + 2] };
    float rgb_out[3];

    /* Sanitize input: clamp negative (from CA / sharpening halos) to 0
     * and zero any NaN/Inf. Negatives reachable after demosaic + lens. */
    for(int c = 0; c < 3; c++)
      rgb_in[c] = isfinite(rgb_in[c]) ? fmaxf(rgb_in[c], 0.0f) : 0.0f;

    /* Save original luminance for detail recovery BEFORE the safety net
     * desaturates rgb_in below. Captured from the already-sanitized values
     * so CA fringing / NaN never reach the detail computation. This MUST
     * happen at this exact point to match the GPU kernel (kernel_3dcf,
     * "before safety net modifies rgb_in") — capturing it later from the
     * raw "in" buffer, as the previous version did, silently diverged from
     * OpenCL on any pixel with negative/NaN input channels. */
    float lum_orig = 0.0f;
    if(hl_detail_recovery > 0.0f)
    {
      const float *lc = d->ctx.luma_coeff;
      lum_orig = lc[0] * rgb_in[0] + lc[1] * rgb_in[1] + lc[2] * rgb_in[2];
    }

    /* Pre-pipeline safety net: sigmoid rolloff toward luma-gray */
    {
      const float lum = fmaxf(fmaxf(rgb_in[0], rgb_in[1]), rgb_in[2]);
      const float w = st_desat_weight(lum * d->ctx.exposure_factor, d->ctx.hl_desat, d->ctx.hl_desat_threshold);
      if(w > 0.0f && isfinite(w))
      {
        const float t = fminf(w, 1.0f);
        const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
        if(isfinite(ts) && ts > 0.0f)
        {
          for(int c = 0; c < 3; c++)
            rgb_in[c] = rgb_in[c] * (1.0f - ts) + lum * ts;
        }
      }
    }

    dt_st_pipeline_eval(rgb_in, rgb_out, &d->ctx);

    if(mat)
    {
      const float r = rgb_out[0], g = rgb_out[1], b = rgb_out[2];
      const float tr = r * mat[0] + g * mat[1] + b * mat[2];
      const float tg = r * mat[3] + g * mat[4] + b * mat[5];
      const float tb = r * mat[6] + g * mat[7] + b * mat[8];

      rgb_out[0] = r * (1.0f - look_opacity) + tr * look_opacity;
      rgb_out[1] = g * (1.0f - look_opacity) + tg * look_opacity;
      rgb_out[2] = b * (1.0f - look_opacity) + tb * look_opacity;

      for(int i = 0; i < 3; i++) rgb_out[i] = fmaxf(rgb_out[i], 0.0f);
    }

    /* HL detail recovery: re-inject guided-filter detail with gain compensation.
     * lum_orig was captured right after sanitization, above — same value
     * and same pipeline point as the GPU kernel. */
    if(hl_detail_recovery > 0.0f)
    {
      const float *lc = d->ctx.luma_coeff;
      const float lum_tm = lc[0] * rgb_out[0] + lc[1] * rgb_out[1] + lc[2] * rgb_out[2];
      if(lum_tm > 1e-6f && lum_orig > 1e-6f)
      {
        const float detail = lum_orig - base_orig_g.data[k];
        const float gain = lum_tm / lum_orig;
        const float lum_final = lum_tm + detail * hl_detail_recovery * gain;
        if(lum_final > 1e-6f)
        {
          const float scale = fmaxf(lum_final / lum_tm, 0.25f);
          rgb_out[0] *= scale;
          rgb_out[1] *= scale;
          rgb_out[2] *= scale;
        }
      }
    }

    out[idx]     = rgb_out[0];
    out[idx + 1] = rgb_out[1];
    out[idx + 2] = rgb_out[2];
    if(ch == 4) out[idx + 3] = in[idx + 3];
  }

  // Free buffers used by HL detail recovery
  if(hl_detail_recovery > 0.0f)
  {
    free_gray_image(&lum_orig_g);
    free_gray_image(&base_orig_g);
    dt_free_align(guide_sanitized);
  }
}

#ifdef HAVE_OPENCL
/* Pack the precomputed CPU context (+ active color look) into the GPU struct.
 * Mirrors exactly what process() reads from d->ctx and d->params. */
static void st_fill_cl_params(const dt_iop_3dcf_data_t *d,
                              dt_st_cl_params_t *clp)
{
  const dt_st_context_t *ctx = &d->ctx;

  for(int i = 0; i < 9; i++) clp->input_matrix[i]  = ctx->input_matrix[i];
  for(int i = 0; i < 9; i++) clp->output_matrix[i] = ctx->output_matrix[i];
  for(int i = 0; i < 3; i++) clp->luma_coeff[i]    = ctx->luma_coeff[i];

  clp->exposure_factor = ctx->exposure_factor;
  clp->contrast        = ctx->contrast;
  clp->contrast_pivot  = ctx->contrast_pivot;
  clp->hl_desat        = ctx->hl_desat;
  clp->hl_desat_threshold = ctx->hl_desat_threshold;
  clp->hl_rotation     = ctx->hl_rotation;
  clp->white_chroma_x  = ctx->white_chroma_x;
  clp->white_chroma_z  = ctx->white_chroma_z;
  clp->gray_point      = ctx->gray_point;
  clp->gray_gamma      = ctx->gray_gamma;
  clp->vibrance        = ctx->vibrance;
  clp->gamut_knee      = ctx->gamut_knee;
  clp->gamut_steepness = ctx->gamut_steepness;
  clp->toe_power       = ctx->toe_power;
  clp->shoulder_power  = ctx->shoulder_power;
  clp->hl_detail_recovery = ctx->hl_detail_recovery;

  for(int i = 0; i < 9; i++) clp->gamut_fwd[i] = ctx->gamut_fwd[i];
  for(int i = 0; i < 9; i++) clp->gamut_inv[i] = ctx->gamut_inv[i];
  clp->gamut_enable    = ctx->gamut_enable;

  clp->ssts_s_2 = (float)ctx->ssts.s_2;
  clp->ssts_m_2 = (float)ctx->ssts.m_2;
  clp->ssts_g   = (float)ctx->ssts.g;
  clp->ssts_t_1 = (float)ctx->ssts.t_1;
  clp->ssts_n_r = (float)ctx->ssts.n_r;
  clp->ssts_n   = (float)ctx->ssts.n;

  for(int i = 0; i < 360; i++) clp->spectral_boundary[i] = ctx->spectral_boundary[i];

  /* Same clamp as process(): color_looks has 11 entries (0..10) */
  const int look_idx = (d->params.color_look > 0 && d->params.color_look <= 10)
                       ? d->params.color_look : 0;
  clp->look_idx     = look_idx;
  clp->look_opacity = d->params.look_opacity;
  if(look_idx > 0)
    for(int i = 0; i < 9; i++) clp->color_look_mat[i] = color_looks[look_idx][i];
  else
    for(int i = 0; i < 9; i++) clp->color_look_mat[i] = (i % 4 == 0) ? 1.0f : 0.0f;
}

int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_3dcf_global_data_t *gd = self->global_data;
  const dt_iop_3dcf_data_t *d = piece->data;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  /* Safety guard mirroring process(): if commit_params() has not yet run the
   * context is all-zero — fall back to the CPU path (which copies through). */
  if(!d || d->ctx.ssts.n <= 0.0)
    return DT_OPENCL_PROCESS_CL;

  dt_st_cl_params_t clp;
  memset(&clp, 0, sizeof(clp));
  st_fill_cl_params(d, &clp);

  cl_int err = CL_SUCCESS;

  /* HL detail recovery: extract luminance, run guided filter on GPU */
  cl_mem dev_lum = NULL;
  cl_mem dev_base = NULL;
  cl_mem dev_in_sanitized = NULL;
  if(clp.hl_detail_recovery > 0.0f)
  {
    dev_lum = dt_opencl_alloc_device(devid, width, height, sizeof(float));
    dev_base = dt_opencl_alloc_device(devid, width, height, sizeof(float));
    /* Sanitized RGBA copy of dev_in, used as the guided-filter GUIDE instead
     * of the raw dev_in. Mirrors the CPU's guide_sanitized buffer: negative/
     * NaN channels from CA fringing must not reach the filter's local
     * mean/variance regression on either platform. */
    dev_in_sanitized = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
    if(!dev_lum || !dev_base || !dev_in_sanitized)
    {
      dt_opencl_release_mem_object(dev_lum);
      dt_opencl_release_mem_object(dev_base);
      dt_opencl_release_mem_object(dev_in_sanitized);
      return DT_OPENCL_PROCESS_CL;
    }

    err = dt_opencl_enqueue_kernel_2d_args(
      devid, gd->kernel_3dcf_extract_lum, width, height,
      CLARG(dev_in), CLARG(dev_in_sanitized), CLARG(dev_lum), CLARG(width), CLARG(height), CLARG(clp));
    if(err != CL_SUCCESS) goto error;

    /* Normalize guided filter radius to sensor resolution (ref: 36 MP 3:2 K1) */
    const float diag = sqrtf((float)piece->iwidth * piece->iwidth
                            + (float)piece->iheight * piece->iheight);
    const int gf_radius = fmaxf(4.0f, 8.0f * diag / 8848.0f);

    err = guided_filter_cl(devid, dev_in_sanitized, dev_lum, dev_base,
                           width, height, 4, gf_radius, 0.05f, 1.0f, 0.0f, CL_FLT_MAX);
    if(err != CL_SUCCESS) goto error;
  }

  err = dt_opencl_enqueue_kernel_2d_args(
    devid, gd->kernel_3dcf, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(clp),
    CLARG(dev_base));
  if(err != CL_SUCCESS) goto error;

error:
  dt_opencl_release_mem_object(dev_lum);
  dt_opencl_release_mem_object(dev_base);
  dt_opencl_release_mem_object(dev_in_sanitized);
  return err;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 45; // 3dcf.cl, from programs.conf
  dt_iop_3dcf_global_data_t *gd = malloc(sizeof(dt_iop_3dcf_global_data_t));
  self->data = gd;
  gd->kernel_3dcf = dt_opencl_create_kernel(program, "kernel_3dcf");
  gd->kernel_3dcf_extract_lum = dt_opencl_create_kernel(program, "kernel_3dcf_extract_lum");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_3dcf_global_data_t *gd = self->data;
  if(gd)
  {
    dt_opencl_free_kernel(gd->kernel_3dcf);
    dt_opencl_free_kernel(gd->kernel_3dcf_extract_lum);
    free(self->data);
    self->data = NULL;
  }
}
#endif // HAVE_OPENCL

void init_presets(dt_iop_module_so_t *self)
{
  self->pref_based_presets = TRUE;

  const char *workflow = dt_conf_get_string_const("plugins/darkroom/workflow");
  const gboolean auto_apply_st = workflow && strcmp(workflow, "scene-referred (3DCF)") == 0;

  dt_iop_3dcf_params_t p;
  memset(&p, 0, sizeof(p));
  p.contrast = 2.25f;
  p.spectral_brilliance = 5.0f;
  p.gray_point = 0.0f;
  p.vibrance = 1.0f;
  p.hl_desaturation = 0.60f;
  p.hl_desat_threshold = 0.45f;
  p.hl_hue_shift = 
  0.0f;
  p.gamut_knee = 0.20f; //CB
  p.gamut_steepness = 0.50f;
  p.output_cs = DT_ST_CS_REC2020;
  p.color_look = 0;
  p.look_opacity = 1.0f;
  p.contrast_pivot = 0.5f;
  p.toe_power = 1.0f;
  p.shoulder_power = 1.0f;
  p.hl_detail_recovery = 0.35f;

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

  dt_gui_presets_add_generic(_("default 3D Colorimetric Film"), self->op,
                              self->version(),
                              &p, sizeof(p),
                              TRUE, DEVELOP_BLEND_CS_RGB_SCENE);
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
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;

  dt_bauhaus_slider_set(g->contrast, p->contrast);
  dt_bauhaus_slider_set(g->contrast_pivot, p->contrast_pivot);
  dt_bauhaus_slider_set(g->shoulder_power, p->shoulder_power);
  dt_bauhaus_slider_set(g->toe_power, p->toe_power);
  dt_bauhaus_slider_set(g->spectral_brilliance, p->spectral_brilliance);
  dt_bauhaus_slider_set(g->mid_tone, p->gray_point);
  dt_bauhaus_slider_set(g->vibrance, p->vibrance);
  dt_bauhaus_slider_set(g->hl_desaturation, p->hl_desaturation);
  dt_bauhaus_slider_set(g->hl_desat_threshold, p->hl_desat_threshold);
  dt_bauhaus_slider_set(g->hl_hue_shift, p->hl_hue_shift);
  dt_bauhaus_slider_set(g->hl_detail_recovery, p->hl_detail_recovery);
  dt_bauhaus_slider_set(g->gamut_knee, p->gamut_knee);
  dt_bauhaus_slider_set(g->gamut_steepness, p->gamut_steepness);
  dt_bauhaus_combobox_set(g->color_space, p->output_cs);
  dt_bauhaus_combobox_set(g->color_look, p->color_look);
  dt_bauhaus_slider_set(g->look_opacity, p->look_opacity);

  gui_changed(self, NULL, NULL);
}

/* ========================================================================
 * Curve drawing callback — calibrated on AGX pattern
 * ======================================================================== */
static gboolean _draw_curve(GtkWidget *widget, cairo_t *crf,
                            const dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;

  /* precompute context from current params */
  dt_st_context_t ctx;
  st_compute_context(p, &ctx);

  gtk_widget_get_allocation(widget, &g->allocation);
  g->allocation.height -= DT_RESIZE_HANDLE_SIZE;

  cairo_surface_t *cst = dt_cairo_image_surface_create(
    CAIRO_FORMAT_ARGB32, g->allocation.width, g->allocation.height);
  PangoFontDescription *desc = pango_font_description_copy_static(
    darktable.bauhaus->pango_font_desc);
  cairo_t *cr = cairo_create(cst);
  PangoLayout *layout = pango_cairo_create_layout(cr);
  pango_layout_set_font_description(layout, desc);
  pango_cairo_context_set_resolution(
    pango_layout_get_context(layout), darktable.gui->dpi);
  g->context = gtk_widget_get_style_context(widget);

  const gint font_size = pango_font_description_get_size(desc);
  pango_font_description_set_size(desc, 0.95 * font_size);
  pango_layout_set_font_description(layout, desc);

  char text[32];
  g_strlcpy(text, "X", sizeof(text));
  pango_layout_set_text(layout, text, -1);
  pango_layout_get_pixel_extents(layout, &g->ink, NULL);
  const float line_height = g->ink.height;

  const int inset = DT_PIXEL_APPLY_DPI(4);
  const float margin_left   = 3.0f * line_height + 2.0f * inset;
  const float margin_bottom = 2.0f * line_height + 2.0f * inset;
  const float margin_top    = inset + 0.5f * line_height;
  const float margin_right  = inset;

  const float graph_width  = g->allocation.width  - margin_right - margin_left;
  const float graph_height = g->allocation.height - margin_bottom - margin_top;

  if(graph_width < 1.0f || graph_height < 1.0f)
    goto cleanup;

  gtk_render_background(g->context, cr, 0, 0,
    g->allocation.width, g->allocation.height);

  cairo_translate(cr, margin_left, margin_top + graph_height);
  cairo_scale(cr, 1.0, -1.0);

  cairo_rectangle(cr, 0.0, 0.0, graph_width, graph_height);
  set_color(cr, darktable.bauhaus->graph_bg);
  cairo_fill_preserve(cr);
  set_color(cr, darktable.bauhaus->graph_border);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));
  cairo_stroke(cr);

  /* horizontal guides + Y labels */
  cairo_save(cr);
  cairo_set_source_rgba(cr,
    darktable.bauhaus->graph_fg.red,
    darktable.bauhaus->graph_fg.green,
    darktable.bauhaus->graph_fg.blue, 0.4);
  const double dashes[] = { 4.0 / darktable.gui->ppd, 4.0 / darktable.gui->ppd };
  cairo_set_dash(cr, dashes, 2, 0);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));

  for(int i = 0; i <= 4; i++)
  {
    const float y_pct = i / 4.0f;
    const float y_graph = y_pct * graph_height;
    cairo_move_to(cr, 0, y_graph);
    cairo_line_to(cr, graph_width, y_graph);
    cairo_stroke(cr);

    cairo_save(cr);
    cairo_identity_matrix(cr);
    set_color(cr, darktable.bauhaus->graph_fg);
    snprintf(text, sizeof(text), "%.0f%%", 100.0f * y_pct);
    pango_layout_set_text(layout, text, -1);
    pango_layout_get_pixel_extents(layout, &g->ink, NULL);
    const float lx = margin_left - g->ink.width - inset / 2.0f;
    float ly = margin_top + graph_height - y_graph
               - g->ink.height / 2.0f - g->ink.y;
    ly = CLAMPF(ly,
      margin_top - g->ink.height / 2.0f - g->ink.y,
      margin_top + graph_height - g->ink.height / 2.0f - g->ink.y);
    cairo_move_to(cr, lx, ly);
    pango_cairo_show_layout(cr, layout);
    cairo_restore(cr);
  }
  cairo_restore(cr);

  /* EV scale */
  const float ev_min = -6.0f;
  const float ev_max =  4.0f;
  const float ev_range = ev_max - ev_min;

  /* vertical EV guide lines + X labels */
  cairo_save(cr);
  cairo_set_source_rgba(cr,
    darktable.bauhaus->graph_fg.red,
    darktable.bauhaus->graph_fg.green,
    darktable.bauhaus->graph_fg.blue, 0.4);
  cairo_set_dash(cr, dashes, 2, 0);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));

  for(int ev = (int)ceilf(ev_min); ev <= (int)floorf(ev_max); ev++)
  {
    const float x_norm = (ev - ev_min) / ev_range;
    const float x_graph = x_norm * graph_width;
    cairo_move_to(cr, x_graph, 0);
    cairo_line_to(cr, x_graph, graph_height);
    cairo_stroke(cr);

    /* label every 2 EV */
    if(ev % 2 == 0)
    {
      cairo_save(cr);
      cairo_identity_matrix(cr);
      set_color(cr, darktable.bauhaus->graph_fg);
      snprintf(text, sizeof(text), "%+d", ev);
      pango_layout_set_text(layout, text, -1);
      pango_layout_get_pixel_extents(layout, &g->ink, NULL);
      float lx = margin_left + x_graph - g->ink.width / 2.0f - g->ink.x;
      lx = CLAMPF(lx,
        margin_left - g->ink.width / 2.0f - g->ink.x,
        margin_left + graph_width - g->ink.width / 2.0f - g->ink.x);
      const float ly = margin_top + graph_height + inset / 2.0f;
      cairo_move_to(cr, lx, ly);
      pango_cairo_show_layout(cr, layout);
      cairo_restore(cr);
    }
  }
  cairo_restore(cr);

  /* three superimposed curves (200 samples, uniform in EV) */
  const int steps = 200;

  /* helper: draw one curve from sampled Y values */
  #define DRAW_CURVE(lw, r, g, b, alpha, eval_fn) \
    do { \
      cairo_set_source_rgba(cr, (r), (g), (b), (alpha)); \
      cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(lw)); \
      cairo_move_to(cr, 0, 0); \
      for(int k = 1; k <= steps; k++) \
      { \
        const float ev = ev_min + ev_range * (float)k / steps; \
        const float y_scene = 0.18f * powf(2.0f, ev); \
        const float x_norm = (ev - ev_min) / ev_range; \
        const float y_norm = fminf(fmaxf((eval_fn), 0.0f), 1.0f); \
        cairo_line_to(cr, x_norm * graph_width, y_norm * graph_height); \
      } \
      cairo_stroke(cr); \
    } while(0)

  /* 1. SSTS only (faint) — spectral brilliance changes the base roll-off */
  DRAW_CURVE(1.0f,
    darktable.bauhaus->graph_fg.red,
    darktable.bauhaus->graph_fg.green,
    darktable.bauhaus->graph_fg.blue, 0.25f,
    ({
      float _y = dt_st_ssts_fwd(&ctx.ssts, y_scene * ctx.exposure_factor) / (float)ctx.ssts.n;
      powf(fmaxf(_y, 0.0f), 1.0f / 2.4f);
    }));

  /* 2. SSTS + contrast S-curve (medium) */
  DRAW_CURVE(1.5f,
    darktable.bauhaus->graph_fg.red,
    darktable.bauhaus->graph_fg.green,
    darktable.bauhaus->graph_fg.blue, 0.55f,
    dt_st_compute_y_tm(y_scene, &ctx));

  /* 3. Full curve including mid-tones gamma (solid) */
  DRAW_CURVE(2.0f,
    darktable.bauhaus->graph_fg_active.red,
    darktable.bauhaus->graph_fg_active.green,
    darktable.bauhaus->graph_fg_active.blue, 1.0f,
    ({
      float _y = dt_st_compute_y_tm(y_scene, &ctx);
      if(ctx.gray_point != 0.0f)
        _y = powf(fminf(fmaxf(_y, 0.0f), 1.0f), ctx.gray_gamma);
      _y;
    }));

  #undef DRAW_CURVE

  /* helper: compute tone-mapped Y with gray_gamma (full curve) */
  #define TONE_MAP(ys) ({ \
    float _y = dt_st_compute_y_tm((ys), &ctx); \
    if(ctx.gray_point != 0.0f) _y = powf(fminf(fmaxf(_y, 0.0f), 1.0f), ctx.gray_gamma); \
    fminf(fmaxf(_y, 0.0f), 1.0f); \
  })

  /* ========== markers and reference lines ========== */

  /* 18% gray (EV 0) — primary marker */
  {
    const float y_gray = TONE_MAP(0.18f);
    const float y_graph = y_gray * graph_height;
    const float x_graph = (0.0f - ev_min) / ev_range * graph_width;

    cairo_save(cr);
    cairo_set_source_rgba(cr,
      darktable.bauhaus->graph_border.red,
      darktable.bauhaus->graph_border.green,
      darktable.bauhaus->graph_border.blue, 0.4);
    cairo_set_dash(cr, dashes, 2, 0);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));
    cairo_move_to(cr, 0, y_graph);
    cairo_line_to(cr, graph_width, y_graph);
    cairo_stroke(cr);
    cairo_restore(cr);

    cairo_save(cr);
    set_color(cr, darktable.bauhaus->graph_fg_active);
    cairo_arc(cr, x_graph, y_graph, DT_PIXEL_APPLY_DPI(4), 0, 2.0 * M_PI);
    cairo_fill(cr);
    cairo_restore(cr);
  }

  /* shadow marker at EV -3 */
  {
    const float y_scene = 0.18f * powf(2.0f, -3.0f);
    const float y_val = TONE_MAP(y_scene);
    const float x_graph = (-3.0f - ev_min) / ev_range * graph_width;
    const float y_graph = y_val * graph_height;

    cairo_save(cr);
    cairo_set_source_rgba(cr,
      darktable.bauhaus->graph_fg.red,
      darktable.bauhaus->graph_fg.green,
      darktable.bauhaus->graph_fg.blue, 0.6);
    cairo_arc(cr, x_graph, y_graph, DT_PIXEL_APPLY_DPI(2.5), 0, 2.0 * M_PI);
    cairo_fill(cr);
    cairo_restore(cr);
  }

  /* highlight marker at EV +2 */
  {
    const float y_scene = 0.18f * powf(2.0f, 2.0f);
    const float y_val = TONE_MAP(y_scene);
    const float x_graph = (2.0f - ev_min) / ev_range * graph_width;
    const float y_graph = y_val * graph_height;

    cairo_save(cr);
    cairo_set_source_rgba(cr,
      darktable.bauhaus->graph_fg.red,
      darktable.bauhaus->graph_fg.green,
      darktable.bauhaus->graph_fg.blue, 0.6);
    cairo_arc(cr, x_graph, y_graph, DT_PIXEL_APPLY_DPI(2.5), 0, 2.0 * M_PI);
    cairo_fill(cr);
    cairo_restore(cr);
  }

  /* pivot indicator — horizontal line at contrast_pivot output level */
  {
    const float pivot_y = ctx.contrast_pivot;
    const float y_graph = pivot_y * graph_height;

    cairo_save(cr);
    cairo_set_source_rgba(cr, 0.6f, 0.6f, 0.2f, 0.5f);
    cairo_set_dash(cr, dashes, 2, 0);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    cairo_move_to(cr, 0, y_graph);
    cairo_line_to(cr, graph_width, y_graph);
    cairo_stroke(cr);

    cairo_restore(cr);

    /* label "pivot" on the right margin */
    cairo_save(cr);
    cairo_identity_matrix(cr);
    cairo_set_source_rgba(cr, 0.6f, 0.6f, 0.2f, 0.7f);
    snprintf(text, sizeof(text), "pivot");
    pango_layout_set_text(layout, text, -1);
    pango_layout_get_pixel_extents(layout, &g->ink, NULL);
    const float lx_p = margin_left + graph_width + inset / 2.0f;
    float ly_p = margin_top + graph_height - y_graph
                 - g->ink.height / 2.0f - g->ink.y;
    ly_p = CLAMPF(ly_p,
      margin_top - g->ink.height / 2.0f - g->ink.y,
      margin_top + graph_height - g->ink.height / 2.0f - g->ink.y);
    cairo_move_to(cr, lx_p, ly_p);
    pango_cairo_show_layout(cr, layout);
    cairo_restore(cr);
  }

  #undef TONE_MAP

cleanup:
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  g_object_unref(layout);
  pango_font_description_free(desc);

  return FALSE;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = IOP_GUI_ALLOC(3dcf);
  self->gui_data = g;

  GtkWidget *main_vbox = dt_gui_vbox();
  self->widget = main_vbox;

  /* === Graph (always visible at top) === */
  g->graph = GTK_DRAWING_AREA(dt_ui_resize_wrap(NULL, DT_PIXEL_APPLY_DPI(200),
      "plugins/darkroom/3dcf/graph_height"));
  g_object_set_data(G_OBJECT(g->graph), "iop-instance", self);
  dt_action_define_iop(self, NULL, N_("graph"), GTK_WIDGET(g->graph), NULL);
  gtk_widget_set_can_focus(GTK_WIDGET(g->graph), TRUE);
  g_signal_connect(G_OBJECT(g->graph), "draw", G_CALLBACK(_draw_curve), self);
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->graph),
    _("3DC Film curve: scene luminance (X) vs display output (Y)"));
  gtk_box_pack_start(GTK_BOX(main_vbox), GTK_WIDGET(g->graph), TRUE, TRUE, 0);

  /* === TONE section === */
  dt_gui_box_add(GTK_BOX(main_vbox), dt_ui_section_label_new(C_("section", "tone")));

  g->spectral_brilliance = dt_bauhaus_slider_from_params(self, "spectral_brilliance");
  dt_bauhaus_slider_set_format(g->spectral_brilliance, "%");
  gtk_widget_set_tooltip_text(g->spectral_brilliance,
    _("Perceptual brightness: auto-exposure compensation with tone scale character. \n"
      "Higher values increase highlight headroom with a softer, film-like rolloff. \n"
      "Brightness is stabilised across the full range."));

  g->contrast = dt_bauhaus_slider_from_params(self, "contrast");
  dt_bauhaus_slider_set_factor(g->contrast, 50.0f);
  dt_bauhaus_slider_set_offset(g->contrast, -112.5f);
  dt_bauhaus_slider_set_format(g->contrast, " %");
  dt_bauhaus_slider_set_digits(g->contrast, 0);
  gtk_widget_set_tooltip_text(g->contrast,
    _("S-curve contrast pivoted at mid-gray. -100% = minimum contrast, \n"
      "0% = neutral, +100% = maximum. Negative values soften, positive values sharpen."));

  g->contrast_pivot = dt_bauhaus_slider_from_params(self, "contrast_pivot");
  dt_bauhaus_slider_set_factor(g->contrast_pivot, 100.0f);
  dt_bauhaus_slider_set_offset(g->contrast_pivot, -50.0f);
  dt_bauhaus_slider_set_format(g->contrast_pivot, " %");
  dt_bauhaus_slider_set_digits(g->contrast_pivot, 0);
  gtk_widget_set_tooltip_text(g->contrast_pivot,
    _("Pivot point for the contrast S-curve. Higher values (right) shift the fulcrum towards \n"
      "shadows, brightening the image. Lower values (left) shift it towards highlights, darkening it."));

  g->shoulder_power = dt_bauhaus_slider_from_params(self, "shoulder_power");
  dt_bauhaus_slider_set_factor(g->shoulder_power, 100.0f);
  dt_bauhaus_slider_set_offset(g->shoulder_power, -100.0f);
  dt_bauhaus_slider_set_format(g->shoulder_power, " %");
  dt_bauhaus_slider_set_digits(g->shoulder_power, 1);
  gtk_widget_set_tooltip_text(g->shoulder_power,
    _("Shoulder (highlight) contrast multiplier relative to master contrast. \n"
      "100% = identical to master, 0% = no contrast in highlights, 200% = double contrast."));

  g->toe_power = dt_bauhaus_slider_from_params(self, "toe_power");
  dt_bauhaus_slider_set_factor(g->toe_power, 100.0f);
  dt_bauhaus_slider_set_offset(g->toe_power, -100.0f);
  dt_bauhaus_slider_set_format(g->toe_power, " %");
  dt_bauhaus_slider_set_digits(g->toe_power, 1);
  gtk_widget_set_tooltip_text(g->toe_power,
    _("Toe (shadow) contrast multiplier relative to master contrast. \n"
      "100% = identical to master, 0% = no contrast in shadows, 200% = double contrast."));

  g->mid_tone = dt_bauhaus_slider_from_params(self, "gray_point");
  dt_bauhaus_slider_set_factor(g->mid_tone, 100.0f);
  dt_bauhaus_slider_set_format(g->mid_tone, " %");
  dt_bauhaus_slider_set_digits(g->mid_tone, 0);
  gtk_widget_set_tooltip_text(g->mid_tone,
    _("Mid-tone brightness adjustment (gamma). \n"
      "Moving to the right (+100%) brightens mid-tones, \n"
      "moving to the left (-100%) darkens them."));

  /* === COLOR section === */
  dt_gui_box_add(GTK_BOX(main_vbox), dt_ui_section_label_new(C_("section", "color")));

  g->vibrance = dt_bauhaus_slider_from_params(self, "vibrance");
  dt_bauhaus_slider_set_factor(g->vibrance, 100.0f);
  dt_bauhaus_slider_set_offset(g->vibrance, -100.0f);
  dt_bauhaus_slider_set_format(g->vibrance, " %");
  dt_bauhaus_slider_set_digits(g->vibrance, 0);
  gtk_widget_set_tooltip_text(g->vibrance,
    _("Smart saturation boost. Protects already-saturated colors while enhancing pastels."));

  g->color_look = dt_bauhaus_combobox_from_params(self, "color_look");
  gtk_widget_set_tooltip_text(g->color_look, _("Apply a color style to the image."));

  g->look_opacity = dt_bauhaus_slider_from_params(self, "look_opacity");
  dt_bauhaus_widget_set_label(g->look_opacity, NULL, _("look opacity"));
  dt_bauhaus_slider_set_format(g->look_opacity, "%");
  dt_bauhaus_slider_set_factor(g->look_opacity, 100.0);
  gtk_widget_set_tooltip_text(g->look_opacity, _("Adjust the strength of the selected color style."));

  /* === Advanced section === */
  dt_gui_new_collapsible_section(&g->advanced_section,
                                 "plugins/darkroom/3dcf/expand_advanced",
                                 _("advanced"),
                                 GTK_BOX(main_vbox),
                                 DT_ACTION(self));

  self->widget = GTK_WIDGET(g->advanced_section.container);

  /* highlights sub-group */
  dt_gui_box_add(GTK_BOX(self->widget), dt_ui_section_label_new(C_("section", "highlights")));

  g->hl_desaturation = dt_bauhaus_slider_from_params(self, "hl_desaturation");
  dt_bauhaus_slider_set_factor(g->hl_desaturation, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_desaturation, " %");
  dt_bauhaus_slider_set_digits(g->hl_desaturation, 0);
  gtk_widget_set_tooltip_text(g->hl_desaturation,
    _("Highlight roll-off combining luminance desaturation and vibrance-negative. \n"
      "0% = off, 15% = natural rolloff, 100% = maximum desaturation."));

  g->hl_desat_threshold = dt_bauhaus_slider_from_params(self, "hl_desat_threshold");
  dt_bauhaus_slider_set_factor(g->hl_desat_threshold, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_desat_threshold, " %");
  dt_bauhaus_slider_set_digits(g->hl_desat_threshold, 0);
  gtk_widget_set_tooltip_text(g->hl_desat_threshold,
    _("Luminance threshold at which highlight desaturation begins. \n"
      "Lower values desaturate earlier (more protection), higher values preserve saturation longer."));

  g->hl_hue_shift = dt_bauhaus_slider_from_params(self, "hl_hue_shift");
  dt_bauhaus_slider_set_factor(g->hl_hue_shift, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_hue_shift, " %");
  dt_bauhaus_slider_set_digits(g->hl_hue_shift, 0);
  gtk_widget_set_tooltip_text(g->hl_hue_shift,
    _("Abney rotation in highlights, modulated by pixel saturation. \n"
      "Positive rotates toward cool (blue), negative toward warm (salmon). \n"
      "Vibrance-negative desaturation: saturated pixels desaturate further. \n"
      "Independent of highlight roll-off."));

  g->hl_detail_recovery = dt_bauhaus_slider_from_params(self, "hl_detail_recovery");
  dt_bauhaus_slider_set_factor(g->hl_detail_recovery, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_detail_recovery, " %");
  dt_bauhaus_slider_set_digits(g->hl_detail_recovery, 0);
  gtk_widget_set_tooltip_text(g->hl_detail_recovery,
    _("Restore local contrast in highlights smoothed by tone mapping. \n"
      "Uses guided filter to extract detail from the original scene luminance \n"
      "and re-injects it after tone mapping with gain compensation. \n"
      "0% = off, 100% = full detail recovery."));

  /* gamut sub-group */
  dt_gui_box_add(GTK_BOX(self->widget), dt_ui_section_label_new(C_("section", "gamut")));

  g->gamut_knee = dt_bauhaus_slider_from_params(self, "gamut_knee");
  dt_bauhaus_slider_set_factor(g->gamut_knee, 100.0f);
  dt_bauhaus_slider_set_format(g->gamut_knee, " %");
  dt_bauhaus_slider_set_digits(g->gamut_knee, 0);
  gtk_widget_set_tooltip_text(g->gamut_knee,
    _("Knee point for gamut roll-off. Lower values start compression earlier in the luminance range."));

  g->gamut_steepness = dt_bauhaus_slider_from_params(self, "gamut_steepness");
  dt_bauhaus_slider_set_factor(g->gamut_steepness, 100.0f);
  dt_bauhaus_slider_set_format(g->gamut_steepness, " %");
  dt_bauhaus_slider_set_digits(g->gamut_steepness, 0);
  gtk_widget_set_tooltip_text(g->gamut_steepness,
    _("Steepness of the gamut roll-off curve. Higher values result in a harder transition towards the gamut boundary."));

  g->color_space = dt_bauhaus_combobox_from_params(self, "output_cs");
  gtk_widget_set_tooltip_text(g->color_space,
    _("Output color space for luma coefficients in vibrance computation. "
      "Also protects against out-of-gamut colors by clipping to the "
      "selected primaries. sRGB/Rec. 709 is the narrowest, Rec. 2020 "
      "the widest (no clipping)."));

  self->widget = main_vbox;
}

/* NOTE: no gui_cleanup() is needed.
 * IOP_GUI_ALLOC() allocates gui_data via dt_calloc_aligned() → dt_alloc_aligned(),
 * which on Windows is _aligned_malloc() (and posix_memalign() on Linux/macOS).
 * The framework already frees it for us in dt_iop_gui_cleanup_module() with the
 * matching dt_free_align(). Freeing it here with free()/g_free() would release an
 * _aligned_malloc() block with the wrong deallocator → heap corruption (c0000374)
 * at module load on Windows. The gui_data struct only holds GtkWidget* and an
 * embedded collapsible_section (no separately-owned resources), so the default
 * widget destruction is sufficient. */

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;

  if(!w || w == g->color_look)
  {
    gtk_widget_set_visible(g->look_opacity, p->color_look > DT_ST_LOOK_NEUTRAL);
  }

  /* redraw curve graph on any parameter change */
  gtk_widget_queue_draw(GTK_WIDGET(g->graph));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
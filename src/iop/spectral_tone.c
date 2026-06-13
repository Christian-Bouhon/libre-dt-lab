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
#include "common/colorspaces_inline_conversions.h"
#include "common/imagebuf.h"
#include "common/matrices.h"
#include "develop/imageop.h"
#include "develop/tiling.h"
#include "develop/imageop_gui.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(3, dt_iop_spectral_tone_params_t)

/* Type definitions for the ACES 2.0 SSTS pipeline context.
 * spectral_tone_data.c and spectral_tone_pipeline.c have been merged
 * into this single translation unit for Windows/MinGW compatibility. */

typedef enum dt_iop_st_colorspace_t
{
  DT_ST_CS_REC709 = 0,    // $DESCRIPTION: "Rec. 709"
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

typedef struct dt_iop_spectral_tone_params_t
{
  float contrast;              // $MIN: 0.25 $MAX: 4.25 $DEFAULT: 2.25 $DESCRIPTION: "contrast"
  float gray_point;            // $MIN: -1 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "mid-tones"
  float vibrance;              // $MIN: 0 $MAX: 2 $DEFAULT: 1.0 $DESCRIPTION: "vibrance"
  float spectral_brilliance;   // $MIN: 0 $MAX: 100 $DEFAULT: 5 $DESCRIPTION: "perceptual brightness"
  float hl_hue_shift;          // $MIN: -1 $MAX: 1 $DEFAULT: 0 $STEP: 0.01 $DESCRIPTION: "Abney rotation"
  float hl_desaturation;       // $MIN: 0 $MAX: 1 $DEFAULT: 0.50 $DESCRIPTION: "highlight roll-off"
  float gamut_knee;            // $MIN: 0 $MAX: 1 $DEFAULT: 0.15 $DESCRIPTION: "gamut knee"
  float gamut_steepness;       // $MIN: 0 $MAX: 1 $DEFAULT: 0.50 $DESCRIPTION: "gamut steepness"
  dt_iop_st_colorspace_t output_cs;  // $DEFAULT: DT_ST_CS_REC2020 $DESCRIPTION: "color space"
  dt_iop_st_look_t color_look;       // $DEFAULT: DT_ST_LOOK_NEUTRAL $DESCRIPTION: "color look"
  float look_opacity;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "look opacity"
  float contrast_pivot;        // $MIN: 0.01 $MAX: 0.99 $DEFAULT: 0.5 $DESCRIPTION: "contrast pivot"
} dt_iop_spectral_tone_params_t;

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
  float hl_rotation;
  float white_chroma_x;
  float white_chroma_z;
  float gray_point;
  float gray_gamma;
  float vibrance;
  float gamut_knee;
  float gamut_steepness;
  dt_st_ssts_params_t ssts;
} dt_st_context_t;

typedef struct dt_iop_spectral_tone_data_t
{
  dt_iop_spectral_tone_params_t params;
  dt_st_context_t ctx;
} dt_iop_spectral_tone_data_t;

/* GPU-side context — MUST match dt_st_cl_params_t in spectral_tone.cl byte for
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
  float hl_rotation;
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
  int   look_idx;
} dt_st_cl_params_t;

typedef struct dt_iop_spectral_tone_global_data_t
{
  int kernel_spectral_tone;
} dt_iop_spectral_tone_global_data_t;

typedef struct dt_iop_spectral_tone_gui_data_t
{
  GtkWidget *contrast;
  GtkWidget *contrast_pivot;
  GtkWidget *spectral_brilliance;
  GtkWidget *mid_tone;
  GtkWidget *vibrance;
  GtkWidget *hl_desaturation;
  GtkWidget *hl_hue_shift;
  GtkWidget *gamut_knee;
  GtkWidget *gamut_steepness;
  dt_gui_collapsible_section_t advanced_section;
  GtkWidget *color_space;
  GtkWidget *color_look;
  GtkWidget *look_opacity;
} dt_iop_spectral_tone_gui_data_t;

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

double dt_st_ssts_fwd(const dt_st_ssts_params_t *p, double x)
{
  if(x <= 0.0) return 0.0;

  /* Michaelis-Menten segment */
  const double f = p->m_2 * pow(x / (x + p->s_2), p->g);

  /* Flare compensation */
  const double h = (f * f) / (f + p->t_1);

  /* Display luminance (cd/m^2) */
  return h * p->n_r;
}

/* Compute tone-mapped Y from scene-linear Y (SSTS + BT.1886 + contrast) */
double dt_st_compute_y_tm(double y_scene, const dt_st_context_t *ctx)
{
  /* Guard: ssts.n is set by dt_st_ssts_init() to peak_luminance (>=100).
   * A value of 0 means the context was never initialised — return black
   * rather than raising a FPE (0.0/0.0) which crashes on Windows. */
  if(ctx->ssts.n <= 0.0) return 0.0;

  double y_disp = dt_st_ssts_fwd(&ctx->ssts, y_scene * ctx->exposure_factor);
  double y_tm = y_disp / ctx->ssts.n;

  /* BT.1886 OETF */
  y_tm = pow(fmax(y_tm, 0.0), 1.0 / 2.4);

  /* Post-SSTS contrast S-curve pivoted at mid-gray (0.5) */
  if(ctx->contrast != 1.0f)
  {
    const double c = (double)ctx->contrast;
    const double p = (double)ctx->contrast_pivot;
    if(y_tm <= p)
      y_tm = p * pow(fmax(y_tm / p, 0.0), c);
    else
      y_tm = 1.0 - (1.0 - p) * pow(fmax((1.0 - y_tm) / (1.0 - p), 0.0), c);
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
static inline double st_desat_weight(double y_norm, float hl_desat)
{
  const double threshold = 0.45; // CB On commence a desaturer beaucoup plus tôt
  if(hl_desat <= 0.0f || y_norm <= threshold) return 0.0;
  if(!isfinite(y_norm)) return 0.0;
  const double t = fmax(y_norm - threshold, 0.0) / y_norm;
  const double x = fmin(t * (double)hl_desat, 1.0);
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

/* Spectral gamut: film-like chromaticity roll-off in CIE xyY space
 *
 * Emulates film dye absorption: colors far from the D50 white point in CIE
 * chromaticity (x,y) desaturate naturally because dyes have limited density.
 * Applied in xyY space (perceptually uniform), compressing the distance from
 * white with a smooth asymptotic knee, then mapped back to XYZ preserving Y.
 */
static inline void st_spectral_gamut(
  double *x_tm, double *z_tm, double y_tm,
  const double white_x_ratio, const double white_z_ratio,
  const double knee, const double steepness)
{
  if(y_tm <= 0.0) return;
  if(!isfinite(*x_tm) || !isfinite(*z_tm)) return;

  /* Convert current XYZ to CIE xy chromaticity */
  const double sum = *x_tm + y_tm + *z_tm;
  if(sum <= 0.0) return;
  const double cie_x = *x_tm / sum;
  const double cie_z = *z_tm / sum;

  /* D50 white point in CIE xy (from white ratios: X/Y, Z/Y) */
  const double wy = 1.0;
  const double wx = white_x_ratio;
  const double wz = white_z_ratio;
  const double wsum = wx + wy + wz;
  const double white_cie_x = wx / wsum;
  const double white_cie_z = wz / wsum;

  /* Distance from white in uniform CIE xy space */
  const double dx = cie_x - white_cie_x;
  const double dz = cie_z - white_cie_z;
  const double chroma_sq = dx * dx + dz * dz;

  if(chroma_sq > knee * knee)
  {
    const double chroma = sqrt(chroma_sq);
    const double excess = chroma - knee;
    const double compression = excess / (excess + steepness);
    const double scale = (chroma - compression * excess) / chroma;

    /* New chromaticity in CIE xy, then back to XYZ preserving Y */
    const double x_new = white_cie_x + scale * dx;
    const double z_new = white_cie_z + scale * dz;
    const double y_new = 1.0 - x_new - z_new;

    if(y_new > 0.0)
    {
      const double S_new = y_tm / y_new;
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
  const double r = (double)rgb_in[0], g = (double)rgb_in[1], b = (double)rgb_in[2];
  const float *M_in = ctx->input_matrix;
  double x_abs = M_in[0] * r + M_in[1] * g + M_in[2] * b;
  double y_abs = M_in[3] * r + M_in[4] * g + M_in[5] * b;
  double z_abs = M_in[6] * r + M_in[7] * g + M_in[8] * b;

  if(!(y_abs > 1e-10) || !isfinite(y_abs))
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  /* Chromaticity ratios */
  const double x_ratio = fmin(fmax(x_abs / y_abs, -100.0), 100.0);
  const double z_ratio = fmin(fmax(z_abs / y_abs, -100.0), 100.0);

  /* Step 2-3: Tone-mapped Y (SSTS + BT.1886 + contrast) */
  double y_tm = dt_st_compute_y_tm(y_abs, ctx);

  /* Step 4: Mid-tone adjustment — gamma pivot */
  if(ctx->gray_point != 0.0f)
  {
    float y_lvl = fminf(fmaxf((float)y_tm, 0.0f), 1.0f);
    y_lvl = powf(y_lvl, ctx->gray_gamma);
    y_tm = (double)y_lvl;
  }

  /* Step 5: Scale chromaticity with tone-mapped luminance */
  double x_tm = x_ratio * y_tm;
  double z_tm = z_ratio * y_tm;

  /* Step 6: Spectral gamut — film-like chromaticity roll-off in CIE xy */
  st_spectral_gamut(&x_tm, &z_tm, y_tm,
                    (double)ctx->white_chroma_x,
                    (double)ctx->white_chroma_z,
                    (double)ctx->gamut_knee,
                    (double)ctx->gamut_steepness);

  /* Step 7: XYZ -> Output RGB via precomputed matrix */
  const float *M = ctx->output_matrix;
  float rgb[3];
  rgb[0] = M[0] * (float)x_tm + M[1] * (float)y_tm + M[2] * (float)z_tm;
  rgb[1] = M[3] * (float)x_tm + M[4] * (float)y_tm + M[5] * (float)z_tm;
  rgb[2] = M[6] * (float)x_tm + M[7] * (float)y_tm + M[8] * (float)z_tm;

  /* Step 8: Film-print highlight desaturation toward white */
  {
    const double y_exposed = y_abs * ctx->exposure_factor;
    const double w = st_desat_weight(y_exposed, ctx->hl_desat);
    if(w > 0.0 && isfinite(w))
    {
      /* Progressive Abney hue rotation — weight indépendant de hl_desat */
      if(ctx->hl_rotation != 0.0f)
      {
        const float wr = fminf((float)st_desat_weight(y_exposed, 1.0f), 1.0f);
        const float angle = ctx->hl_rotation * 0.15f * wr; //CB
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

      /* Vibrance négative : désature plus les pixels saturés, activée par hl_desat et hl_hue_shift */
      double w_final = w;
      if(ctx->hl_desat > 0.0f || ctx->hl_rotation != 0.0f)
      {
        const float maxc = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
        const float minc = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
        const float sat = (maxc > 0.0f) ? (maxc - minc) / maxc : 0.0f;
        const float ss = (sat * sat) / (sat * sat + (1.0f - sat) * (1.0f - sat) + 1e-6f);

        if(ctx->hl_desat > 0.0f)
        {
          const float vib_neg = ctx->hl_desat * ss * 0.5f;
          const double w_vib = st_desat_weight(y_exposed, 1.0f) * vib_neg;
          w_final = fmin(w_final + w_vib, 1.0);
        }
        if(ctx->hl_rotation != 0.0f)
        {
          const float vib_neg = fabsf(ctx->hl_rotation) * ss * 1.0f; //CB
          const double w_rot = st_desat_weight(y_exposed, 1.0f) * vib_neg;
          w_final = fmax(w_final, w_rot);
        }
      }

      /* Blend toward white with sigmoidal curve — film-like */
      const float t = fminf((float)w_final, 1.0f);
      const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
      if(isfinite(ts))
      {
        rgb[0] = rgb[0] * (1.0f - ts) + 1.0f * ts;
        rgb[1] = rgb[1] * (1.0f - ts) + 1.0f * ts;
        rgb[2] = rgb[2] * (1.0f - ts) + 1.0f * ts;
      }
    }
  }

  /* Step 9: Vibrance — saturation with high-sat protection */
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

  /* Step 10: Gamut compression */
  st_gamut_compress(rgb);

  /* Step 11: Clamp negative channels (safety) */
  rgb_out[0] = isfinite(rgb[0]) ? fmaxf(rgb[0], 0.0f) : 0.0f;
  rgb_out[1] = isfinite(rgb[1]) ? fmaxf(rgb[1], 0.0f) : 0.0f;
  rgb_out[2] = isfinite(rgb[2]) ? fmaxf(rgb[2], 0.0f) : 0.0f;
}

/* Given input RGB in working space, compute the context */
static void st_compute_context(dt_iop_spectral_tone_params_t *p,
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
  ctx->hl_rotation = p->hl_hue_shift;
  ctx->gamut_knee = p->gamut_knee;
  ctx->gamut_steepness = p->gamut_steepness;

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
  if(old_version == 1 || old_version == 2)
  {
    typedef struct dt_iop_spectral_tone_params_v2_t
    {
      float contrast, gray_point, vibrance, spectral_brilliance;
      float hl_hue_shift, hl_desaturation, gamut_knee, gamut_steepness;
      int output_cs, color_look;
      float look_opacity;
    } dt_iop_spectral_tone_params_v2_t;

    typedef struct dt_iop_spectral_tone_params_v1_t
    {
      float contrast, gray_point, vibrance, spectral_brilliance;
      float hl_hue_shift, hl_desaturation, gamut_knee, gamut_steepness;
      int output_cs;
    } dt_iop_spectral_tone_params_v1_t;

    /* Must use malloc(): the framework frees the result with free()
     * (see dt_iop_legacy_params / imageop.c). Allocating with g_malloc0()
     * and letting free() release it mixes allocators → heap corruption on
     * Windows. calloc() matches the previous zero-init semantics. */
    dt_iop_spectral_tone_params_t *n = calloc(1, sizeof(dt_iop_spectral_tone_params_t));
    if(!n) return 1;

    if(old_version == 1)
    {
      const dt_iop_spectral_tone_params_v1_t *o = (const dt_iop_spectral_tone_params_v1_t*)old_params;
      n->contrast = o->contrast;
      n->gray_point = o->gray_point;
      n->vibrance = o->vibrance;
      n->spectral_brilliance = o->spectral_brilliance;
      n->hl_hue_shift = o->hl_hue_shift;
      n->hl_desaturation = o->hl_desaturation;
      n->gamut_knee = o->gamut_knee;
      n->gamut_steepness = o->gamut_steepness;
      n->output_cs = (dt_iop_st_colorspace_t)o->output_cs;
      n->color_look = DT_ST_LOOK_NEUTRAL;
      n->look_opacity = 1.0f;
    }
    else /* v2 */
    {
      const dt_iop_spectral_tone_params_v2_t *o = (const dt_iop_spectral_tone_params_v2_t*)old_params;
      n->contrast = o->contrast;
      n->gray_point = o->gray_point;
      n->vibrance = o->vibrance;
      n->spectral_brilliance = o->spectral_brilliance;
      n->hl_hue_shift = o->hl_hue_shift;
      n->hl_desaturation = o->hl_desaturation;
      n->gamut_knee = o->gamut_knee;
      n->gamut_steepness = o->gamut_steepness;
      n->output_cs = (dt_iop_st_colorspace_t)o->output_cs;
      n->color_look = (dt_iop_st_look_t)o->color_look;
      n->look_opacity = o->look_opacity;
    }

    n->contrast_pivot = 0.5f;
    *new_params = n;
    *new_params_size = sizeof(dt_iop_spectral_tone_params_t);
    *new_version = 3;
    return 0;
  }
  return 1;
}

const char *name()
{
  return _("spectral tone");
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
  piece->data = dt_alloc1_align_type(dt_iop_spectral_tone_data_t);
  if(!piece->data) return;
  dt_iop_spectral_tone_data_t *d = piece->data;
  memset(d, 0, sizeof(dt_iop_spectral_tone_data_t));
  memcpy(&d->params, self->default_params, sizeof(dt_iop_spectral_tone_params_t));
  
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
  dt_iop_spectral_tone_params_t *p = (dt_iop_spectral_tone_params_t *)p1;
  dt_iop_spectral_tone_data_t *d = piece->data;

  st_compute_context(p, &d->ctx);
  memcpy(&d->params, p, sizeof(dt_iop_spectral_tone_params_t));
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
  dt_iop_spectral_tone_data_t *d = piece->data;
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

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    shared(in, out, width, height, ch, d, npixels, mat, look_opacity)
  #endif
  for(size_t k = 0; k < npixels; k++)
  {
    const size_t idx = k * ch;
    float rgb_in[3] = { in[idx], in[idx + 1], in[idx + 2] };
    float rgb_out[3];

    /* Pre-pipeline safety net: sigmoid rolloff toward luma-gray */
    {
      const float lum = fmaxf(fmaxf(rgb_in[0], rgb_in[1]), rgb_in[2]);
      const double w = st_desat_weight(lum * d->ctx.exposure_factor, d->ctx.hl_desat);
      if(w > 0.0 && isfinite(w))
      {
        const float t = fminf((float)w, 1.0f);
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

    out[idx]     = rgb_out[0];
    out[idx + 1] = rgb_out[1];
    out[idx + 2] = rgb_out[2];
    if(ch == 4) out[idx + 3] = in[idx + 3];
  }
}

#ifdef HAVE_OPENCL
/* Pack the precomputed CPU context (+ active color look) into the GPU struct.
 * Mirrors exactly what process() reads from d->ctx and d->params. */
static void st_fill_cl_params(const dt_iop_spectral_tone_data_t *d,
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
  clp->hl_rotation     = ctx->hl_rotation;
  clp->white_chroma_x  = ctx->white_chroma_x;
  clp->white_chroma_z  = ctx->white_chroma_z;
  clp->gray_point      = ctx->gray_point;
  clp->gray_gamma      = ctx->gray_gamma;
  clp->vibrance        = ctx->vibrance;
  clp->gamut_knee      = ctx->gamut_knee;
  clp->gamut_steepness = ctx->gamut_steepness;

  clp->ssts_s_2 = (float)ctx->ssts.s_2;
  clp->ssts_m_2 = (float)ctx->ssts.m_2;
  clp->ssts_g   = (float)ctx->ssts.g;
  clp->ssts_t_1 = (float)ctx->ssts.t_1;
  clp->ssts_n_r = (float)ctx->ssts.n_r;
  clp->ssts_n   = (float)ctx->ssts.n;

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
  const dt_iop_spectral_tone_global_data_t *gd = self->global_data;
  const dt_iop_spectral_tone_data_t *d = piece->data;

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

  return dt_opencl_enqueue_kernel_2d_args(
    devid, gd->kernel_spectral_tone, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(clp));
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 42; // spectral_tone.cl, from programs.conf
  dt_iop_spectral_tone_global_data_t *gd = malloc(sizeof(dt_iop_spectral_tone_global_data_t));
  self->data = gd;
  gd->kernel_spectral_tone = dt_opencl_create_kernel(program, "spectral_tone");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_spectral_tone_global_data_t *gd = self->data;
  if(gd)
  {
    dt_opencl_free_kernel(gd->kernel_spectral_tone);
    free(self->data);
    self->data = NULL;
  }
}
#endif // HAVE_OPENCL

void init_presets(dt_iop_module_so_t *self)
{
  self->pref_based_presets = TRUE;

  const char *workflow = dt_conf_get_string_const("plugins/darkroom/workflow");
  const gboolean auto_apply_st = workflow && strcmp(workflow, "scene-referred (spectral tone)") == 0;

  dt_iop_spectral_tone_params_t p;
  memset(&p, 0, sizeof(p));
  p.contrast = 2.25f;
  p.spectral_brilliance = 5.0f;
  p.gray_point = 0.0f;
  p.vibrance = 1.0f;
  p.hl_desaturation = 0.50f;
  p.hl_hue_shift = 0.0f;
  p.gamut_knee = 0.15f;
  p.gamut_steepness = 0.50f;
  p.output_cs = DT_ST_CS_REC2020;
  p.color_look = 0;
  p.look_opacity = 1.0f;
  p.contrast_pivot = 0.5f;

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

  dt_gui_presets_add_generic(_("default spectral tone"), self->op,
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
  dt_iop_spectral_tone_gui_data_t *g = self->gui_data;
  dt_iop_spectral_tone_params_t *p = self->params;

  dt_bauhaus_slider_set(g->contrast, p->contrast);
  dt_bauhaus_slider_set(g->contrast_pivot, p->contrast_pivot);
  dt_bauhaus_slider_set(g->spectral_brilliance, p->spectral_brilliance);
  dt_bauhaus_slider_set(g->mid_tone, p->gray_point);
  dt_bauhaus_slider_set(g->vibrance, p->vibrance);
  dt_bauhaus_slider_set(g->hl_desaturation, p->hl_desaturation);
  dt_bauhaus_slider_set(g->hl_hue_shift, p->hl_hue_shift);
  dt_bauhaus_slider_set(g->gamut_knee, p->gamut_knee);
  dt_bauhaus_slider_set(g->gamut_steepness, p->gamut_steepness);
  dt_bauhaus_combobox_set(g->color_space, p->output_cs);
  dt_bauhaus_combobox_set(g->color_look, p->color_look);
  dt_bauhaus_slider_set(g->look_opacity, p->look_opacity);

  gui_changed(self, NULL, NULL);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_spectral_tone_gui_data_t *g = IOP_GUI_ALLOC(spectral_tone);
  self->gui_data = g;

  GtkWidget *main_vbox = dt_gui_vbox();
  self->widget = main_vbox;

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

  g->mid_tone = dt_bauhaus_slider_from_params(self, "gray_point");
  dt_bauhaus_slider_set_factor(g->mid_tone, 100.0f);
  dt_bauhaus_slider_set_format(g->mid_tone, " %");
  dt_bauhaus_slider_set_digits(g->mid_tone, 0);
  gtk_widget_set_tooltip_text(g->mid_tone,
    _("Mid-tone brightness adjustment (gamma). \n"
      "Moving to the right (+100%) brightens mid-tones, \n"
      "moving to the left (-100%) darkens them."));

  g->vibrance = dt_bauhaus_slider_from_params(self, "vibrance");
  dt_bauhaus_slider_set_factor(g->vibrance, 100.0f);
  dt_bauhaus_slider_set_offset(g->vibrance, -100.0f);
  dt_bauhaus_slider_set_format(g->vibrance, " %");
  dt_bauhaus_slider_set_digits(g->vibrance, 0);
  gtk_widget_set_tooltip_text(g->vibrance,
    _("Smart saturation boost. Protects already-saturated colors while enhancing pastels."));
  
    g->spectral_brilliance = dt_bauhaus_slider_from_params(self, "spectral_brilliance");
  dt_bauhaus_slider_set_format(g->spectral_brilliance, "%");
  gtk_widget_set_tooltip_text(g->spectral_brilliance,
    _("Perceptual brightness: auto-exposure compensation with tone scale character. \n"
      "Higher values increase highlight headroom with a softer, film-like rolloff. \n"
      "Brightness is stabilised across the full range."));

  /* color_look est maintenant un enum dt_iop_st_look_t avec $DESCRIPTION sur chaque
   * valeur. dt_bauhaus_combobox_from_params lit l'introspection et peuple les 11
   * entrées automatiquement — aucun combobox_add manuel nécessaire. */
  g->color_look = dt_bauhaus_combobox_from_params(self, "color_look");
  gtk_widget_set_tooltip_text(g->color_look, _("Apply a color style to the image."));

  g->look_opacity = dt_bauhaus_slider_from_params(self, "look_opacity");
  dt_bauhaus_widget_set_label(g->look_opacity, NULL, _("look opacity"));
  dt_bauhaus_slider_set_format(g->look_opacity, "%");
  dt_bauhaus_slider_set_factor(g->look_opacity, 100.0);
  gtk_widget_set_tooltip_text(g->look_opacity, _("Adjust the strength of the selected color style."));

  // Advanced section
  dt_gui_new_collapsible_section(&g->advanced_section,
                                 "plugins/darkroom/spectral_tone/expand_advanced",
                                 _("advanced"),
                                 GTK_BOX(main_vbox),
                                 DT_ACTION(self));

  self->widget = GTK_WIDGET(g->advanced_section.container);

  g->hl_hue_shift = dt_bauhaus_slider_from_params(self, "hl_hue_shift");
  dt_bauhaus_slider_set_factor(g->hl_hue_shift, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_hue_shift, " %");
  dt_bauhaus_slider_set_digits(g->hl_hue_shift, 0);
  gtk_widget_set_tooltip_text(g->hl_hue_shift,
    _("Abney rotation in highlights, modulated by pixel saturation. \n"
      "Positive rotates toward cool (blue), negative toward warm (salmon). \n"
      "Vibrance-negative desaturation: saturated pixels desaturate further. \n"
      "Independent of highlight roll-off."));

  g->hl_desaturation = dt_bauhaus_slider_from_params(self, "hl_desaturation");
  dt_bauhaus_slider_set_factor(g->hl_desaturation, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_desaturation, " %");
  dt_bauhaus_slider_set_digits(g->hl_desaturation, 0);
  gtk_widget_set_tooltip_text(g->hl_desaturation,
    _("Highlight roll-off combining luminance desaturation and vibrance-negative. \n"
      "0% = off, 15% = natural rolloff, 100% = maximum desaturation."));

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
    _("Output color space used for luma coefficients in vibrance computation."));

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
  dt_iop_spectral_tone_gui_data_t *g = self->gui_data;
  dt_iop_spectral_tone_params_t *p = self->params;

  if(!w || w == g->color_look)
  {
    gtk_widget_set_visible(g->look_opacity, p->color_look > DT_ST_LOOK_NEUTRAL);
  }
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
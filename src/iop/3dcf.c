/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers
    Libre DT-lab Edition (C) 2026, Designed and developed by Christian Bouhon,
    with the assistance of artificial intelligence tools for code optimization 
    and OpenCL porting.

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

    - XYZ sigmoid curve (GIMP 3 Python plug-in) : discuss.pixls.us (2025).
      Inspiration for the X/Z chroma contrast feature — applying independent
      sigmoid contrast curves on the CIE X and Z chromaticity axes (with the
      X/Z balance slider and the default X↔Z link) to steer the asymmetry of
      the chromatic response.
      https://discuss.pixls.us/t/python-plug-in-for-gimp3-xyz-sigmoid-curve/60096

    ---------------------------------------------------------------------------

    */

#include "bauhaus/bauhaus.h"
#include "common/colorspaces.h"
#include "common/guided_filter.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/imagebuf.h"
#include "common/math.h"
#include "common/matrices.h"
#include "dtgtk/paint.h"
#include "develop/imageop.h"
#include "develop/tiling.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/color_picker_proxy.h"
#include "gui/draw.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

DT_MODULE_INTROSPECTION(10, dt_iop_3dcf_params_t)

/* Type definitions for the ACES 2.0 SSTS pipeline context.
   spectral_tone_data.c and spectral_tone_pipeline.c have been merged into
   3dcf.c into this single translation unit for Windows/MinGW
   compatibility. */

typedef enum dt_iop_st_colorspace_t
{
  DT_ST_CS_REC709 = 0,    // $DESCRIPTION: "sRGB"
  DT_ST_CS_REC2020,       // $DESCRIPTION: "Rec. 2020"
  DT_ST_CS_DISPLAYP3,     // $DESCRIPTION: "Display P3"
  DT_ST_CS_PROPHOTO,      // $DESCRIPTION: "ProPhoto RGB"
  DT_ST_CS_ADOBERGB,      // $DESCRIPTION: "Adobe RGB"
} dt_iop_st_colorspace_t;

/* MUST be a proper enum, NOT int. dt_bauhaus_combobox_from_params uses
   introspection: for a plain int it creates an empty combobox and
   immediately tries to set the default index on it. Setting index 0 on an
   empty combobox is a silent no-op on Linux/GTK but an assertion failure or
   out-of-bounds access on Windows/GTK -- crash at module mount, before any
   image is loaded. */
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

typedef enum dt_iop_st_auto_mode_t
{
  DT_ST_AUTO_CONTRASTY = 0,
  DT_ST_AUTO_NEUTRAL,
  DT_ST_AUTO_SOFT,
} dt_iop_st_auto_mode_t;

typedef struct dt_iop_3dcf_params_t
{
  float contrast;              // $MIN: 0.25 $MAX: 4.25 $DEFAULT: 2.25 $DESCRIPTION: "contrast"
  float gamma;                 // $MIN: -1 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "gamma"
  float vibrance;              // $MIN: 0 $MAX: 2 $DEFAULT: 1.0 $DESCRIPTION: "vibrance"
  float chromatic_boost;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $STEP: 0.01 $DESCRIPTION: "chromatic boost"
  float peak_luminance;     // $MIN: -100 $MAX: 100 $DEFAULT: 0 $DESCRIPTION: "peak luminance"
  float input_exposure;      // $MIN: -2 $MAX: 2 $DEFAULT: 0 $STEP: 0.05 $DESCRIPTION: "input exposure"
  float hl_hue_shift;        // $MIN: -1 $MAX: 1 $DEFAULT: 0 $STEP: 0.01 $DESCRIPTION: "Abney rotation"
  float hl_desaturation;     // $MIN: 0 $MAX: 1 $DEFAULT: 0.25 $DESCRIPTION: "highlight roll-off"
  float hl_desat_threshold;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.50 $DESCRIPTION: "desaturation threshold"
  float gamut_knee;          // $MIN: 0 $MAX: 1 $DEFAULT: 0.20 $DESCRIPTION: "gamut knee"
  float gamut_steepness;     // $MIN: 0 $MAX: 1 $DEFAULT: 0.50 $DESCRIPTION: "gamut steepness"
  dt_iop_st_colorspace_t output_cs;  // $DEFAULT: DT_ST_CS_REC2020 $DESCRIPTION: "display target"
  dt_iop_st_look_t color_look;       // $DEFAULT: DT_ST_LOOK_NEUTRAL $DESCRIPTION: "color look"
  float look_opacity;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "look opacity"
  float contrast_pivot;        // $MIN: 0.01 $MAX: 0.99 $DEFAULT: 0.5 $DESCRIPTION: "contrast pivot"
  float toe_power;             // $MIN: 0.25 $MAX: 3.0 $DEFAULT: 1.0 $DESCRIPTION: "toe power"
  float shoulder_power;        // $MIN: 0.25 $MAX: 3.0 $DEFAULT: 1.0 $DESCRIPTION: "shoulder power"
  float hl_detail_recovery;    // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.20 $DESCRIPTION: "detail recovery"

  /* Chroma contrast — independent sigmoid on the CIE-xz offset from white,
     inserted between chromaticity scaling (step 5) and spectral gamut
     roll-off (step 6). Operates on the same normalized chroma axes used by
     st_spectral_gamut(), not on raw XYZ. Default 0.0 = no-op. */
  float chroma_x_contrast;     // $MIN: 0.0 $MAX: 10.0 $DEFAULT: 0.0 $STEP: 0.01 $DESCRIPTION: "X-axis chroma contrast"
  float chroma_x_pivot;        // $MIN: 25.0 $MAX: 75.0 $DEFAULT: 50.0 $STEP: 0.1 $DESCRIPTION: "X-axis contrast pivot"
  float chroma_z_contrast;     // $MIN: 0.0 $MAX: 10.0 $DEFAULT: 0.0 $STEP: 0.01 $DESCRIPTION: "Z-axis chroma contrast"
  float chroma_z_pivot;        // $MIN: 25.0 $MAX: 75.0 $DEFAULT: 50.0 $STEP: 0.1 $DESCRIPTION: "Z-axis contrast pivot"
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
  float gamma;
  float gamma_power;
  float vibrance;
  float chromatic_boost;
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

  /* Chroma contrast sigmoid — precomputed at commit time so the per-pixel
     kernel only does one exp() per axis instead of three. */
  float chroma_x_gain;         // = max(chroma_x_contrast, 1e-4)
  float chroma_x_shift;        // = chroma_x_pivot / 100
  float chroma_x_sig0;         // = sigmoid(0, gain, shift)
  float chroma_x_inv_range;    // = 1 / (sigmoid(1, gain, shift) - sig0)
  float chroma_z_gain;
  float chroma_z_shift;
  float chroma_z_sig0;
  float chroma_z_inv_range;
} dt_st_context_t;

typedef struct dt_iop_3dcf_data_t
{
  dt_iop_3dcf_params_t params;
  dt_st_context_t ctx;
} dt_iop_3dcf_data_t;

/* GPU-side context -- MUST match dt_st_cl_params_t in 3dcf.cl byte for
   byte. Only 4-byte members (float / int) are used so host and device
   layouts are identical (no padding); the struct is passed to the kernel by
   value. The SSTS parameters are double in dt_st_context_t but narrowed to
   float here. */
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
  float spectral_boundary[360];
  float chroma_x_gain;
  float chroma_x_shift;
  float chroma_x_sig0;
  float chroma_x_inv_range;
  float chroma_z_gain;
  float chroma_z_shift;
  float chroma_z_sig0;
  float chroma_z_inv_range;
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
  GtkWidget *peak_luminance;
  GtkWidget *input_exposure;
  GtkWidget *mid_tone;
  GtkWidget *vibrance;
  GtkWidget *chromatic_boost;
  GtkWidget *hl_desaturation;
  GtkWidget *hl_desat_threshold;
  GtkWidget *hl_hue_shift;
  GtkWidget *chroma_x_contrast;
  GtkWidget *chroma_x_pivot;
  GtkWidget *chroma_z_contrast;
  GtkWidget *chroma_z_pivot;
  /* Transient slider (not a module parameter) shown only while the X and Z
     axes are locked together; pans the shared contrast between the two axes. */
  GtkWidget *chroma_balance;
  GtkWidget *hl_detail_recovery;
  GtkWidget *gamut_knee;
  GtkWidget *gamut_steepness;
  dt_gui_collapsible_section_t advanced_section;
  GtkWidget *color_space;
  GtkWidget *color_look;
  GtkWidget *look_opacity;
  gboolean chroma_linked;
  gboolean chroma_syncing;
  GtkDrawingArea *graph;
  GtkAllocation allocation;
  PangoRectangle ink;
  GtkStyleContext *context;
  /* GUI-only combobox, not a module parameter. Also carries the live
     grey-point picker via its quad icon (see color_picker_apply below) and
     selects the contrast rendering that picker uses. Changing the selection
     alone triggers nothing. */
  GtkWidget *auto_mode_combo;
  /* Set by color_picker_apply() on each live sample, consumed in process().
     Written from the GUI thread, read from the pixelpipe thread; not
     otherwise synchronized. */
  gboolean auto_apply_requested;
  /* Preset read from auto_mode_combo when the flag above is set. */
  dt_iop_st_auto_mode_t auto_requested_mode;
  /* Picked grey-point luminance (scene-referred Y, module working space).
     0 = no pick, the estimator falls back to the scene average.
     Deliberately not a module parameter: only the resulting sliders
     (input_exposure, contrast...) are recorded in history, so editing stays
     reproducible; only the picked point itself is lost on reopen. */
  float picked_grey_y;
  /* Last picked_grey_y value actually sent to auto_adjust_3dcf_params().
     Performance guard: skips a full-image scan when the picked value hasn't
     changed meaningfully since the last computation (see
     color_picker_apply()). */
  float picked_grey_y_last_applied;
  /* Last image id seen by gui_update(). gui_update() fires far more often
     than on image change alone, so picked_grey_y is reset here only when
     the displayed image actually changed. 0 (NO_IMGID) on the first call is
     distinct from any real id, so the initial reset happens naturally. */
  dt_imgid_t last_imgid;
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

/* Rec.2020 RGB -> CIE XYZ (D50) — from primaries + D50 white point */
const double dt_st_rec2020_d50_to_xyz[9] = {
  6.9458363963246861e-01, 1.4268890101747944e-01, 1.2693945377125121e-01,
  2.8646669883147013e-01, 6.6895914182900651e-01, 4.4574159339523321e-02,
  0.0,                    2.7698433726922409e-02, 7.9748985079190637e-01
};

/* Chromatic adaptation matrix CIE CAT16: D65 -> D50 */
const double dt_st_cat_d65_to_d50[9] = {
  1.047839954303051e+00,  2.289791610380174e-02, -5.018079725046408e-02,
  2.955368681442254e-02,  9.904924221623178e-01, -1.706631418019539e-02,
  -9.245918452778928e-03,  1.506326034916465e-02,  7.518388616796452e-01
};

/* Pre-computed combined matrices for output gamut protection:
     gamut_fwd[cs]  : Rec.2020 D65 RGB -> Target D65 RGB
     gamut_inv[cs]  : Target D65 RGB -> Rec.2020 D65 RGB
   Indexed by dt_iop_st_colorspace_t. Slot REC2020 is identity (unused).
   Round-trip verified: fwd @ inv = I to within ~3e-16. */
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
   ACES 2.0 SSTS (Single-Stage Tone Scale) and spectral pipeline functions
   ======================================================================== */

/* ACES 2.0 SSTS (Single-Stage Tone Scale) -- official ACES 2.0 RRT tone
   scale.

   Parametric MM (Michaelis-Menten) curve with flare compensation.
   Reference: aces-core/lib/Lib.Academy.Tonescale.ctl

     f = m_2 * (x / (x + s_2))^g
     h = f^2 / (f + t_1)
     Y = h * n_r          (output: display luminance in cd/m^2)

   The SSTS defines the "texture of light" -- the character of the tone
   reproduction from scene-linear to display. Its shape is determined by
   the peak luminance of the target display. */
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
             / (float)ctx->ssts.n_r;

  y_tm = powf(fmaxf(y_tm, 0.0f), 1.0f / 2.4f);

  /* Clamp to [0, 1] before contrast curve: the shoulder formula assumes
     y_tm in [0,1] and produces NaN/Inf when y_tm > 1 (powf(0, negative)). */
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

/* ========================================================================
   Auto-adjustment: derive tone/contrast params from scene content
   ========================================================================

   Note on the peak_luminance slider's direction: dt_st_compute_y_tm()
   normalizes by ctx->ssts.n_r, a CONSTANT (100.0, set in dt_st_ssts_init),
   instead of ctx->ssts.n (the actual chosen peak). For a fixed scene value,
   y_tm grows with peak_luminance, so a higher peak_luminance clips EARLIER,
   not later -- the tooltip and the bisection below are both written to
   match this actual, verified behavior. If dt_st_compute_y_tm() is ever
   changed to normalize by ctx->ssts.n instead, the bisection direction
   below must flip accordingly (see the comment at that spot).

   Computation domains:
   - Scene Y: linear luminance, Rec.2020 D50 weighting -- reuses the Y row
     of dt_st_rec2020_d50_to_xyz (not Rec.709, not st_luma_rec2020, which
     are the Rec.2020 D65 coefficients used for the OUTPUT space, a
     different space).
   - peak_luminance (param): percentage [-100,100], 0% = 200 nits, converted
     to nits via peak_nits = 200*(1+p/100), see st_compute_context(). Solved
     by bisection to bring the scene's high percentile just under clipping
     in the actual y_tm domain, rather than assigning a linear scene value
     to a field that represents a nits target.
   - contrast_pivot (param): the pipeline uses it inverted
     (ctx->contrast_pivot = 1 - p->contrast_pivot), in the y_tm domain (post
     SSTS + 1/2.4 gamma), not the scene domain. So this computes where scene
     middle grey (0.18) lands after SSTS+gamma with the peak_luminance
     determined above, then inverts it.

   Runs in ~220ms on a 24MP image, single-threaded. The contrast-pass
   constants (target_std_dev, bounds, toe/shoulder thresholds) are
   reasonable starting points, not calibrated against real images with this
   module. */

#define ST_AUTO_HIST_BINS      2048
#define ST_AUTO_HIST_EV_MIN    (-20.0)
#define ST_AUTO_HIST_EV_MAX    (16.0)
#define ST_AUTO_HL_PERCENTILE  0.99   /* high percentile rather than raw max: robust to hot/specular pixels */
#define ST_AUTO_HEADROOM       0.92f  /* target margin under clipping in the y_tm domain (1.0 = hard clip) */

/* --- Contrast rendering presets for the auto-adjust picker ----------------

   These presets modulate ONLY the contrast rendering. The scene
   measurements (exposure, peak_luminance, contrast_pivot) are identical in
   all three cases -- they describe the scene, not a look, and have no
   reason to change with the chosen rendering.

   DT_ST_AUTO_CONTRASTY: sustained master contrast. Shadows harden further
   as they spread out (backlit / high-dynamic-range scenes), highlights
   soften further as they spread out. Suited to landscapes and
   wide-dynamic-range scenes.

   DT_ST_AUTO_SOFT: never hardens on either side; both toe and shoulder only
   soften as their side spreads out, capped at 1.0.

   DT_ST_AUTO_NEUTRAL: interpolates between the two extremes. Not validated
   against real images -- a constructed midpoint, not a measured setting.

   toe_base: the toe/shoulder value for a lightly spread side (>1.0 =
   hardens). toe_amplitude/shoulder_amplitude: how much that side moves away
   from toe_base as it spreads out; sign decides the direction (positive =
   softens as it spreads, negative = hardens as it spreads). Shadows on a
   backlit scene benefit from extra punch as they get more spread out, hence
   toe_amplitude is negative for contrasty/neutral; highlights instead need
   a gentler rolloff as they spread out (the SSTS has already compressed
   them heavily), hence shoulder_amplitude stays positive for all three
   presets. soft's toe_amplitude stays positive too, since it must never
   harden either side. */
typedef struct dt_st_auto_preset_t
{
  float target_sd;       /* scene spread (in stops) that receives the module's default contrast */
  float gain;            /* factor applied to the computed contrast; 1.0 = the "contrasty" reference */
  float contrast_min;    /* bounds applied to the master contrast */
  float contrast_max;
  float toe_base;             /* toe/shoulder value for a lightly spread side (>1.0 = hardens) */
  float toe_amplitude;        /* shadow-side response to spread; see header comment for sign */
  float shoulder_amplitude;   /* highlight-side response to spread; always softens (positive) */
} dt_st_auto_preset_t;

/* The three presets share the same response shape (linear ratio, same
   target_sd) and differ only by a multiplicative gain and their bounds.
   This is deliberate: a multiplicative gain guarantees soft < neutral <
   contrasty at every scene spread, which varying the target and a damping
   exponent per preset does not. */
static const dt_st_auto_preset_t st_auto_presets[3] = {
  /* CONTRASTY -- gain 1.0, the reference rendering. toe_amplitude negative:
     shadows harden (up to 1.55) as they spread out. */
  { 1.8f, 1.00f, 0.6f, 3.5f, 1.15f, -0.40f, 0.40f },
  /* NEUTRAL -- interpolated between the two extremes. Same shadow-hardening
     direction as contrasty, smaller magnitude (up to 1.43). */
  { 1.8f, 0.88f, 0.7f, 3.1f, 1.08f, -0.35f, 0.35f },
  /* SOFT -- never hardens on either side (toe_amplitude stays positive). */
  { 1.8f, 0.78f, 0.8f, 2.8f, 1.00f, -0.30f, 0.30f },
};

/* Reproduces exactly the "tone mapping" part of dt_st_compute_y_tm(),
   without the contrast S-curve, to evaluate where a given scene value lands
   in the y_tm domain for a given peak_luminance (in nits). */
static float st_auto_eval_y_tm(float y_scene, double peak_nits)
{
  dt_st_ssts_params_t ssts;
  dt_st_ssts_init(&ssts, peak_nits);
  float y_tm = dt_st_ssts_fwd(&ssts, y_scene) / (float)ssts.n_r;
  y_tm = powf(fmaxf(y_tm, 0.0f), 1.0f / 2.4f);
  return fminf(y_tm, 1.0f);
}

/* Matches st_compute_context(): hp = peak_luminance/100; peak = 200*(1+hp). */
static inline double st_auto_percent_to_nits(float percent) { return 200.0 * (1.0 + (double)percent / 100.0); }
static inline float  st_auto_nits_to_percent(double nits)   { return (float)((nits / 200.0 - 1.0) * 100.0); }

/**
   Computes a set of tone/contrast settings from scene content and writes
   them into params.

   @param in      Linear RGB buffer, module working space (Rec.2020 D50) --
                   the buffer BEFORE 3DCF processing, i.e. exactly what
                   process() receives in ivoid.
   @param width   roi_in->width
   @param height  roi_in->height
   @param ch      piece->colors (3 or 4; only R,G,B are read)
   @param params  Module parameters to update
   @param mode    Contrast rendering preset (see st_auto_presets[]). Affects
                   ONLY contrast and toe/shoulder; exposure, peak_luminance
                   and contrast_pivot are identical across all three modes.
   @param picked_grey_y  Picked mid-grey luminance Y (scene, module working
                   space). > 0: used as a direct exposure anchor (no bias),
                   instead of the trimmed geometric mean. <= 0: trimmed
                   geometric mean (automatic behavior). */
static void auto_adjust_3dcf_params(const float *in, int width, int height, int ch,
                                     dt_iop_3dcf_params_t *params,
                                     dt_iop_st_auto_mode_t mode,
                                     float picked_grey_y)
{
  const size_t total_pixels = (size_t)width * (size_t)height;
  if(!in || total_pixels == 0 || !params || ch < 3) return;

  /* Defensive bound: an out-of-range preset would index past the array. */
  const int mode_idx = (mode >= DT_ST_AUTO_CONTRASTY && mode <= DT_ST_AUTO_SOFT)
                         ? (int)mode : (int)DT_ST_AUTO_NEUTRAL;
  const dt_st_auto_preset_t *preset = &st_auto_presets[mode_idx];

  /* Y row of dt_st_rec2020_d50_to_xyz (Rec.2020 D50 RGB -> XYZ D50): the
     actual luminance coefficients for the module's working space, not the
     Rec.2020 D65 coefficients (st_luma_rec2020) used elsewhere for the
     output space. */
  const double luma_r = dt_st_rec2020_d50_to_xyz[3];
  const double luma_g = dt_st_rec2020_d50_to_xyz[4];
  const double luma_b = dt_st_rec2020_d50_to_xyz[5];

  const size_t step = (total_pixels > 1000000) ? 4 : 1;

  /* -----------------------------------------------------------------
     PASS 1: EV histogram, used both for the trimmed geometric mean
     (exposure) and the high percentile (peak_luminance).
     ----------------------------------------------------------------- */
  size_t sample_count = 0;
  uint32_t hist[ST_AUTO_HIST_BINS] = { 0 };
  const double hist_scale = ST_AUTO_HIST_BINS / (ST_AUTO_HIST_EV_MAX - ST_AUTO_HIST_EV_MIN);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    reduction(+:sample_count) reduction(+:hist[:ST_AUTO_HIST_BINS])
#endif
  for(size_t i = 0; i < total_pixels; i += step)
  {
    const float r = in[ch * i + 0];
    const float g = in[ch * i + 1];
    const float b = in[ch * i + 2];
    const float y = (float)(luma_r * r + luma_g * g + luma_b * b);

    if(y > 1e-6f)
    {
      sample_count++;

      const double ev = log2((double)y);
      long bin = (long)((ev - ST_AUTO_HIST_EV_MIN) * hist_scale);
      if(bin < 0) bin = 0;
      if(bin >= ST_AUTO_HIST_BINS) bin = ST_AUTO_HIST_BINS - 1;
      hist[bin]++;
    }
  }

  if(sample_count == 0) return;

  const float target_middle_grey = 0.18f;

  /* -----------------------------------------------------------------
     1.1 Exposure: trimmed geometric mean of the scene, in EV, rather than
         the median. The median ignores pixel VALUE (only rank matters): a
         large flat area near the 50th percentile (sky, wall, blurred
         background...) can pull it far from the actual subject with no
         indication. The trimmed geometric mean instead weights each pixel
         in the [15%, 95%] rank range by its actual EV value -- the 15%
         darkest (noise, background, vignetting) and 5% brightest
         (speculars) are excluded, but the rest of the frame contributes
         by its real weight, not just its rank.

     1.2 High percentile (peak_luminance): from the same histogram (hi_bin
         = first bin where the cumulative count reaches 99%).
     ----------------------------------------------------------------- */
  const size_t low_trim_count  = (size_t)((double)sample_count * 0.15);
  const size_t high_trim_count = (size_t)((double)sample_count * 0.95);
  const size_t hl_count = (size_t)((double)sample_count * ST_AUTO_HL_PERCENTILE);

  double sum_ev_trimmed = 0.0;
  size_t count_trimmed = 0;
  int hi_bin = ST_AUTO_HIST_BINS - 1;
  gboolean hi_bin_found = FALSE;
  {
    size_t cumulative = 0;
    for(int b = 0; b < ST_AUTO_HIST_BINS; b++)
    {
      const uint32_t bin_count = hist[b];
      if(bin_count == 0) continue;

      const size_t cum_start = cumulative;
      cumulative += bin_count;
      const size_t cum_end = cumulative;

      /* This bin's contribution to the trimmed mean: only the part of
         bin_count that falls in [low_trim_count, high_trim_count). */
      if(cum_end > low_trim_count && cum_start < high_trim_count)
      {
        size_t valid_in_bin = bin_count;
        if(cum_start < low_trim_count) valid_in_bin -= (low_trim_count - cum_start);
        if(cum_end > high_trim_count)  valid_in_bin -= (cum_end - high_trim_count);

        const double ev_center = ST_AUTO_HIST_EV_MIN + ((double)b + 0.5) / hist_scale;
        sum_ev_trimmed += (double)valid_in_bin * ev_center;
        count_trimmed += valid_in_bin;
      }

      if(!hi_bin_found && cum_end >= hl_count) { hi_bin = b; hi_bin_found = TRUE; }
    }
  }

  /* Exposure anchor: the picked grey point if there is one -- used
     directly, with no correction bias, since the user designated THE point
     they want at 18% (semantic information, not statistical) -- otherwise
     the trimmed geometric mean above. */
  const gboolean use_picked = (picked_grey_y > 1e-6f);
  float scene_grey_y;
  if(use_picked)
    scene_grey_y = picked_grey_y;
  else if(count_trimmed > 0)
    scene_grey_y = exp2f((float)(sum_ev_trimmed / (double)count_trimmed));
  else
    scene_grey_y = target_middle_grey; /* safety fallback: degenerate scene */

  const float ev_shift = log2f(target_middle_grey / scene_grey_y);
  params->input_exposure = fmaxf(fminf(ev_shift, 2.0f), -2.0f); /* slider bound */

  /* High percentile of the scene (before exposure), re-expressed in linear
     AFTER the exposure shift just set. */
  {
    const double ev_high_scene = ST_AUTO_HIST_EV_MIN + ((double)hi_bin + 0.5) / hist_scale;
    const float y_high_shifted = exp2f((float)ev_high_scene + ev_shift);

    /* Bisection on peak_luminance (in nits). See the section header comment:
       y_tm GROWS with peak_nits for a fixed scene value, so reducing
       clipping means LOWERING peak_nits. If dt_st_compute_y_tm() is ever
       changed to normalize by ctx->ssts.n instead of ctx->ssts.n_r, flip
       the two branches below (and the y_tm_at_hi/y_tm_at_lo test). */
    double nits_lo = 1.0, nits_hi = 400.0; /* upper bound = peak_luminance param at +100% */
    const float y_tm_at_lo = st_auto_eval_y_tm(y_high_shifted, nits_lo);
    const float y_tm_at_hi = st_auto_eval_y_tm(y_high_shifted, nits_hi);

    double peak_nits;
    if(y_tm_at_hi <= ST_AUTO_HEADROOM)
    {
      /* No clipping risk even at the max allowed nits: leave the default
         setting (0% = 200 nits) rather than saturate the slider. */
      peak_nits = 200.0;
    }
    else if(y_tm_at_lo >= ST_AUTO_HEADROOM)
    {
      /* Already clipping even at the minimum nits (extreme highlight): the
         best value reachable within the slider's range. */
      peak_nits = nits_lo;
    }
    else
    {
      for(int it = 0; it < 24; it++)
      {
        const double mid = 0.5 * (nits_lo + nits_hi);
        const float y_tm = st_auto_eval_y_tm(y_high_shifted, mid);
        if(y_tm > ST_AUTO_HEADROOM) nits_hi = mid; else nits_lo = mid;
      }
      peak_nits = 0.5 * (nits_lo + nits_hi);
    }
    params->peak_luminance = fmaxf(fminf(st_auto_nits_to_percent(peak_nits), 100.0f), -100.0f);
  }

  /* 1.3 Contrast pivot: where the middle grey point, exactly as exposed
     (target_middle_grey -- ev_shift lands it exactly on 0.18 by
     construction, see above), lands in the y_tm domain with the
     peak_luminance determined above. The pipeline uses this parameter
     inverted (ctx->contrast_pivot = 1 - p->contrast_pivot). Anchoring the
     pivot on anything other than the actually-exposed point misaligns the
     S-curve from the subject. */
  {
    const double peak_nits = st_auto_percent_to_nits(params->peak_luminance);
    const float y_tm_grey = st_auto_eval_y_tm(target_middle_grey, peak_nits);
    params->contrast_pivot = fmaxf(fminf(1.0f - y_tm_grey, 0.99f), 0.01f);
  }

  /* -----------------------------------------------------------------
     PASS 2: standard deviation in EV (log2) of the re-exposed luminance,
     for contrast. The constants below are not calibrated against real
     images with this module.
     ----------------------------------------------------------------- */
  double sum_sq_diff = 0.0;
  size_t var_count = 0;
  double sum_sq_shadow = 0.0, sum_sq_highlight = 0.0;
  size_t shadow_count = 0, highlight_count = 0;
  const float log_pivot = log2f(target_middle_grey);
  const float exposure_factor = exp2f(ev_shift);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    reduction(+:sum_sq_diff, var_count, sum_sq_shadow, sum_sq_highlight, shadow_count, highlight_count)
#endif
  for(size_t i = 0; i < total_pixels; i += step)
  {
    const float r = in[ch * i + 0];
    const float g = in[ch * i + 1];
    const float b = in[ch * i + 2];
    const float y = (float)(luma_r * r + luma_g * g + luma_b * b) * exposure_factor;

    if(y > 1e-3f)
    {
      const float diff = log2f(y) - log_pivot;
      sum_sq_diff += (double)diff * (double)diff;
      var_count++;

      if(diff < 0.0f) { sum_sq_shadow    += (double)diff * (double)diff; shadow_count++;    }
      else             { sum_sq_highlight += (double)diff * (double)diff; highlight_count++; }
    }
  }

  const float std_dev = (var_count > 0) ? sqrtf((float)(sum_sq_diff / (double)var_count)) : 1.5f;
  /* std_dev computed separately for each side of the pivot. Falls back to
     the preset's target if one side is empty -- should only happen on a
     degenerate scene (uniformly above or below the exposed grey point). */
  const float std_dev_shadow    = (shadow_count > 0)    ? sqrtf((float)(sum_sq_shadow    / (double)shadow_count))    : preset->target_sd;
  const float std_dev_highlight = (highlight_count > 0) ? sqrtf((float)(sum_sq_highlight / (double)highlight_count)) : preset->target_sd;

  /* 2.1 Master contrast. Parameter range: [0.25,4.25], default 2.25. The
     constants come from the chosen preset (see st_auto_presets[]):
     target_sd sets which scene spread receives the module's default
     contrast, gain scales the whole thing toward soft (< 1.0) without
     distorting the curve shape. */
  const float contrast_factor =
    preset->gain * 2.25f * (preset->target_sd / fmaxf(std_dev, 0.4f));
  params->contrast = fminf(fmaxf(contrast_factor, preset->contrast_min), preset->contrast_max);

  /* 2.2 Toe/Shoulder -- asymmetric, computed separately from the spread
     (std_dev) of EACH side of the pivot, not a single shared std_dev.

     Meaning: 1.0 = master contrast applied as-is in that zone; below =
     softened; above = hardened on top of the master contrast. The two
     sides move in OPPOSITE directions as their spread grows: shadows
     harden further on contrasty/neutral (useful on a backlit /
     wide-dynamic-range scene, where master contrast is low and shadow
     detail needs extra punch), while highlights always soften further
     (the SSTS has already compressed them heavily, so a harder highlight
     rolloff there would clip). soft never hardens on either side. The
     sign is baked into toe_amplitude/shoulder_amplitude per preset -- see
     st_auto_presets[]. */
  const float t_shadow    = fminf(fmaxf((std_dev_shadow    - 1.2f) / (2.5f - 1.2f), 0.0f), 1.0f);
  const float t_highlight = fminf(fmaxf((std_dev_highlight - 1.2f) / (2.5f - 1.2f), 0.0f), 1.0f);
  params->toe_power      = preset->toe_base - preset->toe_amplitude * t_shadow;
  params->shoulder_power = preset->toe_base - preset->shoulder_amplitude * t_highlight;
}

/* ------------------------------------------------------------------------
   GUI <-> process() wiring for the live auto-adjust picker. Placed here
   (before process()) because process() consumes _st_auto_result_t and
   _auto_apply_to_gui -- the declaration must precede use in C.
   ------------------------------------------------------------------------ */

/* Single entry point for an auto-adjust request, called from
   color_picker_apply() on each live picker sample (see below). Computes
   nothing itself: runs on the GUI thread and has no access to the
   pixelpipe's input buffer (ivoid only exists in process()). Sets a flag
   plus the requested preset, and triggers a preview recompute; process()
   reads both, computes on ivoid, and hands back to the GUI thread via
   g_idle_add(). */
static void _auto_request(dt_iop_module_t *self, dt_iop_st_auto_mode_t mode)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  if(!g) return;

  g->auto_requested_mode = mode;
  g->auto_apply_requested = TRUE;
  dt_dev_reprocess_preview(self->dev);
}

/* Changing the combobox selection doesn't move the picker's sampled area:
   the framework (_record_point_area() in gui/color_picker_proxy.c) only
   calls color_picker_apply() again when the sampled position/area changes,
   so the new mode would never be read while the picker sits still -- hence
   this dedicated handler. Triggers nothing if this module's picker isn't
   currently armed (the combobox alone does nothing); otherwise re-runs
   immediately with the last picked point (or 0 if none, in which case the
   auto-adjust falls back to the automatic estimate) and the new mode, so
   contrasty/neutral/soft can be compared on the same area without
   re-dragging the picker. */
static void _auto_mode_changed(GtkWidget *widget, dt_iop_module_t *self)
{
  (void)widget;
  if(self->request_color_pick == DT_REQUEST_COLORPICK_OFF) return;

  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  if(!g) return;

  const int combo_idx = dt_bauhaus_combobox_get(g->auto_mode_combo);
  const dt_iop_st_auto_mode_t mode =
    (combo_idx >= DT_ST_AUTO_CONTRASTY && combo_idx <= DT_ST_AUTO_SOFT)
    ? (dt_iop_st_auto_mode_t)combo_idx : DT_ST_AUTO_NEUTRAL;

  g->picked_grey_y_last_applied = g->picked_grey_y;
  _auto_request(self, mode);
}

/* Passed to g_idle_add() to hand the result computed in process() (pixelpipe
   thread) back to the GTK widgets (GUI thread). */
typedef struct _st_auto_result_t
{
  dt_iop_module_t *self;
  dt_iop_3dcf_params_t params;
} _st_auto_result_t;
static gboolean _auto_apply_to_gui(gpointer user_data)
{
  _st_auto_result_t *res = (_st_auto_result_t *)user_data;
  dt_iop_module_t *self = res->self;
  dt_iop_3dcf_gui_data_t *g = self->gui_data;

  if(g)
  {
    memcpy(self->params, &res->params, sizeof(dt_iop_3dcf_params_t));

    /* Targeted update, not dt_iop_gui_update(self): same pattern as
       _exposure_set_white()/_exposure_set_black() in iop/exposure.c, which
       only touch the widget(s) their picker concerns rather than
       resyncing the whole panel. A full dt_iop_gui_update() here would
       disarm the picker on every drag frame instead of it staying active
       for a continuous drag. DT_ENTER_GUI_UPDATE()/DT_LEAVE_GUI_UPDATE():
       standard darktable guard so dt_bauhaus_slider_set() doesn't cascade
       into gui_changed() during a programmatic update. */
    DT_ENTER_GUI_UPDATE();
    dt_bauhaus_slider_set(g->contrast, res->params.contrast);
    dt_bauhaus_slider_set(g->contrast_pivot, res->params.contrast_pivot);
    dt_bauhaus_slider_set(g->toe_power, res->params.toe_power);
    dt_bauhaus_slider_set(g->shoulder_power, res->params.shoulder_power);
    dt_bauhaus_slider_set(g->input_exposure, res->params.input_exposure);
    dt_bauhaus_slider_set(g->peak_luminance, res->params.peak_luminance);
    DT_LEAVE_GUI_UPDATE();

    dt_dev_add_history_item(darktable.develop, self, TRUE);

    gtk_widget_queue_draw(GTK_WIDGET(g->graph));
  }

  free(res);
  return G_SOURCE_REMOVE;
}

/* Desaturation weight for highlights.

   Desaturation activates above a scene luminance threshold (in SSTS-exposed
   space) and ramps up quadratically as luminance increases. hl_desat
   controls the strength: 0 = off, 1 = full desat well above threshold,
   > 1 = faster desaturation. */
static inline float st_desat_weight(float y_norm, float hl_desat, float threshold)
{
  if(hl_desat <= 0.0f || y_norm <= threshold) return 0.0f;
  if(!isfinite(y_norm)) return 0.0f;
  const float t = fmaxf(y_norm - threshold, 0.0f) / y_norm;
  const float x = fminf(t * hl_desat, 1.0f);
  return x * x;
}

/* Hue-preserving gamut compression: smoothly pull out-of-[0,1] channels back
   in range with a Reinhard-style soft knee, blending toward the PIXEL'S OWN
   luma rather than a fixed white point.

   Why this matters (ACES 2.0-inspired principle, applied without needing a
   JMh/CAM conversion): blending toward a fixed neutral like (1,1,1) is a
   compound of desaturation and a genuine brightening -- the more a pixel is
   pushed toward white, the more its perceived luma rises too, which is what
   makes strongly saturated highlights look like they "wash out" rather than
   gently losing saturation. Blending toward the pixel's own luma instead is
   a pure desaturation along the true hue-preserving axis: hue and luma stay
   put, only chroma is reduced. This mirrors what ACES 2.0's gamut_compress
   achieves via cusp/JMh geometry (compress chroma while leaving the
   lightness axis alone), just done directly in RGB.

   The response is also not linear in the excess: mild excursions (t << 1)
   are barely touched, and only pixels close to the true minimal-correction
   boundary (t -> 1) get pulled hard -- a soft knee instead of a uniform
   blend, which would make even mild out-of-gamut pixels lose a large,
   constant fraction of their saturation. */
static inline void st_gamut_compress(float rgb[3], const float luma_coeff[3])
{
  const float luma = luma_coeff[0] * rgb[0] + luma_coeff[1] * rgb[1] + luma_coeff[2] * rgb[2];
  /* Guard against a degenerate (near-black) luma: falls back to the fixed
     anchor (1,1,1) only in this edge case, where "hue-preserving" is moot. */
  const float anchor = (luma > 1e-4f) ? luma : 1.0f;

  /* Minimal blend-toward-anchor amount `t` needed to bring every channel
     back into [0,1]: the worst channel reaches exactly its boundary at
     t=1, measured against `anchor` instead of a fixed 1.0/0.0. */
  float t = 0.0f;
  for(int c = 0; c < 3; c++)
  {
    if(rgb[c] < 0.0f)
    {
      const float denom = anchor - rgb[c];
      const float ti = (denom > 1e-6f) ? (-rgb[c]) / denom : 1.0f;
      if(ti > t) t = ti;
    }
    else if(rgb[c] > 1.0f)
    {
      const float denom = rgb[c] - anchor;
      const float ti = (denom > 1e-6f) ? (rgb[c] - 1.0f) / denom : 1.0f;
      if(ti > t) t = ti;
    }
  }
  if(t <= 0.0f) return;
  const float blend = fminf(t, 1.0f);

  for(int c = 0; c < 3; c++)
    rgb[c] = (1.0f - blend) * rgb[c] + blend * anchor;
}

/* Output gamut protection: convert from Rec.2020 D65 to the target color
   space, hard-clamp negative (out-of-gamut) channels to zero, then convert
   back. Called after st_gamut_compress() as an additional safety net when
   the user selects a colour space narrower than Rec. 2020. */
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

/* Spectral locus CIE 1931 2-degree xy at 5 nm (380-700 nm) + endpoint at
   780 nm. From 700 nm onward the coordinates plateau at (0.734690,
   0.265310). Used to build the angle-to-max-distance lookup table for
   automatic non-spectral colour detection. Purple-line interpolation is
   added at precompute time. */
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

/* Build 360-bin lookup table: for each integer degree angle (0-359) from the
   D50 white point, store the maximum CIE xz distance of the spectral locus
   (including the purple line). The table is used by st_spectral_gamut() to
   detect and smoothly roll off non-spectral chromaticities. */
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

  /* Purple line: interpolate N segments between endpoints 780 nm and 380 nm
     in CIE XYZ space (not xy), so the interpolation follows the physical
     mixture of two monochromatic lights rather than a straight line in xy
     chromaticity space. The two spectral endpoints converted with Y=1 give
     the correct tristimulus for linear mixing. */
  {
    const float x_r = st_spectral_locus_xy[SPECTRAL_LOCUS_N - 1][0];  /* 780 nm */
    const float y_r = st_spectral_locus_xy[SPECTRAL_LOCUS_N - 1][1];
    const float x_b = st_spectral_locus_xy[0][0];  /* 380 nm */
    const float y_b = st_spectral_locus_xy[0][1];

    const float inv_y_r = 1.0f / y_r;
    const float inv_y_b = 1.0f / y_b;
    const float X_r = x_r * inv_y_r;
    const float Z_r = (1.0f - x_r - y_r) * inv_y_r;
    const float X_b = x_b * inv_y_b;
    const float Z_b = (1.0f - x_b - y_b) * inv_y_b;

    const int NSEG = 32;
    for(int i = 1; i <= NSEG; i++)
    {
      const float t = (float)i / (float)(NSEG + 1);
      const float X = X_r + t * (X_b - X_r);
      const float Z = Z_r + t * (Z_b - Z_r);
      const float sum_xyz = X + 1.0f + Z;
      const float x = X / sum_xyz;
      const float y = 1.0f / sum_xyz;
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

/* ====================================================================
   ACES 2.0-derived per-hue chroma shape (RC1-style static tables)

   Ported from the verified aces20.c implementation:
   - st_gamut_reach[360]: AP1-limited, AP0-reach M table at a fixed 100 nits
     reference, generated offline from the corrected cusp/reach
     table-generation algorithm (AP1 limiting primaries; any_below_zero()
     gamut test only, no upper-bound check).
   - st_chroma_norm(): the exact trigonometric polynomial approximation of
     the AP1 gamut cusp M by hue, verified term-for-term against the
     official aces-core Lib.Academy.OutputTransform.ctl.

   Both are expressed in CAM16/JMh hue space in their origin. 3DCF has no
   CAM/JMh conversion, so here they are deliberately re-purposed: the hue
   fed in is the CIE xy chromaticity angle already computed in
   st_spectral_gamut() (atan2 of the (dz,dx) offset from white), not a true
   CAM16 hue. This means st_reach_from_table()/st_chroma_norm() are used
   here purely as a smooth, plausible per-hue shape for how the display
   gamut's chroma extent varies with hue -- not as numerically exact
   ACES 2.0 reach/Mnorm values, trading numerical exactness for 3DCF's
   lightweight, RGB/xy-native architecture over a full CAM16 hue
   computation.
   ==================================================================== */

static const float st_gamut_reach[360] =
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
  1164.36269f,   1198.98879f,  1237.19312f,   1273.05128f,   1313.63076f,  1351.65816f,
  1394.65221f,   1434.90942f,  1479.36892f,  1521.29923f,   1568.31051f,   1611.98419f,
  1660.62312f,   1705.85267f,  1756.34372f,   1803.03751f,   1855.55009f,  1903.87186f,
  1958.61575f,  2008.99678f,  2065.34081f,   2116.53174f,   2174.75144f,  2227.24046f,
  2286.71107f,  2334.72467f,  2382.30107f,  2412.86075f,   2440.19453f,  2454.06478f,
  2459.10001f,  2451.71571f,  2434.68372f,   2404.68880f,   2367.15629f,  2312.62957f,
  2255.38400f,  2181.09339f,  2107.34164f,   2016.32946f,   1927.02893f,  1817.55349f,
  1712.68006f,  1587.07577f,  1470.86887f,   1330.42737f,   1206.14536f,  1053.68427f,
   932.34317f,    788.29193f,    696.24225f,    611.53390f,    544.99676f,   492.60999f,
   450.65358f,    416.32756f,    387.83403f,    363.87664f,    343.46013f,   325.87004f,
   310.59306f,    297.22935f,    285.46049f,    275.03443f,    265.75110f,   257.45036f,
   250.00035f,    243.29181f,    237.23430f,    231.75173f,    226.77956f,   222.26262f,
   218.15373f,    214.41231f,    210.99364f,    207.86736f,    204.98974f,   202.32907f,
   199.85763f,    197.55163f,    195.39062f,    193.35700f,    191.43587f,   189.61455f,
   187.88248f,    186.23068f,    184.65166f,    183.13911f,    181.68774f,   180.29310f,
   178.95149f,    177.65981f,    176.41537f,    175.21587f,    174.05930f,   172.94393f,
   171.86825f,    170.83094f,    169.83086f,    168.86699f,    167.93843f,   167.04441f,
   166.18422f,    165.35726f,    164.56299f,    163.80093f,    163.07064f,   162.37175f,
};

static inline float st_reach_from_table(float h)
{
  if(!isfinite(h)) return 0.0f;
  float hw = fmodf(h, 360.0f);
  if(hw < 0.0f) hw += 360.0f;
  int i0 = (int)hw;
  int i1 = (i0 + 1) % 360;
  const float t = hw - (float)i0;
  return st_gamut_reach[i0] + t * (st_gamut_reach[i1] - st_gamut_reach[i0]);
}

static inline float st_chroma_norm(float h)
{
  const float hr = h * (float)(M_PI / 180.0);
  const float a = cosf(hr);
  const float b = sinf(hr);
  const float a2 = a * a - b * b;                /* cos(2h) */
  const float b2 = 2.0f * a * b;                 /* sin(2h) */
  const float a3 = 4.0f * a * a * a - 3.0f * a;  /* cos(3h) */
  const float b3 = 3.0f * b - 4.0f * b * b * b;  /* sin(3h) */
  const float m = 11.34072f * a + 16.46899f * a2 + 7.88380f * a3
                + 14.66441f * b - 6.37224f * b2 + 9.19364f * b3
                + 77.12896f;
  return m;
}

/* Mean of st_reach_from_table(h)/st_chroma_norm(h) over h=0..359, precomputed
   offline from the table above, used to normalise the shape factor below to
   ~1.0 on average. */
#define ST_GAMUT_SHAPE_REF  2.090563f

/* Independent sigmoid contrast on the CIE-xz offset from white, applied
   BEFORE st_spectral_gamut() so its knee compression absorbs any excursion
   this creates. Operates on chroma normalized by the spectral locus radius
   for the pixel's hue angle -- NOT on raw x_tm/z_tm, whose absolute scale
   depends on y_tm and would make the contrast luminance-dependent instead
   of saturation-dependent. Y is intentionally left untouched (handled by
   SSTS + toe/shoulder already).

   Normalization mirrors sigmoidAdj() from Ohnishi Yasuo's XYZ sigmoid curve
   GIMP plug-in (GPLv3): the sigmoid is rescaled so f(0)=0, f(1)=1 exactly,
   here applied per-axis to the [-1,1]-normalized chroma offset instead of
   to a raw channel value. */
static inline void st_chroma_contrast_sigmoid(
  float *x_tm, float *z_tm, const float y_tm,
  const float white_x_ratio, const float white_z_ratio,
  const float gain_x, const float shift_x, const float sig0_x, const float inv_range_x,
  const float gain_z, const float shift_z, const float sig0_z, const float inv_range_z,
  const float *spectral_boundary)
{
  if(y_tm <= 0.0f) return;
  if(!isfinite(*x_tm) || !isfinite(*z_tm)) return;

  const float sum = *x_tm + y_tm + *z_tm;
  if(sum <= 0.0f) return;
  const float cie_x = *x_tm / sum;
  const float cie_z = *z_tm / sum;

  const float wy = 1.0f;
  const float wsum = white_x_ratio + wy + white_z_ratio;
  const float white_cie_x = white_x_ratio / wsum;
  const float white_cie_z = white_z_ratio / wsum;

  const float dx = cie_x - white_cie_x;
  const float dz = cie_z - white_cie_z;

  if(!spectral_boundary) return;

  const float angle = atan2f(dz, dx);
  float angle_deg = angle * (180.0f / (float)M_PI);
  if(angle_deg < 0.0f) angle_deg += 360.0f;
  if(angle_deg >= 360.0f) angle_deg -= 360.0f;
  const int bin = (int)angle_deg;
  const int next = (bin + 1) % 360;
  const float frac = angle_deg - (float)bin;
  const float max_dist = spectral_boundary[bin]
                        + frac * (spectral_boundary[next] - spectral_boundary[bin]);
  if(max_dist <= 0.0f) return;

  /* Normalize each axis independently to [-1, 1] by the spectral radius,
     remap to [0, 1] for the sigmoid, then back. */
  const float u = fminf(fmaxf(dx / max_dist, -1.0f), 1.0f);
  const float w = fminf(fmaxf(dz / max_dist, -1.0f), 1.0f);

  const float u01 = 0.5f * (u + 1.0f);
  const float w01 = 0.5f * (w + 1.0f);

  const float sig_u = 1.0f / (1.0f + expf(-gain_x * (u01 - shift_x)));
  const float sig_w = 1.0f / (1.0f + expf(-gain_z * (w01 - shift_z)));

  const float u01_new = (sig_u - sig0_x) * inv_range_x;
  const float w01_new = (sig_w - sig0_z) * inv_range_z;

  const float u_new = 2.0f * u01_new - 1.0f;
  const float w_new = 2.0f * w01_new - 1.0f;

  const float cie_x_new = white_cie_x + u_new * max_dist;
  const float cie_z_new = white_cie_z + w_new * max_dist;
  const float y_new = 1.0f - cie_x_new - cie_z_new;

  if(y_new > 0.0f)
  {
    const float S_new = y_tm / y_new;
    *x_tm = cie_x_new * S_new;
    *z_tm = cie_z_new * S_new;
  }
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

  /* Spectral locus boundary check -- automatic detection of non-spectral
     colours. Runs before the user knee to catch colours whose chromaticity
     ratios lie outside the visible spectrum (e.g. laser primaries,
     narrow-band LEDs). */
  if(spectral_boundary)
  {
    const float angle = atan2f(dz, dx);
    float angle_deg = angle * (180.0f / (float)M_PI);
    if(angle_deg < 0.0f) angle_deg += 360.0f;
    if(angle_deg >= 360.0f) angle_deg -= 360.0f;
    int bin = (int)angle_deg;
    int next = (bin + 1) % 360;
    float frac = angle_deg - (float)bin;
    float max_dist = spectral_boundary[bin] + frac * (spectral_boundary[next] - spectral_boundary[bin]);
    const float target_dist = max_dist * 0.92f;

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

  /* User-controlled knee: smooth circular roll-off controlled by user
     sliders, modulated by a hue-dependent shape factor (reach/chroma_norm
     normalised to mean=1, with sqrt to temper extreme values). */
  {
    const float angle = atan2f(dz, dx);
    float angle_deg = angle * (180.0f / (float)M_PI);
    if(angle_deg < 0.0f) angle_deg += 360.0f;
    if(angle_deg >= 360.0f) angle_deg -= 360.0f;
    const float shape = st_reach_from_table(angle_deg) / fmaxf(st_chroma_norm(angle_deg), 1e-6f);
    const float shape_norm = fmaxf(shape / ST_GAMUT_SHAPE_REF, 0.0f);
    const float knee_mod = knee * sqrtf(shape_norm);

  if(chroma_sq > knee_mod * knee_mod)
  {
    const float chroma = sqrtf(chroma_sq);
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

/* Complete spectral tone mapping pipeline for one pixel.

   Pipeline order:
   1. D50-adapted Rec.2020 RGB -> D50 XYZ (precise matrix)
   2. ACES 2.0 SSTS on luminance Y only (spectral tone scale)
   3. BT.1886 OETF + contrast S-curve
   4. Mid-tone gamma adjustment
   5. Chromaticity ratio scaling: x = ratio * Y
   5b. Chroma contrast: independent sigmoid on CIE-xz offset from white (X/Z only)
   6. Spectral gamut: film-like chromaticity roll-off in CIE xy
   7. XYZ -> output RGB via output matrix
   8. Highlight desaturation (blend toward achromatic luma)
   9. Vibrance (saturation with high-sat protection)
   10. Gamut compression safety net
   11. Clamp to [0, inf) */
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
  if(ctx->gamma != 0.0f)
  {
    float y_lvl = fminf(fmaxf(y_tm, 0.0f), 1.0f);
    y_lvl = powf(y_lvl, ctx->gamma_power);
    y_tm = y_lvl;
  }

  /* Step 5: Scale chromaticity with tone-mapped luminance */
  float x_tm = x_ratio * y_tm;
  float z_tm = z_ratio * y_tm;

  /* Step 5b: Chroma contrast — independent sigmoid, X/Z axes only (Y is
     left untouched, see st_chroma_contrast_sigmoid() documentation) */
  st_chroma_contrast_sigmoid(&x_tm, &z_tm, y_tm,
                              ctx->white_chroma_x, ctx->white_chroma_z,
                              ctx->chroma_x_gain, ctx->chroma_x_shift,
                              ctx->chroma_x_sig0, ctx->chroma_x_inv_range,
                              ctx->chroma_z_gain, ctx->chroma_z_shift,
                              ctx->chroma_z_sig0, ctx->chroma_z_inv_range,
                              ctx->spectral_boundary);

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
    /* Abney hue rotation tied to SSTS compression ratio. weight = pow(comp_factor, 0.6)
       ramps up as soon as compression begins (progressive onset, midtones still 0
       since pow(0, .6) = 0) up to 1 at full compression (clipped highlights):
       midtones/shadows are never touched while the rotation stays clearly visible
       on highlights. hl_hue_shift controls only the rotation angle:
       angle = hl_hue_shift * max_angle * weight
       (max_angle = 0.4 rad ~= 23 deg at full excursion on clipped highlights). */
    const float y_exposed = y_abs * ctx->exposure_factor;
    float hl_weight = 0.0f;
    if(ctx->hl_rotation != 0.0f && y_abs > 1e-10f)
    {
      const float compression = fminf(y_tm / (y_abs * ctx->exposure_factor), 1.0f);
      const float comp_factor = 1.0f - compression;
      hl_weight = powf(comp_factor, 0.6f);
      const float max_angle = 0.4f; //CB
      const float angle = ctx->hl_rotation * max_angle * hl_weight;
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

        /* Chroma clamp: scale uv to keep rotated RGB in [0,1] */
        {
          float t = 1.0f;
          if(ur > 0.0f) t = fminf(t, (1.0f - y) / ur);
          if(ur < 0.0f) t = fminf(t, -y / ur);
          if(vr > 0.0f) t = fminf(t, (1.0f - y) / vr);
          if(vr < 0.0f) t = fminf(t, -y / vr);
          if(lc[1] != 0.0f)
          {
            const float g_d = -(lc[0] * ur + lc[2] * vr) / lc[1];
            if(g_d > 0.0f) t = fminf(t, (1.0f - y) / g_d);
            if(g_d < 0.0f) t = fminf(t, -y / g_d);
          }
          t = fmaxf(t, 0.0f);
          rgb[0] = y + t * ur;
          rgb[2] = y + t * vr;
          if(lc[1] != 0.0f)
            rgb[1] = y - (lc[0] / lc[1]) * t * ur - (lc[2] / lc[1]) * t * vr;
        }
      }
    }

    const float w = st_desat_weight(y_exposed, ctx->hl_desat, ctx->hl_desat_threshold);
    if(w > 0.0f && isfinite(w))
    {
      const float maxc_pre = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
      const float minc_pre = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
      const float sat_pre = (maxc_pre > 0.0f) ? (maxc_pre - minc_pre) / maxc_pre : 0.0f;
      const float ss_pre = (sat_pre * sat_pre) / (sat_pre * sat_pre + (1.0f - sat_pre) * (1.0f - sat_pre) + 1e-6f);

      /* Negative vibrance: desaturate saturated pixels more, driven by hl_desat */
      float w_final = w;
      if(ctx->hl_desat > 0.0f)
      {
        const float vib_neg = ctx->hl_desat * ss_pre * 0.35f;
        const float w_vib = st_desat_weight(y_exposed, 1.0f, ctx->hl_desat_threshold) * vib_neg;
        w_final = fminf(w_final + w_vib, 1.0f);
      }

      /* Blend toward white with linear ramp */
      const float ts = fminf(w_final, 1.0f);
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

  /* Step 9b: chromatic contrast — luminance-adaptive mid-tone saturation boost */
  if(ctx->chromatic_boost > 0.0f)
  {
    const float *lc = ctx->luma_coeff;
    const float luma = lc[0] * rgb[0] + lc[1] * rgb[1] + lc[2] * rgb[2];
    const float maxc = fmaxf(fmaxf(rgb[0], rgb[1]), rgb[2]);
    const float minc = fminf(fminf(rgb[0], rgb[1]), rgb[2]);
    const float sat_m = maxc - minc;
    const float level = fmaxf(maxc, fmaxf(fabsf(minc), fabsf(luma)));
    const float sat_norm = (level > 0.0f) ? sat_m / level : 0.0f;
    const float y_mid = 0.18f, sigma = 1.85f;
    const float log_rel = log2f(fmaxf(y_abs / y_mid, 1e-10f));
    const float w_gauss = expf(-(log_rel * log_rel) / (2.0f * sigma * sigma));
    const float w_mid = (log_rel <= 0.0f) ? 1.0f
                       : 1.328f * w_gauss - 0.328f;
    const float pp = 1.0f - fminf(sat_norm, 1.0f);
    /* Hue modulation: −10% jaunes (60°), +10% bleus (240°) */
    float hue_deg = 0.0f;
    if(sat_m > 0.0f)
    {
      if(maxc == rgb[0])
        hue_deg = 60.0f * fmodf((rgb[1] - rgb[2]) / sat_m, 6.0f);
      else if(maxc == rgb[1])
        hue_deg = 60.0f * ((rgb[2] - rgb[0]) / sat_m + 2.0f);
      else
        hue_deg = 60.0f * ((rgb[0] - rgb[1]) / sat_m + 4.0f);
      if(hue_deg < 0.0f) hue_deg += 360.0f;
    }
    const float hue_mod = 1.0f + 0.1f * cosf((hue_deg - 60.0f) * (M_PI / 180.0f));
    const float gain = 1.0f + ctx->chromatic_boost * w_mid * (pp * pp) * hue_mod;
    rgb[0] = luma + gain * (rgb[0] - luma);
    rgb[1] = luma + gain * (rgb[1] - luma);
    rgb[2] = luma + gain * (rgb[2] - luma);
  }

  st_gamut_compress(rgb, ctx->luma_coeff);

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
  /* Exposure factor from independent EV slider (ACES 2.0 parity) */
  ctx->exposure_factor = exp2f(p->input_exposure);

  /* === Combined input matrix: Rec.2020 D50 RGB → D50 XYZ === */
  {
    for(int i = 0; i < 9; i++)
      ctx->input_matrix[i] = (float)dt_st_rec2020_d50_to_xyz[i];
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
  /* Pivot inverted so that 'right' (higher value) brightens the image */
  ctx->contrast_pivot = 1.0f - fmaxf(fminf(p->contrast_pivot, 0.99f), 0.01f);

  ctx->hl_desat = fmaxf(p->hl_desaturation, 0.0f);
  ctx->hl_desat_threshold = fmaxf(p->hl_desat_threshold, 0.0f);
  ctx->hl_rotation = p->hl_hue_shift;
  ctx->gamut_knee = fmaxf(p->gamut_knee, 0.0f);
  ctx->gamut_steepness = fmaxf(p->gamut_steepness, 1e-6f);
  ctx->toe_power = fmaxf(p->toe_power, 0.0f);
  ctx->shoulder_power = fmaxf(p->shoulder_power, 0.0f);
  ctx->hl_detail_recovery = fmaxf(p->hl_detail_recovery, 0.0f);

  /* Chroma contrast sigmoid — precompute gain/shift/normalization per axis.
     Mirrors sigmoidAdj() from Ohnishi's XYZ sigmoid curve GIMP plug-in:
     gain guarded away from 0 to avoid division by zero in the endpoint
     normalization; as gain -> 0 the normalized curve tends to identity. */
  {
    const float gx = fmaxf(p->chroma_x_contrast, 1e-4f);
    const float sx = p->chroma_x_pivot / 100.0f;
    const float sig0x = 1.0f / (1.0f + expf(-gx * (0.0f - sx)));
    const float sig1x = 1.0f / (1.0f + expf(-gx * (1.0f - sx)));
    ctx->chroma_x_gain = gx;
    ctx->chroma_x_shift = sx;
    ctx->chroma_x_sig0 = sig0x;
    ctx->chroma_x_inv_range = 1.0f / fmaxf(sig1x - sig0x, 1e-6f);

    const float gz = fmaxf(p->chroma_z_contrast, 1e-4f);
    const float sz = p->chroma_z_pivot / 100.0f;
    const float sig0z = 1.0f / (1.0f + expf(-gz * (0.0f - sz)));
    const float sig1z = 1.0f / (1.0f + expf(-gz * (1.0f - sz)));
    ctx->chroma_z_gain = gz;
    ctx->chroma_z_shift = sz;
    ctx->chroma_z_sig0 = sig0z;
    ctx->chroma_z_inv_range = 1.0f / fmaxf(sig1z - sig0z, 1e-6f);
  }

  /* Output gamut protection matrices from pre-computed lookup table */
  {
    const int cs = (p->output_cs >= DT_ST_CS_REC709 && p->output_cs <= DT_ST_CS_ADOBERGB)
                     ? (int)p->output_cs : (int)DT_ST_CS_REC2020;
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

  /* Mid-tone gamma adjustment, inverted so that 'right' brightens (gamma < 1.0) */
  ctx->gamma = -fmaxf(fminf(p->gamma, 1.0f), -1.0f);
  ctx->gamma_power = exp2f(ctx->gamma);

  ctx->vibrance = fmaxf(p->vibrance, 0.0f);
  ctx->chromatic_boost = fmaxf(p->chromatic_boost, 0.0f);

  /* Initialize ACES 2.0 SSTS from peak_luminance (%, 0% = 200 nits) */
  {
    const float hp = p->peak_luminance / 100.0f; /* -1 .. 1 */
    const double peak = 200.0 * (1.0 + (double)hp);
    dt_st_ssts_init(&ctx->ssts, peak);
  }
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
    new_p->gamma               = old->gray_point;
    new_p->vibrance            = old->vibrance;
    new_p->peak_luminance     = 0.0f;
    new_p->input_exposure      = 0.0f;
    new_p->chromatic_boost     = 0.0f;
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
    *new_version = 9;
    return 0;
  }

  // v5 → v6: added hl_exposure field
  if(old_version == 5)
  {
    typedef struct dt_iop_3dcf_params_v5_t
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
      float hl_detail_recovery;
    } dt_iop_3dcf_params_v5_t;

    const dt_iop_3dcf_params_v5_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    new_p->contrast            = old->contrast;
    new_p->gamma               = old->gray_point;
    new_p->vibrance            = old->vibrance;
    new_p->peak_luminance     = 0.0f;
    new_p->input_exposure      = 0.0f;
    new_p->chromatic_boost     = 0.0f;
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
    new_p->hl_detail_recovery  = old->hl_detail_recovery;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 9;
    return 0;
  }

  // v6 → v7: renamed gray_point → gamma field
  if(old_version == 6)
  {
    typedef struct
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
      float hl_detail_recovery;
      float hl_exposure;
    } dt_iop_3dcf_params_v6_t;

    const dt_iop_3dcf_params_v6_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    new_p->contrast            = old->contrast;
    new_p->gamma               = old->gray_point;
    new_p->vibrance            = old->vibrance;
    new_p->peak_luminance     = 0.0f;
    new_p->input_exposure      = 0.0f;
    new_p->chromatic_boost     = 0.0f;
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
    new_p->hl_detail_recovery  = old->hl_detail_recovery;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 9;
    return 0;
  }

  // v7 → v8: added chromatic_boost field
  if(old_version == 7)
  {
    typedef struct
    {
      float contrast;
      float gamma;
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
      float hl_detail_recovery;
      float hl_exposure;
    } dt_iop_3dcf_params_v7_t;

    const dt_iop_3dcf_params_v7_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    new_p->contrast            = old->contrast;
    new_p->gamma               = old->gamma;
    new_p->vibrance            = old->vibrance;
    new_p->peak_luminance     = 0.0f;
    new_p->input_exposure      = 0.0f;
    new_p->chromatic_boost     = 0.0f;
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
    new_p->hl_detail_recovery  = old->hl_detail_recovery;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 9;
    return 0;
  }

  // v8 → v9: spectral_brilliance split into peak_luminance + input_exposure
  if(old_version == 8)
  {
    const dt_iop_3dcf_params_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    *new_p = *old;
    new_p->peak_luminance = 0.0f;
    new_p->input_exposure = 0.0f;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 9;
    return 0;
  }

  // v9 → v10: added chroma_x_contrast/pivot, chroma_z_contrast/pivot
  if(old_version == 9)
  {
    const dt_iop_3dcf_params_t *old = old_params;
    dt_iop_3dcf_params_t *new_p = malloc(sizeof(dt_iop_3dcf_params_t));

    *new_p = *old;
    new_p->chroma_x_contrast = 0.0f;
    new_p->chroma_x_pivot    = 50.0f;
    new_p->chroma_z_contrast = 0.0f;
    new_p->chroma_z_pivot    = 50.0f;

    *new_params = new_p;
    *new_params_size = sizeof(dt_iop_3dcf_params_t);
    *new_version = 10;
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
  
  /* Initialize the context with the default params NOW, not later at
     commit, to avoid race conditions on Windows. */
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

  /* A pending "auto" request needs a histogram/variance pass over the host
     float buffer (auto_adjust_3dcf_params()), which only process() performs
     -- process_cl() has no GPU equivalent. Left alone, the flag set by the
     picker callback is simply never consumed when OpenCL handles the
     preview pipe. Force this preview run through the CPU path instead, same
     pattern as channelmixerrgb.c's run_profile/checker_ready handling. */
  {
    const dt_iop_3dcf_gui_data_t *g = self->gui_data;
    if(g && g->auto_apply_requested && dt_pipe_is_preview(pipe))
      piece->process_cl_ready = FALSE;
  }
}

void tiling_callback(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in, const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_3dcf_data_t *d = piece->data;

  /* HL detail recovery allocates one extra full-size guide buffer plus two
     single-channel buffers (luminance + smoothed base). Account for that
     extra memory so tiling doesn't under-estimate on very large images. */
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
     (init_pipe only zeroes the struct). Pass the image through unchanged
     rather than dividing by zero or crashing. Hits only on the very first
     preview render on startup; commit_params() corrects this immediately. */
  if(!d || d->ctx.ssts.n <= 0.0)
  {
    memcpy(ovoid, ivoid, sizeof(float) * npixels * ch);
    return;
  }

  /* Auto-adjust: triggered by color_picker_apply() via a flag on gui_data.
     self->gui_data is NULL outside darkroom (export, thumbnails), so this
     path never runs outside an interactive session. ivoid here is the
     module's INPUT buffer (before 3DCF processing), in the Rec.2020 D50
     working space -- exactly what auto_adjust_3dcf_params() expects.

     Restricted to the preview pipe, matching commit_params()'s own
     dt_pipe_is_preview() check for this same flag: _auto_request() only
     invalidates the preview pipe (dt_dev_reprocess_preview()), so this scan
     is meant to run on that small buffer, not on whichever pipe happens to
     call process() first while the flag is set. Left unguarded, the full
     pipe could consume the flag and run this scan against the full-
     resolution buffer instead. */
  {
    dt_iop_3dcf_gui_data_t *g = self->gui_data;
    if(g && g->auto_apply_requested && dt_pipe_is_preview(piece->pipe))
    {
      const dt_iop_st_auto_mode_t mode = g->auto_requested_mode;
      const float picked_grey = g->picked_grey_y;
      g->auto_apply_requested = FALSE;

      _st_auto_result_t *res = (_st_auto_result_t *)malloc(sizeof(_st_auto_result_t));
      if(res)
      {
        res->self = self;
        memcpy(&res->params, &d->params, sizeof(dt_iop_3dcf_params_t));
        auto_adjust_3dcf_params((const float *)ivoid, width, height, ch, &res->params,
                                mode, picked_grey);
        g_idle_add(_auto_apply_to_gui, res); /* passe la main au thread GUI */
      }
    }
  }

  /* Bounds check: color_looks has 11 entries (0-10). Clamp defensively in
     case a stale preset carries an out-of-range value. */
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
  int guide_ok = 0;
  if(hl_detail_recovery > 0.0f)
  {
    lum_orig_g = new_gray_image(width, height);
    base_orig_g = new_gray_image(width, height);
    if(!lum_orig_g.data || !base_orig_g.data)
      fprintf(stderr, "[3dcf] failed to allocate detail-recovery luminance buffers, "
                      "detail recovery disabled\n");

    /* Sanitized copy of the full image, brilliance-adjusted, used as the
       guided-filter GUIDE. The brilliance exposure factor is baked in here
       so that the guided filter's base is in the same perceptual domain as
       the tone-mapped output, preventing the gain (lum_tm / lum_orig) from
       exploding when perceptual brilliance is high. Negative/NaN channels
       (CA fringing, sharpening halos) must be cleared here too: the guide
       feeds the local mean/variance regression, and a single stray NaN
       corrupts the entire filter window. Allocated/freed with
       dt_alloc_align/dt_free_align. */
    guide_sanitized = (float *)dt_alloc_aligned(sizeof(float) * (size_t)npixels * ch);
    if(!guide_sanitized)
      fprintf(stderr, "[3dcf] failed to allocate guide_sanitized (%zu bytes), detail recovery disabled\n",
              sizeof(float) * (size_t)npixels * ch);
    const float bf = d->ctx.exposure_factor;

    const float *lc = d->ctx.luma_coeff;
    #ifdef _OPENMP
    #pragma omp parallel for default(none) \
      shared(in, lum_orig_g, guide_sanitized, npixels, ch, lc, bf)
    #endif
    for(size_t k = 0; k < npixels; k++)
    {
      const size_t idx = k * ch;
      float r = in[idx], g = in[idx + 1], b = in[idx + 2];
      r = isfinite(r) ? fmaxf(r, 0.0f) : 0.0f;
      g = isfinite(g) ? fmaxf(g, 0.0f) : 0.0f;
      b = isfinite(b) ? fmaxf(b, 0.0f) : 0.0f;
      r *= bf;
      g *= bf;
      b *= bf;

      if(guide_sanitized)
      {
        guide_sanitized[idx]     = r;
        guide_sanitized[idx + 1] = g;
        guide_sanitized[idx + 2] = b;
        if(ch == 4) guide_sanitized[idx + 3] = in[idx + 3];
      }

      if(lum_orig_g.data)
        lum_orig_g.data[k] = lc[0] * r + lc[1] * g + lc[2] * b;
    }

    /* Normalize guided filter radius to sensor resolution (ref: 36 MP 3:2 K1) */
    const float diag = sqrtf((float)piece->iwidth * piece->iwidth
                            + (float)piece->iheight * piece->iheight);
    const int gf_radius = fmaxf(4.0f, 8.0f * diag / 8848.0f);

    if(guide_sanitized && lum_orig_g.data && base_orig_g.data)
    {
      guided_filter(guide_sanitized, lum_orig_g.data, base_orig_g.data,
                    width, height, ch, gf_radius, 0.05f, 1.0f, 0.0f, FLT_MAX);
      guide_ok = 1;
    }
  }

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    shared(in, out, width, height, ch, d, npixels, mat, look_opacity, \
           hl_detail_recovery, lum_orig_g, base_orig_g, guide_ok)
  #endif
  for(size_t k = 0; k < npixels; k++)
  {
    const size_t idx = k * ch;
    float rgb_in[3] = { in[idx], in[idx + 1], in[idx + 2] };
    float rgb_out[3];

    /* Sanitize input: clamp negative (from CA / sharpening halos) to 0 and
       zero any NaN/Inf. Negatives reachable after demosaic + lens. */
    for(int c = 0; c < 3; c++)
      rgb_in[c] = isfinite(rgb_in[c]) ? fmaxf(rgb_in[c], 0.0f) : 0.0f;

    /* Save original luminance for detail recovery BEFORE the safety net
       desaturates rgb_in below. Captured from the already-sanitized values
       so CA fringing / NaN never reach the detail computation. This must
       happen at this exact point to match the GPU kernel (kernel_3dcf,
       "before safety net modifies rgb_in") -- capturing it later from the
       raw "in" buffer would silently diverge from OpenCL on any pixel with
       negative/NaN input channels. */
    float lum_orig = 0.0f;
    if(hl_detail_recovery > 0.0f)
    {
      const float *lc = d->ctx.luma_coeff;
      lum_orig = (lc[0] * rgb_in[0] + lc[1] * rgb_in[1] + lc[2] * rgb_in[2]) * d->ctx.exposure_factor;
    }

    /* Pre-pipeline safety net: sigmoid rolloff toward luma-gray. Track the
       desaturation factor so the detail recovery below can attenuate
       proportionally -- a pixel that was heavily desaturated (e.g.
       out-of-gamut blue to white) would otherwise receive a huge gain
       boost from the pre-desat luminance vs post-tm luminance mismatch,
       creating a halo outside the object. */
    float hl_desat_factor = 0.0f;
    {
      const float lum = fmaxf(fmaxf(rgb_in[0], rgb_in[1]), rgb_in[2]);
      const float w = st_desat_weight(lum * d->ctx.exposure_factor, d->ctx.hl_desat, d->ctx.hl_desat_threshold);
      if(w > 0.0f && isfinite(w))
      {
        const float t = fminf(w, 1.0f);
        const float ts = (t * t) / (t * t + (1.0f - t) * (1.0f - t) + 1e-6f);
        hl_desat_factor = ts;
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

    /* HL detail recovery: re-inject guided-filter detail with gain
       compensation. lum_orig was captured right after sanitization, above
       -- same value and same pipeline point as the GPU kernel. guide_ok
       ensures we skip this when the guided-filter pre-pass failed (e.g. OOM
       on guide_sanitized), preventing use of an uninitialized base. */
    if(hl_detail_recovery > 0.0f && guide_ok)
    {
      const float *lc = d->ctx.luma_coeff;
      const float lum_tm = lc[0] * rgb_out[0] + lc[1] * rgb_out[1] + lc[2] * rgb_out[2];
      if(lum_tm > 1e-6f && lum_orig > 1e-6f)
      {
        float detail = lum_orig - base_orig_g.data[k];
        /* Soft clamp detail amplitude: the threshold tightens as the
           slider increases, allowing more micro-contrast at low settings
           while preventing halos at high settings. Same soft shoulder
           formula as the spectral gamut and knee. */
        {
          const float t = fminf(hl_detail_recovery, 1.0f);
          const float detail_frac = 0.10f * (1.0f - t * 0.50f);
          const float bsteep_frac = 0.25f * (1.0f - t * 0.40f);
          const float dl = fmaxf(lum_orig * detail_frac, 1e-6f);
          const float da = fabsf(detail);
          if(da > dl)
          {
            const float excess = da - dl;
            const float bsteep = fmaxf(dl * bsteep_frac, 0.001f);
            const float compression = excess / (excess + bsteep);
            const float clamped_abs = dl + compression * bsteep;
            detail = copysignf(clamped_abs, detail);
          }
        }
        const float gain = lum_tm / lum_orig;
        const float lum_final = lum_tm + detail * hl_detail_recovery * (1.0f - hl_desat_factor) * gain;
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
/* Pack the precomputed CPU context (+ active color look) into the GPU
   struct. Mirrors exactly what process() reads from d->ctx and d->params. */
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
  clp->gamma           = ctx->gamma;
  clp->gamma_power     = ctx->gamma_power;
  clp->vibrance        = ctx->vibrance;
  clp->chromatic_boost = ctx->chromatic_boost;
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

  clp->chroma_x_gain      = ctx->chroma_x_gain;
  clp->chroma_x_shift     = ctx->chroma_x_shift;
  clp->chroma_x_sig0      = ctx->chroma_x_sig0;
  clp->chroma_x_inv_range = ctx->chroma_x_inv_range;
  clp->chroma_z_gain      = ctx->chroma_z_gain;
  clp->chroma_z_shift     = ctx->chroma_z_shift;
  clp->chroma_z_sig0      = ctx->chroma_z_sig0;
  clp->chroma_z_inv_range = ctx->chroma_z_inv_range;

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

  /* Safety guard mirroring process(): if commit_params() has not yet run,
     the context is all-zero -- fall back to the CPU path (which copies
     through). */
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
       of the raw dev_in. Mirrors the CPU's guide_sanitized buffer:
       negative/NaN channels from CA fringing must not reach the filter's
       local mean/variance regression on either platform. */
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
  p.peak_luminance = 0.0f;
  p.input_exposure = 0.0f;
  p.gamma = 0.0f;
  p.vibrance = 1.0f;
  p.chromatic_boost = 0.0f;
  p.hl_desaturation = 0.25f; //CB
  p.hl_desat_threshold = 0.50f;
  p.hl_hue_shift = 0.0f;
  p.gamut_knee = 0.20f; //CB
  p.gamut_steepness = 0.50f;
  p.output_cs = DT_ST_CS_REC2020;
  p.color_look = 0;
  p.look_opacity = 1.0f;
  p.contrast_pivot = 0.5f;
  p.toe_power = 1.0f;
  p.shoulder_power = 1.0f;
  p.hl_detail_recovery = 0.20f; //CB
  /* Keep the chroma axes at their introspection defaults (pivot 50, contrast 0)
     so that re-applying this auto-preset (e.g. after a history reset) restores
     the pivot sliders to 50%. Without these, the zeroed struct would put them
     at 0. */
  p.chroma_x_contrast = 0.0f;
  p.chroma_x_pivot = 50.0f;
  p.chroma_z_contrast = 0.0f;
  p.chroma_z_pivot = 50.0f;

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
  (void)pipe;
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  if(!g || picker != g->auto_mode_combo) return;

  /* self->picked_color[] holds the average of the sampled area, in the
     module's INPUT space (Rec.2020 D50 scene-linear), the same space as the
     ivoid buffer auto_adjust_3dcf_params() reads. dt_color_picker_new() in
     gui_init() must be configured in AREA mode for the average to be
     computed; in POINT mode the value is a single pixel, much noisier on a
     RAW. */
  const double luma_r = dt_st_rec2020_d50_to_xyz[3];
  const double luma_g = dt_st_rec2020_d50_to_xyz[4];
  const double luma_b = dt_st_rec2020_d50_to_xyz[5];

  const float y = (float)(luma_r * (double)self->picked_color[0]
                        + luma_g * (double)self->picked_color[1]
                        + luma_b * (double)self->picked_color[2]);

  /* A sample on pure black (or an invalid value) would give an infinite
     ev_shift: ignored rather than producing an aberrant exposure. */
  if(!(y > 1e-6f) || !isfinite(y)) return;

  g->picked_grey_y = y;

  /* Live trigger: this callback fires continuously while the user drags the
     sampled area (once per pipe update as long as the position/area
     changed -- see _record_point_area() in gui/color_picker_proxy.c, which
     filters out redundant calls and rules out any reprocess loop: our own
     dt_dev_reprocess_preview() below only causes a single "echo" pass that
     self-terminates, since the area hasn't moved during that echo). */
  const int combo_idx = dt_bauhaus_combobox_get(g->auto_mode_combo);
  const dt_iop_st_auto_mode_t mode =
    (combo_idx >= DT_ST_AUTO_CONTRASTY && combo_idx <= DT_ST_AUTO_SOFT)
    ? (dt_iop_st_auto_mode_t)combo_idx : DT_ST_AUTO_NEUTRAL;

  /* Re-triggers if the picked value changed meaningfully (dragged to a new
     area) OR the rendering preset changed (comparing contrasty/neutral/soft
     on the SAME area without moving the picker) -- auto_requested_mode
     holds the last mode actually sent, so this comparison doesn't
     re-trigger needlessly when neither changed (e.g. echo calls). */
  const float delta = fabsf(y - g->picked_grey_y_last_applied);
  if(delta > 1e-4f || mode != g->auto_requested_mode)
  {
    g->picked_grey_y_last_applied = y;
    _auto_request(self, mode);
  }
}

void gui_reset(dt_iop_module_t *self)
{
  /* The module's reset button goes through dt_iop_gui_reset() (which calls
     this hook if it exists) BEFORE dt_iop_gui_update() -- see
     _gui_reset_callback() in develop/imageop.c. This clears the picker's
     armed/picked state on an explicit module reset, in addition to the
     image-change case handled in gui_update(). */
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  g->picked_grey_y = 0.0f;
  g->picked_grey_y_last_applied = 0.0f;
  /* The X/Z lock is a GUI preference, not a parameter: keep it across an
     explicit module reset too (it defaults to linked). */
  dt_iop_color_picker_reset(self, TRUE);
}

/* Derived asymmetry helpers for the transient X/Z balance widget. The
   balance is not a parameter: it is recomputed from the stored X/Z contrast
   values and, when X and Z are locked together, decides how a shared
   contrast level is split between the two axes. */
static inline float _chroma_balance_from_xz(const float x, const float z)
{
  const float sum = x + z;
  return sum > 1e-6f ? CLAMPF((x - z) / sum, -1.0f, 1.0f) : 0.0f;
}

/* Z contrast that preserves the current balance when X contrast changes:
   b = (X-Z)/(X+Z)  =>  Z = X*(1-b)/(1+b).  b is clamped away from the
   singularities at ±1 so the ratio always stays finite. */
static inline float _chroma_z_from_x(const float x, const float b)
{
  const float bf = CLAMPF(b, -0.999f, 0.999f);
  return CLAMPF(x * (1.0f - bf) / (1.0f + bf), 0.0f, 10.0f);
}

/* Mirror of _chroma_z_from_x() for the (hidden while linked) Z slider. */
static inline float _chroma_x_from_z(const float z, const float b)
{
  const float bf = CLAMPF(b, -0.999f, 0.999f);
  return CLAMPF(z * (1.0f + bf) / (1.0f - bf), 0.0f, 10.0f);
}

/* While X and Z are locked together the X axis alone drives both, so the
   balance slider replaces the redundant sliders (Z contrast, Z pivot and X
   pivot are hidden); unlocking brings them back and hides the balance. */
static void _chroma_set_visibility(dt_iop_3dcf_gui_data_t *g, const gboolean linked)
{
  gtk_widget_set_visible(g->chroma_x_pivot, !linked);
  gtk_widget_set_visible(g->chroma_z_contrast, !linked);
  gtk_widget_set_visible(g->chroma_z_pivot, !linked);
  gtk_widget_set_visible(g->chroma_balance, linked);
  /* While locked, X alone drives both axes: drop the axis qualifier from the
     X contrast label (and restore it on unlock). */
  dt_bauhaus_widget_set_label(g->chroma_x_contrast, NULL,
                              linked ? _("chroma contrast") : _("X-axis chroma contrast"));
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;

  dt_bauhaus_slider_set(g->contrast, p->contrast);
  dt_bauhaus_slider_set(g->contrast_pivot, p->contrast_pivot);
  dt_bauhaus_slider_set(g->shoulder_power, p->shoulder_power);
  dt_bauhaus_slider_set(g->toe_power, p->toe_power);
  dt_bauhaus_slider_set(g->input_exposure, p->input_exposure);
  dt_bauhaus_slider_set(g->peak_luminance, p->peak_luminance);
  dt_bauhaus_slider_set(g->mid_tone, p->gamma);
  dt_bauhaus_slider_set(g->vibrance, p->vibrance);
  dt_bauhaus_slider_set(g->chromatic_boost, p->chromatic_boost);
  dt_bauhaus_slider_set(g->hl_desaturation, p->hl_desaturation);
  dt_bauhaus_slider_set(g->hl_desat_threshold, p->hl_desat_threshold);
  dt_bauhaus_slider_set(g->hl_hue_shift, p->hl_hue_shift);
  dt_bauhaus_slider_set(g->chroma_x_contrast, p->chroma_x_contrast);
  dt_bauhaus_slider_set(g->chroma_x_pivot, p->chroma_x_pivot);
  dt_bauhaus_slider_set(g->chroma_z_contrast, p->chroma_z_contrast);
  dt_bauhaus_slider_set(g->chroma_z_pivot, p->chroma_z_pivot);
  /* The X/Z balance is transient: recompute it from the stored contrasts. */
  dt_bauhaus_slider_set(g->chroma_balance,
                        _chroma_balance_from_xz(p->chroma_x_contrast, p->chroma_z_contrast));
  dt_bauhaus_slider_set(g->hl_detail_recovery, p->hl_detail_recovery);
  dt_bauhaus_slider_set(g->gamut_knee, p->gamut_knee);
  dt_bauhaus_slider_set(g->gamut_steepness, p->gamut_steepness);
  dt_bauhaus_combobox_set(g->color_space, p->output_cs);
  dt_bauhaus_combobox_set(g->color_look, p->color_look);
  dt_bauhaus_slider_set(g->look_opacity, p->look_opacity);

  /* The grey-point picker is transient module state, not a parameter (see
     the note at picked_grey_y's declaration): it must not survive an image
     change, or the auto-adjust would anchor on a luminance picked from a
     different photo.

     gui_update() fires far more often than on image change alone, so an
     unconditional reset here would clear the picked value almost
     immediately, making the picker unusable. Reset only when the displayed
     image actually changed (see last_imgid). */
  const dt_imgid_t current_imgid = self->dev->image_storage.id;
  if(g->last_imgid != current_imgid)
  {
    g->picked_grey_y = 0.0f;
    g->picked_grey_y_last_applied = 0.0f;
    /* The X/Z lock is a GUI preference, kept across image changes. */
    dt_iop_color_picker_reset(self, TRUE);
    g->last_imgid = current_imgid;
  }

  dt_bauhaus_widget_set_quad_active(g->chroma_x_contrast, g->chroma_linked);
  _chroma_set_visibility(g, g->chroma_linked);

  gui_changed(self, NULL, NULL);
}

/* ========================================================================
   Curve drawing callback -- calibrated on AGX pattern
   ======================================================================== */
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
      float _y = dt_st_ssts_fwd(&ctx.ssts, y_scene * ctx.exposure_factor) / (float)ctx.ssts.n_r;
      powf(fmaxf(_y, 0.0f), 1.0f / 2.4f);
    }));

  /* 2. SSTS + contrast S-curve (medium) */
  DRAW_CURVE(1.5f,
    darktable.bauhaus->graph_fg.red,
    darktable.bauhaus->graph_fg.green,
    darktable.bauhaus->graph_fg.blue, 0.55f,
    dt_st_compute_y_tm(y_scene, &ctx));

  /* 3. Full curve including gamma (solid) */
  DRAW_CURVE(2.0f,
    darktable.bauhaus->graph_fg_active.red,
    darktable.bauhaus->graph_fg_active.green,
    darktable.bauhaus->graph_fg_active.blue, 1.0f,
    ({
      float _y = dt_st_compute_y_tm(y_scene, &ctx);
      if(ctx.gamma != 0.0f)
        _y = powf(fminf(fmaxf(_y, 0.0f), 1.0f), ctx.gamma_power);
      _y;
    }));

  #undef DRAW_CURVE

  /* helper: compute tone-mapped Y with gamma (full curve) */
  #define TONE_MAP(ys) ({ \
    float _y = dt_st_compute_y_tm((ys), &ctx); \
    if(ctx.gamma != 0.0f) _y = powf(fminf(fmaxf(_y, 0.0f), 1.0f), ctx.gamma_power); \
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

/* Chroma link padlock: when active, X and Z axes track each other and the
   Z sliders are replaced by the X/Z balance slider. */
static void _chroma_link_toggled(GtkWidget *quad, dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;

  /* bauhaus already toggled CPF_ACTIVE before emitting quad-pressed. */
  g->chroma_linked = dt_bauhaus_widget_get_quad_active(quad);

  /* Locking preserves the current X and Z values: recompute the transient
     balance slider from them instead of copying X → Z and resetting the
     asymmetry to zero (which would drop an existing X/Z split). */
  if(g->chroma_linked && !g->chroma_syncing)
  {
    g->chroma_syncing = TRUE;
    dt_bauhaus_slider_set(g->chroma_balance,
                          _chroma_balance_from_xz(p->chroma_x_contrast, p->chroma_z_contrast));
    g->chroma_syncing = FALSE;
  }

  _chroma_set_visibility(g, g->chroma_linked);
}

static void _chroma_x_changed(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;
  (void)slider;

  if(!g->chroma_linked || g->chroma_syncing) return;
  g->chroma_syncing = TRUE;
  /* Preserve the X/Z balance while X follows the slider: Z is derived from
     the new X with the same asymmetry. The stored values are pre-written so
     the (hidden) Z sliders do not add another history step. */
  p->chroma_z_contrast = _chroma_z_from_x(p->chroma_x_contrast,
                                          dt_bauhaus_slider_get(g->chroma_balance));
  dt_bauhaus_slider_set(g->chroma_z_contrast, p->chroma_z_contrast);
  p->chroma_z_pivot = p->chroma_x_pivot;
  dt_bauhaus_slider_set(g->chroma_z_pivot, p->chroma_z_pivot);
  g->chroma_syncing = FALSE;
}

static void _chroma_z_changed(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;
  (void)slider;

  if(!g->chroma_linked || g->chroma_syncing) return;
  g->chroma_syncing = TRUE;
  p->chroma_x_contrast = _chroma_x_from_z(p->chroma_z_contrast,
                                          dt_bauhaus_slider_get(g->chroma_balance));
  dt_bauhaus_slider_set(g->chroma_x_contrast, p->chroma_x_contrast);
  p->chroma_x_pivot = p->chroma_z_pivot;
  dt_bauhaus_slider_set(g->chroma_x_pivot, p->chroma_x_pivot);
  g->chroma_syncing = FALSE;
}

/* Balance slider: pans the shared contrast level between the X and Z axes.
   base = (X+Z)/2 stays constant; X = base*(1+b), Z = base*(1-b). Only X is
   driven here -- _chroma_x_changed() propagates the derived Z, so the two
   axes stay coherent with a single history step. */
static void _chroma_balance_changed(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = self->gui_data;
  dt_iop_3dcf_params_t *p = self->params;
  (void)slider;

  if(g->chroma_syncing) return;

  /* The balance slider is only shown while linked. */
  if(!g->chroma_linked)
  {
    dt_bauhaus_slider_set(g->chroma_balance,
                          _chroma_balance_from_xz(p->chroma_x_contrast, p->chroma_z_contrast));
    return;
  }

  const float b = dt_bauhaus_slider_get(g->chroma_balance);
  const float base = 0.5f * (p->chroma_x_contrast + p->chroma_z_contrast);
  const float newx = CLAMPF(base * (1.0f + b), 0.0f, 10.0f);
  dt_bauhaus_slider_set(g->chroma_x_contrast, newx);
}

/* Reset quad on the X/Z balance slider: restores the four X/Z chroma sliders
   (contrast and pivot on both axes) to their defaults. The padlock state
   (chroma_linked) is a GUI preference and is deliberately left untouched, so
   the user keeps their locked/unlocked choice. A single undoable history step
   records the reset. */
static void _chroma_reset_clicked(GtkWidget *quad, dt_iop_module_t *self)
{
  dt_iop_3dcf_params_t *p = self->params;
  (void)quad;

  if(p->chroma_x_contrast == 0.0f && p->chroma_z_contrast == 0.0f
     && p->chroma_x_pivot == 50.0f && p->chroma_z_pivot == 50.0f)
    return;

  p->chroma_x_contrast = 0.0f;
  p->chroma_z_contrast = 0.0f;
  p->chroma_x_pivot = 50.0f;
  p->chroma_z_pivot = 50.0f;

  dt_iop_gui_update(self);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_3dcf_gui_data_t *g = IOP_GUI_ALLOC(3dcf);
  self->gui_data = g;
  /* X and Z are locked together by default. */
  g->chroma_linked = TRUE;

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

  /* Preset combobox and live picker, in the SAME widget: the scene
     measurements (exposure, peak_luminance, pivot) are identical whatever
     the chosen preset, only the contrast rendering depends on it (see
     st_auto_presets[]). The combobox only selects the preset -- it
     triggers nothing on its own; it's the picker, dragged over the image,
     that computes and applies live (see color_picker_apply()).

     The picker is attached to the combobox via its "quad" icon, exactly
     like g->exposure in iop/exposure.c (dt_color_picker_new() wraps the
     widget and adds the icon rather than creating a separate widget). Every
     bauhaus widget reserves the same quad-icon column internally whether it
     shows an icon there or not, so this keeps the row the same width as the
     sliders below it. */
  g->auto_mode_combo = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, dt_bauhaus_combobox_new(self));
  dt_bauhaus_widget_set_label(g->auto_mode_combo, NULL, N_("auto-adjust"));
  dt_bauhaus_combobox_add(g->auto_mode_combo, _("contrasty"));
  dt_bauhaus_combobox_add(g->auto_mode_combo, _("neutral"));
  dt_bauhaus_combobox_add(g->auto_mode_combo, _("soft"));
  dt_bauhaus_combobox_set(g->auto_mode_combo, DT_ST_AUTO_NEUTRAL);
  gtk_widget_set_tooltip_text(g->auto_mode_combo,
    _("Contrast rendering used by the grey-point picker (icon on the right).\n"
      "Selecting an entry here does nothing on its own. While the picker\n"
      "is active, switching entries re-applies immediately on the same\n"
      "picked area, so you can compare renderings without re-picking."));
  dt_bauhaus_widget_set_quad_tooltip(g->auto_mode_combo,
    _("Click, then drag over a mid-grey reference in the image: tone\n"
      "settings analyze and apply live as you move the picker, using the\n"
      "contrast rendering selected above. Click again to stop.\n"
      "Useful on wide dynamic range scenes, where the automatic estimate\n"
      "can be pulled off by large bright or dark areas."));
  g_signal_connect(G_OBJECT(g->auto_mode_combo), "value-changed", G_CALLBACK(_auto_mode_changed), self);
  dt_gui_box_add(main_vbox, g->auto_mode_combo);

  g->input_exposure = dt_bauhaus_slider_from_params(self, "input_exposure");
  dt_bauhaus_slider_set_format(g->input_exposure, _(" EV"));
  dt_bauhaus_slider_set_digits(g->input_exposure, 2);
  gtk_widget_set_tooltip_text(g->input_exposure,
    _("Pre-SSTS exposure in EV steps. Positive values brighten the image \n"
      "and push more signal into the SSTS. Negative values darken it. \n"
      "0 EV = no compensation (ACES 2.0 reference)."));

  g->peak_luminance = dt_bauhaus_slider_from_params(self, "peak_luminance");
  dt_bauhaus_slider_set_format(g->peak_luminance, _("%"));
  gtk_widget_set_tooltip_text(g->peak_luminance,
    _("Peak luminance: controls SSTS peak luminance via the tone scale character. \n"
      "0% = 200 nits (ACES 2.0 reference). -100% = 0 nits (soft), +100% = 400 nits.\n"
      "Lower values give more highlight headroom with a gentler roll-off;\n"
      "higher values clip earlier."));

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

  g->mid_tone = dt_bauhaus_slider_from_params(self, "gamma");
  dt_bauhaus_slider_set_factor(g->mid_tone, 100.0f);
  dt_bauhaus_slider_set_format(g->mid_tone, " %");
  dt_bauhaus_slider_set_digits(g->mid_tone, 0);
  gtk_widget_set_tooltip_text(g->mid_tone,
    _("Global gamma adjustment of the tone curve. \n"
      "Moving to the right (+100%) brightens the image, \n"
      "moving to the left (-100%) darkens it."));

  /* === COLOR section === */
  dt_gui_box_add(GTK_BOX(main_vbox), dt_ui_section_label_new(C_("section", "color")));

  g->color_look = dt_bauhaus_combobox_from_params(self, "color_look");
  gtk_widget_set_tooltip_text(g->color_look, _("Apply a color style to the image."));

  g->look_opacity = dt_bauhaus_slider_from_params(self, "look_opacity");
  dt_bauhaus_widget_set_label(g->look_opacity, NULL, _("look opacity"));
  dt_bauhaus_slider_set_format(g->look_opacity, "%");
  dt_bauhaus_slider_set_factor(g->look_opacity, 100.0);
  gtk_widget_set_tooltip_text(g->look_opacity, _("Adjust the strength of the selected color style."));

  g->vibrance = dt_bauhaus_slider_from_params(self, "vibrance");
  dt_bauhaus_slider_set_factor(g->vibrance, 100.0f);
  dt_bauhaus_slider_set_offset(g->vibrance, -100.0f);
  dt_bauhaus_slider_set_format(g->vibrance, " %");
  dt_bauhaus_slider_set_digits(g->vibrance, 0);
  gtk_widget_set_tooltip_text(g->vibrance,
    _("Smart saturation boost. Protects already-saturated colors while enhancing pastels."));

  g->chromatic_boost = dt_bauhaus_slider_from_params(self, "chromatic_boost");
  dt_bauhaus_slider_set_factor(g->chromatic_boost, 100.0f);
  dt_bauhaus_slider_set_format(g->chromatic_boost, " %");
  dt_bauhaus_slider_set_digits(g->chromatic_boost, 0);
  gtk_widget_set_tooltip_text(g->chromatic_boost,
    _("Chromatic accentuation of midtones with a slight contrast in hue in the highlights"));

  g->chroma_x_contrast = dt_bauhaus_slider_from_params(self, "chroma_x_contrast");
  /* Display as a percentage of the axis' chroma range: the stored 0..10 maps
     to 0..100%, e.g. 2.02 shows as 20.2%. The parameter is unchanged. */
  dt_bauhaus_slider_set_factor(g->chroma_x_contrast, 10.0f);
  dt_bauhaus_slider_set_format(g->chroma_x_contrast, " %");
  dt_bauhaus_slider_set_digits(g->chroma_x_contrast, 1);
  gtk_widget_set_tooltip_text(g->chroma_x_contrast,
    _("Independent contrast on the CIE X-axis chroma offset from white, applied \n"
      "right after chromaticity scaling and before spectral gamut roll-off. \n"
      "0 % = off (no-op)."));
  /* Padlock quad: locks X and Z axes together */
  dt_bauhaus_widget_set_quad(g->chroma_x_contrast, self, dtgtk_cairo_paint_lock,
                             TRUE, _chroma_link_toggled,
     _("Lock X and Z axes: when active, both contrast and pivot track together."));
  g_signal_connect_after(G_OBJECT(g->chroma_x_contrast), "value-changed",
                         G_CALLBACK(_chroma_x_changed), self);

  g->chroma_x_pivot = dt_bauhaus_slider_from_params(self, "chroma_x_pivot");
  dt_bauhaus_slider_set_default(g->chroma_x_pivot, 50.0f);
  /* Display relative to the symmetric 50% position: 50% shows as 0%, a 40%
     pivot as -10%, a 60% one as +10%. The stored parameter is unchanged. */
  dt_bauhaus_slider_set_offset(g->chroma_x_pivot, -50.0f);
  dt_bauhaus_slider_set_format(g->chroma_x_pivot, " %");
  dt_bauhaus_slider_set_digits(g->chroma_x_pivot, 1);
  gtk_widget_set_tooltip_text(g->chroma_x_pivot,
    _("Pivot point for the X-axis chroma contrast sigmoid, relative to the \n"
      "symmetric position (0% = symmetric, the stored 50%)."));
  g_signal_connect_after(G_OBJECT(g->chroma_x_pivot), "value-changed",
                         G_CALLBACK(_chroma_x_changed), self);
  /* Reset quad on the X pivot slider: same mechanism as the padlock and the
     balance reset quad, so it is exactly the same size and sits at the right
     edge of the slider. The X pivot is only shown while unlocked, whereas the
     balance (with its own reset quad) is only shown while locked, so together
     the reset icon is always available. */
  dt_bauhaus_widget_set_quad(g->chroma_x_pivot, self, dtgtk_cairo_paint_reset,
                             FALSE, _chroma_reset_clicked,
                             _("Reset the X/Z chroma sliders (contrast and pivot) to their default values."));

  /* X/Z balance: transient slider (not a module parameter), shown only while
     X and Z are locked. It pans the shared contrast level between the two
     axes; the effective X/Z contrast values are written into the existing
     chroma_x/z_contrast parameters. */
  g->chroma_balance = dt_bauhaus_slider_new_with_range(self, -1.0f, 1.0f, 0.01f, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->chroma_balance, NULL, _("X/Z balance"));
  dt_bauhaus_slider_set_factor(g->chroma_balance, 100.0f);
  dt_bauhaus_slider_set_format(g->chroma_balance, " %");
  dt_bauhaus_slider_set_digits(g->chroma_balance, 0);
  gtk_widget_set_tooltip_text(g->chroma_balance,
    _("Split the chroma contrast between the X and Z axes around their average.\n"
      "0% = symmetric (X = Z). Negative favors Z, positive favors X.\n"
      "Only available while X and Z are locked together (padlock)."));
  g_signal_connect(G_OBJECT(g->chroma_balance), "value-changed",
                   G_CALLBACK(_chroma_balance_changed), self);
  /* Reset quad on the balance slider (same mechanism as the padlock quad on
     the X contrast slider, so it is exactly the same size and sits at the
     right edge of the slider): restores the four X/Z chroma sliders to their
     defaults. As it lives inside the balance widget, it shares the same
     visibility (shown only while the axes are locked). */
  dt_bauhaus_widget_set_quad(g->chroma_balance, self, dtgtk_cairo_paint_reset,
                             FALSE, _chroma_reset_clicked,
                             _("Reset the X/Z chroma sliders (contrast and pivot) to their default values."));
  dt_gui_box_add(main_vbox, g->chroma_balance);

  g->chroma_z_contrast = dt_bauhaus_slider_from_params(self, "chroma_z_contrast");
  /* Display as a percentage of the axis' chroma range (see chroma_x_contrast). */
  dt_bauhaus_slider_set_factor(g->chroma_z_contrast, 10.0f);
  dt_bauhaus_slider_set_format(g->chroma_z_contrast, " %");
  dt_bauhaus_slider_set_digits(g->chroma_z_contrast, 1);
  gtk_widget_set_tooltip_text(g->chroma_z_contrast,
    _("Independent contrast on the CIE Z-axis chroma offset from white, applied \n"
      "right after chromaticity scaling and before spectral gamut roll-off. \n"
      "0 % = off (no-op)."));
  g_signal_connect_after(G_OBJECT(g->chroma_z_contrast), "value-changed",
                         G_CALLBACK(_chroma_z_changed), self);

  g->chroma_z_pivot = dt_bauhaus_slider_from_params(self, "chroma_z_pivot");
  dt_bauhaus_slider_set_default(g->chroma_z_pivot, 50.0f);
  /* Display relative to the symmetric 50% position (see chroma_x_pivot). */
  dt_bauhaus_slider_set_offset(g->chroma_z_pivot, -50.0f);
  dt_bauhaus_slider_set_format(g->chroma_z_pivot, " %");
  dt_bauhaus_slider_set_digits(g->chroma_z_pivot, 1);
  gtk_widget_set_tooltip_text(g->chroma_z_pivot,
    _("Pivot point for the Z-axis chroma contrast sigmoid, relative to the \n"
      "symmetric position (0% = symmetric, the stored 50%)."));
  g_signal_connect_after(G_OBJECT(g->chroma_z_pivot), "value-changed",
                         G_CALLBACK(_chroma_z_changed), self);

  /* Abney rotation, kept at the very bottom of the color section. */
  g->hl_hue_shift = dt_bauhaus_slider_from_params(self, "hl_hue_shift");
  dt_bauhaus_slider_set_factor(g->hl_hue_shift, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_hue_shift, " %");
  dt_bauhaus_slider_set_digits(g->hl_hue_shift, 0);
  gtk_widget_set_tooltip_text(g->hl_hue_shift,
    _("Abney rotation in highlights, tied to the SSTS compression ratio. \n"
      "Positive rotates toward cool (blue), negative toward warm (salmon). \n"
      "Ramps up progressively as highlights compress (power 0.6), up to ~23 deg \n"
      "on clipped highlights; midtones and shadows stay untouched. \n"
      "Independent of highlight roll-off."));

  /* Apply the initial show/hide for the lock-dependent chroma widgets. */
  _chroma_set_visibility(g, g->chroma_linked);

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

/* NOTE: no gui_cleanup() is needed. IOP_GUI_ALLOC() allocates gui_data via
   dt_calloc_aligned() -> dt_alloc_aligned(), which on Windows is
   _aligned_malloc() (and posix_memalign() on Linux/macOS). The framework
   already frees it for us in dt_iop_gui_cleanup_module() with the matching
   dt_free_align(). Freeing it here with free()/g_free() would release an
   _aligned_malloc() block with the wrong deallocator -- heap corruption
   (c0000374) at module load on Windows. The gui_data struct only holds
   GtkWidget* and an embedded collapsible_section (no separately-owned
   resources), so the default widget destruction is sufficient. */

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
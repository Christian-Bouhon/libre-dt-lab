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

    - Mallett2019 spectral basis : spectral reconstruction from RGB values
      using precomputed basis functions for accurate spectral rendering.
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

#include "spectral_tone/spectral_tone.h"

DT_MODULE_INTROSPECTION(2, dt_iop_spectral_tone_params_t)

typedef enum dt_iop_st_colorspace_t
{
  DT_ST_CS_REC709 = 0,    // $DESCRIPTION: "Rec. 709"
  DT_ST_CS_REC2020,       // $DESCRIPTION: "Rec. 2020"
  DT_ST_CS_DISPLAYP3,     // $DESCRIPTION: "Display P3"
  DT_ST_CS_PROPHOTO,      // $DESCRIPTION: "ProPhoto RGB"
  DT_ST_CS_ADOBERGB,      // $DESCRIPTION: "Adobe RGB"
} dt_iop_st_colorspace_t;

typedef struct dt_iop_spectral_tone_params_t
{
  float contrast;              // $MIN: 0.25 $MAX: 4.25 $DEFAULT: 2.25 $DESCRIPTION: "contrast"
  float spectral_brilliance;   // $MIN: 0 $MAX: 100 $DEFAULT: 5 $DESCRIPTION: "spectral brilliance"
  float gray_point;            // $MIN: -1 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "mid-tone"
  float vibrance;              // $MIN: 0 $MAX: 2 $DEFAULT: 1.0 $DESCRIPTION: "vibrance"
  float hl_desaturation;       // $MIN: 0 $MAX: 2 $DEFAULT: 1 $DESCRIPTION: "HL desaturation"
  float hl_hue_shift;          // $MIN: -1 $MAX: 1 $DEFAULT: 0 $STEP: 0.01 $DESCRIPTION: "HL hue shift"
  float gamut_knee;            // $MIN: 0 $MAX: 1 $DEFAULT: 0.25 $DESCRIPTION: "gamut knee"
  float gamut_steepness;       // $MIN: 0 $MAX: 1 $DEFAULT: 0.25 $DESCRIPTION: "gamut steepness"
  dt_iop_st_colorspace_t output_cs;  // $DEFAULT: DT_ST_CS_REC2020 $DESCRIPTION: "color space"
} dt_iop_spectral_tone_params_t;

typedef struct dt_iop_spectral_tone_data_t
{
  dt_iop_spectral_tone_params_t params;
  dt_st_context_t ctx;
} dt_iop_spectral_tone_data_t;

typedef struct dt_iop_spectral_tone_gui_data_t
{
  GtkWidget *contrast;
  GtkWidget *spectral_brilliance;
  GtkWidget *mid_tone;
  GtkWidget *vibrance;
  GtkWidget *hl_desaturation;
  GtkWidget *hl_hue_shift;
  GtkWidget *gamut_knee;
  GtkWidget *gamut_steepness;
  GtkWidget *color_space;
} dt_iop_spectral_tone_gui_data_t;

/* Conversion matrices */
static const float st_luma_rec709[3]    = { 0.2126f,  0.7152f,  0.0722f };
static const float st_luma_rec2020[3]   = { 0.2627f,  0.6780f,  0.0593f };
static const float st_luma_displayp3[3] = { 0.2289f,  0.6918f,  0.0793f };
static const float st_luma_prophoto[3]  = { 0.2880f,  0.7119f,  0.0001f };
static const float st_luma_adobergb[3]  = { 0.2973f,  0.6274f,  0.0753f };

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
    mat3inv(ctx->output_matrix, M_in_ws);
  }

  /* Luma coefficients matching the output color space */
  const float *lc = st_get_luma_coeff(p->output_cs);
  ctx->luma_coeff[0] = lc[0];
  ctx->luma_coeff[1] = lc[1];
  ctx->luma_coeff[2] = lc[2];

  ctx->contrast = fmaxf(p->contrast, 0.001f);
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

  /* Mid-tone gamma adjustment */
  ctx->gray_point = fmaxf(fminf(p->gray_point, 1.0f), -1.0f);
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
  if(old_version == 1)
  {
    dt_iop_spectral_tone_params_t *p = calloc(1, sizeof(dt_iop_spectral_tone_params_t));
    memcpy(p, old_params, 6 * sizeof(float)); // copie contrast -> hl_hue_shift
    p->gamut_knee = 0.25f;
    p->gamut_steepness = 0.25f;
    p->output_cs = ((const int *)old_params)[6]; // l'enum était le 7ème champ
    *new_params = p;
    *new_params_size = sizeof(dt_iop_spectral_tone_params_t);
    *new_version = 2;
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
  return _("tone mapping|spectral|highlight reconstruction");
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

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_alloc1_align_type(dt_iop_spectral_tone_data_t);
  dt_iop_spectral_tone_data_t *d = piece->data;
  memset(&d->ctx, 0, sizeof(dt_st_context_t)); // Ensure context is initialized
  memcpy(&d->params, self->default_params, sizeof(dt_iop_spectral_tone_params_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
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
  tiling->factor = 1.0f;
  tiling->factor_cl = 1.0f;
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

  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;

  #ifdef _OPENMP
  #pragma omp parallel for default(none) \
    shared(in, out, width, height, ch, d, npixels)
  #endif
  for(size_t k = 0; k < npixels; k++)
  {
    const size_t idx = k * ch;
    float rgb_in[3] = { in[idx], in[idx + 1], in[idx + 2] };
    float rgb_out[3];
    dt_st_pipeline_eval(rgb_in, rgb_out, &d->ctx);
    out[idx]     = rgb_out[0];
    out[idx + 1] = rgb_out[1];
    out[idx + 2] = rgb_out[2];
    if(ch == 4) out[idx + 3] = in[idx + 3];
  }
}

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
  p.hl_desaturation = 1.0f;
  p.hl_hue_shift = 0.0f;
  p.gamut_knee = 0.25f;
  p.gamut_steepness = 0.25f;
  p.output_cs = DT_ST_CS_REC2020;

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
  dt_bauhaus_slider_set(g->spectral_brilliance, p->spectral_brilliance);
  dt_bauhaus_slider_set(g->mid_tone, p->gray_point);
  dt_bauhaus_slider_set(g->vibrance, p->vibrance);
  dt_bauhaus_slider_set(g->hl_desaturation, p->hl_desaturation);
  dt_bauhaus_slider_set(g->hl_hue_shift, p->hl_hue_shift);
  dt_bauhaus_slider_set(g->gamut_knee, p->gamut_knee);
  dt_bauhaus_slider_set(g->gamut_steepness, p->gamut_steepness);
  dt_bauhaus_combobox_set(g->color_space, p->output_cs);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_spectral_tone_gui_data_t *g = IOP_GUI_ALLOC(spectral_tone);
  self->gui_data = g;

  g->contrast = dt_bauhaus_slider_from_params(self, "contrast");
  dt_bauhaus_slider_set_factor(g->contrast, 50.0f);
  dt_bauhaus_slider_set_offset(g->contrast, -112.5f);
  dt_bauhaus_slider_set_format(g->contrast, " %");
  dt_bauhaus_slider_set_digits(g->contrast, 0);
  gtk_widget_set_tooltip_text(g->contrast,
    _("S-curve contrast pivoted at mid-gray. -100% = minimum contrast, "
      "0% = neutral, +100% = maximum. Negative values soften, positive values sharpen."));

  g->spectral_brilliance = dt_bauhaus_slider_from_params(self, "spectral_brilliance");
  dt_bauhaus_slider_set_format(g->spectral_brilliance, "%");
  gtk_widget_set_tooltip_text(g->spectral_brilliance,
    _("Tone curve character with auto-exposure compensation. Higher values increase "
      "highlight headroom with a softer, film-like rolloff. Brightness is automatically "
      "stabilised across the full range."));

  g->mid_tone = dt_bauhaus_slider_from_params(self, "gray_point");
  dt_bauhaus_slider_set_factor(g->mid_tone, 100.0f);
  dt_bauhaus_slider_set_format(g->mid_tone, " %");
  dt_bauhaus_slider_set_digits(g->mid_tone, 0);
  gtk_widget_set_tooltip_text(g->mid_tone,
    _("Mid-tone brightness adjustment. -100% = darker mid-tones, "
      "0% = neutral, +100% = brighter mid-tones."));

  g->vibrance = dt_bauhaus_slider_from_params(self, "vibrance");
  dt_bauhaus_slider_set_factor(g->vibrance, 100.0f);
  dt_bauhaus_slider_set_offset(g->vibrance, -100.0f);
  dt_bauhaus_slider_set_format(g->vibrance, " %");
  dt_bauhaus_slider_set_digits(g->vibrance, 0);
  gtk_widget_set_tooltip_text(g->vibrance,
    _("Smart saturation boost. Protects already-saturated colors while enhancing pastels."));

  g->hl_desaturation = dt_bauhaus_slider_from_params(self, "hl_desaturation");
  dt_bauhaus_slider_set_factor(g->hl_desaturation, 100.0f);
  dt_bauhaus_slider_set_offset(g->hl_desaturation, -100.0f);
  dt_bauhaus_slider_set_format(g->hl_desaturation, " %");
  dt_bauhaus_slider_set_digits(g->hl_desaturation, 0);
  gtk_widget_set_tooltip_text(g->hl_desaturation,
    _("Desaturates highlights toward achromatic luma to prevent out-of-gamut colors. "
      "-100% = off, 0% = natural rolloff, +100% = maximum desaturation."));

  g->hl_hue_shift = dt_bauhaus_slider_from_params(self, "hl_hue_shift");
  dt_bauhaus_slider_set_factor(g->hl_hue_shift, 100.0f);
  dt_bauhaus_slider_set_format(g->hl_hue_shift, " %");
  dt_bauhaus_slider_set_digits(g->hl_hue_shift, 0);
  gtk_widget_set_tooltip_text(g->hl_hue_shift,
    _("Abney hue shift in highlights. Positive rotates toward cool (blue), "
      "negative toward warm (salmon). Independent of desaturation strength."));

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
}

void gui_cleanup(dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_spectral_tone_gui_data_t *g = self->gui_data;
  (void)g;
  (void)w;
  (void)previous;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
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

#ifndef DT_SPECTRAL_TONE_H
#define DT_SPECTRAL_TONE_H

/* Precomputed SSTS (ACES 2.0 Single-Stage Tone Scale) parameters */
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
  float contrast;          /* post-SSTS contrast S-curve */
  float contrast_pivot;    /* pivot for contrast S-curve */
  float input_matrix[9];   /* D50 Rec.2020 RGB → D50 XYZ (CAT_adapted) */
  float output_matrix[9];  /* D50 XYZ → D50 output RGB (CAT_adapted) */
  float luma_coeff[3];     /* luma coefficients for output space */
  float hl_desat;          /* highlight desaturation amount */
  float hl_rotation;       /* Abney hue rotation (−1..+1, cold/warm) */
  float white_chroma_x;    /* D50 white chromaticity x = X/Y */
  float white_chroma_z;    /* D50 white chromaticity z = Z/Y */
  float gray_point;        /* mid-tone gamma adjustment */
  float vibrance;          /* vibrance (saturation with high-sat protection) */
  float gamut_knee;        /* gamut compression knee point */
  float gamut_steepness;   /* gamut compression steepness */
  dt_st_ssts_params_t ssts; /* ACES 2.0 SSTS precomputed params */
} dt_st_context_t;

extern const double dt_st_rec2020_to_xyz[9];
extern const double dt_st_cat_d65_to_d50[9];

/* Init SSTS params for a given peak luminance (nits) */
void dt_st_ssts_init(dt_st_ssts_params_t *p, double peak_luminance);

/* Forward SSTS tone scale: scene-linear x → display luminance (cd/m²) */
double dt_st_ssts_fwd(const dt_st_ssts_params_t *p, double x);

/* Compute tone-mapped Y from scene-linear Y (SSTS + BT.1886 + contrast) */
double dt_st_compute_y_tm(double y_scene, const dt_st_context_t *ctx);

/* Pipeline function */
void dt_st_pipeline_eval(const float rgb_in[3], float rgb_out[3],
                          const dt_st_context_t *ctx);

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
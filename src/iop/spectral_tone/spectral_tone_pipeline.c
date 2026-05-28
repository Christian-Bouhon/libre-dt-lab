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

#include <math.h>
#include "spectral_tone.h"

/* ACES 2.0 SSTS (Single-Stage Tone Scale) — official ACES 2.0 RRT tone scale
 *
 * Parametric MM (Michaelis-Menten) curve with flare compensation.
 * Reference: aces-core/lib/Lib.Academy.Tonescale.ctl
 *
 *   f = m_2 * (x / (x + s_2))^g
 *   h = f^2 / (f + t_1)
 *   Y = h * n_r          (output: display luminance in cd/m²)
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

  /* Display luminance (cd/m²) */
  return h * p->n_r;
}

/* Compute tone-mapped Y from scene-linear Y (SSTS + BT.1886 + contrast) */
double dt_st_compute_y_tm(double y_scene, const dt_st_context_t *ctx)
{
  double y_disp = dt_st_ssts_fwd(&ctx->ssts, y_scene * ctx->exposure_factor);
  double y_tm = y_disp / ctx->ssts.n;

  /* BT.1886 OETF */
  y_tm = pow(fmax(y_tm, 0.0), 1.0 / 2.4);

  /* Post-SSTS contrast S-curve pivoted at mid-gray (0.5) */
  if(ctx->contrast != 1.0f)
  {
    const double c = (double)ctx->contrast;
    const double p = 0.5;
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
  if(hl_desat <= 0.0f || y_norm <= 0.7) return 0.0;
  const double t = fmax(y_norm - 0.7, 0.0) / y_norm; /* 0 at threshold, ~0.3 at white, approaches 1 asymptotically */
  const double x = fmin(t * (double)hl_desat, 1.0);
  return x * x; /* quadratic ramp */
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

  /* Convert current XYZ to CIE xy chromaticity */
  const double sum = *x_tm + y_tm + *z_tm;
  if(sum <= 0.0) return;
  const double cie_x = *x_tm / sum;
  const double cie_z = *z_tm / sum;  /* z = 1 - x - y, we don't need y directly */

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

  /* Knee: chromaticity distance below which no compression occurs.
   * In CIE xy space, natural colors (skin, grass, sky) have chroma < 0.25,
   * saturated primaries reach ~0.37-0.40. Knee=0.25 protects natural colors
   * while compressing very saturated/artificial colors beyond Rec.2020 gamut.
   * Steepness: higher = more aggressive compression. */

  if(chroma_sq > knee * knee)
  {
    const double chroma = sqrt(chroma_sq);
    const double excess = chroma - knee;
    const double compression = excess / (excess + steepness);
    const double scale = (chroma - compression * excess) / chroma;

    /* New chromaticity in CIE xy, then back to XYZ preserving Y */
    const double x_new = white_cie_x + scale * dx;
    const double z_new = white_cie_z + scale * dz;
    const double y_new = 1.0 - x_new - z_new;  /* CIE y */

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
 *   1. D50-adapted Rec.2020 RGB → D50 XYZ (precise matrix)
 *   2. ACES 2.0 SSTS on luminance Y only (spectral tone scale)
 *   3. BT.1886 OETF + contrast S-curve
 *   4. Mid-tone gamma adjustment
 *   5. Chromaticity ratio scaling: x = ratio * Y
 *   6. Spectral gamut: film-like chromaticity roll-off in CIE xy
  *   7. XYZ → output RGB via output matrix
  *   8. Highlight desaturation (blend toward achromatic luma)
  *   9. Vibrance (saturation with high-sat protection)
  *  10. Gamut compression safety net
  *  11. Clamp to [0, ∞)
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

  if(y_abs <= 0.0)
  {
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return;
  }

  /* Chromaticity ratios */
  const double x_ratio = x_abs / y_abs;
  const double z_ratio = z_abs / y_abs;

  /* Step 2-3: Tone-mapped Y (SSTS + BT.1886 + contrast) */
  double y_tm = dt_st_compute_y_tm(y_abs, ctx);

  /* Step 4: Mid-tone adjustment — gamma pivot */
  if(ctx->gray_point != 0.0f)
  {
    float y_lvl = fminf(fmaxf((float)y_tm, 0.0f), 1.0f);
    y_lvl = powf(y_lvl, exp2f(ctx->gray_point));
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

  /* Step 7: XYZ → Output RGB via precomputed matrix */
  const float *M = ctx->output_matrix;
  float rgb[3];
  rgb[0] = M[0] * (float)x_tm + M[1] * (float)y_tm + M[2] * (float)z_tm;
  rgb[1] = M[3] * (float)x_tm + M[4] * (float)y_tm + M[5] * (float)z_tm;
  rgb[2] = M[6] * (float)x_tm + M[7] * (float)y_tm + M[8] * (float)z_tm;

  /* Step 8: Highlight desaturation with Abney hue correction
   *         Converts to YUV (using output luma coeffs), applies a constant
   *         hue rotation (correcting Abney effect), then desaturates toward
   *         luma. Rotation is independent of the desaturation weight so the
   *         hue shift is uniform across all luminance levels, fading only as
   *         chroma approaches zero (fully desaturated).
   *         Uses SSTS-exposed scene Y so the threshold is consistent. */
  {
    const double y_exposed = y_abs * ctx->exposure_factor;
    const double w = st_desat_weight(y_exposed, ctx->hl_desat);
    if(w > 0.0)
    {
      const float *lc = ctx->luma_coeff;
      const float y = lc[0] * rgb[0] + lc[1] * rgb[1] + lc[2] * rgb[2];
      float u = rgb[0] - y;
      float v = rgb[2] - y;

      /* Abney hue rotation (constant angle, independent of luminance) */
      const float angle = ctx->hl_rotation * 0.3f;
      if(angle != 0.0f)
      {
        const float ca = cosf(angle);
        const float sa = sinf(angle);
        const float ur = u * ca - v * sa;
        const float vr = u * sa + v * ca;
        u = ur;
        v = vr;
      }

      /* Desaturate toward luma (luminance-dependent) */
      u *= (float)(1.0 - w);
      v *= (float)(1.0 - w);

      /* YUV → RGB (G reconstructed from Y, U, V) */
      rgb[0] = y + u;
      rgb[2] = y + v;
      rgb[1] = y - (lc[0] / lc[1]) * u - (lc[2] / lc[1]) * v;
    }
  }

  /* Step 9: Vibrance — saturation with high-sat protection
   *         sat > 1: protects already-saturated colors (more gain for pastels)
   *         sat < 1: uniform desaturation
   * Protection is based on relative saturation sat_norm = sat_m / level,
   * which works correctly even when RGB values exceed 1.0 (super-white). */
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

  /* Step 10: Gamut compression — desaturate out-of-gamut colors toward white */
  st_gamut_compress(rgb);

  /* Step 11: Clamp negative channels (safety) */
  rgb_out[0] = fmaxf(rgb[0], 0.0f);
  rgb_out[1] = fmaxf(rgb[1], 0.0f);
  rgb_out[2] = fmaxf(rgb[2], 0.0f);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
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
*/

/*
 * Experimental green-guided red/blue residual interpolation for RCD.
 *
 * This is a CPU prototype inspired by the residual red/blue interpolation
 * described for the RapidRAW v1.7 ("Furka") pipeline. It is an independent
 * GPL reimplementation for darktable; no AGPL code was copied.
 *
 * The idea implemented here is to interpolate the colour differences
 * D = C - G at the known C sites with a small joint-bilateral filter guided by
 * the green plane. D is zero in neutral areas, so the correction removes the
 * false colour left by the directional RCD interpolation without touching
 * luminance. The green term rejects samples across luminance edges, while the
 * local variance of D provides a confidence that keeps the RCD estimate in
 * ambiguous areas. Known CFA sites are never modified.
 *
 * This is deliberately a conservative prototype: it is disabled by default and
 * only runs on the CPU for now.
 */

#include <math.h>

#define RCD_RB_RADIUS 4
#define RCD_RB_DIAMETER (2 * RCD_RB_RADIUS + 1)
#define RCD_RB_SIGMA 1.5f
#define RCD_RB_GSIGMA 0.02f
#define RCD_RB_SHOT 1e-4f
#define RCD_RB_EPS 1e-5f
#define RCD_RB_TINY 1e-20f

static void _rcd_residual_rb(float *const out,
                             const int width,
                             const int height,
                             const uint32_t filters)
{
  const size_t npix = (size_t)width * height;
  float kernel[RCD_RB_DIAMETER * RCD_RB_DIAMETER];
  float ksum = 0.0f;
  for(int dy = -RCD_RB_RADIUS; dy <= RCD_RB_RADIUS; dy++)
  {
    for(int dx = -RCD_RB_RADIUS; dx <= RCD_RB_RADIUS; dx++)
    {
      const float v = expf(-0.5f * (float)(dx * dx + dy * dy) / (RCD_RB_SIGMA * RCD_RB_SIGMA));
      kernel[(dy + RCD_RB_RADIUS) * RCD_RB_DIAMETER + dx + RCD_RB_RADIUS] = v;
      ksum += v;
    }
  }
  for(int k = 0; k < RCD_RB_DIAMETER * RCD_RB_DIAMETER; k++) kernel[k] /= ksum;

  float *const mask = dt_alloc_align_float(npix);
  float *const diff = dt_alloc_align_float(npix);

  if(!mask || !diff)
  {
    dt_print(DT_DEBUG_PIPE, "[rcd_residual_rb] not enough memory, skipping");
    goto cleanup;
  }

  for(int C = RED; C <= BLUE; C += 2)
  {
    DT_OMP_FOR()
    for(int y = 0; y < height; y++)
    {
      for(int x = 0; x < width; x++)
      {
        const size_t idx = (size_t)y * width + x;
        if(FC(y, x, filters) == C)
        {
          mask[idx] = 1.0f;
          diff[idx] = out[idx * 4 + C] - out[idx * 4 + 1];
        }
        else
        {
          mask[idx] = 0.0f;
          diff[idx] = 0.0f;
        }
      }
    }

    DT_OMP_FOR()
    for(int y = 0; y < height; y++)
    {
      for(int x = 0; x < width; x++)
      {
        const size_t idx = (size_t)y * width + x;
        if(FC(y, x, filters) == C) continue;

        const float G = out[idx * 4 + 1];
        float wsum = 0.0f;
        float wdsum = 0.0f;
        float wd2sum = 0.0f;

        for(int dy = -RCD_RB_RADIUS; dy <= RCD_RB_RADIUS; dy++)
        {
          const int yy = CLAMP(y + dy, 0, height - 1);
          for(int dx = -RCD_RB_RADIUS; dx <= RCD_RB_RADIUS; dx++)
          {
            const int xx = CLAMP(x + dx, 0, width - 1);
            const size_t n = (size_t)yy * width + xx;
            if(mask[n] == 0.0f) continue;

            const float gd = out[n * 4 + 1] - G;
            if(fabsf(gd) > 4.0f * RCD_RB_GSIGMA) continue;

            const float w = kernel[(dy + RCD_RB_RADIUS) * RCD_RB_DIAMETER + dx + RCD_RB_RADIUS]
                            * expf(-0.5f * gd * gd / (RCD_RB_GSIGMA * RCD_RB_GSIGMA));
            const float d = diff[n];
            wsum += w;
            wdsum += w * d;
            wd2sum += w * d * d;
          }
        }

        if(wsum < RCD_RB_TINY) continue;

        const float dmean = wdsum / wsum;
        const float dvar = fmaxf(wd2sum / wsum - dmean * dmean, 0.0f);
        const float crit = 1.0f / (1.0f + dvar / (RCD_RB_SHOT + RCD_RB_EPS));
        const float support = fminf(1.0f, wsum / 0.25f);
        const float conf = support * crit;
        const float orig = out[idx * 4 + C];
        const float cand = G + dmean;
        const float limit = 0.10f * fmaxf(G, 0.02f);
        const float corr = CLAMP(cand - orig, -limit, limit);

        out[idx * 4 + C] = fmaxf(0.0f, orig + conf * corr);
      }
    }
  }

cleanup:
  dt_free_align(mask);
  dt_free_align(diff);
}

#undef RCD_RB_RADIUS
#undef RCD_RB_DIAMETER
#undef RCD_RB_SIGMA
#undef RCD_RB_GSIGMA
#undef RCD_RB_SHOT
#undef RCD_RB_EPS
#undef RCD_RB_TINY

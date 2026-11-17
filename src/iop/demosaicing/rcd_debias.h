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
 * Green debiasing for RCD demosaicing.
 *
 * The idea of measuring the residual high-frequency bias of the interpolated
 * green with an alternating binomial filter comes from the description of the
 * RCD debiasing step introduced by RapidRAW v1.7 ("Furka"). This is an
 * independent GPL reimplementation written for darktable, no AGPL code was
 * copied.
 *
 * The filter is a 7-tap binomial kernel {1, 6, 15, 20, 15, 6, 1}. The
 * alternating sign is applied to green CFA sites in the horizontal pass. Both
 * passes are normalised by the kernel sum so that mod/level are local weighted
 * averages; this makes the correction equal to the local green bias and keeps
 * DEBIAS_LIMIT meaningful as a fraction of the local green level.
 */

#pragma once

#include <math.h>
#include <stdint.h>

#include "develop/imageop_math.h" // for FC and GREEN

#define RCD_DEBIAS_RADIUS 3
#define RCD_DEBIAS_SAFE 7 // first valid green row (4) + filter radius (3)
#define RCD_DEBIAS_NORM (1.0f / 64.0f)
#define RCD_DEBIAS_LIMIT 0.04f

static const float rcd_debias_weights[7] = { 1.0f, 6.0f, 15.0f, 20.0f, 15.0f, 6.0f, 1.0f };

static inline void rcd_debias_apply(float *const restrict green,
                                    float *const restrict mod,
                                    float *const restrict level,
                                    const int tile_rows,
                                    const int tile_cols,
                                    const int stride,
                                    const int origin_row,
                                    const int origin_col,
                                    const uint32_t filters)
{
  // RCD step 3 only fills green in [4, tile_rows-5] x [4, tile_cols-5].
  // The horizontal pass uses rows [4, tile_rows-4) and columns
  // [7, tile_cols-7) so its support stays inside that filled area. The
  // vertical pass then stays within [7, tile_rows-7) x [7, tile_cols-7).
  if(tile_rows <= 2 * RCD_DEBIAS_SAFE || tile_cols <= 2 * RCD_DEBIAS_SAFE) return;

  for(int row = 4; row < tile_rows - 4; row++)
  {
    const size_t row_off = (size_t)row * stride;
    for(int col = RCD_DEBIAS_SAFE; col < tile_cols - RCD_DEBIAS_SAFE; col++)
    {
      float m = 0.0f;
      float l = 0.0f;
      for(int k = 0; k < 7; k++)
      {
        const int x = col + k - RCD_DEBIAS_RADIUS;
        const float g = green[row_off + x];
        const float signed_g = (FC(origin_row + row, origin_col + x, filters) == GREEN) ? -g : g;
        m += rcd_debias_weights[k] * signed_g;
        l += rcd_debias_weights[k] * g;
      }
      const size_t indx = row_off + col;
      mod[indx] = m * RCD_DEBIAS_NORM;
      level[indx] = l * RCD_DEBIAS_NORM;
    }
  }

  for(int row = RCD_DEBIAS_SAFE; row < tile_rows - RCD_DEBIAS_SAFE; row++)
  {
    const size_t row_off = (size_t)row * stride;
    for(int col = RCD_DEBIAS_SAFE; col < tile_cols - RCD_DEBIAS_SAFE; col++)
    {
      if(FC(origin_row + row, origin_col + col, filters) == GREEN) continue;

      float m = 0.0f;
      float l = 0.0f;
      for(int k = 0; k < 7; k++)
      {
        const size_t n = (size_t)(row + k - RCD_DEBIAS_RADIUS) * stride + col;
        m += rcd_debias_weights[k] * mod[n];
        l += rcd_debias_weights[k] * level[n];
      }

      const float limit = RCD_DEBIAS_LIMIT * fmaxf(0.0f, l) * RCD_DEBIAS_NORM;
      const float correction = fminf(fmaxf(2.0f * m * RCD_DEBIAS_NORM, -limit), limit);
      green[row_off + col] -= correction;
    }
  }
}

/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.
    Libre DT-lab Edition (C) 2026 Christian Bouhon.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "common/colorspaces.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/dttypes.h"
#include "common/math.h"

/* Compress chromaticity toward the D50 white point in CIE 1960 uv space.
   Shared by color calibration and the hidden chromatic adaptation module. */
DT_OMP_DECLARE_SIMD(aligned(input, output:16) uniform(compression, clip))
static inline void dt_wb_gamut_mapping(const dt_aligned_pixel_t input,
                                       const float compression,
                                       const gboolean clip,
                                       dt_aligned_pixel_t output)
{
  // Get the sum XYZ
  const float sum = input[0] + input[1] + input[2];
  const float Y = input[1];

  // use chromaticity coordinates of reference white for sum == 0
  dt_aligned_pixel_t xyY = {  sum > 0.0f ? input[0] / sum : D50xyY.x,
                              sum > 0.0f ? input[1] / sum : D50xyY.y,
                              Y,
                              0.0f };

  // Convert to uvY
  dt_aligned_pixel_t uvY;
  dt_xyY_to_uvY(xyY, uvY);

  // Get the chromaticity difference with white point uv
  const float D50[2] DT_ALIGNED_PIXEL = { 0.20915914598542354f, 0.488075320769787f };
  const float delta[2] DT_ALIGNED_PIXEL = { D50[0] - uvY[0], D50[1] - uvY[1] };
  const float Delta = Y * (sqf(delta[0]) + sqf(delta[1]));

  // Compress chromaticity (move toward white point)
  const float correction = (compression == 0.0f) ? 0.f : powf(Delta, compression);
  for(size_t c = 0; c < 2; c++)
  {
    // Ensure the correction does not bring our uyY vector the other side of D50
    // that would switch to the opposite color, so we clip at D50
    // correction * delta[c] + uvY[c]
    const float tmp = DT_FMA(correction, delta[c], uvY[c]);
    uvY[c] = (uvY[c] > D50[c])  ? fmaxf(tmp, D50[c])
                                : fminf(tmp, D50[c]);
  }

  // Convert back to xyY
  dt_uvY_to_xyY(uvY, xyY);

  // Clip upon request
  if(clip) for(size_t c = 0; c < 2; c++) xyY[c] = fmaxf(xyY[c], 0.0f);

  // Check sanity of y
  // since we later divide by y, it can't be zero
  xyY[1] = fmaxf(xyY[1], NORM_MIN);

  // Check sanity of x and y :
  // since Z = Y (1 - x - y) / y, if x + y >= 1, Z will be negative
  const float scale = xyY[0] + xyY[1];
  const int sanitize = (scale >= 1.f);
  for(size_t c = 0; c < 2; c++) xyY[c] = (sanitize) ? xyY[c] / scale : xyY[c];

  // Convert back to XYZ
  dt_xyY_to_XYZ(xyY, output);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

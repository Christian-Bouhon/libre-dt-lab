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

#include "common/dttypes.h"

/* Ratios that take raw data gained by the applied white balance coefficients
   to the camera D65 reference. Channel 3 is left neutral: on 3-color raws its
   multipliers can be non-finite, and colorin's OpenCL kernels would write the
   result to alpha. */
static inline void dt_wb_get_rgb_correction(const dt_aligned_pixel_t applied,
                                            const double D65[4],
                                            dt_aligned_pixel_t correction)
{
  for_three_channels(k)
    correction[k] = (applied[k] > 1e-6f) ? (float)(D65[k] / applied[k]) : 1.0f;
  correction[3] = 1.0f;
}

/* Inverse of dt_wb_get_rgb_correction(), used where clip thresholds are
   expressed in D65-referenced units. */
static inline void dt_wb_get_rgb_inverse_correction(const dt_aligned_pixel_t applied,
                                                    const double D65[4],
                                                    dt_aligned_pixel_t correction)
{
  for_three_channels(k)
    correction[k] = (D65[k] > 1e-6) ? (float)(applied[k] / D65[k]) : 1.0f;
  correction[3] = 1.0f;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

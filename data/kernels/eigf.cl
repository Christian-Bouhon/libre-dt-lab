/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.
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
    OpenCL building blocks for the Exposure-Independent Guided Filter (EIGF),
    mirroring the CPU reference in src/common/eigf.h.

    Only the "no mask" path (quantization == 0) is implemented here, because
    that is the path used by the modules driving it (e.g. contrast & texture,
    which always calls fast_eigf_surface_blur with quantization = 0).

    The host (fast_eigf_surface_blur_cl in src/common/eigf.h) orchestrates one
    iteration of the filter as:
      1. eigf_bilinear1   : down-scale the guide image  (1 channel)
      2. eigf_build_moments : pack (g, g*g) for the gaussian  (-> 2 channels)
      3. dt_gaussian_blur_cl_buffer (2 channels)             [reused infra]
      4. eigf_finish_variance : var = E[g^2] - E[g]^2        (in place)
      5. eigf_bilinear2   : up-scale (avg, var) to full res  (2 channels)
      6. eigf_blending_no_mask : guided blend back into the image
*/

#include "common.h"

/* MIN_FLOAT = exp2f(-16) — must match src/common/luminance_mask.h / fast_guided_filter.h */
#define EIGF_MIN_FLOAT 1.52587890625e-05f

/* Bilinear resampling (1 channel). Matches interpolate_bilinear() in
 * src/common/fast_guided_filter.h byte for byte (same node clamping and weights). */
__kernel void eigf_bilinear1(global const float *in,
                             const int width_in,
                             const int height_in,
                             global float *out,
                             const int width_out,
                             const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const float x_out = (float)x / (float)width_out;
  const float y_out = (float)y / (float)height_out;
  const float x_in = x_out * (float)width_in;
  const float y_in = y_out * (float)height_in;

  int x_prev = (int)floor(x_in);
  int x_next = x_prev + 1;
  int y_prev = (int)floor(y_in);
  int y_next = y_prev + 1;

  x_prev = (x_prev < width_in)  ? x_prev : width_in - 1;
  x_next = (x_next < width_in)  ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float Q_NW = in[mad24(y_prev, width_in, x_prev)];
  const float Q_NE = in[mad24(y_prev, width_in, x_next)];
  const float Q_SE = in[mad24(y_next, width_in, x_next)];
  const float Q_SW = in[mad24(y_next, width_in, x_prev)];

  const float Dy_next = (float)y_next - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = (float)x_next - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[mad24(y, width_out, x)] =
      Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
    + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}

/* Bilinear resampling (2 channels: average + variance). */
__kernel void eigf_bilinear2(global const float2 *in,
                             const int width_in,
                             const int height_in,
                             global float2 *out,
                             const int width_out,
                             const int height_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width_out || y >= height_out) return;

  const float x_out = (float)x / (float)width_out;
  const float y_out = (float)y / (float)height_out;
  const float x_in = x_out * (float)width_in;
  const float y_in = y_out * (float)height_in;

  int x_prev = (int)floor(x_in);
  int x_next = x_prev + 1;
  int y_prev = (int)floor(y_in);
  int y_next = y_prev + 1;

  x_prev = (x_prev < width_in)  ? x_prev : width_in - 1;
  x_next = (x_next < width_in)  ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float2 Q_NW = in[mad24(y_prev, width_in, x_prev)];
  const float2 Q_NE = in[mad24(y_prev, width_in, x_next)];
  const float2 Q_SE = in[mad24(y_next, width_in, x_next)];
  const float2 Q_SW = in[mad24(y_next, width_in, x_prev)];

  const float Dy_next = (float)y_next - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = (float)x_next - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[mad24(y, width_out, x)] =
      Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
    + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}

/* Pack the guide image into the 2-channel (g, g*g) buffer expected by the
 * gaussian blur. Mirrors the first loop of eigf_variance_analysis_no_mask(). */
__kernel void eigf_build_moments(global const float *guide,
                                 global float2 *moments,
                                 const int width,
                                 const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int k = mad24(y, width, x);
  const float g = guide[k];
  moments[k] = (float2)(g, g * g);
}

/* Turn blurred moments (E[g], E[g^2]) into (average, variance), in place.
 * Mirrors the second loop of eigf_variance_analysis_no_mask(). */
__kernel void eigf_finish_variance(global float2 *av,
                                   const int width,
                                   const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int k = mad24(y, width, x);
  const float2 m = av[k];
  av[k] = (float2)(m.x, m.y - m.x * m.x);
}

/* Guided blend (no-mask, guide == mask). Mirrors eigf_blending_no_mask().
 * filter: 0 = DT_GF_BLENDING_LINEAR, 1 = DT_GF_BLENDING_GEOMEAN. */
__kernel void eigf_blending_no_mask(global float *image,
                                    global const float2 *av,
                                    const int width,
                                    const int height,
                                    const float feathering,
                                    const int filter)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int k = mad24(y, width, x);

  const float img = image[k];
  const float avg_g = av[k].x;
  const float var_g = av[k].y;

  const float norm_g = fmax(avg_g * img, 1e-6f);
  const float normalized_var_guide = var_g / norm_g;
  const float a = normalized_var_guide / (normalized_var_guide + feathering);
  const float b = avg_g - a * avg_g;

  if(filter == 0) // DT_GF_BLENDING_LINEAR
  {
    image[k] = fmax(img * a + b, EIGF_MIN_FLOAT);
  }
  else // DT_GF_BLENDING_GEOMEAN
  {
    const float v = img * fmax(img * a + b, EIGF_MIN_FLOAT);
    image[k] = sqrt(v);
  }
}

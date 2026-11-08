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
*/

#include "common/chromatic_adaptation.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/dttypes.h"
#include "common/illuminants.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/control.h"
#include "common/wb_gamut_mapping.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "iop/iop_api.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INVERSE_SQRT_3 0.5773502691896258f

DT_MODULE_INTROSPECTION(1, dt_iop_cat_params_t)

/* hidden processing module: performs the chromatic adaptation published by the
   white balance module. It has no user interface and no history entry. */

typedef struct dt_iop_cat_params_t
{
  int dummy;
} dt_iop_cat_params_t;

const char *name(void)
{
  return C_("modulename", "chromatic adaptation");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("apply the scene illuminant chromatic adaptation driven by white balance"),
     _("corrective"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group(void)
{
  return IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL;
}

int flags(void)
{
  return IOP_FLAGS_HIDDEN | IOP_FLAGS_ONE_INSTANCE
         | IOP_FLAGS_NO_HISTORY_STACK;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_cat_params_t *d = self->default_params;
  d->dummy = 0;

  // always present in the pipe, like gamma. It turns itself off for legacy
  // edits or when white balance does not provide a CAT.
  self->default_enabled = TRUE;
  self->hide_enable_button = TRUE;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  // processing reads the current white balance state at runtime, so there is
  // nothing to cache here. Just enable or disable the pass.
  piece->enabled = pipe->wb.active && pipe->wb.has_cat;
}

static inline void _cat_zero_luma_chroma(const dt_aligned_pixel_t input,
                                         dt_aligned_pixel_t output)
{
  // exact equivalent of color calibration's _luma_chroma() with zero
  // saturation and lightness, version 3. It is mathematically an identity but
  // preserves the same floating-point rounding.
  const float norm0 = euclidean_norm(input);
  const float avg = fmaxf((input[0] + input[1] + input[2]) / 3.0f, NORM_MIN);

  if(norm0 > 0.f && avg > 0.f)
  {
    float norm = norm0 * INVERSE_SQRT_3;
    for_three_channels(c)
      output[c] = input[c] / norm;

    // zero colorfulness: only the negative-ratio guard remains
    for_three_channels(c)
    {
      const float min_ratio = (output[c] < 0.0f) ? output[c] : 0.0f;
      output[c] = fmaxf(output[c], min_ratio);
    }

    norm /= euclidean_norm(output) * INVERSE_SQRT_3;
    for_three_channels(c)
      output[c] *= norm;
  }
  else
  {
    for_three_channels(c)
      output[c] = input[c];
  }
  output[3] = input[3];
}

static inline void _cat_adapt(const dt_aligned_pixel_t XYZ,
                              dt_aligned_pixel_t out,
                              const dt_aligned_pixel_t illuminant,
                              const dt_adaptation_t adaptation,
                              const float p)
{
  dt_aligned_pixel_t t1 = { 0.0f, 0.0f, 0.0f, 0.0f };
  dt_aligned_pixel_t t2 = { 0.0f, 0.0f, 0.0f, 0.0f };
  const float Y = XYZ[1];

  switch(adaptation)
  {
    case DT_ADAPTATION_FULL_BRADFORD:
    {
      convert_XYZ_to_bradford_LMS(XYZ, t2);
      downscale_vector(t2, Y);
      bradford_adapt_D50(t2, illuminant, p, TRUE, t1);
      upscale_vector(t1, Y);
      convert_bradford_LMS_to_XYZ(t1, out);
      break;
    }
    case DT_ADAPTATION_LINEAR_BRADFORD:
    {
      convert_XYZ_to_bradford_LMS(XYZ, t2);
      bradford_adapt_D50(t2, illuminant, p, FALSE, t1);
      convert_bradford_LMS_to_XYZ(t1, out);
      break;
    }
    case DT_ADAPTATION_CAT16:
    {
      convert_XYZ_to_CAT16_LMS(XYZ, t2);
      CAT16_adapt_D50(t2, illuminant, 1.0f, TRUE, t1);
      convert_CAT16_LMS_to_XYZ(t1, out);
      break;
    }
    case DT_ADAPTATION_XYZ:
    {
      XYZ_adapt_D50(XYZ, illuminant, out);
      break;
    }
    case DT_ADAPTATION_RGB:
    case DT_ADAPTATION_LAST:
    default:
      copy_pixel(out, XYZ);
      break;
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const size_t npixels = (size_t)roi_out->width * roi_out->height;
  const float *const in = (const float *const)ivoid;
  float *const out = (float *const)ovoid;

  const dt_dev_pixelpipe_t *const pipe = piece->pipe;
  const gboolean calibrating = self->dev && self->dev->chroma.calibrating;
  const gboolean active = pipe->wb.active && pipe->wb.has_cat && !calibrating;

  if(!active)
  {
    dt_iop_image_copy_by_size(ovoid, ivoid, roi_out->width, roi_out->height, 4);
    return;
  }

  const dt_adaptation_t adaptation = (dt_adaptation_t)pipe->wb.adaptation;
  dt_aligned_pixel_t illuminant;
  {
    dt_aligned_pixel_t XYZ;
    illuminant_xy_to_XYZ(pipe->wb.illuminant_xy[0], pipe->wb.illuminant_xy[1], XYZ);
    convert_any_XYZ_to_LMS(XYZ, illuminant, adaptation);
    illuminant[3] = 0.0f;
  }
  const float p = powf(0.818155f / illuminant[2], 0.0834f);

  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid,
                                        roi_in, roi_out))
    return;

  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_current_profile_info(self, piece->pipe);
  if(!work_profile)
  {
    dt_iop_image_copy_by_size(ovoid, ivoid, roi_out->width, roi_out->height, 4);
    return;
  }

  dt_colormatrix_t RGB_to_XYZ;
  dt_colormatrix_t XYZ_to_RGB;
  memcpy(RGB_to_XYZ, work_profile->matrix_in, sizeof(RGB_to_XYZ));
  memcpy(XYZ_to_RGB, work_profile->matrix_out, sizeof(XYZ_to_RGB));

  dt_colormatrix_t RGB_to_XYZ_trans;
  dt_colormatrix_t XYZ_to_RGB_trans;
  dt_colormatrix_transpose(RGB_to_XYZ_trans, RGB_to_XYZ);
  dt_colormatrix_transpose(XYZ_to_RGB_trans, XYZ_to_RGB);

  const float gamut = 1.0f;
  const gboolean clip = TRUE;

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    dt_aligned_pixel_t pix;
    dt_aligned_pixel_t XYZ;
    dt_aligned_pixel_t adapted;
    dt_aligned_pixel_t mapped;
    dt_aligned_pixel_t tmp;

    // match color calibration: clamp negative input values
    const dt_aligned_pixel_t zero = { 0.0f, 0.0f, 0.0f, 0.0f };
    dt_vector_max_nan(pix, &in[k], zero);

    // pipeline RGB -> XYZ
    dt_apply_transposed_color_matrix(pix, RGB_to_XYZ_trans, XYZ);

    // chromatic adaptation, illumant provided in LMS
    _cat_adapt(XYZ, adapted, illuminant, adaptation, p);
    dt_vector_clipneg_nan(adapted);

    // gamut compression in XYZ
    dt_wb_gamut_mapping(adapted, gamut, clip, mapped);

    // go to the adaptation space (LMS or XYZ), clip, then come back to XYZ
    switch(adaptation)
    {
      case DT_ADAPTATION_FULL_BRADFORD:
      case DT_ADAPTATION_LINEAR_BRADFORD:
      case DT_ADAPTATION_CAT16:
      case DT_ADAPTATION_XYZ:
      {
        convert_any_XYZ_to_LMS(mapped, tmp, adaptation);
        dt_vector_clipneg_nan(tmp);
        _cat_zero_luma_chroma(tmp, mapped);
        dt_vector_clipneg_nan(mapped);
        convert_any_LMS_to_XYZ(mapped, tmp, adaptation);
        copy_pixel(mapped, tmp);
        break;
      }
      case DT_ADAPTATION_RGB:
      case DT_ADAPTATION_LAST:
      default:
        break;
    }

    dt_vector_clipneg_nan(mapped);

    // XYZ -> pipeline RGB
    dt_apply_transposed_color_matrix(mapped, XYZ_to_RGB_trans, pix);
    dt_vector_clipneg_nan(pix);

    pix[3] = in[k + 3];
    copy_pixel_nontemporal(&out[k], pix);
  }
}

#ifdef HAVE_OPENCL

typedef struct dt_iop_cat_global_data_t
{
  int kernel_cat16;
  int kernel_bradford_full;
  int kernel_bradford_linear;
  int kernel_xyz;
  int kernel_rgb;
} dt_iop_cat_global_data_t;

void init_global(dt_iop_module_so_t *self)
{
  const int program = 32; // channelmixer.cl in programs.conf
  dt_iop_cat_global_data_t *gd = malloc(sizeof(dt_iop_cat_global_data_t));
  self->data = gd;
  gd->kernel_cat16 = dt_opencl_create_kernel(program, "channelmixerrgb_CAT16");
  gd->kernel_bradford_full =
    dt_opencl_create_kernel(program, "channelmixerrgb_bradford_full");
  gd->kernel_bradford_linear =
    dt_opencl_create_kernel(program, "channelmixerrgb_bradford_linear");
  gd->kernel_xyz = dt_opencl_create_kernel(program, "channelmixerrgb_XYZ");
  gd->kernel_rgb = dt_opencl_create_kernel(program, "channelmixerrgb_RGB");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_cat_global_data_t *gd = self->data;
  if(!gd) return;
  dt_opencl_free_kernel(gd->kernel_cat16);
  dt_opencl_free_kernel(gd->kernel_bradford_full);
  dt_opencl_free_kernel(gd->kernel_bradford_linear);
  dt_opencl_free_kernel(gd->kernel_xyz);
  dt_opencl_free_kernel(gd->kernel_rgb);
  free(self->data);
  self->data = NULL;
}

int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               const cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_dev_pixelpipe_t *const pipe = piece->pipe;
  const int devid = pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const gboolean calibrating = self->dev && self->dev->chroma.calibrating;
  const gboolean active = pipe->wb.active && pipe->wb.has_cat && !calibrating;

  if(!active)
  {
    const size_t region[2] = { width, height };
    return dt_opencl_enqueue_copy_image(devid, dev_in, dev_out,
                                        CLIMG_ORIGIN, CLIMG_ORIGIN, region);
  }

  if(piece->colors != 4)
  {
    dt_control_log(_("chromatic adaptation works only on RGB input"));
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_current_profile_info(self, piece->pipe);
  if(!work_profile)
  {
    const size_t region[2] = { width, height };
    return dt_opencl_enqueue_copy_image(devid, dev_in, dev_out,
                                        CLIMG_ORIGIN, CLIMG_ORIGIN, region);
  }

  const dt_iop_cat_global_data_t *const gd = self->global_data;

  dt_colormatrix_t MIX = { { 1.0f, 0.0f, 0.0f, 0.0f },
                           { 0.0f, 1.0f, 0.0f, 0.0f },
                           { 0.0f, 0.0f, 1.0f, 0.0f },
                           { 0.0f, 0.0f, 0.0f, 0.0f } };
  const dt_aligned_pixel_t zero = { 0.0f, 0.0f, 0.0f, 0.0f };

  const dt_adaptation_t adaptation = (dt_adaptation_t)pipe->wb.adaptation;
  dt_aligned_pixel_t illuminant;
  dt_aligned_pixel_t XYZ;
  illuminant_xy_to_XYZ(pipe->wb.illuminant_xy[0], pipe->wb.illuminant_xy[1], XYZ);
  convert_any_XYZ_to_LMS(XYZ, illuminant, adaptation);
  illuminant[3] = 0.0f;
  const float p = powf(0.818155f / illuminant[2], 0.0834f);

  int kernel = gd->kernel_rgb;
  switch(adaptation)
  {
    case DT_ADAPTATION_FULL_BRADFORD:
      kernel = gd->kernel_bradford_full;
      break;
    case DT_ADAPTATION_LINEAR_BRADFORD:
      kernel = gd->kernel_bradford_linear;
      break;
    case DT_ADAPTATION_CAT16:
      kernel = gd->kernel_cat16;
      break;
    case DT_ADAPTATION_XYZ:
      kernel = gd->kernel_xyz;
      break;
    case DT_ADAPTATION_RGB:
    case DT_ADAPTATION_LAST:
    default:
      kernel = gd->kernel_rgb;
      break;
  }

  cl_mem input_matrix_cl = NULL;
  cl_mem output_matrix_cl = NULL;
  cl_mem mix_cl = NULL;
  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  input_matrix_cl = dt_opencl_copy_host_to_device_constant
    (devid, 12 * sizeof(float), (float *)work_profile->matrix_in);
  output_matrix_cl = dt_opencl_copy_host_to_device_constant
    (devid, 12 * sizeof(float), (float *)work_profile->matrix_out);
  mix_cl = dt_opencl_copy_host_to_device_constant
    (devid, 12 * sizeof(float), MIX);

  if(input_matrix_cl == NULL || output_matrix_cl == NULL || mix_cl == NULL)
    goto error;

  err = dt_opencl_enqueue_kernel_2d_args
    (devid, kernel, width, height,
     CLARG(dev_in), CLARG(dev_out),
     CLARG(width), CLARG(height),
     CLARG(input_matrix_cl), CLARG(output_matrix_cl), CLARG(mix_cl),
     CLARG(illuminant), CLARG(zero), CLARG(zero), CLARG(zero),
     CLARG(p), CLARGFLOAT(1.0f), CLARGINT(1), CLARGINT(0), CLARGINT(2));

error:
  dt_opencl_release_mem_object(input_matrix_cl);
  dt_opencl_release_mem_object(output_matrix_cl);
  dt_opencl_release_mem_object(mix_cl);
  return err;
}

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

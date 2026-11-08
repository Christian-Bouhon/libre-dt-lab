/*
    This file is part of darktable,
    Copyright (C) 2016-2020 darktable developers.

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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "common/darktable.h"

struct dt_dev_pixelpipe_iop_t;
struct dt_dev_pixelpipe_t;
struct dt_iop_module_t;

/** white balance state published by the white balance module for the modules
    that complete the correction (colorin) and perform the chromatic
    adaptation (channelmixerrgb). It is stored per pipe so that processing
    threads never have to read shared, mutable developer state. */
typedef struct dt_dev_wb_t
{
  gboolean active;          // white balance module is active in this pipe
  gboolean has_cat;         // the scene illuminant is valid for chromatic adaptation
  float illuminant_xy[2];   // scene illuminant chromaticity for the CAT
  int adaptation;           // dt_adaptation_t, kept as int here to avoid
                            // pulling chromatic adaptation into this header
} dt_dev_wb_t;

typedef enum dt_iop_buffer_type_t {
  TYPE_UNKNOWN,
  TYPE_FLOAT,
  TYPE_UINT16,
} dt_iop_buffer_type_t;

typedef struct dt_iop_buffer_dsc_t
{
  /** how many channels the data has? 1 or 4 */
  unsigned int channels;
  /** what is the datatype? */
  dt_iop_buffer_type_t datatype;
  /** Bayer demosaic pattern */
  uint32_t filters;
  /** filter for Fuji X-Trans images, only used if filters == 9u */
  uint8_t xtrans[6][6];

  struct
  {
    uint16_t raw_black_level;
    uint16_t raw_white_point;
  } rawprepare;

  struct
  {
    gboolean enabled;
    dt_aligned_pixel_t coeffs;
  } temperature;

  /** sensor saturation, propagated through the operations */
  dt_aligned_pixel_t processed_maximum;

  /** colorspace of the image */
  int cst;

} dt_iop_buffer_dsc_t;

size_t dt_iop_buffer_dsc_to_bpp(const struct dt_iop_buffer_dsc_t *dsc);

void default_input_format(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_t *pipe,
                          struct dt_dev_pixelpipe_iop_t *piece, struct dt_iop_buffer_dsc_t *dsc);

void default_output_format(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_t *pipe,
                           struct dt_dev_pixelpipe_iop_t *piece, struct dt_iop_buffer_dsc_t *dsc);

int default_input_colorspace(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_t *pipe, struct dt_dev_pixelpipe_iop_t *piece);
int default_output_colorspace(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_t *pipe, struct dt_dev_pixelpipe_iop_t *piece);
int default_blend_colorspace(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_t *pipe, struct dt_dev_pixelpipe_iop_t *piece);


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on


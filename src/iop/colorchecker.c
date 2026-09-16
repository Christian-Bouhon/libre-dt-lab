/*
    This file is part of darktable,
    Copyright (C) 2016-2026 darktable developers.

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

#include "bauhaus/bauhaus.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/colorchecker.h"
#include "common/math.h"
#include "common/opencl.h"
#include "common/exif.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/imageop_math.h"
#include "develop/openmp_maths.h"
#include "develop/tiling.h"
#include "dtgtk/drawingarea.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "iop/gaussian_elimination.h"
#include "chart/common.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gtk/gtk.h>
#include <inttypes.h>

DT_MODULE_INTROSPECTION(3, dt_iop_colorchecker_params_t)

static const int colorchecker_patches = 24;
static const float colorchecker_Lab[] =
{ // from argyll ColorChecker.cie
 37.99,   13.56,  14.06, // dark skin
 65.71,   18.13,  17.81, // light skin
 49.93,   -4.88, -21.93, // blue sky
 43.14,  -13.10,  21.91, // foliage
 55.11,    8.84, -25.40, // blue flower
 70.72,  -33.40, -0.20 , // bluish green
 62.66,   36.07,  57.10, // orange
 40.02,   10.41, -45.96, // purple red
 51.12,   48.24,  16.25, // moderate red
 30.33,   22.98, -21.59, // purple
 72.53,  -23.71,  57.26, // yellow green
 71.94,  19.36 ,  67.86, // orange yellow
 28.78,  14.18 , -50.30, // blue
 55.26,  -38.34,  31.37, // green
 42.10,  53.38 ,  28.19, // red
 81.73,  4.04  ,  79.82, // yellow
 51.94,  49.99 , -14.57, // magenta
 51.04,  -28.63, -28.64, // cyan
 96.54,  -0.43 ,  1.19 , // white
 81.26,  -0.64 , -0.34 , // neutral 8
 66.77,  -0.73 , -0.50 , // neutral 65
 50.87,  -0.15 , -0.27 , // neutral 5
 35.66,  -0.42 , -1.23 , // neutral 35
 20.46,  -0.08 , -0.97   // black
};

typedef enum dt_iop_colorchecker_colorspace_t
{
  DT_IOP_CC_CS_LAB = 0,     // $DESCRIPTION: "CIE Lab"
  DT_IOP_CC_CS_XYZ = 1,     // $DESCRIPTION: "CIE XYZ"
} dt_iop_colorchecker_colorspace_t;

typedef enum dt_iop_colorchecker_anchor_t
{
  DT_IOP_CC_ANCHOR_WHITE = 0, // $DESCRIPTION: "white patch"
  DT_IOP_CC_ANCHOR_GRAY = 1,  // $DESCRIPTION: "middle gray"
  DT_IOP_CC_ANCHOR_NONE = 2,  // $DESCRIPTION: "none"
} dt_iop_colorchecker_anchor_t;

// we came to the conclusion that more than 7x7 patches will not be
// manageable in the gui. the fitting experiments show however that you
// can do significantly better with 49 than you can with 24 patches,
// especially when considering max delta E.
#define MAX_PATCHES 49
typedef struct dt_iop_colorchecker_params_t
{
  float source_L[MAX_PATCHES];
  float source_a[MAX_PATCHES];
  float source_b[MAX_PATCHES];
  float target_L[MAX_PATCHES];
  float target_a[MAX_PATCHES];
  float target_b[MAX_PATCHES];
  int32_t num_patches;
  dt_iop_colorchecker_colorspace_t colorspace; // $DEFAULT: DT_IOP_CC_CS_XYZ $DESCRIPTION: "color space"
  dt_iop_colorchecker_anchor_t anchor;         // $DEFAULT: DT_IOP_CC_ANCHOR_WHITE $DESCRIPTION: "exposure reference"
} dt_iop_colorchecker_params_t;

typedef struct dt_iop_colorchecker_gui_data_t
{
  GtkWidget *area, *combobox_patch;
  GtkWidget *scale_L, *scale_a, *scale_b, *scale_C, *combobox_target;
  GtkWidget *combobox_colorspace, *combobox_anchor;
  int patch, drawn_patch;
  int absolute_target; // 0: show relative offsets in sliders, 1: show
                       // absolute Lab values

  // ---- color chart calibration ----
  dt_color_checker_t *checker;   // selected reference chart
  point_t box[4];                // current (possibly non rectangular) chart corners in preview pixels
  point_t ideal_box[4];          // unit rectangle the corners map from
  point_t center_box;
  point_t click_start, click_end;
  gboolean active_node[4];       // cursor close to a corner
  gboolean is_cursor_close;
  gboolean drag_drop;
  float homography[9];
  float inverse_homography[9];
  float safety_margin;
  gboolean run_profile;          // request patch extraction at next preview recompute
  gboolean profile_ready;        // a measured profile is available
  gboolean preview_pending;      // a second reprocess is needed to show the measurement
  gboolean checker_ready;        // the bounding box has been initialised
  gboolean is_profiling_started; // the calibration section is expanded
  float *measured_lab;           // 3 * MAX_PATCHES measured patch colors (Lab)
  float *measured_XYZ;           // 3 * MAX_PATCHES same patches, raw CIE XYZ
                                  // (only filled/valid when the calibration
                                  // ran in XYZ mode, see measured_xyz_valid)
  gboolean measured_xyz_valid;   // whether measured_XYZ reflects the colorspace
                                  // mode that was active at extraction time
  float *delta_E_in;             // per-patch delta E before correction
  gchar *delta_E_label_text;
  // cache of the last formatted report (gui_post_expose runs on every redraw)
  float delta_E_avg, delta_E_max;
  int delta_E_worst;
  gboolean delta_E_valid;

  GtkWidget *checkers_list, *safety, *label_delta_E;
  GtkWidget *button_profile, *button_commit, *button_reuse;
  dt_gui_collapsible_section_t cs;
} dt_iop_colorchecker_gui_data_t;

typedef struct dt_iop_colorchecker_data_t
{
  int32_t num_patches;
  int32_t colorspace;
  int32_t anchor;
  float source_Lab[3*MAX_PATCHES];
  float coeff_L[MAX_PATCHES+4];
  float coeff_a[MAX_PATCHES+4];
  float coeff_b[MAX_PATCHES+4];
  float scale_in;   // multiply input by this before the spline (XYZ mode)
  float scale_out;  // multiply spline output by this after (XYZ mode)
  float matrix_in[9];  // RGB -> XYZ (XYZ mode)
  float matrix_out[9]; // XYZ -> RGB (XYZ mode)
  gboolean has_matrix;
} dt_iop_colorchecker_data_t;

typedef struct dt_iop_colorchecker_global_data_t
{
  int kernel_colorchecker;
} dt_iop_colorchecker_global_data_t;


const char *name()
{
  return _("color look up table");
}

const char *aliases()
{
  return _("profile|lut|color grading");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self, _("perform color space corrections and apply looks"),
     _("corrective or creative"),
     _("linear or non-linear, Lab or XYZ, scene or display-referred"),
     _("defined by profile, Lab or XYZ"),
     _("linear or non-linear, Lab or XYZ, scene or display-referred"));
}

int default_group()
{
  return IOP_GROUP_COLOR | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_colorchecker_params_t *const p = self->params;
  if(!p || p->colorspace == DT_IOP_CC_CS_LAB)
    return IOP_CS_LAB;

  // XYZ mode works on linear RGB, but only if we have a valid work profile
  // to convert it to CIE XYZ. Otherwise stay in Lab to avoid a mismatch.
  if(pipe)
  {
    const dt_iop_order_iccprofile_info_t *const wp = dt_ioppr_get_pipe_work_profile_info(pipe);
    if(!wp || !dt_is_valid_colormatrix(wp->matrix_in[0][0]))
      return IOP_CS_LAB;
  }
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  static const float colorchecker_Lab_v1[] = {
    39.19, 13.76,  14.29,  // dark skin
    65.18, 19.00,  17.32,  // light skin
    49.46, -4.23,  -22.95, // blue sky
    42.85, -13.33, 22.12,  // foliage
    55.18, 9.44,   -24.94, // blue flower
    70.36, -32.77, -0.04,  // bluish green
    62.92, 35.49,  57.10,  // orange
    40.75, 11.41,  -46.03, // purple red
    52.10, 48.11,  16.89,  // moderate red
    30.67, 21.19,  -20.81, // purple
    73.08, -23.55, 56.97,  // yellow green
    72.43, 17.48,  68.20,  // orange yellow
    30.97, 12.67,  -46.30, // blue
    56.43, -40.66, 31.94,  // green
    43.40, 50.68,  28.84,  // red
    82.45, 2.41,   80.25,  // yellow
    51.98, 50.68,  -14.84, // magenta
    51.02, -27.63, -28.03, // cyan
    95.97, -0.40,  1.24,   // white
    81.10, -0.83,  -0.43,  // neutral 8
    66.81, -1.08,  -0.70,  // neutral 65
    50.98, -0.19,  -0.30,  // neutral 5
    35.72, -0.69,  -1.11,  // neutral 35
    21.46, 0.06,   -0.95,  // black
  };

  // NOTE: versions 3 to 5 of the params struct (intermediate steps of the
  // colorspace/anchor/solving-strategy rework) were never published, so
  // there is no on-disk XMP/database data using them. Only v1 and v2 -
  // the versions that actually shipped - need a migration path to the
  // current v3 struct (source/target Lab + num_patches + colorspace +
  // anchor).
  if(old_version >= 1 && old_version <= 2)
  {
    dt_iop_colorchecker_params_t *n = malloc(sizeof(dt_iop_colorchecker_params_t));
    memset(n, 0, sizeof(dt_iop_colorchecker_params_t));
    n->colorspace = DT_IOP_CC_CS_LAB;
    n->anchor = DT_IOP_CC_ANCHOR_WHITE;

    if(old_version == 1)
    {
      typedef struct dt_iop_colorchecker_params_v1_t
      {
        float target_L[24];
        float target_a[24];
        float target_b[24];
      } dt_iop_colorchecker_params_v1_t;

      const dt_iop_colorchecker_params_v1_t *o = old_params;
      n->num_patches = 24;
      for(int k=0; k<24; k++)
      {
        n->target_L[k] = o->target_L[k];
        n->target_a[k] = o->target_a[k];
        n->target_b[k] = o->target_b[k];
        n->source_L[k] = colorchecker_Lab_v1[3 * k + 0];
        n->source_a[k] = colorchecker_Lab_v1[3 * k + 1];
        n->source_b[k] = colorchecker_Lab_v1[3 * k + 2];
      }
    }
    else // old_version == 2
    {
      typedef struct dt_iop_colorchecker_params_v2_t
      {
        float source_L[MAX_PATCHES];
        float source_a[MAX_PATCHES];
        float source_b[MAX_PATCHES];
        float target_L[MAX_PATCHES];
        float target_a[MAX_PATCHES];
        float target_b[MAX_PATCHES];
        int32_t num_patches;
      } dt_iop_colorchecker_params_v2_t;

      const dt_iop_colorchecker_params_v2_t *o = old_params;
      n->num_patches = o->num_patches;
      for(int k=0; k<MAX_PATCHES; k++)
      {
        n->source_L[k] = o->source_L[k];
        n->source_a[k] = o->source_a[k];
        n->source_b[k] = o->source_b[k];
        n->target_L[k] = o->target_L[k];
        n->target_a[k] = o->target_a[k];
        n->target_b[k] = o->target_b[k];
      }
    }
    *new_params = n;
    *new_params_size = sizeof(dt_iop_colorchecker_params_t);
    *new_version = 3;
    return 0;
  }
  return 1;
}

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_colorchecker_params_t p;
  memset(&p, 0, sizeof(p));
  p.num_patches = 24;
  p.target_L[ 0] = p.source_L[ 0] = 17.460945129394531;
  p.target_L[ 1] = p.source_L[ 1] = 26.878498077392578;
  p.target_L[ 2] = p.source_L[ 2] = 34.900054931640625;
  p.target_L[ 3] = p.source_L[ 3] = 21.692604064941406;
  p.target_L[ 4] = p.source_L[ 4] = 32.18853759765625;
  p.target_L[ 5] = p.source_L[ 5] = 62.531227111816406;
  p.target_L[ 6] = p.source_L[ 6] = 18.933284759521484;
  p.target_L[ 7] = p.source_L[ 7] = 53.936111450195312;
  p.target_L[ 8] = p.source_L[ 8] = 69.154266357421875;
  p.target_L[ 9] = p.source_L[ 9] = 43.381229400634766;
  p.target_L[10] = p.source_L[10] = 57.797889709472656;
  p.target_L[11] = p.source_L[11] = 73.27630615234375;
  p.target_L[12] = p.source_L[12] = 53.175498962402344;
  p.target_L[13] = p.source_L[13] = 49.111373901367188;
  p.target_L[14] = p.source_L[14] = 63.169830322265625;
  p.target_L[15] = p.source_L[15] = 61.896102905273438;
  p.target_L[16] = p.source_L[16] = 67.852409362792969;
  p.target_L[17] = p.source_L[17] = 72.489517211914062;
  p.target_L[18] = p.source_L[18] = 70.935714721679688;
  p.target_L[19] = p.source_L[19] = 70.173004150390625;
  p.target_L[20] = p.source_L[20] = 77.78826904296875;
  p.target_L[21] = p.source_L[21] = 76.070747375488281;
  p.target_L[22] = p.source_L[22] = 68.645004272460938;
  p.target_L[23] = p.source_L[23] = 74.502906799316406;
  p.target_a[ 0] = p.source_a[ 0] = 8.4928874969482422;
  p.target_a[ 1] = p.source_a[ 1] = 27.94782829284668;
  p.target_a[ 2] = p.source_a[ 2] = 43.8824462890625;
  p.target_a[ 3] = p.source_a[ 3] = 16.723676681518555;
  p.target_a[ 4] = p.source_a[ 4] = 39.174972534179688;
  p.target_a[ 5] = p.source_a[ 5] = 24.966419219970703;
  p.target_a[ 6] = p.source_a[ 6] = 8.8226642608642578;
  p.target_a[ 7] = p.source_a[ 7] = 34.451812744140625;
  p.target_a[ 8] = p.source_a[ 8] = 18.39008903503418;
  p.target_a[ 9] = p.source_a[ 9] = 28.272598266601562;
  p.target_a[10] = p.source_a[10] = 10.193824768066406;
  p.target_a[11] = p.source_a[11] = 13.241470336914062;
  p.target_a[12] = p.source_a[12] = 43.655307769775391;
  p.target_a[13] = p.source_a[13] = 23.247600555419922;
  p.target_a[14] = p.source_a[14] = 23.308664321899414;
  p.target_a[15] = p.source_a[15] = 11.138319969177246;
  p.target_a[16] = p.source_a[16] = 18.200069427490234;
  p.target_a[17] = p.source_a[17] = 15.363990783691406;
  p.target_a[18] = p.source_a[18] = 11.173545837402344;
  p.target_a[19] = p.source_a[19] = 11.313735961914062;
  p.target_a[20] = p.source_a[20] = 15.059500694274902;
  p.target_a[21] = p.source_a[21] = 4.7686996459960938;
  p.target_a[22] = p.source_a[22] = 3.0603706836700439;
  p.target_a[23] = p.source_a[23] = -3.687053918838501;
  p.target_b[ 0] = p.source_b[ 0] = -0.023579597473144531;
  p.target_b[ 1] = p.source_b[ 1] = 14.991056442260742;
  p.target_b[ 2] = p.source_b[ 2] = 26.443553924560547;
  p.target_b[ 3] = p.source_b[ 3] = 7.3905587196350098;
  p.target_b[ 4] = p.source_b[ 4] = 23.309671401977539;
  p.target_b[ 5] = p.source_b[ 5] = 19.262432098388672;
  p.target_b[ 6] = p.source_b[ 6] = 3.136211633682251;
  p.target_b[ 7] = p.source_b[ 7] = 31.949621200561523;
  p.target_b[ 8] = p.source_b[ 8] = 16.144514083862305;
  p.target_b[ 9] = p.source_b[ 9] = 25.893926620483398;
  p.target_b[10] = p.source_b[10] = 12.271202087402344;
  p.target_b[11] = p.source_b[11] = 16.763805389404297;
  p.target_b[12] = p.source_b[12] = 53.904998779296875;
  p.target_b[13] = p.source_b[13] = 36.537342071533203;
  p.target_b[14] = p.source_b[14] = 32.930683135986328;
  p.target_b[15] = p.source_b[15] = 19.008804321289062;
  p.target_b[16] = p.source_b[16] = 32.259223937988281;
  p.target_b[17] = p.source_b[17] = 25.815582275390625;
  p.target_b[18] = p.source_b[18] = 26.509498596191406;
  p.target_b[19] = p.source_b[19] = 40.572704315185547;
  p.target_b[20] = p.source_b[20] = 88.354469299316406;
  p.target_b[21] = p.source_b[21] = 33.434604644775391;
  p.target_b[22] = p.source_b[22] = 9.5750093460083008;
  p.target_b[23] = p.source_b[23] = 41.285167694091797;
  dt_gui_presets_add_generic(_("it8 skin tones"), self->op,
                             self->version(), &p, sizeof(p),
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);

  memset(&p, 0, sizeof(p));
  p.num_patches = 49;
  // red ramp in first row
  p.target_L[0] = p.source_L[0] = 10;
  p.target_L[1] = p.source_L[1] = 20;
  p.target_L[2] = p.source_L[2] = 30;
  p.target_L[3] = p.source_L[3] = 50;
  p.target_L[4] = p.source_L[4] = 70;
  p.target_L[5] = p.source_L[5] = 80;
  p.target_L[6] = p.source_L[6] = 90;
  p.target_a[0] = p.source_a[0] = 48;
  p.target_a[1] = p.source_a[1] = 72;
  p.target_a[2] = p.source_a[2] = 72;
  p.target_a[3] = p.source_a[3] = 72;
  p.target_a[4] = p.source_a[4] = 72;
  p.target_a[5] = p.source_a[5] = 72;
  p.target_a[6] = p.source_a[6] = 72;
  p.target_b[0] = p.source_b[0] = 16;
  p.target_b[1] = p.source_b[1] = 24;
  p.target_b[2] = p.source_b[2] = 24;
  p.target_b[3] = p.source_b[3] = 24;
  p.target_b[4] = p.source_b[4] = 24;
  p.target_b[5] = p.source_b[5] = 24;
  p.target_b[6] = p.source_b[6] = 24;

  // blue ramp in second row
  p.target_L[ 7] = p.source_L[ 7] = 10;
  p.target_L[ 8] = p.source_L[ 8] = 20;
  p.target_L[ 9] = p.source_L[ 9] = 30;
  p.target_L[10] = p.source_L[10] = 50;
  p.target_L[11] = p.source_L[11] = 70;
  p.target_L[12] = p.source_L[12] = 80;
  p.target_L[13] = p.source_L[13] = 90;
  p.target_a[ 7] = p.source_a[ 7] = 7;
  p.target_a[ 8] = p.source_a[ 8] = 14;
  p.target_a[ 9] = p.source_a[ 9] = 21;
  p.target_a[10] = p.source_a[10] = 21;
  p.target_a[11] = p.source_a[11] = 21;
  p.target_a[12] = p.source_a[12] = 21;
  p.target_a[13] = p.source_a[13] = 14;
  p.target_b[ 7] = p.source_b[ 7] = -25;
  p.target_b[ 8] = p.source_b[ 8] = -50;
  p.target_b[ 9] = p.source_b[ 9] = -75;
  p.target_b[10] = p.source_b[10] = -75;
  p.target_b[11] = p.source_b[11] = -75;
  p.target_b[12] = p.source_b[12] = -75;
  p.target_b[13] = p.source_b[13] = -50;

  // green ramp in third row
  p.target_L[14] = p.source_L[14] = 10;
  p.target_L[15] = p.source_L[15] = 20;
  p.target_L[16] = p.source_L[16] = 30;
  p.target_L[17] = p.source_L[17] = 50;
  p.target_L[18] = p.source_L[18] = 70;
  p.target_L[19] = p.source_L[19] = 80;
  p.target_L[20] = p.source_L[20] = 90;
  p.target_a[14] = p.source_a[14] = -20;
  p.target_a[15] = p.source_a[15] = -40;
  p.target_a[16] = p.source_a[16] = -40;
  p.target_a[17] = p.source_a[17] = -40;
  p.target_a[18] = p.source_a[18] = -40;
  p.target_a[19] = p.source_a[19] = -40;
  p.target_a[20] = p.source_a[20] = -40;
  p.target_b[14] = p.source_b[14] = 16;
  p.target_b[15] = p.source_b[15] = 32;
  p.target_b[16] = p.source_b[16] = 32;
  p.target_b[17] = p.source_b[17] = 32;
  p.target_b[18] = p.source_b[18] = 32;
  p.target_b[19] = p.source_b[19] = 32;
  p.target_b[20] = p.source_b[20] = 32;

  // orange/yellow/cyan tones in fourth row
  p.target_L[21] = p.source_L[21] = 63;	// orange
  p.target_a[21] = p.source_a[21] = 36;
  p.target_b[21] = p.source_b[21] = 57;
  p.target_L[22] = p.source_L[22] = 72; // orange yellow
  p.target_a[22] = p.source_a[22] = 19;
  p.target_b[22] = p.source_b[22] = 68;
  p.target_L[23] = p.source_L[23] = 82; // yellow
  p.target_a[23] = p.source_a[23] =  4;
  p.target_b[23] = p.source_b[23] = 80;
  p.target_L[24] = p.source_L[24] = 72; // yellow green
  p.target_a[24] = p.source_a[24] = -24;
  p.target_b[24] = p.source_b[24] = 57;
  p.target_L[25] = p.source_L[25] = 43; // foliage
  p.target_a[25] = p.source_a[25] = -13;
  p.target_b[25] = p.source_b[25] = 22;
  p.target_L[26] = p.source_L[26] = 71; // bluish green
  p.target_a[26] = p.source_a[26] = -33;
  p.target_b[26] = p.source_b[26] = 0;
  p.target_L[27] = p.source_L[27] = 51; // cyan
  p.target_a[27] = p.source_a[27] = -60;
  p.target_b[27] = p.source_b[27] = -60;

  // fifth row: CC24-like skin tone and misc patches
  p.target_L[28] = p.source_L[28] = 39;  // CC24 dark skin
  p.target_a[28] = p.source_a[28] = 14;
  p.target_b[28] = p.source_b[28] = 14;
  p.target_L[29] = p.source_L[29] = 65;  // CC24 light skin
  p.target_a[29] = p.source_a[29] = 19;
  p.target_b[29] = p.source_b[29] = 17;
  p.target_L[30] = p.source_L[30] = 49;  // blue sky
  p.target_a[30] = p.source_a[30] = -4;
  p.target_b[30] = p.source_b[30] = -23;
  p.target_L[31] = p.source_L[31] = 55;  // blue flower
  p.target_a[31] = p.source_a[31] =  9;
  p.target_b[31] = p.source_b[31] = -25;
  p.target_L[32] = p.source_L[32] = 52;  // magenta
  p.target_a[32] = p.source_a[32] = 75;
  p.target_b[32] = p.source_b[32] = -21;
  p.target_L[33] = p.source_L[33] = 31;  // purple
  p.target_a[33] = p.source_a[33] = 50;
  p.target_b[33] = p.source_b[33] = -50;
  p.target_L[34] = p.source_L[34] = 41;  // purple red
  p.target_a[34] = p.source_a[34] = 33;
  p.target_b[34] = p.source_b[34] = -66;

  // IT8 skin tones in sixth row
  p.target_L[35] = p.source_L[35] = 17;
  p.target_a[35] = p.source_a[35] = 8;
  p.target_b[35] = p.source_b[35] = 0;
  p.target_L[36] = p.source_L[36] = 30;
  p.target_a[36] = p.source_a[36] = 9;
  p.target_b[36] = p.source_b[36] = 3;
  p.target_L[37] = p.source_L[37] = 26;
  p.target_a[37] = p.source_a[37] = 28;
  p.target_b[37] = p.source_b[37] = 15;
  p.target_L[38] = p.source_L[38] = 32;
  p.target_a[38] = p.source_a[38] = 39;
  p.target_b[38] = p.source_b[38] = 23;
  p.target_L[39] = p.source_L[39] = 54;
  p.target_a[39] = p.source_a[39] = 34;
  p.target_b[39] = p.source_b[39] = 32;
  p.target_L[40] = p.source_L[40] = 70;
  p.target_a[40] = p.source_a[40] = 11;
  p.target_b[40] = p.source_b[40] = 41;
  p.target_L[41] = p.source_L[41] = 76;
  p.target_a[41] = p.source_a[41] = 5;
  p.target_b[41] = p.source_b[41] = 33;

  // 7-level gray ramp in last row
  p.target_L[42] = p.source_L[42] = 2.0;
  p.target_L[43] = p.source_L[43] = 18.0;
  p.target_L[44] = p.source_L[44] = 34.0;
  p.target_L[45] = p.source_L[45] = 50.0;
  p.target_L[46] = p.source_L[46] = 66.0;
  p.target_L[47] = p.source_L[47] = 82.0;
  p.target_L[48] = p.source_L[48] = 98.0;
  p.target_a[42] = p.source_a[42] = 0.0;
  p.target_a[43] = p.source_a[43] = 0.0;
  p.target_a[44] = p.source_a[44] = 0.0;
  p.target_a[45] = p.source_a[45] = 0.0;
  p.target_a[46] = p.source_a[46] = 0.0;
  p.target_a[47] = p.source_a[47] = 0.0;
  p.target_a[48] = p.source_a[48] = 0.0;
  p.target_b[42] = p.source_b[42] = 0.0;
  p.target_b[43] = p.source_b[43] = 0.0;
  p.target_b[44] = p.source_b[44] = 0.0;
  p.target_b[45] = p.source_b[45] = 0.0;
  p.target_b[46] = p.source_b[46] = 0.0;
  p.target_b[47] = p.source_b[47] = 0.0;
  p.target_b[48] = p.source_b[48] = 0.0;
  dt_gui_presets_add_generic(_("expanded color checker"), self->op,
                             self->version(), &p, sizeof(p),
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);

  // Helmholtz/Kohlrausch effect applied to black and white conversion.
  // implemented by wmader as an iop and matched as a clut for increased
  // flexibility. this was done using darktable-chart and this is copied
  // from the resulting dtstyle output file:
  const char *hk_params_input =
    "9738b84231c098426fb8814234a82d422ac41d422e3fa04100004843f7daa24257e09a422a1a984225113842f89cc9410836ca4295049542ad1c9242887370427cb32b427c512242b5a40742545bd141808740412cc6964262e484429604c44100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ef6d3bc152c2acc1ef6566c093a522c2e7d4e4c1a87c7cc100000000b4c4dd407af09e40d060df418afc7d421dadd0413ec5124097d79041fcba2642fc9f484183eb92415d6b7040fcdcdc41b8fe2f42b64a1740fc8612c1276defc144432ec100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d237eb4022a72842f5639742396d1442a2660d411c338b40000000006e35ca408df2054289658d4132327a4118427741d4cf08c0f8a4d5c03abed7c13fac36c23b41a6c03c2230c07d5088c26caff7c1e0e9c6bff14ecec073b028c29e0accc10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000085f2b642a4ba9a423c9a8442a6493c428baf28425667b64100004843a836a142a84e9b4226719d421cb15d424c22ee4175fcca4211ae96426e6d9a4243878142ef45354222f82542629527420280ff416c2066417e3996420d838e424182e3410000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000fa370000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8b700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004837000000000000c8b60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000018000000";
  int params_len = 0;
  uint8_t *hk_params = dt_exif_xmp_decode
    (hk_params_input,
     strlen(hk_params_input), &params_len);
  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(hk_params);
  dt_gui_presets_add_generic(_("Helmholtz/Kohlrausch monochrome"), self->op,
                             2, hk_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(hk_params);

  /** The following are based on Jo's Fuji film emulations, without
   *  tonecurve which is let to user choice
   *  https://jo.dreggn.org/blog/darktable-fuji-styles.tar.xz **/

  const char *astia_params_input =
  "20f59e427e278d42a2ae6f4218265742c69f4e4282bb1b4200831942eca40942d85cb641000048430000c842083a964214368d42fb258b42928b73424cad4d4231ab3e42093f3c42d38e0c42d828fb412299b841c6e7ad41b2a0a44296dd90422827874224e97c42f4606f425c795b42088b434229b7154206ff1442f61f074229a70442a620fa4120bc9b4160729b41bc109b41ce889441be73904110486e419878b940fa849142fc3c7d42e4d37442aed36f42c5b50d42877d0742e821a0411ae11341a871a4be4a1979c17d9794c18c26ebc17682e8bfec9823c1d2ae6cc03bca04c27ea111c10000000000000000bcda0b3f18478e40040b023f66ca9741097a96413c7eb14104090b41079b0b4236804a423a1624412c95ab41f8e0323f672c684136a909401fb4dc4134380e4188acfe400e6d3e425f60564040228d40b041904176f8dd41127986420bcc2a42b88bc041e7eaa9402ab50341e5f6f841a2dab840333c36426ae64fc106e5aac1a0eac5c19e42babf844ad8c139be78c198f65fc1101fa8bda089444163890b413a7f8a41c748b741979736422c2798413b18fc4024fde6414f3b73410000000000000000fcfb134234fb754246425b4140dc353f4487ce412cf53142ea844d41089ebb41bc42ed411c3d7641af131b41aea35ac0e48351c13f1a92c0b182a7c1892d8ac158c606c2406af6c1992d3ac1dd9ae2c149a950c2c608e7c0c0ff0dc268aaf3c1bf8b90c1aea004c21f564bc2db46c9c0a8a098bf5ee18cc20b3878c18de1d7c1e0c533c142ba1bc1ecd83cc106d411c20603e9c0907a30c0bea4a142fe288c42d48b6042a4c54e42ac414842f68a1542804a1442510b06429c18ac41264845435e58b24213c197428e4b8d4255e18c42ceb17542d0d64042d3293942f92f364293aa0f4296bc0c42b42fb841ceadb441ca69a542e67e984293338742c2248742a8c07c42ee3c6342923a5a429e07184213dc2042d6901f42301d0d42778a2442d6dfd74108a7b541baecc641de56e841bedfb3417a076f41ec9dc24123d19742081185424e427a427c4578424ab81942c07c224200eea94108d1134170d930bfd5e49ac143b4adc1e3180bc2248b4dbf3e6624c13e266bc034f6c6c1f5a3ecc000803bb9008890baf892bf3eb7ffc0400a16fd3f497ab04161009a416eddc941121a0d417b740d42cbf6354235603e4136ce9c41002c493eda48614199e90640ac88f64135230e41a69fac40dbb23c427bce3540a18b4d40f4ce5a41c7b0d84110816b42b4ddf741d01a98418d2510413dcc8b412331bd41efe896407578e64129fd98c1617010c2242005c23e4d85c05be37ac194fa68bf0178d2c028bacc3d46f2674121d83a413a349f416a60d141d6e0264272e8a2417c590f414c1cc241c4df634100e0f63a00b6003c1df73442b2b97442d4d78f41481be73f06bbca41d39c1642f48c674191c5a8414638b9413cc6794191c3354102e024c0262653c11276b8c07a3ad5c1d4d8c1c1e7b039c28ec129c2b5156ec1d82a26c2160a97c2626400c1bec74ac2fe5bf6c1465e87c13ab90dc2c5c47ec2581a2bc038ea0cbf06b38bc2488593c1f8140dc240a6b6c1689254c182c683c13e216cc2a03dd9c0028e10c031000000";

  uint8_t *astia_params = dt_exif_xmp_decode(
      astia_params_input, strlen(astia_params_input), &params_len);

  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(astia_params);
  dt_gui_presets_add_generic(_("Fuji Astia emulation"), self->op,
                             2, astia_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(astia_params);


  const char *chrome_params_input =
  "d303b542eb5a9742ccdd7d4288707142ee9d40427af718427062d641000048430000c8420d96bc42faeaae429c32aa423a6ca9423c9ba7425993a0424e639542788d9242a722894260eb7f42d2876b420c724442dcba4042b6c02b42a8990b421276de41ac68c2410790a542393b9242a7279242a45d8f42a132864230e57e42002145426c3f44428a0b274204e62342b092fd41d68fcd41e02cbb419e07bb41ac2433413247b742a3ad9242006a924293d98142ae892e422cd42642366a26429c7ec44175d738c170f6d7c16fbc62c0116916c25d263dc13639f4c1352ac7c0000000000000000050176d3fe59a98400047863f168f2a401e8d0a41d72e8c418626bb4110dd5341c02f0e4270d9b03ef8c9fd4116fbb9411f8f6542391bfa41a0872f42815d56415e5f06420deec841b2d5b141de5f0841ee252342db21154160bd43405af34f40d5688e42624ea741f1799641242473400a34294238e8114241ee0f41383f184052f118c1724989c18c3c9ec0cf0decc138a006c29d4f65c0ef399fc1ea1696c17ba0f7405e30a741a026964231230042f235c641d6eee641aa7a5a410000000000000000b421d241467c8142ae6de741f7a0ee40a00da9423cb40742d6f24240461c864112558741c9ae1542089484423d261242e79d0a427392c240668cd341d554b241dd0ced40e72188c1091983c1e40b55c1f7b6cdc1304713c2360f12c0b8ca24c06a8319c232e36dc2a96dffc185040ac00e1ae8c1449c95c2c20370c29c0736bf6cce33c12c2200c2d0235cc177a125c2aa6f4fc11aab49c1bcb428c274a900c14babb542f2118d42489f6a42e4de5442c2153142be3202428ef2be4137584743b41ac3428d7dc042f9e4a7422c8fac425b61b04217c69a42d69e9b4255ec974210fa8c4298b687428a7a714282ef5f4292923942805242423c032d4222a90e421665d841a0dbda4154d9aa4255269e425ac99842d51a9a42a8bf8b4244637e42ea414542eac56a4280184042bb6d3542a4070042bf650242a7c111425a620642466841414be5b34248d59042e58c95422ef8814264842c423bef2542bc3f3742e63ac141fb61aac16444c7c1b455523ff40b0ec259efe8c055ec9cc166182cc00000fab800007ab97fc70fc15aec44c1c0eaa4bf4e5fe84072b9f9c0cf0a0041e0859641ac1d5241bb43b641d2a95840ce0bdb41420ca541583e2842c50aba416d47f641188f51410313b5416eec9f41b120c041284ba040a6b2e3417c0ffbbf711224407cdd2f40d2a2364219c555c0daaef1407be03240a8b5b4412e221e402cc6bcbe3067883f51cbc5c1e74603c2d25b09c188a03bc2be01abc1b07bb0c029248cc131a90ac1320d4a41a82c6e416a983f42cd15b741b8ef8941c00e88415aeaee400080ed390010d63a78ed0242dcc74f427ad0de41c023394128677642a7aecb4154458440d4f8504140563b41a9c3e64150812542f354c6414e45ba41bab6c240b6a49241c3a15c412c6e08410c168ec108f28cc1707549c18795ecc1a2b80cc2b861c2bf40480bc035b8d1c13b7a27c2875cb7c18a91acbfc9cd7ac13b382fc27eed03c2003cbe3abf62ecc03433dec17f0a69c1b58ae7c1fc0df5c09cbf17c143b7d6c124d68ac031000000";

  uint8_t *chrome_params = dt_exif_xmp_decode
    (chrome_params_input, strlen(chrome_params_input), &params_len);

  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(chrome_params);
  dt_gui_presets_add_generic(_("Fuji Classic Chrome emulation"), self->op,
                             2, chrome_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(chrome_params);


  const char *mchrome_params_input =
  "287bc242632bb84226d3b54263b1a142befa904280da8942e09a88426c9d67425e6254420abc3042000048438be5aa4213ca99420d748842548c7c42d00a5942a46147422410444227060042b8bfff41348ec742c672b04293a7aa425e7f9d425e779b421a2c9a422b1f9a42fd0b87420a1e7b426e0772429e404a422a3e4a4220fc47423e8d414290c1e8412c6ddd412422cf41cce0b7419cc96441050bc4427c9fc142cebba142dbe0a04224bba04239449f4206e96e42bcec42428292e341b63ed641ca5f2dc02cfe09bfeab32cc0ca08ccc1a49ebbc1640dfcc09c6465bf7de528c2828667c19a8516c2000000002024e040c553d1419ee5594166cd9d4102e2164294636342ae0a19427699cb41a4e0de3e24a60a3fca0aa24112b99040fe569340f8adb441dc810d42aa00f740e048cc3f226070428bc677410000fa3f1053a840e46ed341aea6494144836441a2fd2f42a702824152a14142a2ea103f00e426c1c897d0c1f462f6c1fbfea9c1cb29f1c1175d1ac1efcfb9c1175407c281b891c19ced14c161f0d04192d26b42863e9a41fd251042c58c5041189b884282c51641d981fa416aa89d413b0e1e4100000000ca02b040c8fafa3ffde2b541a4fc0641c47e2e429fb2da404125b14124141a3f7c06a53fc0aae9be3817c0c16f24a8c09a8cabc1e0f6fac154eb25c2927530c2389b4fc1e97a4cc210946ec23e2934c148e702c2400ce8c1257492c2c1fe84c15e791ac2868f90c2599db5c2f66fe9c082aa61c09e38abc0585464bfcec916c2f6cfb8c16b022bc14d3275c26955a0c11a2946c146d9fac1ccf5be428046ac4247acbe4208b697427529894244c87f421ac5874230733d42722546425c5c07426aca474358f8b9421ea1a6427ee58d42e7208842d2416a426a656742fa625742012c0f4280bafb414f0ec542b457bf42a8eab14292dd9c421c95a242e5e4a54279da9942574c8842ff55914222fd7a420e9c4b42f8c44842c2da59421ae935421a45fa4126010c42ecdbd1418a2bd94140c36041ec10bf424b81a9425cfd8f421fa88b42abfb8742d9a9994298f23242ad2f12422a33bd41c8dabb41008ae3bc00b209bc8045e4bc00e87dbb0028a0ba00606aba0028a0ba0000fab700007ab900b0b3390000fa3880fdefbc00d2d7bb00c406bb00f8a7ba007014ba00b033ba0020cbb900a08c390010a43900349ebb8051e2bc003248bc0044c5bb00f6d1bb00ccd8bb00007abb0010a4ba00d004bb003072ba00803bb900007ab90060eab90000fa3700b0b3390010a4390060ea390060ea3900e8003a0007e4bc0008cfbb00a00cbb00940ebb0010a4ba00f47bbb0000fa3700803b390030f2390000fa3920a14f3e8081733de017503e0041eb3c00ec103c0060d13b0012133c0000c8b80020b23a008419bb00001639404f593e00e6433d0094723c0044133c00ec903b000c943b0068583b00040dbb005421bb001d713de0eb4e3ec097b63d00442c3d807d313d005d453d007ee53c004a123c00ca693c00d8d63b0070ad3a0070ad3a00b8533b008009b9001c22bb00e012bb00d04fbb003847bb00b86cbbc0334f3e802f3e3d004e6d3c0038793c0012133c005fe63c008009b90088dbba007c5dbb00705fbb31000000";

  uint8_t *mchrome_params = dt_exif_xmp_decode
    (mchrome_params_input, strlen(mchrome_params_input), &params_len);

  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(mchrome_params);
  dt_gui_presets_add_generic(_("Fuji Monochrome emulation"), self->op,
                             2, mchrome_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(mchrome_params);


  const char *provia_params_input =
  "aa1fae42b13a98429c8997420bbc8f4264bb81424e3f76423a034642de774542b8522142000048430000c8422467bc42f123b2422c209e4282049842fc5b9342567d8b423c50704286f657424e153842deec2f4239fc0d428857de41de0aca414552bd4233bdb342973099428ddb95420af59442f7df9442f0a89442a73d874206ff75428c79704248b5484214c93e42aaee344234af074246a0d04156a284412c803b41f8d7ba4248029d42ddd3964200e884421e123142485c2c42c80e2c42ce24c441ff528ec1f8f123c14b9869c05c0bfdc18c4191bf6dc517c25d1ad6c1f2cd3ec176a711c200000000000000003242bd3fce19a2407cc67a41c7b6784152e27a41982e1142ecbd9f4142e53142f0da7d423b50ff41e574314270501140f6fad04154c232414eef50402f2ce040164c1c4184deb64190aa8f4048930a42bd5d46409d2f6642a6bd4841704e5c40e18dd441b6b79a42ca88dc41ee6e5542333e7d413cc16d3e39061ec16f90cec1c6736ac1143cefc14e0ad8c180ce9dc181d75dc0f5da2dc1b2ce4141fd67a4414d0d26427e43c6419a48664289f20042a8713f42c7dbc441c3dd52410000000000000000a1cd1242fab58242300db2427767e94004a1cd41aa56844166861442a95c5542b9287a41c117b340f682cb414e54c440fdeb76411c4c0bc1469f58c0cce3f0c1537f02c1c7768ac13a0a9ec1d151cdc1a43e47c0946b09c2e9b036c2b8de42c0a5de98c15c0722c2934588c22a7911c2ef9cddc1377a1ec072313dc18f46f2c125f1f7c0acb628c2367522c1fe682bc2c68d55c1af28ccc1ff7ab44211c69742e6f08d42e2918942b03c7842061e6c4265603b42dd9f3942ae882142cc0e48430e6dc842e4f5c2429960b942005490427ab3994210e68c4225cc86427ea6664270774a42fcf6394250a931427a111642226bce41de78d441963fc3425c07b44204ad9b42b72d9d42f9cb9f42d1f59c42bd9c9c4221488742c23a854240d87f4264c648426cb54a4264ce5642f4d92d429ef80d42accba741007f3b4154cabc42993ba44260959b422b7396421c5a3742f48a4a42397a2c429c51e14190161fc222ff73c16fe39dc0cbbd33c2e00058bffabb4bc283daf8c181095ac138a6f4c10000fa3800007a386881b1c15b5c03c24454f83f04aaa64170cd9141ca3cd641a618bc415d2c2042e1bf5542fd60054232552a42b6da20408ab1c14178bfa140f258b440c0e3ba3d66036e414efafa41aa6a3340158303424c05fe3fcbf3344231607a40a2e66440a045da4109637d425dbb6741f4002542b7c23141b018ff3d9b08fac10b2f6cc231a3c3c11e1a72c21ceed2c1b33887c1346393c0d2a38ac0c4c7b9416c71c34101e52d4208cce641b8fd5842397b14429dda1b42e4a2c841aab68d41000048b8000016b9a12f504214e69c422a9e8d42e6791241c41ed941b39a4a417a52144297102642dc4e2b41a152ca40086ac441748eb3404a6369413aac87c09cef18c1bb1805c2be0f4bc1a7bce6c1bc6701c26233f4c1b6b040c0909a26c2c2e040c290ca65c0aaa4b2c1bce85ac2df088fc2423808c2f7d5b5c1255fbcbfd0ad1cc1eef8eac10e2832c18df519c2df67f4c0accb37c26cf164c1f460a3c131000000";

  uint8_t *provia_params = dt_exif_xmp_decode
    (provia_params_input, strlen(provia_params_input), &params_len);

  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(provia_params);
  dt_gui_presets_add_generic(_("Fuji Provia emulation"), self->op,
                             2, provia_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(provia_params);


  const char *velvia_params_input =
  "3f259c42b92693425c7b83420e107d42f86e4f4252a94b4293c32042db870442269da341000048430000c8427ee97f42ceca7342e81e6b42c9eb3e425514254248600f42c0fc0242ea69e941022bcd414624994222cb8d42f57d8842d77587428cea6e421c546c42b2a668429eda5e42da4a5e42242f2f42f37a1542c0fd0d42d0e30842867bab414eeca34154c46941482b5f41d08646415e552c41c512a5423390964242c7914260c07e42ea6176429c79744286010e4273310b42d6a28541fa0a4a41ca2161c0af9206c045d4f4c07ec5c3c1633ccec0d57efac17e2981c1f8449ec112a734c00000000000000000ad5fd440cb8a9441e0fab740a649a941f85d6b41387b2541888d2e42853cc241c33ad0406843c4408eb22d41c016713d7fd79541da99953f7d70c241ba600142f0d0273fd25e0541ceda4e42456b944138a29d41f76448424a941c41d0cc1642a54ba0412c030c428342874106e0e54032bfbdbfab3a48c13fe059c1d141a0c1e655c1c1ac9c49c190d038c1e3c242c094c185c0217c5ac075074e410485174251beb941c0c422412bf53c4282ada0410571a64130a5d93f584cab3e000000000000000004d88f4229c6ba4053185a41e8d51f4268579f41302c503f87e59a410806fe4085f0cf40e67992c190b1ccc0e75c45c19ee3d1c16677a1c11b6e81c1461c06c26c192cc1ef3128c2378125c29272b0c142de69c2154e7bc120564cc2d4a807c2aa6f15c12e2e82c20fa010c200327cc1fe8a4dc1502e4cc0a6debec11a4609c230e38cc112a5c5c042f01dc2b4aa7ec1fd3986c15abf8dc0282aa242f202994250707d429aed7b42604a51424c8b4e42efac1f4276070e426420a441d3d84443567fae4219ce83425a567b4214286242a8554642f1421e42c3f10d427cab1c426af6f5416221ce416de0a14206bf9242de7e8842d21d9142668d7d42465c7e42acb57c428ada5e42f4516242eaf9514232971f42c7522042028e2b42747af9410c8aef4158809141603adb4150e2a7411e1815413287a7429d2d9a420bea9c429a418d428ea5864280877f42687f3142e5cb0f42d85b9f4160000d41c30fbec0b4246fc03f0f46c19b1c1ac2f36b08c1f2513cc2b239b4c196fda7c1123632c000409cb90010a4ba349c76416a78ea410249f3404dfd00427f41974148854d4140604c42c70edc413bf6064131cc684008178941bcb2653fa9edaf4160fe4d40b8121a4222fd2a420238c03fd436d8405e0577429e85bb41f7b899419b5469426c50c541f7e217425da58e41c99c1442ef1690417ac27e416b5e56c0a5d1a5c12405f6c12c5e1bc26ab106c2c5a59ec142693dc0f43a11c082d65140698887c0efab9c41c5de6842b0e8054221f29041eeab36420440f241673fc6410201b4404822063f00e0123b001a1a3ce60f8242e6631e41ef649b41813329425bfeb741fea0973ff9f8d0419a453f41362007412eee15c128293fc18667b0c12eb0acc14bb20fc213a7ebc1281c0dc29cd587c1f61739c2f7974cc2ac6c08c2003c8fc2389bb6c119b5a2c214a74ec266f4ecc05264b6c2107819c2f476a9c17398a8c05af39dc02d6e5cc16d31cec11095f4c1fe9e20c1bfbd76c2d3adc1c12fea7fc196bf11c131000000";

  uint8_t *velvia_params = dt_exif_xmp_decode
    (velvia_params_input, strlen(velvia_params_input), &params_len);

  assert(params_len == 6 * MAX_PATCHES * (int)sizeof(float) + (int)sizeof(int32_t));
  assert(velvia_params);
  dt_gui_presets_add_generic(_("Fuji Velvia emulation"), self->op,
                             2, velvia_params, params_len,
                             TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);
  free(velvia_params);
}

// thinplate spline kernel \phi(r) = 2 r^2 ln(r)
DT_OMP_DECLARE_SIMD(aligned(x, y))
static inline float kernel(const dt_aligned_pixel_t x,
                           const dt_aligned_pixel_t y)
{
  dt_aligned_pixel_t diff2;
  for_each_channel(c)
  {
    diff2[c] = (x[c] - y[c]);
    diff2[c] *= diff2[c];
  }
  const float r2 = diff2[0] + diff2[1] + diff2[2];
  return r2*fastlog(MAX(1e-8f,r2));
}

// num / den, with den clamped away from zero (keeping its sign) so a
// single-patch calibration (N==1, see commit_params()) whose source
// coordinate happens to be ~0 in the working space (Lab a/b near a
// neutral, or a near-zero XYZ channel) cannot produce Inf/NaN spline
// coefficients that would then propagate into every pixel.
static inline float _safe_ratio(const float num, const float den)
{
  const float eps = 1e-6f;
  const float d = (fabsf(den) > eps) ? den : copysignf(eps, den);
  return num / d;
}

// defined with the other chart-calibration helpers further down
static void _extract_patches(const float *const restrict in,
                             const dt_iop_roi_t *const roi_in,
                             dt_iop_module_t *self,
                             dt_iop_colorchecker_gui_data_t *g,
                             const dt_iop_order_iccprofile_info_t *const work_profile);
static void _update_delta_E_label(dt_iop_module_t *self);
static inline float _delta_E_2000(const float Lab_ref[3], const float Lab_test[3]);

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/,
                                        self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  // color chart calibration: extract the measured patch colors from the
  // preview input on demand (OpenCL is disabled in commit_params meanwhile).
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(self->dev->gui_attached && g && g->checker && g->run_profile
     && dt_pipe_is_preview(piece->pipe))
  {
    const dt_iop_order_iccprofile_info_t *const wp =
      dt_ioppr_get_pipe_work_profile_info(piece->pipe);
    dt_iop_gui_enter_critical_section(self);
    _extract_patches((const float *)ivoid, roi_in, self, g, wp);
    g->run_profile = FALSE;
    g->profile_ready = TRUE;
    dt_iop_gui_leave_critical_section(self);
  }

  const dt_iop_colorchecker_data_t *const data = piece->data;
  // in XYZ mode the pipeline hands us linear RGB: convert to XYZ, apply the
  // spline (trained in normalized XYZ) and convert back. In Lab mode the
  // pipeline already gives us Lab, so this is a pure pass-through.
  const gboolean xyz_mode = (data->colorspace == DT_IOP_CC_CS_XYZ) && data->has_matrix;
  const float scale_in = data->scale_in;
  const float scale_out = data->scale_out;
  const size_t npixels = (size_t)roi_out->height * (size_t)roi_out->width;
  float *const restrict out = (float*)DT_IS_ALIGNED(ovoid);

  // convert patch data from struct of arrays to array of structs so
  // we can vectorize operations
  const int num_patches = data->num_patches;
  dt_aligned_pixel_t *sources = dt_alloc_align_type(dt_aligned_pixel_t, num_patches);
  for(int i = 0; i < num_patches; i++)
  {
    sources[i][0] = data->source_Lab[3 * i];
    sources[i][1] = data->source_Lab[3 * i + 1];
    sources[i][2] = data->source_Lab[3 * i + 2];
    sources[i][3] = 0.0f;
  }
  dt_aligned_pixel_t *patches = dt_alloc_align_type(dt_aligned_pixel_t, (num_patches + 1));
  for(int i = 0; i <= num_patches; i++)
  {
    patches[i][0] = data->coeff_L[i];
    patches[i][1] = data->coeff_a[i];
    patches[i][2] = data->coeff_b[i];
    patches[i][3] = 0.0f;
  }
  const dt_aligned_pixel_t polynomial_L =
    { data->coeff_L[num_patches+1],
      data->coeff_L[num_patches+2],
      data->coeff_L[num_patches+3], 0.0f };
  const dt_aligned_pixel_t polynomial_a =
    { data->coeff_a[num_patches+1],
      data->coeff_a[num_patches+2],
      data->coeff_a[num_patches+3], 0.0f };
  const dt_aligned_pixel_t polynomial_b =
    { data->coeff_b[num_patches+1],
      data->coeff_b[num_patches+2],
      data->coeff_b[num_patches+3], 0.0f };

  const float *const min = data->matrix_in;
  const float *const mout = data->matrix_out;

  DT_OMP_FOR()
  for(int k=0; k < npixels; k++)
  {
    dt_aligned_pixel_t inpx;
    copy_pixel(inpx, ((float *)ivoid) + 4*k);

    dt_aligned_pixel_t working_in;
    if(xyz_mode)
    {
      working_in[0] = (min[0]*inpx[0] + min[1]*inpx[1] + min[2]*inpx[2]) * scale_in;
      working_in[1] = (min[3]*inpx[0] + min[4]*inpx[1] + min[5]*inpx[2]) * scale_in;
      working_in[2] = (min[6]*inpx[0] + min[7]*inpx[1] + min[8]*inpx[2]) * scale_in;
      working_in[3] = 0.0f;
    }
    else
    {
      copy_pixel(working_in, inpx);
    }

    // polynomial part:
    dt_aligned_pixel_t poly_L, poly_a, poly_b;
    for_each_channel(c)
    {
      poly_L[c] = (polynomial_L[c] * working_in[c]);
      poly_a[c] = (polynomial_a[c] * working_in[c]);
      poly_b[c] = (polynomial_b[c] * working_in[c]);
    }
    dt_aligned_pixel_t sums = { poly_L[0] + poly_L[1] + poly_L[2],
                                poly_a[0] + poly_a[1] + poly_a[2],
                                poly_b[0] + poly_b[1] + poly_b[2],
                                0.0f };
    dt_aligned_pixel_t working_res;
    for_each_channel(c)
      working_res[c] = patches[num_patches][c] + sums[c];
    for(int p=0; p < num_patches; p++)
    {
      // rbf from thin plate spline
      const float phi = kernel(working_in, sources[p]);
      for_each_channel(c)
        working_res[c] += patches[p][c] * phi;
    }

    dt_aligned_pixel_t outpx;
    if(xyz_mode)
    {
      const float x = working_res[0] * scale_out;
      const float y = working_res[1] * scale_out;
      const float z = working_res[2] * scale_out;
      outpx[0] = mout[0]*x + mout[1]*y + mout[2]*z;
      outpx[1] = mout[3]*x + mout[4]*y + mout[5]*z;
      outpx[2] = mout[6]*x + mout[7]*y + mout[8]*z;
      outpx[3] = inpx[3];
    }
    else
    {
      copy_pixel(outpx, working_res);
      outpx[3] = inpx[3];
    }
    copy_pixel_nontemporal(out + 4*k, outpx);
  }
  dt_omploop_sfence();
  dt_free_align(patches);
  dt_free_align(sources);
}


#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_colorchecker_data_t *d = piece->data;
  dt_iop_colorchecker_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;
  const int width = roi_out->width;
  const int height = roi_out->height;
  const int num_patches = d->num_patches;
  const int mode = (d->colorspace == DT_IOP_CC_CS_XYZ && d->has_matrix) ? 1 : 0;

  cl_int err = DT_OPENCL_DEFAULT_ERROR;
  cl_mem dev_params = NULL;
  cl_mem dev_matrix_in = NULL;
  cl_mem dev_matrix_out = NULL;

  const size_t params_size =
    (size_t)(4 * (2 * num_patches + 4)) * sizeof(float);
  float *params = malloc(params_size);
  float *idx = params;

  // re-arrange data->source_Lab and data->coeff_{L,a,b} into float4
  for(int n = 0; n < num_patches; n++, idx += 4)
  {
    idx[0] = d->source_Lab[3 * n];
    idx[1] = d->source_Lab[3 * n + 1];
    idx[2] = d->source_Lab[3 * n + 2];
    idx[3] = 0.0f;
  }

  for(int n = 0; n < num_patches + 4; n++, idx += 4)
  {
    idx[0] = d->coeff_L[n];
    idx[1] = d->coeff_a[n];
    idx[2] = d->coeff_b[n];
    idx[3] = 0.0f;
  }

  dev_params = dt_opencl_copy_host_to_device_constant(devid, params_size,
                                                      params);
  if(dev_params == NULL) goto error;

  dev_matrix_in = dt_opencl_copy_host_to_device_constant(devid, 9 * sizeof(float),
                                                         d->matrix_in);
  dev_matrix_out = dt_opencl_copy_host_to_device_constant(devid, 9 * sizeof(float),
                                                          d->matrix_out);
  if(dev_matrix_in == NULL || dev_matrix_out == NULL) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid,
                                         gd->kernel_colorchecker,
                                         width, height,
                                         CLARG(dev_in), CLARG(dev_out),
                                         CLARG(width), CLARG(height),
                                         CLARG(num_patches),
                                         CLARG(dev_params),
                                         CLARG(mode),
                                         CLARG(dev_matrix_in),
                                         CLARG(dev_matrix_out),
                                         CLARG(d->scale_in), CLARG(d->scale_out));
error:
  free(params);
  dt_opencl_release_mem_object(dev_matrix_out);
  dt_opencl_release_mem_object(dev_matrix_in);
  dt_opencl_release_mem_object(dev_params);
  return err;
}
#endif


void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_colorchecker_params_t *p = (const dt_iop_colorchecker_params_t *)p1;
  dt_iop_colorchecker_data_t *d = piece->data;

  // patch extraction is CPU-only: disable OpenCL while a calibration runs
  if(self->dev->gui_attached && self->gui_data)
  {
    const dt_iop_colorchecker_gui_data_t *const g = self->gui_data;
    if(g->run_profile) piece->process_cl_ready = FALSE;
  }

  d->colorspace = p->colorspace;
  d->anchor = p->anchor;
  d->scale_in = 1.0f;
  d->scale_out = 1.0f;
  d->has_matrix = FALSE;
  memset(d->matrix_in, 0, sizeof(d->matrix_in));
  memset(d->matrix_out, 0, sizeof(d->matrix_out));

  // Effective patch set: the committed parameters, or the live measurement
  // when a calibration has just been computed (preview before validation).
  dt_iop_colorchecker_params_t cal = *p;
  // If that measurement was taken in XYZ mode, its raw (pre-Lab) XYZ is
  // still cached on the gui data: reuse it below instead of converting
  // the Lab it was derived from back to XYZ a second time.
  gboolean have_cached_source_xyz = FALSE;
  const float *cached_source_XYZ = NULL;
  if(self->dev->gui_attached && self->gui_data)
  {
    const dt_iop_colorchecker_gui_data_t *const g = self->gui_data;
    if(g->profile_ready && g->measured_lab && g->checker)
    {
      const int n = MIN(MAX_PATCHES, (int)g->checker->patches);
      cal.num_patches = n;
      for(int k = 0; k < n; k++)
      {
        cal.source_L[k] = g->measured_lab[3 * k + 0];
        cal.source_a[k] = g->measured_lab[3 * k + 1];
        cal.source_b[k] = g->measured_lab[3 * k + 2];
        cal.target_L[k] = g->checker->values[k].Lab[0];
        cal.target_a[k] = g->checker->values[k].Lab[1];
        cal.target_b[k] = g->checker->values[k].Lab[2];
      }
      // g->measured_xyz_valid reflects the colorspace mode that was active
      // at extraction time; only trust the cache if that still matches.
      have_cached_source_xyz = g->measured_xyz_valid && g->measured_XYZ;
      cached_source_XYZ = g->measured_XYZ;
    }
  }

  d->num_patches = MIN(MAX_PATCHES, cal.num_patches);
  const unsigned N = MAX(0, d->num_patches);
  const unsigned N4 = N + 4;

  // Working copy of the patch coordinates in the space where the spline is
  // solved: CIE Lab as-is, or CIE XYZ (D50) in XYZ mode. In XYZ mode the Lab
  // references are converted to XYZ and normalized by the luminance of an
  // anchor patch (white or middle gray) so that the spline sees a scale
  // comparable to the reference chart while the image exposure is preserved.
  dt_iop_colorchecker_params_t wp = cal;

  if(p->colorspace == DT_IOP_CC_CS_XYZ)
  {
    const dt_iop_order_iccprofile_info_t *profile = dt_ioppr_get_pipe_work_profile_info(pipe);
    if(profile && dt_is_valid_colormatrix(profile->matrix_in[0][0])
       && dt_is_valid_colormatrix(profile->matrix_out[0][0]))
    {
      d->has_matrix = TRUE;
      for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
        {
          d->matrix_in[3 * r + c] = profile->matrix_in[r][c];
          d->matrix_out[3 * r + c] = profile->matrix_out[r][c];
        }
    }

    if(d->has_matrix)
    {
      for(unsigned k = 0; k < N; k++)
      {
        dt_aligned_pixel_t lab, xyz;
        if(have_cached_source_xyz)
        {
          // already computed once from the raw measurement in _extract_patches()
          xyz[0] = cached_source_XYZ[3 * k + 0];
          xyz[1] = cached_source_XYZ[3 * k + 1];
          xyz[2] = cached_source_XYZ[3 * k + 2];
        }
        else
        {
          lab[0] = wp.source_L[k]; lab[1] = wp.source_a[k]; lab[2] = wp.source_b[k]; lab[3] = 0.0f;
          dt_Lab_to_XYZ(lab, xyz);
        }
        wp.source_L[k] = xyz[0]; wp.source_a[k] = xyz[1]; wp.source_b[k] = xyz[2];

        // the chart reference is only ever available in Lab, so the target
        // always needs this conversion
        lab[0] = wp.target_L[k]; lab[1] = wp.target_a[k]; lab[2] = wp.target_b[k]; lab[3] = 0.0f;
        dt_Lab_to_XYZ(lab, xyz);
        wp.target_L[k] = xyz[0]; wp.target_a[k] = xyz[1]; wp.target_b[k] = xyz[2];
      }

      // generic channel indices: 0 = X, 1 = Y, 2 = Z
      float ws = 1.0f, wt = 1.0f;
      if(p->anchor != DT_IOP_CC_ANCHOR_NONE && N > 0)
      {
        int best = 0;
        float best_score = -FLT_MAX;
        for(unsigned k = 0; k < N; k++)
        {
          const float score = (p->anchor == DT_IOP_CC_ANCHOR_GRAY)
                                ? -fabsf(wp.target_a[k] - 0.18f) // closest to middle gray
                                : wp.target_a[k];                // brightest patch (white)
          if(score > best_score)
          {
            best_score = score;
            best = k;
          }
        }
        ws = wp.source_a[best];
        wt = wp.target_a[best];
      }
      if(ws <= 1e-6f) ws = 1.0f;
      if(wt <= 1e-6f) wt = 1.0f;

      d->scale_in = 1.0f / ws;
      d->scale_out = ws; // preserve the image exposure

      for(unsigned k = 0; k < N; k++)
      {
        wp.source_L[k] /= ws; wp.source_a[k] /= ws; wp.source_b[k] /= ws;
        wp.target_L[k] /= wt; wp.target_a[k] /= wt; wp.target_b[k] /= wt;
      }
    }
  }

  // From here on the spline is solved on the working-space copy.
  p = &wp;

  for(unsigned k = 0; k < N; ++k)
  {
    d->source_Lab[3*k+0] = p->source_L[k];
    d->source_Lab[3*k+1] = p->source_a[k];
    d->source_Lab[3*k+2] = p->source_b[k];
  }

  // initialize coefficients with default values that will be
  // used for N<=4 and if coefficient matrix A is singular
  for(unsigned i = 0; i < N4; ++i)
  {
    d->coeff_L[i] = 0;
    d->coeff_a[i] = 0;
    d->coeff_b[i] = 0;
  }
  d->coeff_L[N + 1] = 1;
  d->coeff_a[N + 2] = 1;
  d->coeff_b[N + 3] = 1;

  /*
      Following

      K. Anjyo, J. P. Lewis, and F. Pighin, "Scattered data
      interpolation for computer graphics," ACM SIGGRAPH 2014 Courses
      on - SIGGRAPH ’14, 2014.
      http://dx.doi.org/10.1145/2614028.2615425
      http://scribblethink.org/Courses/ScatteredInterpolation/scatteredinterpcoursenotes.pdf

      construct the system matrix and the vector of function values and
      solve the set of linear equations

      / R   P \  / c \   / f \
      |       |  |   | = |   |
      \ P^t 0 /  \ d /   \ 0 /

      for the coefficient vector (c d)^t.

      By design of the interpolation scheme the interpolation
      coefficients c for radial non-linear basis functions (the kernel)
      must always vanish for N<=4.  For N<4 the (N+4)x(N+4) coefficient
      matrix A is singular, the linear system has non-unique solutions.
      Thus the cases with N<=4 need special treatment, unique solutions
      are found by setting some of the unknown coefficients to zero and
      solving a smaller linear system.
  */
  switch(N)
  {
  case 0:
    break;
  case 1:
    // interpolation via constant function
    d->coeff_L[N + 1] = _safe_ratio(p->target_L[0], p->source_L[0]);
    d->coeff_a[N + 2] = _safe_ratio(p->target_a[0], p->source_a[0]);
    d->coeff_b[N + 3] = _safe_ratio(p->target_b[0], p->source_b[0]);
    break;
  case 2:
    // interpolation via single constant function and the linear
    // function of the corresponding color channel
    {
      double A[2 * 2] = { 1, p->source_L[0],
                          1, p->source_L[1] };
      double b[2] = { p->target_L[0], p->target_L[1] };
      if(!gauss_solve(A, b, 2)) break;
      d->coeff_L[N + 0] = b[0];
      d->coeff_L[N + 1] = b[1];
    }
    {
      double A[2 * 2] = { 1, p->source_a[0],
                          1, p->source_a[1] };
      double b[2] = { p->target_a[0], p->target_a[1] };
      if(!gauss_solve(A, b, 2)) break;
      d->coeff_a[N + 0] = b[0];
      d->coeff_a[N + 2] = b[1];
    }
    {
      double A[2 * 2] = { 1, p->source_b[0],
                          1, p->source_b[1] };
      double b[2] = { p->target_b[0], p->target_b[1] };
      if(!gauss_solve(A, b, 2)) break;
      d->coeff_b[N + 0] = b[0];
      d->coeff_b[N + 3] = b[1];
    }
    break;
  case 3:
    // interpolation via single constant function, the linear function
    // of the corresponding color channel and the linear functions
    // of the other two color channels having both the same weight
    {
      double A[3 * 3] = { 1, p->source_L[0], p->source_a[0] + p->source_b[0],
                          1, p->source_L[1], p->source_a[1] + p->source_b[1],
                          1, p->source_L[2], p->source_a[2] + p->source_b[2] };
      double b[3] = { p->target_L[0], p->target_L[1], p->target_L[2] };
      if(!gauss_solve(A, b, 3)) break;
      d->coeff_L[N + 0] = b[0];
      d->coeff_L[N + 1] = b[1];
      d->coeff_L[N + 2] = b[2];
      d->coeff_L[N + 3] = b[2];
    }
    {
      double A[3 * 3] = { 1, p->source_a[0], p->source_L[0] + p->source_b[0],
                          1, p->source_a[1], p->source_L[1] + p->source_b[1],
                          1, p->source_a[2], p->source_L[2] + p->source_b[2] };
      double b[3] = { p->target_a[0], p->target_a[1], p->target_a[2] };
      if(!gauss_solve(A, b, 3)) break;
      d->coeff_a[N + 0] = b[0];
      d->coeff_a[N + 1] = b[2];
      d->coeff_a[N + 2] = b[1];
      d->coeff_a[N + 3] = b[2];
    }
    {
      double A[3 * 3] = { 1, p->source_b[0], p->source_L[0] + p->source_a[0],
                          1, p->source_b[1], p->source_L[1] + p->source_a[1],
                          1, p->source_b[2], p->source_L[2] + p->source_a[2] };
      double b[3] = { p->target_b[0], p->target_b[1], p->target_b[2] };
      if(!gauss_solve(A, b, 3)) break;
      d->coeff_b[N + 0] = b[0];
      d->coeff_b[N + 1] = b[2];
      d->coeff_b[N + 2] = b[2];
      d->coeff_b[N + 3] = b[1];
    }
    break;
  case 4:
  {
    // interpolation via constant function and 3 linear functions
    double A[4 * 4] = { 1, p->source_L[0], p->source_a[0], p->source_b[0],
                        1, p->source_L[1], p->source_a[1], p->source_b[1],
                        1, p->source_L[2], p->source_a[2], p->source_b[2],
                        1, p->source_L[3], p->source_a[3], p->source_b[3] };
    int pivot[4];
    if(!gauss_make_triangular(A, pivot, 4)) break;
    {
      double b[4] = { p->target_L[0],
                      p->target_L[1],
                      p->target_L[2],
                      p->target_L[3] };
      gauss_solve_triangular(A, pivot, b, 4);
      d->coeff_L[N + 0] = b[0];
      d->coeff_L[N + 1] = b[1];
      d->coeff_L[N + 2] = b[2];
      d->coeff_L[N + 3] = b[3];
    }
    {
      double b[4] = { p->target_a[0],
                      p->target_a[1],
                      p->target_a[2],
                      p->target_a[3] };
      gauss_solve_triangular(A, pivot, b, 4);
      d->coeff_a[N + 0] = b[0];
      d->coeff_a[N + 1] = b[1];
      d->coeff_a[N + 2] = b[2];
      d->coeff_a[N + 3] = b[3];
    }
    {
      double b[4] = { p->target_b[0],
                      p->target_b[1],
                      p->target_b[2],
                      p->target_b[3] };
      gauss_solve_triangular(A, pivot, b, 4);
      d->coeff_b[N + 0] = b[0];
      d->coeff_b[N + 1] = b[1];
      d->coeff_b[N + 2] = b[2];
      d->coeff_b[N + 3] = b[3];
    }
    break;
  }
  default:
  {
    // setup linear system of equations
    double *A = malloc(sizeof(*A) * N4 * N4);
    double *b = malloc(sizeof(*b) * N4);
    // coefficients from nonlinear radial kernel functions
    for(unsigned j = 0; j < N; ++j)
      for(unsigned i = j; i < N; ++i)
        A[j * N4 + i] = A[i * N4 + j] =
          kernel(d->source_Lab + 3 * i, d->source_Lab + 3 * j);
    // coefficients from constant and linear functions
    for(unsigned i = 0; i < N; ++i)
      A[i * N4 + N + 0] = A[(N + 0) * N4 + i] = 1;
    for(unsigned i = 0; i < N; ++i)
      A[i * N4 + N + 1] = A[(N + 1) * N4 + i] = d->source_Lab[3 * i + 0];
    for(unsigned i = 0; i < N; ++i)
      A[i * N4 + N + 2] = A[(N + 2) * N4 + i] = d->source_Lab[3 * i + 1];
    for(unsigned i = 0; i < N; ++i)
      A[i * N4 + N + 3] = A[(N + 3) * N4 + i] = d->source_Lab[3 * i + 2];
    // lower-right zero block
    for(unsigned j = N; j < N4; ++j)
      for(unsigned i = N; i < N4; ++i) A[j * N4 + i] = 0;

    // make coefficient matrix triangular
    int *pivot = malloc(sizeof(*pivot) * N4);
    if(gauss_make_triangular(A, pivot, N4))
    {
      // calculate coefficients for L channel
      for(unsigned i = 0; i < N; ++i)
        b[i] = p->target_L[i];
      for(unsigned i = N; i < N4; ++i)
        b[i] = 0;
      gauss_solve_triangular(A, pivot, b, N4);
      for(unsigned i = 0; i < N4; ++i)
        d->coeff_L[i] = b[i];
      // calculate coefficients for a channel
      for(unsigned i = 0; i < N; ++i)
        b[i] = p->target_a[i];
      for(unsigned i = N; i < N4; ++i)
        b[i] = 0;
      gauss_solve_triangular(A, pivot, b, N4);
      for(unsigned i = 0; i < N4; ++i)
        d->coeff_a[i] = b[i];
      // calculate coefficients for b channel
      for(unsigned i = 0; i < N; ++i)
        b[i] = p->target_b[i];
      for(unsigned i = N; i < N4; ++i)
        b[i] = 0;
      gauss_solve_triangular(A, pivot, b, N4);
      for(unsigned i = 0; i < N4; ++i)
        d->coeff_b[i] = b[i];
    }
    // free resources
    free(pivot);
    free(b);
    free(A);
  }
  }
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_colorchecker_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_reset(dt_iop_module_t *self)
{
  dt_iop_color_picker_reset(self, TRUE);
}

void _colorchecker_rebuild_patch_list(dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  if(g->patch >= p->num_patches
     || g->patch < 0)
    return;

  if(dt_bauhaus_combobox_length(g->combobox_patch) != p->num_patches)
  {
    dt_bauhaus_combobox_clear(g->combobox_patch);
    char cboxentry[1024];
    for(int k=0;k<p->num_patches;k++)
    {
      snprintf(cboxentry, sizeof(cboxentry), _("patch #%d"), k);
      dt_bauhaus_combobox_add(g->combobox_patch, cboxentry);
    }
    if(p->num_patches <= 24)
      dtgtk_drawing_area_set_aspect_ratio(g->area, 2.0/3.0);
    else
      dtgtk_drawing_area_set_aspect_ratio(g->area, 1.0);
    // FIXME: why not just use g->patch for everything?
    g->drawn_patch = dt_bauhaus_combobox_get(g->combobox_patch);
  }
}

void _colorchecker_update_sliders(dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  // while a measurement is pending, show the measured source and the reference
  // target instead of the committed parameters
  const gboolean preview = g->profile_ready && g->measured_lab && g->checker;
  const int n_patches = preview ? (int)g->checker->patches : p->num_patches;

  if(g->patch >= n_patches
     || g->patch < 0)
    return;

  float src[3], tgt[3];
  if(preview)
  {
    src[0] = g->measured_lab[3 * g->patch + 0];
    src[1] = g->measured_lab[3 * g->patch + 1];
    src[2] = g->measured_lab[3 * g->patch + 2];
    tgt[0] = g->checker->values[g->patch].Lab[0];
    tgt[1] = g->checker->values[g->patch].Lab[1];
    tgt[2] = g->checker->values[g->patch].Lab[2];
  }
  else
  {
    src[0] = p->source_L[g->patch];
    src[1] = p->source_a[g->patch];
    src[2] = p->source_b[g->patch];
    tgt[0] = p->target_L[g->patch];
    tgt[1] = p->target_a[g->patch];
    tgt[2] = p->target_b[g->patch];
  }

  if(g->absolute_target)
  {
    dt_bauhaus_slider_set(g->scale_L, tgt[0]);
    dt_bauhaus_slider_set(g->scale_a, tgt[1]);
    dt_bauhaus_slider_set(g->scale_b, tgt[2]);
    const float Cout = sqrtf(tgt[1] * tgt[1] + tgt[2] * tgt[2]);
    dt_bauhaus_slider_set(g->scale_C, Cout);
  }
  else
  {
    dt_bauhaus_slider_set(g->scale_L, tgt[0] - src[0]);
    dt_bauhaus_slider_set(g->scale_a, tgt[1] - src[1]);
    dt_bauhaus_slider_set(g->scale_b, tgt[2] - src[2]);
    const float Cin = sqrtf(src[1] * src[1] + src[2] * src[2]);
    const float Cout = sqrtf(tgt[1] * tgt[1] + tgt[2] * tgt[2]);
    dt_bauhaus_slider_set(g->scale_C, Cout - Cin);
  }
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  const dt_iop_colorchecker_params_t *const p = self->params;

  _colorchecker_rebuild_patch_list(self);
  _colorchecker_update_sliders(self);
  _update_delta_E_label(self);

  if(g->combobox_anchor)
    gtk_widget_set_visible(g->combobox_anchor, p->colorspace == DT_IOP_CC_CS_XYZ);

  gtk_widget_queue_draw(g->area);
}

void init(dt_iop_module_t *self)
{
  self->params = calloc(1, sizeof(dt_iop_colorchecker_params_t));
  self->default_params = calloc(1, sizeof(dt_iop_colorchecker_params_t));
  self->default_enabled = FALSE;
  self->params_size = sizeof(dt_iop_colorchecker_params_t);
  self->gui_data = NULL;

  dt_iop_colorchecker_params_t *d = self->default_params;
  d->num_patches = colorchecker_patches;
  d->colorspace = DT_IOP_CC_CS_XYZ;
  d->anchor = DT_IOP_CC_ANCHOR_WHITE;
  for(int k = 0; k < d->num_patches; k++)
  {
    d->source_L[k] = d->target_L[k] = colorchecker_Lab[3*k+0];
    d->source_a[k] = d->target_a[k] = colorchecker_Lab[3*k+1];
    d->source_b[k] = d->target_b[k] = colorchecker_Lab[3*k+2];
  }
}

void init_global(dt_iop_module_so_t *self)
{
  dt_iop_colorchecker_global_data_t *gd = malloc(sizeof(dt_iop_colorchecker_global_data_t));
  self->data = gd;

  const int program = 8; // extended.cl, from programs.conf
  gd->kernel_colorchecker = dt_opencl_create_kernel(program, "colorchecker");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_colorchecker_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_colorchecker);
  free(self->data);
  self->data = NULL;
}

// The color picker returns values in the module input colorspace: Lab in Lab
// mode, linear RGB in XYZ mode. Patch values are always stored in Lab, so
// convert the picked RGB to Lab when needed.
static void _picked_to_lab(dt_iop_module_t *self,
                           const float in[3],
                           dt_aligned_pixel_t out)
{
  const dt_iop_colorchecker_params_t *const p = self->params;
  out[3] = 0.0f;

  if(!p || p->colorspace == DT_IOP_CC_CS_LAB)
  {
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2];
    return;
  }

  const dt_iop_order_iccprofile_info_t *const wp =
    self->dev ? dt_ioppr_get_iop_work_profile_info(self, self->dev->iop) : NULL;
  if(wp && dt_is_valid_colormatrix(wp->matrix_in[0][0]))
  {
    const dt_aligned_pixel_t rgb = { in[0], in[1], in[2], 0.0f };
    dt_aligned_pixel_t xyz;
    dot_product(rgb, wp->matrix_in, xyz);
    dt_XYZ_to_Lab(xyz, out);
    out[3] = 0.0f;
  }
  else
  {
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2];
  }
}

void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(p->num_patches <= 0) return;

  // determine patch based on color picker result
  dt_aligned_pixel_t picked_mean;
  _picked_to_lab(self, self->picked_color, picked_mean);
  int best_patch = 0;
  for(int patch = 1; patch < p->num_patches; patch++)
  {
    const dt_aligned_pixel_t Lab = { p->source_L[patch],
                                     p->source_a[patch],
                                     p->source_b[patch] };
    if((self->request_color_pick == DT_REQUEST_COLORPICK_MODULE)
       && (sqf(picked_mean[0] - Lab[0])
               + sqf(picked_mean[1] - Lab[1])
               + sqf(picked_mean[2] - Lab[2])
           < sqf(picked_mean[0] - p->source_L[best_patch])
                 + sqf(picked_mean[1] - p->source_a[best_patch])
                 + sqf(picked_mean[2] - p->source_b[best_patch])))
      best_patch = patch;
  }

  if(best_patch != g->drawn_patch)
  {
    g->patch = g->drawn_patch = best_patch;
    DT_ENTER_GUI_UPDATE();
    dt_bauhaus_combobox_set(g->combobox_patch, g->drawn_patch);
    _colorchecker_update_sliders(self);
    DT_LEAVE_GUI_UPDATE();
    gtk_widget_queue_draw(g->area);
  }
}

// a pending measurement preview is dropped as soon as the user edits anything
static void _cancel_preview(dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(g && g->profile_ready)
  {
    g->profile_ready = FALSE;
    if(g->area) gtk_widget_queue_draw(g->area);
  }
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  _cancel_preview(self);
}

static void target_L_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  _cancel_preview(self);

  if(g->patch >= p->num_patches || g->patch < 0)
    return;
  if(g->absolute_target)
    p->target_L[g->patch] = dt_bauhaus_slider_get(slider);
  else
    p->target_L[g->patch] = p->source_L[g->patch] +
      dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void target_a_callback(GtkWidget *slider,
                              dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  _cancel_preview(self);

  if(g->patch >= p->num_patches
     || g->patch < 0)
    return;

  if(g->absolute_target)
  {
    p->target_a[g->patch] = CLAMP(dt_bauhaus_slider_get(slider),
                                  -128.0, 128.0);
    const float Cout = sqrtf(
        p->target_a[g->patch]*p->target_a[g->patch]+
        p->target_b[g->patch]*p->target_b[g->patch]);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_C, Cout);
    DT_LEAVE_GUI_UPDATE();
  }
  else
  {
    p->target_a[g->patch] = CLAMP(p->source_a[g->patch] +
                                  dt_bauhaus_slider_get(slider),
                                  -128.0, 128.0);
    const float Cin = sqrtf(
        p->source_a[g->patch]*p->source_a[g->patch] +
        p->source_b[g->patch]*p->source_b[g->patch]);
    const float Cout = sqrtf(
        p->target_a[g->patch]*p->target_a[g->patch]+
        p->target_b[g->patch]*p->target_b[g->patch]);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_C, Cout-Cin);
    DT_LEAVE_GUI_UPDATE();
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void target_b_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  _cancel_preview(self);

  if(g->patch >= p->num_patches
     || g->patch < 0)
    return;

  if(g->absolute_target)
  {
    p->target_b[g->patch] = CLAMP(dt_bauhaus_slider_get(slider), -128.0, 128.0);
    const float Cout = sqrtf(p->target_a[g->patch]*p->target_a[g->patch]
                             + p->target_b[g->patch]*p->target_b[g->patch]);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_C, Cout);
    DT_LEAVE_GUI_UPDATE();
  }
  else
  {
    p->target_b[g->patch] = CLAMP(p->source_b[g->patch]
                                  + dt_bauhaus_slider_get(slider),
                                  -128.0, 128.0);
    const float Cin = sqrtf(
        p->source_a[g->patch]*p->source_a[g->patch] +
        p->source_b[g->patch]*p->source_b[g->patch]);
    const float Cout = sqrtf(
        p->target_a[g->patch]*p->target_a[g->patch]+
        p->target_b[g->patch]*p->target_b[g->patch]);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_C, Cout-Cin);
    DT_LEAVE_GUI_UPDATE();
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void target_C_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  _cancel_preview(self);

  if(g->patch >= p->num_patches
     || g->patch < 0)
    return;

  const float Cin = sqrtf(p->source_a[g->patch]*p->source_a[g->patch]
                          + p->source_b[g->patch]*p->source_b[g->patch]);
  const float Cout =
    MAX(1e-4f,
        sqrtf(p->target_a[g->patch] * p->target_a[g->patch]
              + p->target_b[g->patch]*p->target_b[g->patch]));

  if(g->absolute_target)
  {
    const float Cnew = CLAMP(dt_bauhaus_slider_get(slider), 0.01, 128.0);
    p->target_a[g->patch] = CLAMP(p->target_a[g->patch]*Cnew/Cout,
                                  -128.0, 128.0);
    p->target_b[g->patch] = CLAMP(p->target_b[g->patch]*Cnew/Cout,
                                  -128.0, 128.0);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_a, p->target_a[g->patch]);
    dt_bauhaus_slider_set(g->scale_b, p->target_b[g->patch]);
    DT_LEAVE_GUI_UPDATE();
  }
  else
  {
    const float Cnew = CLAMP(Cin + dt_bauhaus_slider_get(slider),
                             0.01, 128.0);
    p->target_a[g->patch] = CLAMP(p->target_a[g->patch]*Cnew/Cout,
                                  -128.0, 128.0);
    p->target_b[g->patch] = CLAMP(p->target_b[g->patch]*Cnew/Cout,
                                  -128.0, 128.0);
    DT_ENTER_GUI_UPDATE(); // avoid history item
    dt_bauhaus_slider_set(g->scale_a, p->target_a[g->patch]
                          - p->source_a[g->patch]);
    dt_bauhaus_slider_set(g->scale_b, p->target_b[g->patch]
                          - p->source_b[g->patch]);
    DT_LEAVE_GUI_UPDATE();
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void target_callback(GtkWidget *combo,
                            dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  g->absolute_target = dt_bauhaus_combobox_get(combo);
  DT_ENTER_GUI_UPDATE();
  _colorchecker_update_sliders(self);
  DT_LEAVE_GUI_UPDATE();
  // switch off colour picker, it'll interfere with other changes of
  // the patch:
  dt_iop_color_picker_reset(self, TRUE);
  gtk_widget_queue_draw(g->area);
}

static void patch_callback(GtkWidget *combo, dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  g->drawn_patch = g->patch = dt_bauhaus_combobox_get(combo);
  DT_ENTER_GUI_UPDATE();
  _colorchecker_update_sliders(self);
  DT_LEAVE_GUI_UPDATE();
  // switch off colour picker, it'll interfere with other changes of
  // the patch:
  dt_iop_color_picker_reset(self, TRUE);
  gtk_widget_queue_draw(g->area);
}

static gboolean checker_draw(GtkWidget *widget,
                             cairo_t *crf,
                             dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  const int width = allocation.width;
  const int height = allocation.height;
  cairo_surface_t *cst =
    dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);
  // clear bg
  cairo_set_source_rgb(cr, .2, .2, .2);
  cairo_paint(cr);

  cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
  // while a measurement is pending, show the measured colors instead of the
  // committed ones so they can be reviewed before accepting
  const gboolean preview = g->profile_ready && g->measured_lab && g->checker;
  const int n_patches = preview ? (int)g->checker->patches : p->num_patches;
  const int cells_x = n_patches > 24 ? 7 : 6;
  const int cells_y = n_patches > 24 ? 7 : 4;
  for(int j = 0; j < cells_y; j++)
  {
    for(int i = 0; i < cells_x; i++)
    {
      const int patch = i + j*cells_x;
      if(patch >= n_patches) continue;

      dt_aligned_pixel_t Lab = { 0.f, 0.f, 0.f, 0.f };
      dt_aligned_pixel_t Lab_tgt = { 0.f, 0.f, 0.f, 0.f };
      if(preview)
      {
        Lab[0] = g->measured_lab[3 * patch + 0];
        Lab[1] = g->measured_lab[3 * patch + 1];
        Lab[2] = g->measured_lab[3 * patch + 2];
        Lab_tgt[0] = g->checker->values[patch].Lab[0];
        Lab_tgt[1] = g->checker->values[patch].Lab[1];
        Lab_tgt[2] = g->checker->values[patch].Lab[2];
      }
      else
      {
        Lab[0] = p->source_L[patch];
        Lab[1] = p->source_a[patch];
        Lab[2] = p->source_b[patch];
        Lab_tgt[0] = p->target_L[patch];
        Lab_tgt[1] = p->target_a[patch];
        Lab_tgt[2] = p->target_b[patch];
      }
      dt_aligned_pixel_t rgb, XYZ;
      dt_Lab_to_XYZ(Lab, XYZ);
      dt_XYZ_to_sRGB(XYZ, rgb);
      cairo_set_source_rgb(cr, rgb[0], rgb[1], rgb[2]);

      cairo_rectangle(cr,
                      width * i / (float)cells_x,
                      height * j / (float)cells_y,
                      width / (float)cells_x - DT_PIXEL_APPLY_DPI(1),
                      height / (float)cells_y - DT_PIXEL_APPLY_DPI(1));
      cairo_fill(cr);
      if(fabsf(Lab_tgt[0] - Lab[0]) > 1e-5f
         || fabsf(Lab_tgt[1] - Lab[1]) > 1e-5f
         || fabsf(Lab_tgt[2] - Lab[2]) > 1e-5f)
      {
        cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.));
        cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
        cairo_rectangle(cr,
            width * i / (float)cells_x + DT_PIXEL_APPLY_DPI(1),
            height * j / (float)cells_y + DT_PIXEL_APPLY_DPI(1),
            width / (float)cells_x - DT_PIXEL_APPLY_DPI(3),
            height / (float)cells_y - DT_PIXEL_APPLY_DPI(3));
        cairo_stroke(cr);
        cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.));
        cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
        cairo_rectangle(cr,
            width * i / (float)cells_x + DT_PIXEL_APPLY_DPI(2),
            height * j / (float)cells_y + DT_PIXEL_APPLY_DPI(2),
            width / (float)cells_x - DT_PIXEL_APPLY_DPI(5),
            height / (float)cells_y - DT_PIXEL_APPLY_DPI(5));
        cairo_stroke(cr);
      }
    }
  }

  if(g->drawn_patch != -1 && g->drawn_patch < n_patches)
  {
    const int draw_i = g->drawn_patch % cells_x;
    const int draw_j = g->drawn_patch / cells_x;
    float color = 1.0;
    const float src_L = preview ? g->measured_lab[3 * g->drawn_patch + 0]
                                : p->source_L[g->drawn_patch];
    if(src_L > 80) color = 0.0;
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.));
    cairo_set_source_rgb(cr, color, color, color);
    cairo_rectangle(cr,
                    width * draw_i / (float) cells_x + DT_PIXEL_APPLY_DPI(5),
                    height * draw_j / (float) cells_y + DT_PIXEL_APPLY_DPI(5),
                    width / (float) cells_x - DT_PIXEL_APPLY_DPI(11),
                    height / (float) cells_y - DT_PIXEL_APPLY_DPI(11));
    cairo_stroke(cr);
  }

  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return TRUE;
}

static gboolean checker_motion_notify(
    GtkWidget *widget,
    GdkEventMotion *event,
    dt_iop_module_t *self)
{
  // highlight?
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  const int width = allocation.width;
  const int height = allocation.height;

  const float mouse_x = CLAMP(event->x, 0, width);
  const float mouse_y = CLAMP(event->y, 0, height);
  int cells_x = 6, cells_y = 4;
  if(p->num_patches > 24)
  {
    cells_x = 7;
    cells_y = 7;
  }
  const float mx = mouse_x * cells_x / (float)width;
  const float my = mouse_y * cells_y / (float)height;
  const int patch = (int)mx + cells_x * (int)my;
  if(patch < 0 || patch >= p->num_patches) return FALSE;
  char tooltip[1024];
  snprintf(tooltip, sizeof(tooltip),
      _("(%2.2f %2.2f %2.2f)\n"
        "altered patches are marked with an outline\n"
        "click to select\n"
        "double-click to reset\n"
        "right-click to delete patch\n"
        "shift+click while color picking to replace patch"),
      p->source_L[patch], p->source_a[patch], p->source_b[patch]);
  gtk_widget_set_tooltip_text(g->area, tooltip);
  return TRUE;
}

static gboolean checker_button_press(
    GtkWidget *widget, GdkEventButton *event,
    dt_iop_module_t *self)
{
  dt_iop_colorchecker_params_t *p = self->params;
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width, height = allocation.height;
  const float mouse_x = CLAMP(event->x, 0, width);
  const float mouse_y = CLAMP(event->y, 0, height);
  int cells_x = 6, cells_y = 4;
  if(p->num_patches > 24)
  {
    cells_x = 7;
    cells_y = 7;
  }
  const float mx = mouse_x * cells_x / (float)width;
  const float my = mouse_y * cells_y / (float)height;
  int patch = (int)mx + cells_x*(int)my;
  if(event->button == GDK_BUTTON_PRIMARY && event->type == GDK_2BUTTON_PRESS)
  { // reset on double click
    if(patch < 0 || patch >= p->num_patches) return FALSE;
    p->target_L[patch] = p->source_L[patch];
    p->target_a[patch] = p->source_a[patch];
    p->target_b[patch] = p->source_b[patch];
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    DT_ENTER_GUI_UPDATE();
    _colorchecker_update_sliders(self);
    DT_LEAVE_GUI_UPDATE();
    gtk_widget_queue_draw(g->area);
    return TRUE;
  }
  else if(event->button == GDK_BUTTON_SECONDARY && (patch < p->num_patches))
  {
    // right click: delete patch, move others up
    if(patch < 0 || patch >= p->num_patches) return FALSE;
    memmove(p->target_L+patch, p->target_L+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    memmove(p->target_a+patch, p->target_a+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    memmove(p->target_b+patch, p->target_b+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    memmove(p->source_L+patch, p->source_L+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    memmove(p->source_a+patch, p->source_a+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    memmove(p->source_b+patch, p->source_b+patch+1,
            sizeof(float)*(p->num_patches-1-patch));
    p->num_patches--;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    DT_ENTER_GUI_UPDATE();
    _colorchecker_rebuild_patch_list(self);
    _colorchecker_update_sliders(self);
    DT_LEAVE_GUI_UPDATE();
    gtk_widget_queue_draw(g->area);
    return TRUE;
  }
  else if((event->button == GDK_BUTTON_PRIMARY) &&
          dt_modifier_is(event->state, GDK_SHIFT_MASK) &&
          (self->request_color_pick == DT_REQUEST_COLORPICK_MODULE))
  {
    // shift-left while colour picking: replace source colour
    // if clicked outside the valid patches: add new one

    // the picker works in the module input colorspace; patches are in Lab
    dt_aligned_pixel_t picked;
    _picked_to_lab(self, self->picked_color, picked);

    // color channels should be nonzero to avoid numerical issues
    int new_color_valid = fabsf(picked[0]) > 1.e-3f &&
                          fabsf(picked[1]) > 1.e-3f &&
                          fabsf(picked[2]) > 1.e-3f;
    // check if the new color is very close to some color already in
    // the colorchecker
    for(int i=0;i<p->num_patches;++i)
    {
      float color[] = { p->source_L[i], p->source_a[i], p->source_b[i] };
      if(fabsf(picked[0] - color[0])
         < 1.e-3f && fabsf(picked[1] - color[1]) < 1.e-3f
         && fabsf(picked[2] - color[2]) < 1.e-3f)
        new_color_valid = FALSE;
    }
    if(new_color_valid)
    {
      if(p->num_patches < MAX_PATCHES
         && (patch < 0 || patch >= p->num_patches))
      {
        p->num_patches = MIN(MAX_PATCHES, p->num_patches + 1);
        patch = p->num_patches - 1;
      }
      p->target_L[patch] = p->source_L[patch] = picked[0];
      p->target_a[patch] = p->source_a[patch] = picked[1];
      p->target_b[patch] = p->source_b[patch] = picked[2];
      dt_dev_add_history_item(darktable.develop, self, TRUE);

      DT_ENTER_GUI_UPDATE();
      _colorchecker_rebuild_patch_list(self);
      dt_bauhaus_combobox_set(g->combobox_patch, patch);
      _colorchecker_update_sliders(self);
      DT_LEAVE_GUI_UPDATE();
      g->patch = g->drawn_patch = patch;
      gtk_widget_queue_draw(g->area);
    }
    return TRUE;
  }
  if(patch >= p->num_patches) patch = p->num_patches-1;
  dt_bauhaus_combobox_set(g->combobox_patch, patch);
  return FALSE;
}

/* -------------------------------------------------------------------------
 * color chart calibration
 * ---------------------------------------------------------------------- */

// CIEDE2000 color difference between two CIE Lab colors
static inline float _delta_E_2000(const float Lab_ref[3], const float Lab_test[3])
{
  const float DL = Lab_ref[0] - Lab_test[0];
  const float L_avg = (Lab_ref[0] + Lab_test[0]) / 2.f;
  const float C_ref = dt_fast_hypotf(Lab_ref[1], Lab_ref[2]);
  const float C_test = dt_fast_hypotf(Lab_test[1], Lab_test[2]);
  const float C_avg = (C_ref + C_test) / 2.f;
  // guard against perfectly neutral colours (a=b=0) so the C_avg^7 term
  // below does not turn into 0/0 = NaN and poison the whole delta E
  const float C_avg_safe = MAX(C_avg, 1e-6f);
  float C_avg_7 = C_avg * C_avg; // C_avg²
  C_avg_7 *= C_avg_7;            // C_avg⁴
  C_avg_7 *= C_avg_7;            // C_avg⁸
  C_avg_7 /= C_avg_safe;         // C_avg⁷
  // 25⁷ = 6103515625
  const float C_avg_7_ratio_sqrt = sqrtf(C_avg_7 / (C_avg_7 + 6103515625.f));
  const float a_ref_prime = Lab_ref[1] * (1.f + 0.5f * (1.f - C_avg_7_ratio_sqrt));
  const float a_test_prime = Lab_test[1] * (1.f + 0.5f * (1.f - C_avg_7_ratio_sqrt));
  const float C_ref_prime = dt_fast_hypotf(a_ref_prime, Lab_ref[2]);
  const float C_test_prime = dt_fast_hypotf(a_test_prime, Lab_test[2]);
  const float DC_prime = C_ref_prime - C_test_prime;
  const float C_avg_prime = (C_ref_prime + C_test_prime) / 2.f;
  float h_ref_prime = atan2f(Lab_ref[2], a_ref_prime);
  float h_test_prime = atan2f(Lab_test[2], a_test_prime);

  if(C_ref_prime == 0.f) h_ref_prime = 0.f;
  if(C_test_prime == 0.f) h_test_prime = 0.f;
  if(h_ref_prime < 0.f) h_ref_prime = DT_2PI_F - h_ref_prime;
  if(h_test_prime < 0.f) h_test_prime = DT_2PI_F - h_test_prime;
  h_ref_prime = rad2degf(h_ref_prime);
  h_test_prime = rad2degf(h_test_prime);

  float Dh_prime = h_test_prime - h_ref_prime;
  float Dh_prime_abs = fabsf(Dh_prime);
  if(C_test_prime == 0.f || C_ref_prime == 0.f)
    Dh_prime = 0.f;
  else if(Dh_prime_abs <= 180.f)
    ;
  else if(Dh_prime_abs > 180.f && (h_test_prime <= h_ref_prime))
    Dh_prime += 360.f;
  else if(Dh_prime_abs > 180.f && (h_test_prime > h_ref_prime))
    Dh_prime -= 360.f;

  Dh_prime_abs = fabsf(Dh_prime);

  const float DH_prime =
    2.f * sqrtf(C_test_prime * C_ref_prime) * sinf(deg2radf(Dh_prime) / 2.f);

  float H_avg_prime = h_ref_prime + h_test_prime;
  if(C_test_prime == 0.f || C_ref_prime == 0.f)
    ;
  else if(Dh_prime_abs <= 180.f)
    H_avg_prime /= 2.f;
  else if(Dh_prime_abs > 180.f && (H_avg_prime < 360.f))
    H_avg_prime = (H_avg_prime + 360.f) / 2.f;
  else if(Dh_prime_abs > 180.f && (H_avg_prime >= 360.f))
    H_avg_prime = (H_avg_prime - 360.f) / 2.f;

  const float T = 1.f
                  - 0.17f * cosf(deg2radf(H_avg_prime - 30))
                  + 0.24f * cosf(2.f * deg2radf(H_avg_prime))
                  + 0.32f * cosf(3.f * deg2radf(H_avg_prime) + deg2radf(6.f))
                  - 0.20f * cosf(4.f * deg2radf(H_avg_prime) - deg2radf(63.f));

  const float S_L = 1.f + (0.015f * sqf(L_avg - 50.f)) / sqrtf(20.f + sqf(L_avg - 50.f));
  const float S_C = 1.f + 0.045f * C_avg_prime;
  const float S_H = 1.f + 0.015f * C_avg_prime * T;
  const float R_T = -2.f * C_avg_7_ratio_sqrt
                    * sinf(deg2radf(60.f) * expf(-sqf((H_avg_prime - 275.f) / 25.f)));

  return sqrtf(sqf(DL / S_L) + sqf(DC_prime / S_C) + sqf(DH_prime / S_H)
               + R_T * (DC_prime / S_C) * (DH_prime / S_H));
}

// bounding box persistence (normalized to the preview size, so the same
// relative chart position is reused across images and sessions)
#define CC_LAST_CALIBRATION_KEY "darkroom/modules/colorchecker/last_calibration"

static const char *_cc_box_keys[4][2] = {
  { "darkroom/modules/colorchecker/box0x", "darkroom/modules/colorchecker/box0y" },
  { "darkroom/modules/colorchecker/box1x", "darkroom/modules/colorchecker/box1y" },
  { "darkroom/modules/colorchecker/box2x", "darkroom/modules/colorchecker/box2y" },
  { "darkroom/modules/colorchecker/box3x", "darkroom/modules/colorchecker/box3y" },
};

static gboolean _box_from_conf(dt_iop_colorchecker_gui_data_t *g,
                               const float width,
                               const float height)
{
  if(!dt_conf_key_exists(_cc_box_keys[0][0])) return FALSE;
  if(dt_conf_get_float(_cc_box_keys[0][0]) < 0.f) return FALSE;
  for(int k = 0; k < 4; k++)
  {
    g->box[k].x = dt_conf_get_float(_cc_box_keys[k][0]) * width;
    g->box[k].y = dt_conf_get_float(_cc_box_keys[k][1]) * height;
  }
  return TRUE;
}

static void _box_to_conf(const dt_iop_colorchecker_gui_data_t *g,
                         const float width,
                         const float height)
{
  if(width <= 0.f || height <= 0.f) return;
  for(int k = 0; k < 4; k++)
  {
    dt_conf_set_float(_cc_box_keys[k][0], g->box[k].x / width);
    dt_conf_set_float(_cc_box_keys[k][1], g->box[k].y / height);
  }
}

static void _box_clear_conf(void)
{
  dt_conf_set_float(_cc_box_keys[0][0], -1.f);
}

static inline void _update_bounding_box(dt_iop_colorchecker_gui_data_t *g,
                                        const float x_increment,
                                        const float y_increment)
{
  for(size_t k = 0; k < 4; k++)
  {
    if(g->active_node[k])
    {
      g->box[k].x += x_increment;
      g->box[k].y += y_increment;
    }
  }
  get_homography(g->ideal_box, g->box, g->homography);
  get_homography(g->box, g->ideal_box, g->inverse_homography);
}

static inline void _init_bounding_box(dt_iop_colorchecker_gui_data_t *g,
                                      const float width,
                                      const float height)
{
  if(!g->checker_ready)
  {
    // reuse the last chart position when available
    if(!_box_from_conf(g, width, height))
    {
      const float handle_offset = 10.0f;
      dt_develop_t *dev = darktable.develop;
      dt_dev_viewport_t *port = &dev->full;

      float x1 = handle_offset;
      float y1 = handle_offset;
      float x2 = width - handle_offset;
      float y2 = height - handle_offset;

      float zoom_x, zoom_y, boxw, boxh;
      if(dt_dev_get_zoom_bounds(port, &zoom_x, &zoom_y, &boxw, &boxh))
      {
        const float h_box = width * boxw;
        const float v_box = height * boxh;
        const float h_border = width - h_box;
        const float v_border = height - v_box;
        const float offx = (h_border / 2.0f) + (zoom_x * width);
        const float offy = (v_border / 2.0f) + (zoom_y * height);
        x1 = offx + handle_offset;
        y1 = offy + handle_offset;
        x2 = offx + h_box - handle_offset;
        y2 = offy + v_box - handle_offset;
      }

      g->box[0].x = x1; g->box[0].y = y1; // top left
      g->box[1].x = x2; g->box[1].y = y1; // top right
      g->box[2].x = x2; g->box[2].y = y2; // bottom right
      g->box[3].x = x1; g->box[3].y = y2; // bottom left
    }
    g->checker_ready = TRUE;
  }

  g->center_box.x = 0.5f;
  g->center_box.y = 0.5f;

  g->ideal_box[0].x = 0.f; g->ideal_box[0].y = 0.f;
  g->ideal_box[1].x = 1.f; g->ideal_box[1].y = 0.f;
  g->ideal_box[2].x = 1.f; g->ideal_box[2].y = 1.f;
  g->ideal_box[3].x = 0.f; g->ideal_box[3].y = 1.f;

  _update_bounding_box(g, 0.f, 0.f);
}

int mouse_moved(dt_iop_module_t *self,
                const float pzx,
                const float pzy,
                const double pressure,
                const int which,
                const float zoom_scale)
{
  if(!self->enabled) return 0;

  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(g == NULL || !g->is_profiling_started) return 0;
  if(g->box[0].x == -1.0f || g->box[1].y == -1.0f) return 0;

  float wd, ht;
  if(!dt_dev_get_preview_size(self->dev, &wd, &ht)) return 0;

  if(g->drag_drop)
  {
    dt_iop_gui_enter_critical_section(self);
    g->click_end.x = pzx * wd;
    g->click_end.y = pzy * ht;
    _update_bounding_box(g, g->click_end.x - g->click_start.x,
                         g->click_end.y - g->click_start.y);
    g->click_start.x = pzx * wd;
    g->click_start.y = pzy * ht;
    dt_iop_gui_leave_critical_section(self);
    dt_control_queue_redraw_center();
    return 1;
  }

  dt_iop_gui_enter_critical_section(self);
  g->is_cursor_close = FALSE;
  for(size_t k = 0; k < 4; k++)
  {
    if(hypotf(pzx * wd - g->box[k].x, pzy * ht - g->box[k].y) < 15.f)
    {
      g->active_node[k] = TRUE;
      g->is_cursor_close = TRUE;
    }
    else
      g->active_node[k] = FALSE;
  }
  dt_iop_gui_leave_critical_section(self);

  if(g->is_cursor_close)
  {
    dt_control_change_cursor("none");
  }
  else
  {
    GdkCursor *const cursor =
      gdk_cursor_new_from_name(gdk_display_get_default(), "default");
    gdk_window_set_cursor(gtk_widget_get_window(dt_ui_main_window(darktable.gui->ui)),
                          cursor);
    g_object_unref(cursor);
  }

  dt_control_queue_redraw_center();
  return 1;
}

int button_pressed(dt_iop_module_t *self,
                   const float pzx,
                   const float pzy,
                   const double pressure,
                   const int which,
                   const int type,
                   const uint32_t state,
                   const float zoom_scale)
{
  if(!self->enabled) return 0;

  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(g == NULL || !g->is_profiling_started) return 0;

  float wd, ht;
  if(!dt_dev_get_preview_size(self->dev, &wd, &ht)) return 0;

  // double click : reset the perspective correction
  if(type == GDK_DOUBLE_BUTTON_PRESS)
  {
    dt_iop_gui_enter_critical_section(self);
    _box_clear_conf();
    g->checker_ready = FALSE;
    g->profile_ready = FALSE;
    _init_bounding_box(g, wd, ht);
    dt_iop_gui_leave_critical_section(self);
    dt_control_queue_redraw_center();
    return 1;
  }

  if(g->box[0].x == -1.0f || g->box[1].y == -1.0f) return 0;
  if(!g->is_cursor_close) return 0;

  dt_iop_gui_enter_critical_section(self);
  g->drag_drop = TRUE;
  g->click_start.x = pzx * wd;
  g->click_start.y = pzy * ht;
  dt_iop_gui_leave_critical_section(self);

  dt_control_queue_redraw_center();
  return 1;
}

int button_released(dt_iop_module_t *self,
                    const float pzx,
                    const float pzy,
                    const int which,
                    const uint32_t state,
                    const float zoom_scale)
{
  if(!self->enabled) return 0;

  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(g == NULL || !g->is_profiling_started) return 0;
  if(g->box[0].x == -1.0f || g->box[1].y == -1.0f) return 0;
  if(!g->is_cursor_close || !g->drag_drop) return 0;

  float wd, ht;
  if(!dt_dev_get_preview_size(self->dev, &wd, &ht)) return 0;

  dt_iop_gui_enter_critical_section(self);
  g->drag_drop = FALSE;
  g->click_end.x = pzx * wd;
  g->click_end.y = pzy * ht;
  _update_bounding_box(g, g->click_end.x - g->click_start.x,
                       g->click_end.y - g->click_start.y);
  _box_to_conf(g, wd, ht);
  dt_iop_gui_leave_critical_section(self);

  dt_control_queue_redraw_center();
  return 1;
}

void gui_post_expose(dt_iop_module_t *self,
                     cairo_t *cr,
                     const float width,
                     const float height,
                     const float pointerx,
                     const float pointery,
                     const float zoom_scale)
{
  if(!self->dev->full.pipe) return;
  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_output_profile_info(self->dev->full.pipe);
  if(work_profile == NULL) return;

  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(!g || !g->is_profiling_started || !g->checker) return;

  // once the measurement is available, reprocess once more so the preview
  // shows the calibrated result before it is accepted
  if(g->preview_pending && g->profile_ready)
  {
    g->preview_pending = FALSE;
    dt_dev_reprocess_preview(self->dev);
  }

  const gboolean showhandle = dt_iop_canvas_not_sensitive(darktable.develop) == FALSE;
  const double lwidth = (showhandle ? 1.0 : 0.5) / zoom_scale;

  cairo_set_line_width(cr, 2.0 * lwidth);
  const double origin = 9. / zoom_scale;
  const double destination = 18. / zoom_scale;

  for(size_t k = 0; k < 4; k++)
  {
    if(g->active_node[k])
    {
      cairo_set_source_rgba(cr, 1., 1., 1., 1.);
      cairo_move_to(cr, g->box[k].x - origin, g->box[k].y);
      cairo_line_to(cr, g->box[k].x - destination, g->box[k].y);
      cairo_move_to(cr, g->box[k].x + origin, g->box[k].y);
      cairo_line_to(cr, g->box[k].x + destination, g->box[k].y);
      cairo_move_to(cr, g->box[k].x, g->box[k].y - origin);
      cairo_line_to(cr, g->box[k].x, g->box[k].y - destination);
      cairo_move_to(cr, g->box[k].x, g->box[k].y + origin);
      cairo_line_to(cr, g->box[k].x, g->box[k].y + destination);
      cairo_stroke(cr);
    }

    if(showhandle)
    {
      cairo_set_source_rgba(cr, 1., 1., 1., 1.);
      cairo_arc(cr, g->box[k].x, g->box[k].y, 8. / zoom_scale, 0, 2. * M_PI);
      cairo_stroke(cr);
      cairo_set_source_rgba(cr, 0., 0., 0., 1.);
      cairo_arc(cr, g->box[k].x, g->box[k].y, 1.5 / zoom_scale, 0, 2. * M_PI);
      cairo_fill(cr);
    }
  }

  // draw symmetry axes
  cairo_set_line_width(cr, 1.5 * lwidth);
  cairo_set_source_rgba(cr, 1., 1., 1., 1.);
  const point_t top_ideal = { 0.5f, 1.f };
  const point_t top = apply_homography(top_ideal, g->homography);
  const point_t bottom_ideal = { 0.5f, 0.f };
  const point_t bottom = apply_homography(bottom_ideal, g->homography);
  cairo_move_to(cr, top.x, top.y);
  cairo_line_to(cr, bottom.x, bottom.y);
  cairo_stroke(cr);

  const point_t left_ideal = { 0.f, 0.5f };
  const point_t left = apply_homography(left_ideal, g->homography);
  const point_t right_ideal = { 1.f, 0.5f };
  const point_t right = apply_homography(right_ideal, g->homography);
  cairo_move_to(cr, left.x, left.y);
  cairo_line_to(cr, right.x, right.y);
  cairo_stroke(cr);

  const float radius_x =
    g->checker->radius * hypotf(1.f, g->checker->ratio) * g->safety_margin;
  const float radius_y = radius_x / g->checker->ratio;

  for(size_t k = 0; k < g->checker->patches; k++)
  {
    const point_t center = { g->checker->values[k].x, g->checker->values[k].y };
    const point_t corners[4] = { {center.x - radius_x, center.y - radius_y},
                                 {center.x + radius_x, center.y - radius_y},
                                 {center.x + radius_x, center.y + radius_y},
                                 {center.x - radius_x, center.y + radius_y} };

    const point_t new_center = apply_homography(center, g->homography);
    const float scaling = sqrtf(apply_homography_scaling(center, g->homography));
    point_t new_corners[4];
    for(size_t c = 0; c < 4; c++)
      new_corners[c] = apply_homography(corners[c], g->homography);

    cairo_set_line_cap(cr, CAIRO_LINE_CAP_SQUARE);
    cairo_set_source_rgba(cr, 0., 0., 0., 1.);
    cairo_move_to(cr, new_corners[0].x, new_corners[0].y);
    cairo_line_to(cr, new_corners[1].x, new_corners[1].y);
    cairo_line_to(cr, new_corners[2].x, new_corners[2].y);
    cairo_line_to(cr, new_corners[3].x, new_corners[3].y);
    cairo_line_to(cr, new_corners[0].x, new_corners[0].y);

    if(g->delta_E_in)
    {
      if(g->delta_E_in[k] > 2.3f)
      {
        cairo_move_to(cr, new_corners[0].x, new_corners[0].y);
        cairo_line_to(cr, new_corners[2].x, new_corners[2].y);
      }
      if(g->delta_E_in[k] > 4.6f)
      {
        cairo_move_to(cr, new_corners[1].x, new_corners[1].y);
        cairo_line_to(cr, new_corners[3].x, new_corners[3].y);
      }
    }

    cairo_set_line_width(cr, 5.0 * lwidth);
    cairo_stroke_preserve(cr);
    cairo_set_line_width(cr, 2.0 * lwidth);
    cairo_set_source_rgba(cr, 1., 1., 1., 1.);
    cairo_stroke(cr);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);

    dt_aligned_pixel_t RGB;
    dt_ioppr_lab_to_rgb_matrix(g->checker->values[k].Lab, RGB,
                               work_profile->matrix_out_transposed, work_profile->lut_out,
                               work_profile->unbounded_coeffs_out, work_profile->lutsize,
                               work_profile->nonlinearlut);
    cairo_set_source_rgba(cr, RGB[0], RGB[1], RGB[2], 1.);
    cairo_arc(cr, new_center.x, new_center.y,
              0.25 * (radius_x + radius_y) * scaling, 0, 2. * M_PI);
    cairo_fill(cr);
  }

  // refresh the delta E readout and the patch sliders
  _update_delta_E_label(self);
  if(g->profile_ready)
  {
    DT_ENTER_GUI_UPDATE();
    _colorchecker_update_sliders(self);
    DT_LEAVE_GUI_UPDATE();
  }
}

// Extract the mean color of every chart patch from the module input and store
// it (in CIE Lab) as the "source" color for the spline. The reference chart
// values are used as "target".
static void _extract_patches(const float *const restrict in,
                             const dt_iop_roi_t *const roi_in,
                             dt_iop_module_t *self,
                             dt_iop_colorchecker_gui_data_t *g,
                             const dt_iop_order_iccprofile_info_t *const work_profile)
{
  const dt_iop_colorchecker_params_t *const p = self->params;
  const gboolean xyz_mode = (p->colorspace == DT_IOP_CC_CS_XYZ)
                            && work_profile
                            && dt_is_valid_colormatrix(work_profile->matrix_in[0][0]);
  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const float radius_x =
    g->checker->radius * hypotf(1.f, g->checker->ratio) * g->safety_margin;
  const float radius_y = radius_x / g->checker->ratio;

  if(!g->measured_lab)
    g->measured_lab = dt_alloc_align_float(3 * MAX_PATCHES);
  if(!g->measured_XYZ)
    g->measured_XYZ = dt_alloc_align_float(3 * MAX_PATCHES);
  if(!g->delta_E_in)
    g->delta_E_in = dt_alloc_align_float(MAX_PATCHES);

  // remember which mode this extraction pass was done in: commit_params()
  // only trusts measured_XYZ when the module is still in XYZ mode at
  // commit time, so it never reuses a cache computed under a different
  // (or invalid) work profile / colorspace setting.
  g->measured_xyz_valid = xyz_mode;

  for(size_t k = 0; k < g->checker->patches; k++)
  {
    const point_t center = { g->checker->values[k].x, g->checker->values[k].y };
    const point_t corners[4] = { {center.x - radius_x, center.y - radius_y},
                                 {center.x + radius_x, center.y - radius_y},
                                 {center.x + radius_x, center.y + radius_y},
                                 {center.x - radius_x, center.y + radius_y} };

    point_t new_corners[4];
    size_t x_min = width - 1, x_max = 0, y_min = height - 1, y_max = 0;
    for(size_t c = 0; c < 4; c++)
    {
      new_corners[c] = apply_homography(corners[c], g->homography);
      x_min = MIN(x_min, (size_t)floorf(new_corners[c].x));
      x_max = MAX(x_max, (size_t)ceilf(new_corners[c].x));
      y_min = MIN(y_min, (size_t)floorf(new_corners[c].y));
      y_max = MAX(y_max, (size_t)ceilf(new_corners[c].y));
    }
    x_min = CLAMP(x_min, 0, width - 1);
    x_max = CLAMP(x_max, 0, width - 1);
    y_min = CLAMP(y_min, 0, height - 1);
    y_max = CLAMP(y_max, 0, height - 1);

    dt_aligned_pixel_t mean = { 0.f, 0.f, 0.f, 0.f };
    size_t num = 0;
    for(size_t j = y_min; j < y_max; j++)
      for(size_t i = x_min; i < x_max; i++)
      {
        point_t current_point = { i + 0.5f, j + 0.5f };
        current_point = apply_homography(current_point, g->inverse_homography);
        current_point.x -= center.x;
        current_point.y -= center.y;
        if(current_point.x < radius_x && current_point.x > -radius_x &&
           current_point.y < radius_y && current_point.y > -radius_y)
        {
          for_three_channels(c)
            mean[c] += in[(j * width + i) * 4 + c];
          num++;
        }
      }

    dt_aligned_pixel_t lab = { 0.f, 0.f, 0.f, 0.f };
    dt_aligned_pixel_t XYZ = { 0.f, 0.f, 0.f, 0.f };
    if(num > 0)
    {
      for_three_channels(c) mean[c] /= (float)num;
      if(xyz_mode)
      {
        dot_product(mean, work_profile->matrix_in, XYZ);
        dt_XYZ_to_Lab(XYZ, lab);
      }
      else
      {
        copy_pixel(lab, mean);
      }
    }

    g->measured_lab[3 * k + 0] = lab[0];
    g->measured_lab[3 * k + 1] = lab[1];
    g->measured_lab[3 * k + 2] = lab[2];
    // cached alongside the Lab value so commit_params() can reuse it
    // directly in XYZ mode instead of converting Lab -> XYZ a second time
    g->measured_XYZ[3 * k + 0] = XYZ[0];
    g->measured_XYZ[3 * k + 1] = XYZ[1];
    g->measured_XYZ[3 * k + 2] = XYZ[2];
    g->delta_E_in[k] = _delta_E_2000(g->checker->values[k].Lab, lab);
  }
}

static void _update_delta_E_label(dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(!g || !g->label_delta_E) return;

  if(!g->delta_E_in || !g->checker)
  {
    if(g->delta_E_valid)
    {
      gtk_label_set_markup(GTK_LABEL(g->label_delta_E), "");
      g->delta_E_valid = FALSE;
    }
    return;
  }

  float avg = 0.f, max = 0.f;
  int worst = 0;
  for(size_t k = 0; k < g->checker->patches; k++)
  {
    avg += g->delta_E_in[k];
    if(g->delta_E_in[k] > max)
    {
      max = g->delta_E_in[k];
      worst = (int)k;
    }
  }
  avg /= (float)g->checker->patches;

  // this runs on every expose: only rebuild the (translatable) string when the
  // measured values actually changed
  if(g->delta_E_valid
     && g->delta_E_avg == avg
     && g->delta_E_max == max
     && g->delta_E_worst == worst)
    return;

  const char *diag = _("bad");
  if(avg <= 1.2f)
    diag = _("very good");
  else if(avg <= 2.3f)
    diag = _("good");
  else if(avg <= 3.4f)
    diag = _("passable");

  g_free(g->delta_E_label_text);
  g->delta_E_label_text =
    g_markup_printf_escaped(_("<b>source accuracy: %s</b>\n"
                              "input ΔE: avg. %.2f; max. %.2f (patch %d)"),
                            diag, avg, max, worst);
  gtk_label_set_markup(GTK_LABEL(g->label_delta_E), g->delta_E_label_text);

  g->delta_E_avg = avg;
  g->delta_E_max = max;
  g->delta_E_worst = worst;
  g->delta_E_valid = TRUE;
}

static void _checker_changed_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  const int i = dt_bauhaus_combobox_get(widget);
  dt_conf_set_int("darkroom/modules/colorchecker/chart", i);
  g->checker = dt_get_color_checker(i);

  float wd, ht;
  if(!dt_dev_get_preview_size(self->dev, &wd, &ht)) return;

  dt_iop_gui_enter_critical_section(self);
  g->profile_ready = FALSE;
  g->checker_ready = FALSE;
  _init_bounding_box(g, wd, ht);
  dt_iop_gui_leave_critical_section(self);

  dt_control_queue_redraw_center();
}

static void _safety_changed_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  g->safety_margin = dt_bauhaus_slider_get(widget);
  dt_iop_gui_leave_critical_section(self);

  dt_conf_set_float("darkroom/modules/colorchecker/safety", g->safety_margin);
  dt_control_queue_redraw_center();
}

static void _start_profiling_callback(GtkWidget *togglebutton, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_request_focus(self);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  float wd, ht;
  if(!dt_dev_get_preview_size(self->dev, &wd, &ht)) return;

  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  g->is_profiling_started = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->cs.toggle));

  dt_iop_gui_enter_critical_section(self);
  g->checker_ready = FALSE;
  _init_bounding_box(g, wd, ht);
  dt_iop_gui_leave_critical_section(self);

  dt_control_queue_redraw_center();
}

static void _run_profile_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  g->run_profile = TRUE;
  g->preview_pending = TRUE;
  dt_iop_gui_leave_critical_section(self);

  dt_dev_reprocess_preview(self->dev);
}

static void _commit_profile_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  dt_iop_colorchecker_params_t *p = self->params;

  if(!g->profile_ready || !g->measured_lab || !g->checker) return;

  dt_iop_gui_enter_critical_section(self);
  const int n = MIN(MAX_PATCHES, (int)g->checker->patches);
  p->num_patches = n;
  for(int k = 0; k < n; k++)
  {
    p->source_L[k] = g->measured_lab[3 * k + 0];
    p->source_a[k] = g->measured_lab[3 * k + 1];
    p->source_b[k] = g->measured_lab[3 * k + 2];
    p->target_L[k] = g->checker->values[k].Lab[0];
    p->target_a[k] = g->checker->values[k].Lab[1];
    p->target_b[k] = g->checker->values[k].Lab[2];
  }
  g->profile_ready = FALSE;
  dt_iop_gui_leave_critical_section(self);

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  // remember this calibration so it can be reused on other images
  {
    int blob_len = 0;
    char *blob = dt_exif_xmp_encode((const unsigned char *)p,
                                    (int)sizeof(dt_iop_colorchecker_params_t), &blob_len);
    if(blob)
    {
      dt_conf_set_string(CC_LAST_CALIBRATION_KEY, blob);
      free(blob);
    }
  }

  DT_ENTER_GUI_UPDATE();
  _colorchecker_rebuild_patch_list(self);
  if(g->patch >= p->num_patches) g->patch = p->num_patches - 1;
  if(g->patch < 0) g->patch = 0;
  dt_bauhaus_combobox_set(g->combobox_patch, g->patch);
  _colorchecker_update_sliders(self);
  DT_LEAVE_GUI_UPDATE();
  gtk_widget_queue_draw(g->area);
  dt_control_queue_redraw_center();
}

static void _reuse_calibration_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  dt_iop_colorchecker_params_t *p = self->params;

  const char *blob = dt_conf_get_string_const(CC_LAST_CALIBRATION_KEY);
  if(!blob || !*blob) return;

  int blob_len = 0;
  unsigned char *decoded = dt_exif_xmp_decode(blob, (int)strlen(blob), &blob_len);
  if(!decoded || blob_len != (int)sizeof(dt_iop_colorchecker_params_t))
  {
    free(decoded);
    return;
  }

  dt_iop_gui_enter_critical_section(self);
  memcpy(p, decoded, sizeof(dt_iop_colorchecker_params_t));
  g->profile_ready = FALSE;
  dt_iop_gui_leave_critical_section(self);
  free(decoded);

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  DT_ENTER_GUI_UPDATE();
  _colorchecker_rebuild_patch_list(self);
  if(g->patch >= p->num_patches) g->patch = p->num_patches - 1;
  if(g->patch < 0) g->patch = 0;
  dt_bauhaus_combobox_set(g->combobox_patch, g->patch);
  _colorchecker_update_sliders(self);
  dt_bauhaus_combobox_set(g->combobox_colorspace, p->colorspace);
  dt_bauhaus_combobox_set(g->combobox_anchor, p->anchor);
  gtk_widget_set_visible(g->combobox_anchor, p->colorspace == DT_IOP_CC_CS_XYZ);
  DT_LEAVE_GUI_UPDATE();
  gtk_widget_queue_draw(g->area);
  dt_control_queue_redraw_center();
}

static void _colorspace_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(!g || !g->combobox_anchor) return;

  const int cs = dt_bauhaus_combobox_get(widget);
  gtk_widget_set_visible(g->combobox_anchor, cs == DT_IOP_CC_CS_XYZ);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = IOP_GUI_ALLOC(colorchecker);
  const dt_iop_colorchecker_params_t *const p = self->default_params;

  // color chart calibration state
  g->checker = dt_get_color_checker(dt_conf_get_int("darkroom/modules/colorchecker/chart"));
  g->safety_margin = dt_conf_get_float("darkroom/modules/colorchecker/safety");
  if(g->safety_margin <= 0.f) g->safety_margin = 0.5f;
  g->box[0].x = -1.0f; // mark the bounding box as not yet initialised
  g->box[1].y = -1.0f;

  // custom 24-patch widget in addition to combo box
  g->area = dtgtk_drawing_area_new_with_aspect_ratio(4.0/6.0);
  gtk_widget_add_events(GTK_WIDGET(g->area),
                        GDK_POINTER_MOTION_MASK
                        | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK
                        | GDK_LEAVE_NOTIFY_MASK);
  g_signal_connect(G_OBJECT(g->area), "draw",
                   G_CALLBACK(checker_draw), self);
  g_signal_connect(G_OBJECT(g->area), "button-press-event",
                   G_CALLBACK(checker_button_press), self);
  g_signal_connect(G_OBJECT(g->area), "motion-notify-event",
                   G_CALLBACK(checker_motion_notify), self);

  g->patch = 0;
  g->drawn_patch = -1;
  g->combobox_patch = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->combobox_patch, NULL, N_("patch"));
  gtk_widget_set_tooltip_text(g->combobox_patch, _("color checker patch"));
  char cboxentry[1024];
  for(int k=0;k<p->num_patches;k++)
  {
    snprintf(cboxentry, sizeof(cboxentry), _("patch #%d"), k);
    dt_bauhaus_combobox_add(g->combobox_patch, cboxentry);
  }

  dt_color_picker_new(self, DT_COLOR_PICKER_POINT_AREA, g->combobox_patch);

  g->scale_L = dt_bauhaus_slider_new_with_range(self, -100.0, 200.0,
                                                0, 0.0f, 2);
  gtk_widget_set_tooltip_text
    (g->scale_L,
     _("adjust target color Lab 'L' channel\n"
       "lower values darken target color while higher brighten it"));
  dt_bauhaus_widget_set_label(g->scale_L, NULL, N_("lightness"));

  g->scale_a = dt_bauhaus_slider_new_with_range(self, -256.0, 256.0,
                                                0, 0.0f, 2);
  gtk_widget_set_tooltip_text
    (g->scale_a,
     _("adjust target color Lab 'a' channel\n"
       "lower values shift target color towards greens while"
       " higher shift towards magentas"));
  dt_bauhaus_widget_set_label(g->scale_a, NULL, N_("green-magenta offset"));
  dt_bauhaus_slider_set_stop(g->scale_a, 0.0, 0.0, 1.0, 0.2);
  dt_bauhaus_slider_set_stop(g->scale_a, 0.5, 1.0, 1.0, 1.0);
  dt_bauhaus_slider_set_stop(g->scale_a, 1.0, 1.0, 0.0, 0.2);

  g->scale_b = dt_bauhaus_slider_new_with_range(self, -256.0, 256.0,
                                                0, 0.0f, 2);
  gtk_widget_set_tooltip_text
    (g->scale_b,
     _("adjust target color Lab 'b' channel\n"
       "lower values shift target color towards blues"
       " while higher shift towards yellows"));
  dt_bauhaus_widget_set_label(g->scale_b, NULL, N_("blue-yellow offset"));
  dt_bauhaus_slider_set_stop(g->scale_b, 0.0, 0.0, 0.0, 1.0);
  dt_bauhaus_slider_set_stop(g->scale_b, 0.5, 1.0, 1.0, 1.0);
  dt_bauhaus_slider_set_stop(g->scale_b, 1.0, 1.0, 1.0, 0.0);

  g->scale_C = dt_bauhaus_slider_new_with_range(self, -128.0, 128.0,
                                                0, 0.0f, 2);
  gtk_widget_set_tooltip_text
    (g->scale_C,
     _("adjust target color saturation\n"
       "adjusts 'a' and 'b' channels of target color in Lab space"
       " simultaneously\n"
       "lower values scale towards lower saturation while higher"
       " scale towards higher saturation"));
  dt_bauhaus_widget_set_label(g->scale_C, NULL, N_("saturation"));

  g->absolute_target = 0;
  g->combobox_target = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->combobox_target, 0, N_("target color"));
  gtk_widget_set_tooltip_text
    (g->combobox_target,
     _("control target color of the patches\n"
       "relative - target color is relative from the patch original color\n"
       "absolute - target color is absolute Lab value"));
  dt_bauhaus_combobox_add(g->combobox_target, _("relative"));
  dt_bauhaus_combobox_add(g->combobox_target, _("absolute"));

  self->widget = dt_gui_vbox(g->area, g->combobox_patch, g->scale_L, g->scale_a,
                              g->scale_b, g->scale_C, g->combobox_target);

  g->combobox_colorspace = dt_bauhaus_combobox_from_params(self, "colorspace");
  dt_bauhaus_widget_set_label(g->combobox_colorspace, NULL, N_("color space"));
  gtk_widget_set_tooltip_text(g->combobox_colorspace,
                              _("select working color space (CIE Lab or CIE XYZ)"));

  g->combobox_anchor = dt_bauhaus_combobox_from_params(self, "anchor");
  dt_bauhaus_widget_set_label(g->combobox_anchor, NULL, N_("exposure reference"));
  gtk_widget_set_tooltip_text(g->combobox_anchor,
                              _("patch used to normalize the exposure in CIE XYZ mode"));
  gtk_widget_set_visible(g->combobox_anchor,
                         ((dt_iop_colorchecker_params_t *)self->params)->colorspace
                           == DT_IOP_CC_CS_XYZ);
  g_signal_connect(G_OBJECT(g->combobox_colorspace), "value-changed",
                   G_CALLBACK(_colorspace_callback), self);

  // ---- color chart calibration ----
  dt_gui_new_collapsible_section
    (&g->cs,
     "darkroom/modules/colorchecker/expand_calibrate",
     _("calibrate with a color chart"),
     GTK_BOX(self->widget),
     DT_ACTION(self));
  gtk_widget_set_tooltip_text(g->cs.expander,
                              _("use a color chart to fill the patches with measured colors"));
  g_signal_connect(G_OBJECT(g->cs.toggle), "toggled",
                   G_CALLBACK(_start_profiling_callback), self);

  g->checkers_list = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->checkers_list, N_("calibrate"), N_("chart"));
  gtk_widget_set_tooltip_text(g->checkers_list,
                              _("choose the vendor and the type of your chart"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Xrite ColorChecker 24 pre-2014"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Xrite/Calibrite ColorChecker 24 post-2014"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Datacolor SpyderCheckr 24 pre-2018"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Datacolor SpyderCheckr 24 post-2018"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Datacolor SpyderCheckr 48 pre-2018"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Datacolor SpyderCheckr 48 post-2018"));
  dt_bauhaus_combobox_add(g->checkers_list, _("Datacolor SpyderCheckr Photo"));
  dt_bauhaus_combobox_set(g->checkers_list,
                          dt_conf_get_int("darkroom/modules/colorchecker/chart"));
  g_signal_connect(G_OBJECT(g->checkers_list), "value-changed",
                   G_CALLBACK(_checker_changed_callback), self);

  g->safety = dt_bauhaus_slider_new_with_range_and_feedback(self, 0., 1., 0, 0.5, 3, TRUE);
  dt_bauhaus_widget_set_label(g->safety, N_("calibrate"), N_("patch scale"));
  gtk_widget_set_tooltip_text
    (g->safety,
     _("reduce the radius of the patches to select the more or less central part"));
  dt_bauhaus_slider_set(g->safety, g->safety_margin);
  g_signal_connect(G_OBJECT(g->safety), "value-changed",
                   G_CALLBACK(_safety_changed_callback), self);

  g->label_delta_E = dt_ui_label_new("");
  gtk_label_set_ellipsize(GTK_LABEL(g->label_delta_E), PANGO_ELLIPSIZE_NONE);
  gtk_label_set_line_wrap(GTK_LABEL(g->label_delta_E), TRUE);
  gtk_label_set_xalign(GTK_LABEL(g->label_delta_E), 0.0f);
  gtk_widget_set_tooltip_text(g->label_delta_E,
                              _("the delta E is using the CIE 2000 formula"));

  g->button_commit = dtgtk_button_new(dtgtk_cairo_paint_check_mark, 0, NULL);
  dt_action_define_iop(self, N_("calibrate"), N_("accept"),
                       g->button_commit, &dt_action_def_button);
  g_signal_connect(G_OBJECT(g->button_commit), "clicked",
                   G_CALLBACK(_commit_profile_callback), (gpointer)self);
  gtk_widget_set_tooltip_text(g->button_commit,
                              _("accept the measured colors and fill the patches"));

  g->button_profile = dtgtk_button_new(dtgtk_cairo_paint_refresh, 0, NULL);
  dt_action_define_iop(self, N_("calibrate"), N_("measure"),
                       g->button_profile, &dt_action_def_button);
  g_signal_connect(G_OBJECT(g->button_profile), "clicked",
                   G_CALLBACK(_run_profile_callback), (gpointer)self);
  gtk_widget_set_tooltip_text(g->button_profile,
                              _("measure the chart in the image"));

  g->button_reuse = dtgtk_button_new(dtgtk_cairo_paint_presets, 0, NULL);
  dt_action_define_iop(self, N_("calibrate"), N_("reuse"),
                       g->button_reuse, &dt_action_def_button);
  g_signal_connect(G_OBJECT(g->button_reuse), "clicked",
                   G_CALLBACK(_reuse_calibration_callback), (gpointer)self);
  gtk_widget_set_tooltip_text(g->button_reuse,
                              _("reuse the last calibration done on another image"));

  dt_gui_box_add(g->cs.container, g->checkers_list, g->safety,
                 g->label_delta_E,
                 dt_gui_hbox(dt_gui_align_right(g->button_reuse),
                             g->button_profile, g->button_commit));

  g_signal_connect(G_OBJECT(g->combobox_patch), "value-changed",
                   G_CALLBACK(patch_callback), self);
  g_signal_connect(G_OBJECT(g->scale_L), "value-changed",
                   G_CALLBACK(target_L_callback), self);
  g_signal_connect(G_OBJECT(g->scale_a), "value-changed",
                   G_CALLBACK(target_a_callback), self);
  g_signal_connect(G_OBJECT(g->scale_b), "value-changed",
                   G_CALLBACK(target_b_callback), self);
  g_signal_connect(G_OBJECT(g->scale_C), "value-changed",
                   G_CALLBACK(target_C_callback), self);
  g_signal_connect(G_OBJECT(g->combobox_target), "value-changed",
                   G_CALLBACK(target_callback), self);
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_colorchecker_gui_data_t *g = self->gui_data;
  if(!g) return;

  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;

  if(g->measured_lab)
  {
    dt_free_align(g->measured_lab);
    g->measured_lab = NULL;
  }
  if(g->measured_XYZ)
  {
    dt_free_align(g->measured_XYZ);
    g->measured_XYZ = NULL;
  }
  if(g->delta_E_in)
  {
    dt_free_align(g->delta_E_in);
    g->delta_E_in = NULL;
  }
  g_free(g->delta_E_label_text);
  g->delta_E_label_text = NULL;
}

#undef MAX_PATCHES

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

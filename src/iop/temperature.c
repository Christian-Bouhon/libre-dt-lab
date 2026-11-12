/*
    This file is part of darktable,
    Copyright (C) 2009-2026 darktable developers.

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

#include <assert.h>
#include <lcms2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/darktable.h"
#include "common/iop_order.h"
#include "common/opencl.h"
#include "common/wb_presets.h"
#include "control/control.h"
#include "control/conf.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop_gui.h"
#include "develop/imageop_math.h"
#include "develop/tiling.h"
#include "dtgtk/expander.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/color_picker_proxy.h"
#include "iop/iop_api.h"

// for Kelvin temperature and bogus WB
#include "common/chromatic_adaptation.h"
#include "common/colorspaces.h"
#include "common/illuminants.h"
#include "external/cie_colorimetric_tables.c"

DT_MODULE_INTROSPECTION(7, dt_iop_temperature_params_t)

#define INITIALBLACKBODYTEMPERATURE 4000

#define DT_IOP_LOWEST_TEMPERATURE 1901
#define DT_IOP_HIGHEST_TEMPERATURE 25000

#define DT_IOP_LOWEST_TINT 0.135
#define DT_IOP_HIGHEST_TINT 2.326

#define DT_IOP_NUM_OF_STD_TEMP_PRESETS 5

#define DT_IOP_WB_CAT_HUE_MAX 360.0f
#define DT_IOP_WB_CAT_CHROMA_MAX 300.0f

// If you reorder presets combo, change this consts
#define DT_IOP_TEMP_UNKNOWN -1
#define DT_IOP_TEMP_AS_SHOT 0
#define DT_IOP_TEMP_SPOT 1
#define DT_IOP_TEMP_USER 2
#define DT_IOP_TEMP_D65 3
#define DT_IOP_TEMP_D65_LATE 4

typedef enum dt_iop_temperature_advanced_mode_t
{
  DT_IOP_TEMP_ADVANCED_NONE = 0,
  DT_IOP_TEMP_ADVANCED_WB_CC,
  DT_IOP_TEMP_ADVANCED_WB_ONLY
} dt_iop_temperature_advanced_mode_t;

static void _gui_sliders_update(dt_iop_module_t *self);
static void _cat_controls_update(dt_iop_module_t *self);
static void _cat_xy_sliders_update(dt_iop_module_t *self, const gboolean sync_values);
static dt_iop_module_t *_find_channelmixerrgb(dt_iop_module_t *self);

typedef struct dt_iop_temperature_params_t
{
  float red;     // $MIN: 0.0 $MAX: 8.0
  float green;   // $MIN: 0.0 $MAX: 8.0
  float blue;    // $MIN: 0.0 $MAX: 8.0
  float various; // $MIN: 0.0 $MAX: 8.0
  int preset;

  // D65+CAT pipeline: reference the raw channels to D65 before demosaicing and
  // let color calibration perform the chromatic adaptation
  gboolean use_cat;                // $DEFAULT: TRUE $DESCRIPTION: "prepare raw in D65 and drive chromatic adaptation"
  dt_illuminant_t cat_illuminant;  // $DEFAULT: DT_ILLUMINANT_FROM_WB $DESCRIPTION: "source"
  dt_illuminant_fluo_t cat_illum_fluo; // $DEFAULT: DT_ILLUMINANT_FLUO_F3 $DESCRIPTION: "F source"
  dt_illuminant_led_t cat_illum_led;   // $DEFAULT: DT_ILLUMINANT_LED_B5 $DESCRIPTION: "LED source"
  dt_adaptation_t cat_adaptation;  // $DEFAULT: DT_ADAPTATION_CAT16 $DESCRIPTION: "adaptation"
  float cat_x;                     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.3127 $DESCRIPTION: "x"
  float cat_y;                     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.3290 $DESCRIPTION: "y"
  float cat_temperature;           // $MIN: 0.0 $MAX: 25000.0 $DEFAULT: 6504.0
} dt_iop_temperature_params_t;

typedef struct dt_iop_temperature_gui_data_t
{
  GtkWidget *scale_k, *scale_tint, *scale_r, *scale_g, *scale_b, *scale_y;
  GtkWidget *temp_label_box;
  GtkWidget *presets;
  GtkWidget *finetune;
  GtkWidget *buttonbar;
  GtkWidget *colorpicker;
  GtkWidget *advanced_wb_cc;
  GtkWidget *advanced_wb_only;
  gboolean advanced_update;
  GtkWidget *btn_asshot; //As Shot
  GtkWidget *btn_user;
  GtkWidget *btn_d65;
  GtkWidget *btn_d65_late;
  GtkWidget *temp_label;
  GtkWidget *balance_label;
  int preset_cnt;
  int preset_num[54];
  double mod_coeff[4];
  float mod_temp, mod_tint;
  float last_daylight_temperature;
  float last_bb_temperature;
  gboolean temp_tint_update;
  double XYZ_to_CAM[4][3], CAM_to_XYZ[3][4];
  gboolean colored_sliders;
  gboolean blackbody_is_confusing;
  gboolean button_bar_visible;
  dt_gui_collapsible_section_t cs;

  // chromatic adaptation section
  GtkWidget *cat_illuminant, *cat_illum_fluo, *cat_illum_led;
  GtkWidget *cat_adaptation, *cat_x, *cat_y;
  dt_gui_collapsible_section_t cat_cs;
  dt_gui_collapsible_section_t advanced_cs;
} dt_iop_temperature_gui_data_t;

typedef struct dt_iop_temperature_data_t
{
  float coeffs[4];
  int preset;
} dt_iop_temperature_data_t;

typedef struct dt_iop_temperature_global_data_t
{
  int kernel_whitebalance_4f;
  int kernel_whitebalance_1f;
  int kernel_whitebalance_1f_xtrans;
} dt_iop_temperature_global_data_t;

typedef struct dt_iop_temperature_preset_data_t
{
  int no_ft_pos;
  int min_ft_pos;
  int max_ft_pos;
} dt_iop_temperature_preset_data_t;

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  typedef struct dt_iop_temperature_params_v3_t
  {
    float red;
    float green;
    float blue;
    float various;
  } dt_iop_temperature_params_v3_t;

  typedef struct dt_iop_temperature_params_v4_t
  {
    float red;
    float green;
    float blue;
    float various;
    int preset;
  } dt_iop_temperature_params_v4_t;

  typedef struct dt_iop_temperature_params_v5_t
  {
    float red;
    float green;
    float blue;
    float various;
    int preset;
    gboolean use_cat;
    gboolean advanced;
    dt_illuminant_t cat_illuminant;
    dt_illuminant_fluo_t cat_illum_fluo;
    dt_illuminant_led_t cat_illum_led;
    dt_adaptation_t cat_adaptation;
    float cat_x;
    float cat_y;
    float cat_temperature;
  } dt_iop_temperature_params_v5_t;

  typedef struct dt_iop_temperature_params_v6_t
  {
    float red;
    float green;
    float blue;
    float various;
    int preset;
    gboolean use_cat;
    dt_illuminant_t cat_illuminant;
    dt_illuminant_fluo_t cat_illum_fluo;
    dt_illuminant_led_t cat_illum_led;
    dt_adaptation_t cat_adaptation;
    float cat_x;
    float cat_y;
  } dt_iop_temperature_params_v6_t;

  if(old_version == 2)
  {
    typedef struct dt_iop_temperature_params_v2_t
    {
      float temp_out;
      float coeffs[3];
    } dt_iop_temperature_params_v2_t;

    const dt_iop_temperature_params_v2_t *o = (dt_iop_temperature_params_v2_t *)old_params;
    dt_iop_temperature_params_v3_t *n = malloc(sizeof(dt_iop_temperature_params_v3_t));

    n->red = o->coeffs[0];
    n->green = o->coeffs[1];
    n->blue = o->coeffs[2];
    n->various = NAN;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_temperature_params_v3_t);
    *new_version = 3;
    return 0;
  }

  if(old_version == 3)
  {
    const dt_iop_temperature_params_v3_t *o = (dt_iop_temperature_params_v3_t *)old_params;
    dt_iop_temperature_params_v4_t *n = malloc(sizeof(dt_iop_temperature_params_v4_t));

    n->red = o->red;
    n->green = o->green;
    n->blue = o->blue;
    n->various = NAN;
    n->preset = DT_IOP_TEMP_UNKNOWN;
    *new_params = n;
    *new_params_size = sizeof(dt_iop_temperature_params_v4_t);
    *new_version = 4;
    return 0;
  }

  if(old_version == 4)
  {
    const dt_iop_temperature_params_v4_t *o = (dt_iop_temperature_params_v4_t *)old_params;
    dt_iop_temperature_params_v5_t *n = malloc(sizeof(dt_iop_temperature_params_v5_t));

    n->red = o->red;
    n->green = o->green;
    n->blue = o->blue;
    n->various = o->various;
    n->preset = o->preset;

    // old edits keep the legacy pipeline and render exactly as before
    n->use_cat = FALSE;
    n->advanced = FALSE;
    n->cat_illuminant = DT_ILLUMINANT_FROM_WB;
    n->cat_illum_fluo = DT_ILLUMINANT_FLUO_F3;
    n->cat_illum_led = DT_ILLUMINANT_LED_B5;
    n->cat_adaptation = DT_ADAPTATION_CAT16;
    n->cat_x = 0.3127f;
    n->cat_y = 0.3290f;
    n->cat_temperature = 6504.0f;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_temperature_params_v5_t);
    *new_version = 5;
    return 0;
  }

  if(old_version == 5)
  {
    const dt_iop_temperature_params_v5_t *o = (dt_iop_temperature_params_v5_t *)old_params;
    dt_iop_temperature_params_t *n = malloc(sizeof(dt_iop_temperature_params_t));

    n->red = o->red;
    n->green = o->green;
    n->blue = o->blue;
    n->various = o->various;
    n->preset = o->preset;
    n->use_cat = o->use_cat;
    n->cat_illuminant = o->cat_illuminant;
    n->cat_illum_fluo = o->cat_illum_fluo;
    n->cat_illum_led = o->cat_illum_led;
    n->cat_adaptation = o->cat_adaptation;
    n->cat_x = o->cat_x;
    n->cat_y = o->cat_y;
    n->cat_temperature = o->cat_temperature;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_temperature_params_t);
    *new_version = 7;
    return 0;
  }

  if(old_version == 6)
  {
    const dt_iop_temperature_params_v6_t *o = (dt_iop_temperature_params_v6_t *)old_params;
    dt_iop_temperature_params_t *n = malloc(sizeof(dt_iop_temperature_params_t));

    n->red = o->red;
    n->green = o->green;
    n->blue = o->blue;
    n->various = o->various;
    n->preset = o->preset;
    n->use_cat = o->use_cat;
    n->cat_illuminant = o->cat_illuminant;
    n->cat_illum_fluo = o->cat_illum_fluo;
    n->cat_illum_led = o->cat_illum_led;
    n->cat_adaptation = o->cat_adaptation;
    n->cat_x = o->cat_x;
    n->cat_y = o->cat_y;
    // no CAT temperature was stored in v6: keep the legacy coefficient-based
    // derivation until the user touches the temperature slider
    n->cat_temperature = 0.0f;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_temperature_params_t);
    *new_version = 7;
    return 0;
  }
  return 1;
}

static inline void _temp_params_from_array(dt_iop_temperature_params_t *p,
                                           const double a[4])
{
  float *coeffs = (float *)p;
  for_four_channels(c)
    coeffs[c] = (float)a[c];
}

static inline void _temp_array_from_params(double a[4],
                                           const dt_iop_temperature_params_t *p)
{
  float *coeffs = (float *)p;
  for_four_channels(c)
   a[c] = coeffs[c];
}

// derive the scene illuminant chromaticity from camera WB coefficients
static gboolean _coeffs_to_illuminant_xy(const dt_image_t *img,
                                         const dt_aligned_pixel_t coeffs,
                                         float *x, float *y)
{
  if(!dt_image_is_matrix_correction_supported(img)) return FALSE;

  float XYZ_to_CAM[4][3];
  dt_mark_colormatrix_invalid(&XYZ_to_CAM[0][0]);

  if(dt_is_valid_colormatrix(img->d65_color_matrix[0]))
  {
    XYZ_to_CAM[0][0] = img->d65_color_matrix[0];
    XYZ_to_CAM[0][1] = img->d65_color_matrix[1];
    XYZ_to_CAM[0][2] = img->d65_color_matrix[2];
    XYZ_to_CAM[1][0] = img->d65_color_matrix[3];
    XYZ_to_CAM[1][1] = img->d65_color_matrix[4];
    XYZ_to_CAM[1][2] = img->d65_color_matrix[5];
    XYZ_to_CAM[2][0] = img->d65_color_matrix[6];
    XYZ_to_CAM[2][1] = img->d65_color_matrix[7];
    XYZ_to_CAM[2][2] = img->d65_color_matrix[8];
  }
  else
  {
    for(int k = 0; k < 4; k++)
      for(int i = 0; i < 3; i++)
        XYZ_to_CAM[k][i] = img->adobe_XYZ_to_CAM[k][i];
  }

  if(!dt_is_valid_colormatrix(XYZ_to_CAM[0][0])) return FALSE;

  float CAM_to_XYZ[4][3];
  dt_mark_colormatrix_invalid(&CAM_to_XYZ[0][0]);
  matrice_pseudoinverse(XYZ_to_CAM, CAM_to_XYZ, 3);
  if(!dt_is_valid_colormatrix(CAM_to_XYZ[0][0])) return FALSE;

  WB_coeffs_to_illuminant_xy(CAM_to_XYZ, coeffs, x, y);
  return (x && y && *x > 0.0f && *y > 0.0f);
}

static gboolean _is_usable_illuminant_xy(const float x, const float y)
{
  const float z = 1.f - x - y;
  if(!(x > 0.f && y > 0.f && z > 0.f)) return FALSE;

  dt_aligned_pixel_t XYZ = { 0.f };
  dt_aligned_pixel_t LMS = { 0.f };
  illuminant_xy_to_XYZ(x, y, XYZ);
  convert_XYZ_to_CAT16_LMS(XYZ, LMS);
  return LMS[0] > 0.f && LMS[1] > 0.f && LMS[2] > 0.f;
}

static gboolean _source_illuminant_xy(const dt_image_t *img,
                                      const dt_iop_temperature_params_t *p,
                                      const dt_illuminant_t source,
                                      float *x, float *y)
{
  if(source == DT_ILLUMINANT_FROM_WB)
  {
    const dt_aligned_pixel_t M = { p->red, p->green, p->blue, p->various };
    return _coeffs_to_illuminant_xy(img, M, x, y)
           && _is_usable_illuminant_xy(*x, *y);
  }
  else if(source == DT_ILLUMINANT_CUSTOM)
  {
    *x = p->cat_x;
    *y = p->cat_y;
    return _is_usable_illuminant_xy(*x, *y);
  }
  else if((source == DT_ILLUMINANT_D || source == DT_ILLUMINANT_BB)
          && p->cat_temperature <= 0.0f)
  {
    // v6 edits had no CAT temperature: keep their coefficient-derived source
    const dt_aligned_pixel_t M = { p->red, p->green, p->blue, p->various };
    return _coeffs_to_illuminant_xy(img, M, x, y)
           && _is_usable_illuminant_xy(*x, *y);
  }

  float cx = p->cat_x;
  float cy = p->cat_y;
  const gboolean found =
    illuminant_to_xy(source, img, NULL, &cx, &cy, p->cat_temperature,
                     p->cat_illum_fluo, p->cat_illum_led)
    && _is_usable_illuminant_xy(cx, cy);
  if(found)
  {
    *x = cx;
    *y = cy;
  }
  return found;
}

static float _cct_from_xy(const float x, const float y)
{
  float t = xy_to_CCT(x, y);
  if(t < 3000.f && t > 1667.f)
    t = CCT_reverse_lookup(x, y);

  if(!isfinite(t) || t <= 0.0f)
    return 0.0f;

  if(t < DT_IOP_LOWEST_TEMPERATURE) t = DT_IOP_LOWEST_TEMPERATURE;
  if(t > DT_IOP_HIGHEST_TEMPERATURE) t = DT_IOP_HIGHEST_TEMPERATURE;
  return t;
}

static gboolean _ignore_missing_wb(dt_image_t *img)
{
  // Ignore files that end with "-hdr.dng" since these are broken files we
  // generated without any proper WB tagged
  if(g_str_has_suffix(img->filename,"-hdr.dng"))
    return TRUE;

  // If we failed to read the image correctly, don't complain about WB
  if(img->load_status != DT_IMAGEIO_OK && img->load_status != DT_IMAGEIO_CACHE_FULL)
    return TRUE;

  static const char *const ignored_cameras[] = {
    "Canon PowerShot A610",
    "Canon PowerShot S3 IS",
    "Canon PowerShot A620",
    "Canon PowerShot A720 IS",
    "Canon PowerShot A630",
    "Canon PowerShot A640",
    "Canon PowerShot A650",
    "Canon PowerShot SX110 IS",
    "Mamiya ZD",
    "Canon EOS D2000C",
    "Kodak EOS DCS 1",
    "Kodak DCS560C",
    "Kodak DCS460D",
    "Nikon E5700",
    "Sony DSC-F828",
    "GITUP GIT2",
  };

  for(int i=0; i < sizeof(ignored_cameras)/sizeof(ignored_cameras[1]); i++)
    if(!strcmp(img->camera_makermodel, ignored_cameras[i]))
      return TRUE;

  return FALSE;
}


const char *name()
{
  return C_("modulename", "white balance");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("scale raw RGB channels to balance white and help demosaicing"),
     _("corrective"),
     _("linear, raw, scene-referred"),
     _("linear, raw"),
     _("linear, raw, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_GRADING;
}

int flags()
{
  return IOP_FLAGS_ALLOW_TILING | IOP_FLAGS_ONE_INSTANCE | IOP_FLAGS_UNSAFE_COPY;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  // This module may work in RAW or RGB (e.g. for TIFF files)
  // depending on the input The module does not change the color space
  // between the input and output, therefore implement it here
  if(piece && piece->dsc_in.cst != IOP_CS_RAW)
    return IOP_CS_RGB;
  return IOP_CS_RAW;
}

/*
 * Spectral power distribution functions
 * https://en.wikipedia.org/wiki/Spectral_power_distribution
 */
typedef double((*spd)(unsigned long int wavelength, double TempK));

/*
 * Bruce Lindbloom, "Spectral Power Distribution of a Blackbody Radiator"
 * http://www.brucelindbloom.com/Eqn_Blackbody.html
 */
static double _spd_blackbody(unsigned long int wavelength, double TempK)
{
  // convert wavelength from nm to m
  const long double lambda = (double)wavelength * 1e-9;

/*
 * these 2 constants were computed using following Sage code:
 *
 * (from http://physics.nist.gov/cgi-bin/cuu/Value?h)
 * h = 6.62606957 * 10^-34 # Planck
 * c= 299792458 # speed of light in vacuum
 * k = 1.3806488 * 10^-23 # Boltzmann
 *
 * c_1 = 2 * pi * h * c^2
 * c_2 = h * c / k
 *
 * print 'c_1 = ', c_1, ' ~= ', RealField(128)(c_1)
 * print 'c_2 = ', c_2, ' ~= ', RealField(128)(c_2)
 */

#define c1 3.7417715246641281639549488324352159753e-16L
#define c2 0.014387769599838156481252937624049081933L

  return (double)(c1 / (powl(lambda, 5) * (expl(c2 / (lambda * TempK)) - 1.0L)));

#undef c2
#undef c1
}

/*
 * Bruce Lindbloom, "Spectral Power Distribution of a CIE D-Illuminant"
 * http://www.brucelindbloom.com/Eqn_DIlluminant.html
 * and https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D
 */
static double _spd_daylight(unsigned long int wavelength, double TempK)
{
  cmsCIExyY WhitePoint = { D65xyY.x, D65xyY.y, 1.0 };

  /*
   * Bruce Lindbloom, "TempK to xy"
   * http://www.brucelindbloom.com/Eqn_T_to_xy.html
   */
  cmsWhitePointFromTemp(&WhitePoint, TempK);

  const double M = (0.0241 + 0.2562 * WhitePoint.x - 0.7341 * WhitePoint.y),
               m1 = (-1.3515 - 1.7703 * WhitePoint.x + 5.9114 * WhitePoint.y) / M,
               m2 = (0.0300 - 31.4424 * WhitePoint.x + 30.0717 * WhitePoint.y) / M;

  const unsigned long int j
      = ((wavelength - cie_daylight_components[0].wavelength)
         / (cie_daylight_components[1].wavelength
            - cie_daylight_components[0].wavelength));

  return (cie_daylight_components[j].S[0] + m1 * cie_daylight_components[j].S[1]
          + m2 * cie_daylight_components[j].S[2]);
}

/*
 * Bruce Lindbloom, "Computing XYZ From Spectral Data (Emissive Case)"
 * http://www.brucelindbloom.com/Eqn_Spect_to_XYZ.html
 */
static cmsCIEXYZ _spectrum_to_XYZ(double TempK, spd I)
{
  cmsCIEXYZ Source = {.X = 0.0, .Y = 0.0, .Z = 0.0 };

  /*
   * Color matching functions
   * https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions
   */
  for(size_t i = 0; i < cie_1931_std_colorimetric_observer_count; i++)
  {
    const unsigned long int lambda =
      cie_1931_std_colorimetric_observer[0].wavelength
      + (cie_1931_std_colorimetric_observer[1].wavelength
         - cie_1931_std_colorimetric_observer[0].wavelength) * i;

    const double P = I(lambda, TempK);
    Source.X += P * cie_1931_std_colorimetric_observer[i].xyz.X;
    Source.Y += P * cie_1931_std_colorimetric_observer[i].xyz.Y;
    Source.Z += P * cie_1931_std_colorimetric_observer[i].xyz.Z;
  }

  // normalize so that each component is in [0.0, 1.0] range
  const double _max = fmax(fmax(Source.X, Source.Y), Source.Z);
  Source.X /= _max;
  Source.Y /= _max;
  Source.Z /= _max;

  return Source;
}

// TODO: temperature and tint cannot be disjoined! (here it assumes no tint)
static cmsCIEXYZ _temperature_to_XYZ(double TempK)
{
  if(TempK < DT_IOP_LOWEST_TEMPERATURE) TempK = DT_IOP_LOWEST_TEMPERATURE;
  if(TempK > DT_IOP_HIGHEST_TEMPERATURE) TempK = DT_IOP_HIGHEST_TEMPERATURE;

  if(TempK < INITIALBLACKBODYTEMPERATURE)
  {
    // if temperature is less than 4000K we use blackbody,
    // because there will be no Daylight reference below 4000K...
    return _spectrum_to_XYZ(TempK, _spd_blackbody);
  }
  else
  {
    return _spectrum_to_XYZ(TempK, _spd_daylight);
  }
}

static cmsCIEXYZ _temperature_tint_to_XYZ(double TempK, double tint)
{
  cmsCIEXYZ xyz = _temperature_to_XYZ(TempK);

  xyz.Y /= tint; // TODO: This is baaad!

  return xyz;
}

// binary search inversion
static void _XYZ_to_temperature(cmsCIEXYZ XYZ, float *TempK, float *tint)
{
  double maxtemp = DT_IOP_HIGHEST_TEMPERATURE, mintemp = DT_IOP_LOWEST_TEMPERATURE;
  cmsCIEXYZ _xyz;

  for(*TempK = (maxtemp + mintemp) / 2.0;
      (maxtemp - mintemp) > 1.0;
      *TempK = (maxtemp + mintemp) / 2.0)
  {
    _xyz = _temperature_to_XYZ(*TempK);
    if(_xyz.Z / _xyz.X > XYZ.Z / XYZ.X)
      maxtemp = *TempK;
    else
      mintemp = *TempK;
  }

  // TODO: Fix this to move orthogonally to planckian locus
  *tint = (_xyz.Y / _xyz.X) / (XYZ.Y / XYZ.X);


  if(*TempK < DT_IOP_LOWEST_TEMPERATURE) *TempK = DT_IOP_LOWEST_TEMPERATURE;
  if(*TempK > DT_IOP_HIGHEST_TEMPERATURE) *TempK = DT_IOP_HIGHEST_TEMPERATURE;
  if(*tint < DT_IOP_LOWEST_TINT) *tint = DT_IOP_LOWEST_TINT;
  if(*tint > DT_IOP_HIGHEST_TINT) *tint = DT_IOP_HIGHEST_TINT;
}

static void _xyz2mul(dt_iop_module_t *self,
                    cmsCIEXYZ xyz,
                    double mul[4])
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  double XYZ[3] = { xyz.X, xyz.Y, xyz.Z };

  double CAM[4];
  for(int k = 0; k < 4; k++)
  {
    CAM[k] = 0.0;
    for(int i = 0; i < 3; i++)
    {
      CAM[k] += g->XYZ_to_CAM[k][i] * XYZ[i];
    }
  }

  for(int k = 0; k < 4; k++) mul[k] = 1.0 / CAM[k];
}

static void _temp2mul(dt_iop_module_t *self,
                     const double TempK,
                     const double tint,
                     double mul[4])
{
  cmsCIEXYZ xyz = _temperature_to_XYZ(TempK);

  xyz.Y /= tint; // TODO: This is baaad!
  /**
   * TODO:
   * problem here is that tint as it is is just a nasty hack modifying Y component
   * and therefore changing RGB coefficients in wrong way,
   * because modifying only Y in that way doesn’t move XYZ point orthogonally
   * to planckian locus. That means it actually changes temperature and thus it lies!
   */

  _xyz2mul(self, xyz, mul);
}

static cmsCIEXYZ _mul2xyz(dt_iop_module_t *self,
                         const dt_iop_temperature_params_t *p)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  double CAM[4];
  _temp_array_from_params(CAM, p);

  for(int k = 0; k < 4; k++)
    CAM[k] = CAM[k] > 0.0f ? 1.0 / CAM[k] : 0.0f;

  double XYZ[3];
  for(int k = 0; k < 3; k++)
  {
    XYZ[k] = 0.0;
    for(int i = 0; i < 4; i++)
    {
      XYZ[k] += g->CAM_to_XYZ[k][i] * CAM[i];
    }
  }

  return (cmsCIEXYZ){ XYZ[0], XYZ[1], XYZ[2] };
}

static void _mul2temp(dt_iop_module_t *self,
                     dt_iop_temperature_params_t *p,
                     float *TempK,
                     float *tint)
{
  _XYZ_to_temperature(_mul2xyz(self, p), TempK, tint);
}

DT_OMP_DECLARE_SIMD(aligned(inp,outp))
static inline void scaled_copy_4wide(float *const outp,
                                     const float *const inp,
                                     const float *const coeffs)
{
  // this needs to be in a separate function to make GCC8 vectorize it
  // at -O2 as well as -O3
  for_four_channels(c, aligned(inp, coeffs, outp))
    outp[c] = inp[c] * coeffs[c];
}

static inline void _publish_chroma(dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_temperature_data_t *const d = piece->data;
  dt_iop_module_t *self = piece->module;
  dt_dev_chroma_t *chr = &self->dev->chroma;

  piece->pipe->dsc.temperature.enabled = piece->enabled;
  for_four_channels(k)
  {
    piece->pipe->dsc.temperature.coeffs[k] = d->coeffs[k];
    piece->pipe->dsc.processed_maximum[k] =
      d->coeffs[k] * piece->pipe->dsc.processed_maximum[k];
    chr->wb_coeffs[k] = d->coeffs[k];
  }
  chr->late_correction = !piece->pipe->wb.active && piece->enabled
                          && (d->preset == DT_IOP_TEMP_D65_LATE);
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const uint32_t filters = piece->filters;
  const uint8_t(*const xtrans)[6] = piece->xtrans;
  const dt_iop_temperature_data_t *const d = piece->data;

  const float *const in = (const float *const)ivoid;
  float *const out = (float *const)ovoid;
  const float *const d_coeffs = d->coeffs;
  const int width = roi_out->width;
  const int height = roi_out->height;

  if(filters == 9u)
  { // xtrans float mosaiced
    DT_OMP_FOR()
    for(int j = 0; j < height; j++)
    {
      const float DT_ALIGNED_PIXEL coeffs[3][4] =
      {
        { d_coeffs[FCNxtrans(j, 0, xtrans)],
          d_coeffs[FCNxtrans(j, 1, xtrans)],
          d_coeffs[FCNxtrans(j, 2, xtrans)],
          d_coeffs[FCNxtrans(j, 3, xtrans)] },
        { d_coeffs[FCNxtrans(j, 4, xtrans)],
          d_coeffs[FCNxtrans(j, 5, xtrans)],
          d_coeffs[FCNxtrans(j, 6, xtrans)],
          d_coeffs[FCNxtrans(j, 7, xtrans)] },
        { d_coeffs[FCNxtrans(j, 8, xtrans)],
          d_coeffs[FCNxtrans(j, 9, xtrans)],
          d_coeffs[FCNxtrans(j, 10, xtrans)],
          d_coeffs[FCNxtrans(j, 11, xtrans)] },
      };
      // process sensels four at a time (note that attempting to
      //ensure alignment for this main loop actually slowed things
      //down marginally)
      int i = 0;
      for(int coeff = 0; i + 4 < width; i += 4, coeff = (coeff+1)%3)
      {
        const size_t p = (size_t)j * width + i;
        for_four_channels(c) // in and out are NOT aligned when width is not a multiple of 4
          out[p+c] = in[p+c] * coeffs[coeff][c];
      }
      // process the leftover sensels
      for(; i < width; i++)
      {
        const size_t p = (size_t)j * width + i;
        out[p] = in[p] * d_coeffs[FCNxtrans(j, i, xtrans)];
      }
    }
  }
  else if(filters)
  { // bayer float mosaiced
    DT_OMP_FOR()
    for(int j = 0; j < height; j++)
    {
      int i = 0;

      const int alignment = 3 & (4 - ((j*width) & 3));

      // process the unaligned sensels at the start of the row (when
      // width is not a multiple of 4)
      for(; i < alignment; i++)
      {
        const size_t p = (size_t)j * width + i;
        out[p] = in[p] * d_coeffs[FC(j, i, filters)];
      }
      const dt_aligned_pixel_t coeffs =
        { d_coeffs[FC(j, i, filters)],
          d_coeffs[FC(j, i + 1,filters)],
          d_coeffs[FC(j, i + 2, filters)],
          d_coeffs[FC(j, i + 3, filters)] };

      // process sensels four at a time
      for(; i < width - 4; i += 4)
      {
        const size_t p = (size_t)j * width + i;
        scaled_copy_4wide(out + p,in + p, coeffs);
      }

      // process the leftover sensels
      for(; i < width; i++)
      {
        const size_t p = (size_t)j * width + i;
        out[p] = in[p] * d_coeffs[FC(j, i, filters)];
      }
    }
  }
  else
  { // non-mosaiced
    const size_t npixels = (size_t)height * width;

    DT_OMP_FOR()
    for(size_t k = 0; k < 4*npixels; k += 4)
    {
      for_each_channel(c,aligned(in,out))
      {
        out[k+c] = in[k+c] * d_coeffs[c];
      }
    }
  }

  _publish_chroma(piece);
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_temperature_data_t *d = piece->data;
  const dt_iop_temperature_global_data_t *gd = self->global_data;
  dt_dev_pixelpipe_t *pipe = piece->pipe;

  const int devid = pipe->devid;
  const uint32_t filters = piece->filters;
  cl_mem dev_coeffs = NULL;
  cl_mem dev_xtrans = NULL;
  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  int kernel = -1;
  if(filters == 9u) kernel = gd->kernel_whitebalance_1f_xtrans;
  else if(filters)  kernel = gd->kernel_whitebalance_1f;
  else              kernel = gd->kernel_whitebalance_4f;

  if(filters == 9u)
  {
    dev_xtrans = dt_opencl_copy_host_to_device_constant(devid, sizeof(piece->xtrans), piece->xtrans);
    if(dev_xtrans == NULL) goto error;
  }

  dev_coeffs = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * 4, d->coeffs);
  if(dev_coeffs == NULL) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, kernel, roi_in->width, roi_in->height,
                                         CLARG(dev_in), CLARG(dev_out),
                                         CLARG(roi_in->width), CLARG(roi_in->height),
                                         CLARG(dev_coeffs), CLARG(filters), CLARG(dev_xtrans));
  if(err != CL_SUCCESS) goto error;

  _publish_chroma(piece);

error:
  dt_opencl_release_mem_object(dev_coeffs);
  dt_opencl_release_mem_object(dev_xtrans);
  return err;
}
#endif

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_temperature_params_t *p = (dt_iop_temperature_params_t *)p1;
  dt_iop_temperature_data_t *d = piece->data;
  float *tcoeffs = (float *)p;

  if(self->hide_enable_button)
    piece->enabled = FALSE;

  dt_dev_chroma_t *chr = &self->dev->chroma;
  const dt_image_t *img = &self->dev->image_storage;
  const gboolean true_monochrome =
    dt_image_monochrome_flags(img) & DT_IMAGE_MONOCHROME;
  const gboolean is_raw = dt_image_is_rawprepare_supported(img);
  const gboolean active = piece->enabled && !true_monochrome;

  // always start from a clean per-pipe state
  pipe->wb.active = FALSE;
  pipe->wb.has_cat = FALSE;

  if(active && p->use_cat)
  {
    // raw channels are always referenced to the camera D65 white before
    // demosaicing, and color calibration performs the scene adaptation
    double D65[4] = { 2.0, 1.0, 1.5, 1.0 };
    if(chr->D65coeffs[0] > 0.0 && chr->D65coeffs[1] > 0.0 && chr->D65coeffs[2] > 0.0)
      for_four_channels(k) D65[k] = chr->D65coeffs[k];

    float x = 0.3127f;
    float y = 0.3290f;
    const gboolean found = _source_illuminant_xy(img, p, p->cat_illuminant, &x, &y);

    if(!found)
    {
      x = 0.3127f;
      y = 0.3290f;
    }

    for_four_channels(k)
      d->coeffs[k] = is_raw ? (float)D65[k] : 1.0f;
    d->preset = p->preset;

    chr->late_correction = FALSE;
    chr->cat_pipeline = TRUE;
    for_four_channels(k)
      chr->wb_coeffs[k] = d->coeffs[k];

    const gboolean has_cat =
      dt_image_is_matrix_correction_supported(img)
      && p->cat_adaptation != DT_ADAPTATION_RGB;

    pipe->wb.active = TRUE;
    pipe->wb.has_cat = has_cat;
    pipe->wb.illuminant_xy[0] = x;
    pipe->wb.illuminant_xy[1] = y;
    pipe->wb.adaptation = has_cat ? (int)p->cat_adaptation : (int)DT_ADAPTATION_RGB;

    chr->temperature = self;

    if(dt_pipe_is_preview(pipe) && !piece->enabled)
      dt_iop_set_module_trouble_message(self, NULL, NULL, NULL);
    return;
  }

  // legacy pipeline
  chr->cat_pipeline = FALSE;

  if(self->hide_enable_button)
  {
    for_four_channels(k)
      chr->wb_coeffs[k] = 1.0f;
    return;
  }

  for_four_channels(k)
  {
    d->coeffs[k] = tcoeffs[k];
    chr->wb_coeffs[k] = piece->enabled ? d->coeffs[k] : 1.0f;
  }

  // 4Bayer images not implemented in OpenCL yet
  if(self->dev->image_storage.flags & DT_IMAGE_4BAYER)
    piece->process_cl_ready = FALSE;

  d->preset = p->preset;

  /* Make sure the chroma information stuff is valid
     If piece is disabled we always clear the trouble message and
     make sure chroma does know there is no temperature module.
  */
  chr->late_correction = piece->enabled
                          && p->preset == DT_IOP_TEMP_D65_LATE;
  chr->temperature = piece->enabled ? self : NULL;
  if(dt_pipe_is_preview(pipe) && !piece->enabled)
    dt_iop_set_module_trouble_message(self, NULL, NULL, NULL);
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_temperature_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

static int _generate_preset_combo(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  int presets_found = 0;

  const char *wb_name = NULL;
  if(!dt_image_is_ldr(&self->dev->image_storage))
    for(int i = 0; i < dt_wb_presets_count(); i++)
    {
      if(presets_found >= 50) break;

      const dt_wb_data *wbp = dt_wb_preset(i);
      if(!strcmp(wbp->make, self->dev->image_storage.camera_maker)
         && !strcmp(wbp->model, self->dev->image_storage.camera_model))
      {
        if(!wb_name) // This is first found preset for maker/model. add section.
        {
          char *section = g_strdup_printf("%s %s",
                                          self->dev->image_storage.camera_maker,
                                          self->dev->image_storage.camera_model);
          dt_bauhaus_combobox_add_section(g->presets, section);
          g_free(section);
          g->preset_cnt++;
        }
        if(!wb_name || strcmp(wb_name, wbp->name))
        {
          // new preset found
          dt_iop_temperature_preset_data_t *preset =
            malloc(sizeof(dt_iop_temperature_preset_data_t));
          wb_name = wbp->name;
          preset->no_ft_pos = i;
          preset->max_ft_pos = i;
          preset->min_ft_pos = i;
          if(wbp->tuning != 0)
          {
            // finetuning found.
            // min finetuning is always first, since wb_preset is ordered.
            int ft_pos = i;
            int last_ft = wbp->tuning;
            preset->min_ft_pos = ft_pos++;
            while(strcmp(wb_name, dt_wb_preset(ft_pos)->name) == 0)
            {
              if(dt_wb_preset(ft_pos)->tuning == 0)
              {
                preset->no_ft_pos = ft_pos;
              }
              if(dt_wb_preset(ft_pos)->tuning > last_ft)
              {
                preset->max_ft_pos = ft_pos;
                last_ft = dt_wb_preset(ft_pos)->tuning;
              }
              ft_pos++;
            }

          }
          dt_bauhaus_combobox_add_full(g->presets, _(wbp->name),
                                       DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT, preset, free, TRUE);
          g->preset_num[g->preset_cnt] = i;
          g->preset_cnt++;
          presets_found++;
        }
      }
    }


  return presets_found;
}

static void _color_finetuning_slider(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  dt_bauhaus_slider_clear_stops(g->finetune);
  dt_bauhaus_slider_set_feedback(g->finetune, !g->colored_sliders);

  if(!g->colored_sliders) return;

  dt_iop_temperature_preset_data_t *preset = dt_bauhaus_combobox_get_data(g->presets);
  if(GPOINTER_TO_UINT(preset) >= DT_IOP_NUM_OF_STD_TEMP_PRESETS)
  {
    //we can do realistic/exaggerated.

    double min_tune[3] = {0.0};
    double no_tune[3] = {0.0};
    double max_tune[3] = {0.0};

    const dt_wb_data *wb_min = dt_wb_preset(preset->min_ft_pos);
    const dt_wb_data *wb_no = dt_wb_preset(preset->no_ft_pos);
    const dt_wb_data *wb_max = dt_wb_preset(preset->max_ft_pos);

    if(!g->blackbody_is_confusing)
    {
      //realistic
      const double neutral[3] = {
          1 / wb_no->channels[0],
          1 / wb_no->channels[1],
          1 / wb_no->channels[2],
      };
      for(int ch=0; ch<3; ch++)
      {
        min_tune[ch] = neutral[ch] * wb_min->channels[ch];
        no_tune[ch]  = neutral[ch] * wb_no->channels[ch];
        max_tune[ch] = neutral[ch] * wb_max->channels[ch];
      }

      const float maxsRGBmin_tune = max3(min_tune);
      const float maxsRGBmax_tune = max3(max_tune);

      for(int ch=0; ch<3; ch++)
      {
        min_tune[ch] = min_tune[ch] / maxsRGBmin_tune;
        no_tune[ch]  = 1.0;
        max_tune[ch] = max_tune[ch] / maxsRGBmax_tune;
      }
    }
    else
    {
      //exaggerated

      for(int ch=0; ch<3; ch++)
      {
        min_tune[ch] = 0.5;
        no_tune[ch]  = 0.9;
        max_tune[ch] = 0.5;
      }

      if(wb_min->channels[0] < wb_max->channels[0])
      {
        // from blue to red
        min_tune[0] = 0.1;
        min_tune[2] = 0.9;
        max_tune[0] = 0.9;
        max_tune[2] = 0.1;
      }
      else
      {
        //from red to blue
        min_tune[0] = 0.9;
        min_tune[2] = 0.1;
        max_tune[0] = 0.1;
        max_tune[2] = 0.9;
      }
    }

    dt_bauhaus_slider_set_stop(g->finetune, 0.0, min_tune[0], min_tune[1], min_tune[2]);
    dt_bauhaus_slider_set_stop(g->finetune, 0.5, no_tune[0],  no_tune[1],  no_tune[2]);
    dt_bauhaus_slider_set_stop(g->finetune, 1.0, max_tune[0], max_tune[1], max_tune[2]);
  }
  if(gtk_widget_get_visible(GTK_WIDGET(g->finetune)))
  {
    gtk_widget_queue_draw(GTK_WIDGET(g->finetune));
  }
}

static void _color_rgb_sliders(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  const gboolean color_rgb = g->colored_sliders &&
                             !(self->dev->image_storage.flags & DT_IMAGE_4BAYER);

  dt_bauhaus_slider_clear_stops(g->scale_r);
  dt_bauhaus_slider_clear_stops(g->scale_g);
  dt_bauhaus_slider_clear_stops(g->scale_b);
  dt_bauhaus_slider_clear_stops(g->scale_y);
  dt_bauhaus_slider_set_feedback(g->scale_r, !color_rgb);
  dt_bauhaus_slider_set_feedback(g->scale_g, !color_rgb);
  dt_bauhaus_slider_set_feedback(g->scale_b, !color_rgb);
  dt_bauhaus_slider_set_feedback(g->scale_y, !color_rgb);

  if(!color_rgb) return;

  // there are 3 ways to do colored sliders: naive (independent 0->1),
  // smart(er) (dependent 0->1) and real (coeff)

  if(FALSE)
  {
  //naive:
    dt_bauhaus_slider_set_stop(g->scale_r, 0.0, 0.0, 0.0, 0.0);
    dt_bauhaus_slider_set_stop(g->scale_r, 1.0, 1.0, 0.0, 0.0);

    dt_bauhaus_slider_set_stop(g->scale_g, 0.0, 0.0, 0.0, 0.0);
    dt_bauhaus_slider_set_stop(g->scale_g, 1.0, 0.0, 1.0, 0.0);

    dt_bauhaus_slider_set_stop(g->scale_b, 0.0, 0.0, 0.0, 0.0);
    dt_bauhaus_slider_set_stop(g->scale_b, 1.0, 0.0, 0.0, 1.0);

    dt_bauhaus_slider_set_stop(g->scale_y, 0.0, 0.0, 0.0, 0.0);
    dt_bauhaus_slider_set_stop(g->scale_y, 1.0, 0.0, 1.0, 0.0);
  }
  if(!g->blackbody_is_confusing)
  {
    //smart(er) than naive
    const float rchan =
      dt_bauhaus_slider_get(g->scale_r) / dt_bauhaus_slider_get_hard_max(g->scale_r);
    const float gchan =
      dt_bauhaus_slider_get(g->scale_g) / dt_bauhaus_slider_get_hard_max(g->scale_g);
    const float bchan =
      dt_bauhaus_slider_get(g->scale_b) / dt_bauhaus_slider_get_hard_max(g->scale_b);

    dt_bauhaus_slider_set_stop(g->scale_r, 0.0, 0.0, gchan, bchan);
    dt_bauhaus_slider_set_stop(g->scale_r, 1.0, 1.0, gchan, bchan);

    dt_bauhaus_slider_set_stop(g->scale_g, 0.0, rchan, 0.0, bchan);
    dt_bauhaus_slider_set_stop(g->scale_g, 1.0, rchan, 1.0, bchan);

    dt_bauhaus_slider_set_stop(g->scale_b, 0.0, rchan, gchan, 0.0);
    dt_bauhaus_slider_set_stop(g->scale_b, 1.0, rchan, gchan, 1.0);
  }
  else
  {
     const dt_dev_chroma_t *chr = &self->dev->chroma;
    //real (ish)
    //we consider daylight wb to be "reference white"
    const double white[3] = {
      1.0 / chr->D65coeffs[0],
      1.0 / chr->D65coeffs[1],
      1.0 / chr->D65coeffs[2],
    };

    const float rchanmul = dt_bauhaus_slider_get(g->scale_r);
    const float rchanmulmax = dt_bauhaus_slider_get_hard_max(g->scale_r);
    const float gchanmul = dt_bauhaus_slider_get(g->scale_g);
    const float gchanmulmax = dt_bauhaus_slider_get_hard_max(g->scale_g);
    const float bchanmul = dt_bauhaus_slider_get(g->scale_b);
    const float bchanmulmax = dt_bauhaus_slider_get_hard_max(g->scale_g);

    dt_bauhaus_slider_set_stop
      (g->scale_r, 0.0, white[0]*0.0, white[1]*gchanmul, white[2]*bchanmul);
    dt_bauhaus_slider_set_stop
      (g->scale_r, chr->D65coeffs[0]/rchanmulmax,
       white[0]*chr->D65coeffs[0], white[1]*gchanmul, white[2]*bchanmul);
    dt_bauhaus_slider_set_stop
      (g->scale_r, 1.0, white[0]*1.0,
       white[1]*(gchanmul/gchanmulmax), white[2]*(bchanmul/bchanmulmax));

    dt_bauhaus_slider_set_stop
      (g->scale_g, 0.0, white[0]*rchanmul, white[1]*0.0, white[2]*bchanmul);
    dt_bauhaus_slider_set_stop
      (g->scale_g, chr->D65coeffs[1]/bchanmulmax,
       white[0]*rchanmul, white[1]*chr->D65coeffs[1], white[2]*bchanmul);
    dt_bauhaus_slider_set_stop
      (g->scale_g, 1.0, white[0]*(rchanmul/rchanmulmax),
       white[1]*1.0, white[2]*(bchanmul/bchanmulmax));

    dt_bauhaus_slider_set_stop
      (g->scale_b, 0.0, white[0]*rchanmul, white[1]*gchanmul, white[2]*0.0);
    dt_bauhaus_slider_set_stop
      (g->scale_b, chr->D65coeffs[2]/bchanmulmax,
       white[0]*rchanmul, white[1]*gchanmul, white[2]*chr->D65coeffs[2]);
    dt_bauhaus_slider_set_stop
      (g->scale_b, 1.0, white[0]*(rchanmul/rchanmulmax),
       white[1]*(gchanmul/gchanmulmax), white[2]*1.0);
  }

  if(gtk_widget_get_visible(GTK_WIDGET(g->scale_r)))
  {
    gtk_widget_queue_draw(GTK_WIDGET(g->scale_r));
    gtk_widget_queue_draw(GTK_WIDGET(g->scale_g));
    gtk_widget_queue_draw(GTK_WIDGET(g->scale_b));
  }
}

static void _color_temptint_sliders(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  dt_bauhaus_slider_clear_stops(g->scale_k);
  dt_bauhaus_slider_clear_stops(g->scale_tint);
  dt_bauhaus_slider_set_feedback(g->scale_k, !g->colored_sliders);
  dt_bauhaus_slider_set_feedback(g->scale_tint, !g->colored_sliders);

  if(!g->colored_sliders) return;

  const double temp_step = (double)(DT_IOP_HIGHEST_TEMPERATURE - DT_IOP_LOWEST_TEMPERATURE) / (DT_BAUHAUS_SLIDER_MAX_STOPS - 1.0);
  const double tint_step = (double)(DT_IOP_HIGHEST_TINT - DT_IOP_LOWEST_TINT)
    / (DT_BAUHAUS_SLIDER_MAX_STOPS - 1.0);
  const gboolean blackbody_is_confusing = g->blackbody_is_confusing;

  const float cur_temp = dt_bauhaus_slider_get(g->scale_k);
  const float cur_tint = dt_bauhaus_slider_get(g->scale_tint);

  const dt_dev_chroma_t *chr = &self->dev->chroma;
  //we consider daylight wb to be "reference white"
  const double dayligh_white[3] = {
    1.0 / chr->D65coeffs[0],
    1.0 / chr->D65coeffs[1],
    1.0 / chr->D65coeffs[2],
  };

  double cur_coeffs[4] = {0.0};
  _temp2mul(self, cur_temp, 1.0, cur_coeffs);
  const double cur_white[3] = {
    1.0 / cur_coeffs[0],
    1.0 / cur_coeffs[1],
    1.0 / cur_coeffs[2],
  };

  if(blackbody_is_confusing)
  {
    // show effect of adjustment on temp/tint sliders
    for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
    {
      const float stop = i / (DT_BAUHAUS_SLIDER_MAX_STOPS - 1.0);
      const double K = DT_IOP_LOWEST_TEMPERATURE + i * temp_step;
      const double tint = DT_IOP_LOWEST_TINT + i * tint_step;

      double coeffs_K[4];
      double coeffs_tint[4];
      _temp2mul(self, K, cur_tint, coeffs_K);
      _temp2mul(self, cur_temp, tint, coeffs_tint);
      coeffs_K[0] /= coeffs_K[1];
      coeffs_K[2] /= coeffs_K[1];
      coeffs_K[3] /= coeffs_K[1];
      coeffs_K[1] = 1.0;
      coeffs_tint[0] /= coeffs_tint[1];
      coeffs_tint[2] /= coeffs_tint[1];
      coeffs_tint[3] /= coeffs_tint[1];
      coeffs_tint[1] = 1.0;

      dt_aligned_pixel_t sRGB_K =
        { dayligh_white[0]*coeffs_K[0],
          dayligh_white[1]*coeffs_K[1],
          dayligh_white[2]*coeffs_K[2] };
      dt_aligned_pixel_t sRGB_tint =
        {cur_white[0]*coeffs_tint[0],
         cur_white[1]*coeffs_tint[1],
         cur_white[2]*coeffs_tint[2]};

      const float maxsRGB_K = max3f(sRGB_K);
      const float maxsRGB_tint = max3f(sRGB_tint);

      if(maxsRGB_K > 1.f)
      {
        for(int ch=0; ch<3; ch++)
        {
          sRGB_K[ch] = fmaxf(sRGB_K[ch] / maxsRGB_K, 0.f);
        }
      }
      if(maxsRGB_tint > 1.f)
      {
        for(int ch=0; ch<3; ch++)
        {
          sRGB_tint[ch] = fmaxf(sRGB_tint[ch] / maxsRGB_tint, 0.f);
        }
      }
      dt_bauhaus_slider_set_stop(g->scale_k, stop,
                                 sRGB_K[0], sRGB_K[1], sRGB_K[2]);
      dt_bauhaus_slider_set_stop(g->scale_tint, stop,
                                 sRGB_tint[0], sRGB_tint[1], sRGB_tint[2]);
    }
  }
  else
  {
    // reflect actual black body colors for the temperature slider
    for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
    {
      const float stop = i / (DT_BAUHAUS_SLIDER_MAX_STOPS - 1.0);
      const double K = DT_IOP_LOWEST_TEMPERATURE + i * temp_step;
      const double tint = DT_IOP_LOWEST_TINT + i * tint_step;

      const cmsCIEXYZ cmsXYZ_temp = _temperature_tint_to_XYZ(K,cur_tint);
      const cmsCIEXYZ cmsXYZ_tint = _temperature_tint_to_XYZ(cur_temp, tint);
      dt_aligned_pixel_t XYZ_temp = {cmsXYZ_temp.X, cmsXYZ_temp.Y, cmsXYZ_temp.Z};
      dt_aligned_pixel_t XYZ_tint = {cmsXYZ_tint.X, cmsXYZ_tint.Y, cmsXYZ_tint.Z};
      dt_aligned_pixel_t sRGB_temp;
      dt_aligned_pixel_t sRGB_tint;

      dt_XYZ_to_Rec709_D65(XYZ_temp, sRGB_temp);
      dt_XYZ_to_Rec709_D65(XYZ_tint, sRGB_tint);

      const float maxsRGB_temp = max3f(sRGB_temp);
      const float maxsRGB_tint = max3f(sRGB_tint);

      if(maxsRGB_temp > 1.f)
      {
        for(int ch=0; ch<3; ch++)
        {
          sRGB_temp[ch] = fmaxf(sRGB_temp[ch] / maxsRGB_temp, 0.f);
        }
      }

      if(maxsRGB_tint > 1.f)
      {
        for(int ch=0; ch<3; ch++)
        {
          sRGB_tint[ch] = fmaxf(sRGB_tint[ch] / maxsRGB_tint, 0.f);
        }
      }

      dt_bauhaus_slider_set_stop(g->scale_k, stop,
                                 sRGB_temp[0], sRGB_temp[1], sRGB_temp[2]);
      dt_bauhaus_slider_set_stop(g->scale_tint, stop,
                                 sRGB_tint[0], sRGB_tint[1], sRGB_tint[2]);
    }
  }

  if(gtk_widget_get_visible(GTK_WIDGET(g->scale_k)))
  {
    gtk_widget_queue_draw(GTK_WIDGET(g->scale_k));
    gtk_widget_queue_draw(GTK_WIDGET(g->scale_tint));
  }
}

static inline const char *_preset_to_str(const int preset)
{
  switch(preset)
  {
    case DT_IOP_TEMP_AS_SHOT:   return "as shot";
    case DT_IOP_TEMP_SPOT:      return "by spot";
    case DT_IOP_TEMP_USER:      return "user defined";
    case DT_IOP_TEMP_D65:       return "camera reference";
    case DT_IOP_TEMP_D65_LATE:  return "as shot to reference";
    default:                    return "other";
  }
}

static void _update_preset(dt_iop_module_t *self, int mode)
{
  dt_iop_temperature_params_t *p = self->params;
  dt_dev_chroma_t *chr = &self->dev->chroma;

  p->preset = mode;
  chr->late_correction = self->enabled && mode == DT_IOP_TEMP_D65_LATE;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;
  dt_iop_temperature_params_t *d = self->default_params;

  g->last_daylight_temperature = 0.0f;
  g->last_bb_temperature = 0.0f;

  d->preset = p->use_cat
              ? DT_IOP_TEMP_AS_SHOT
              : (dt_is_scene_referred() ? DT_IOP_TEMP_D65_LATE
                                        : DT_IOP_TEMP_AS_SHOT);

  const gboolean true_monochrome =
    dt_image_monochrome_flags(&self->dev->image_storage) & DT_IMAGE_MONOCHROME;
  const gboolean is_raw =
    dt_image_is_matrix_correction_supported(&self->dev->image_storage);
  self->hide_enable_button = true_monochrome;
  self->default_enabled = is_raw;
  gtk_stack_set_visible_child_name(GTK_STACK(self->widget),
                                   self->hide_enable_button ? "disabled" : "enabled");

  if(self->hide_enable_button) return;

  dt_iop_color_picker_reset(self, TRUE);

  float tempK, tint;
  _mul2temp(self, p, &tempK, &tint);

  dt_bauhaus_slider_set(g->scale_k, tempK);
  dt_bauhaus_slider_set(g->scale_tint, tint);
  dt_bauhaus_slider_set(g->scale_r, p->red);
  dt_bauhaus_slider_set(g->scale_g, p->green);
  dt_bauhaus_slider_set(g->scale_b, p->blue);
  dt_bauhaus_slider_set(g->scale_y, p->various);

  dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_UNKNOWN);
  dt_bauhaus_slider_set(g->finetune, 0);

  gboolean show_finetune = FALSE;

  gboolean found = FALSE;
  const dt_dev_chroma_t *chr = &self->dev->chroma;
  // is this a "as shot" white balance?
  if(dt_dev_equal_chroma((float *)p, chr->as_shot) && (p->preset == DT_IOP_TEMP_D65_LATE))
  {
    dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_D65_LATE);
    found = TRUE;
  }

  else if(dt_dev_equal_chroma((float *)p, chr->as_shot))
  {
    dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_AS_SHOT);
    p->preset = DT_IOP_TEMP_AS_SHOT;
    found = TRUE;
  }

  // is this a "D65 white balance"?
  else if(dt_dev_equal_chroma((float *)p, chr->D65coeffs))
  {
    dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_D65);
    p->preset = DT_IOP_TEMP_D65;
    found = TRUE;
  }

  if(!found)
  {
    // look through all added presets
    for(int j = DT_IOP_NUM_OF_STD_TEMP_PRESETS; !found && (j < g->preset_cnt); j++)
    {
      // look through all variants of this preset, with different tuning
      for(int i = g->preset_num[j];
          !found
          && (i < dt_wb_presets_count())
          && !strcmp(dt_wb_preset(i)->make, self->dev->image_storage.camera_maker)
          && !strcmp(dt_wb_preset(i)->model, self->dev->image_storage.camera_model)
          && !strcmp(dt_wb_preset(i)->name, dt_wb_preset(g->preset_num[j])->name);
          i++)
      {
        const dt_wb_data *wbp = dt_wb_preset(i);
        if(dt_dev_equal_chroma((float *)p, wbp->channels))
        {
          // got exact match!
          dt_bauhaus_combobox_set(g->presets, j);
          dt_iop_temperature_preset_data_t *preset =
            dt_bauhaus_combobox_get_data(g->presets);
          if(GPOINTER_TO_UINT(preset) >= DT_IOP_NUM_OF_STD_TEMP_PRESETS)
          {
            show_finetune = preset->min_ft_pos != preset->max_ft_pos;
            if(show_finetune)
            {
              const dt_wb_data *wb_min = dt_wb_preset(preset->min_ft_pos);
              const dt_wb_data *wb_no = dt_wb_preset(preset->no_ft_pos);
              const dt_wb_data *wb_max = dt_wb_preset(preset->max_ft_pos);

              dt_bauhaus_slider_set_hard_min(g->finetune, wb_min->tuning);
              dt_bauhaus_slider_set_hard_max(g->finetune, wb_max->tuning);
              dt_bauhaus_slider_set_default(g->finetune, wb_no->tuning);
            }
          }

          dt_bauhaus_slider_set(g->finetune, wbp->tuning);
          found = TRUE;
          break;
        }
      }
    }

    if(!found)
    {
      // ok, we haven't found exact match, maybe this was interpolated?

      // look through all added presets
      for(int j = DT_IOP_NUM_OF_STD_TEMP_PRESETS; !found && (j < g->preset_cnt); j++)
      {
        // look through all variants of this preset, with different tuning
        int i = g->preset_num[j] + 1;
        while(!found
              && (i < dt_wb_presets_count())
              && !strcmp(dt_wb_preset(i)->make, self->dev->image_storage.camera_maker)
              && !strcmp(dt_wb_preset(i)->model, self->dev->image_storage.camera_model)
              && !strcmp(dt_wb_preset(i)->name, dt_wb_preset(g->preset_num[j])->name))
        {
          // let's find gaps
          if(dt_wb_preset(i - 1)->tuning + 1 == dt_wb_preset(i)->tuning)
          {
            i++;
            continue;
          }

          // we have a gap!

          // we do not know what finetuning value was set, we need to
          // bruteforce to find it
          for(int tune = dt_wb_preset(i - 1)->tuning + 1;
              !found && (tune < dt_wb_preset(i)->tuning);
              tune++)
          {
            dt_wb_data interpolated = {.tuning = tune };
            dt_wb_preset_interpolate(dt_wb_preset(i - 1),
                                     dt_wb_preset(i), &interpolated);

            if(dt_dev_equal_chroma((float *)p, interpolated.channels))
            {
              // got exact match!

              dt_bauhaus_combobox_set(g->presets, j);
              dt_iop_temperature_preset_data_t *preset =
                dt_bauhaus_combobox_get_data(g->presets);
              if(GPOINTER_TO_UINT(preset) >= DT_IOP_NUM_OF_STD_TEMP_PRESETS)
              {
                show_finetune = preset->min_ft_pos != preset->max_ft_pos;
                if(show_finetune)
                {
                  const dt_wb_data *wb_min = dt_wb_preset(preset->min_ft_pos);
                  const dt_wb_data *wb_no = dt_wb_preset(preset->no_ft_pos);
                  const dt_wb_data *wb_max = dt_wb_preset(preset->max_ft_pos);

                  dt_bauhaus_slider_set_hard_min(g->finetune, wb_min->tuning);
                  dt_bauhaus_slider_set_hard_max(g->finetune, wb_max->tuning);
                  dt_bauhaus_slider_set_default(g->finetune, wb_no->tuning);
                }
              }
              dt_bauhaus_slider_set(g->finetune, tune);
              found = TRUE;
              break;
            }
          }
          i++;
        }
      }
    }
    if(!found) //since we haven't got a match - it's user-set
    {
      dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_USER);
      p->preset = DT_IOP_TEMP_USER;
    }
  }

  if(!found || g->mod_temp != -FLT_MAX) // reset or initialize user-defined
  {
    g->mod_temp = tempK;
    g->mod_tint = tint;
    _temp_array_from_params(g->mod_coeff, p);
  }

  gtk_widget_set_visible(GTK_WIDGET(g->finetune), show_finetune);
  gtk_widget_set_visible(g->buttonbar, g->button_bar_visible);

  _update_preset(self, p->preset);

  dt_bauhaus_combobox_set(g->presets, p->preset);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_asshot),
                               p->preset == DT_IOP_TEMP_AS_SHOT);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_user),
                               p->preset == DT_IOP_TEMP_USER);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65),
                               p->preset == DT_IOP_TEMP_D65);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65_late),
                               p->preset == DT_IOP_TEMP_D65_LATE);

  _color_temptint_sliders(self);
  _color_rgb_sliders(self);
  _color_finetuning_slider(self);

  dt_print_pipe(DT_DEBUG_PIPE,
    "used preset", NULL, self, DT_DEVICE_NONE, NULL, NULL,
    "preset='%s': D65 %.3f %.3f %.3f, AS-SHOT %.3f %.3f %.3f",
    _preset_to_str(p->preset),
    chr->D65coeffs[0], chr->D65coeffs[1], chr->D65coeffs[2], chr->as_shot[0], chr->as_shot[1], chr->as_shot[2]);

  dt_gui_update_collapsible_section(&g->cs);
  dt_gui_update_collapsible_section(&g->cat_cs);
  dt_gui_update_collapsible_section(&g->advanced_cs);
  _cat_controls_update(self);

  gtk_widget_queue_draw(self->widget);
}

static gboolean _calculate_bogus_daylight_wb(dt_iop_module_t *self, double bwb[4])
{
  if(!dt_image_is_matrix_correction_supported(&self->dev->image_storage))
  {
    bwb[0] = 1.0;
    bwb[2] = 1.0;
    bwb[1] = 1.0;
    bwb[3] = 1.0;

    return FALSE;
  }

  double mul[4];
  if(dt_colorspaces_conversion_matrices_rgb
     (self->dev->image_storage.adobe_XYZ_to_CAM,
      NULL, NULL,
      self->dev->image_storage.d65_color_matrix, mul))
  {
    // normalize green:
    bwb[0] = mul[0] / mul[1];
    bwb[2] = mul[2] / mul[1];
    bwb[1] = 1.0;
    bwb[3] = mul[3] / mul[1];

    return FALSE;
  }

  return TRUE;
}

static void _prepare_matrices(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  // sRGB D65
  const double RGB_to_XYZ[3][4] = { { 0.4124564, 0.3575761, 0.1804375, 0 },
                                    { 0.2126729, 0.7151522, 0.0721750, 0 },
                                    { 0.0193339, 0.1191920, 0.9503041, 0 } };

  // sRGB D65
  const double XYZ_to_RGB[4][3] = { { 3.2404542, -1.5371385, -0.4985314 },
                                    { -0.9692660, 1.8760108, 0.0415560 },
                                    { 0.0556434, -0.2040259, 1.0572252 },
                                    { 0, 0, 0 } };

  if(!dt_image_is_raw(&self->dev->image_storage))
  {
    // let's just assume for now(TM) that if it is not raw, it is sRGB
    memcpy(g->XYZ_to_CAM, XYZ_to_RGB, sizeof(g->XYZ_to_CAM));
    memcpy(g->CAM_to_XYZ, RGB_to_XYZ, sizeof(g->CAM_to_XYZ));
    return;
  }

  if(!dt_colorspaces_conversion_matrices_xyz(self->dev->image_storage.adobe_XYZ_to_CAM,
                                              self->dev->image_storage.d65_color_matrix,
                                              g->XYZ_to_CAM, g->CAM_to_XYZ))
  {
    if(self->dev->image_storage.load_status == DT_IMAGEIO_OK)  // suppress spurious error messages
    {
      char *camera = self->dev->image_storage.camera_makermodel;
      dt_print(DT_DEBUG_ALWAYS, "[temperature] `%s' color matrix not found for image", camera);
      dt_control_log(_("`%s' color matrix not found for image"), camera);
    }
  }
}

static void _find_coeffs(dt_iop_module_t *self, double coeffs[4])
{
  const dt_image_t *img = &self->dev->image_storage;

  // the raw should provide wb coeffs:
  gboolean ok = TRUE;
  // Only check the first three values, the fourth is usually NAN for RGB
  const int num_coeffs = (img->flags & DT_IMAGE_4BAYER) ? 4 : 3;
  for(int k = 0; ok && k < num_coeffs; k++)
  {
    if(!dt_isnormal(img->wb_coeffs[k]) || img->wb_coeffs[k] == 0.0f)
      ok = FALSE;
  }
  if(ok)
  {
    for_four_channels(k)
      coeffs[k] = img->wb_coeffs[k];
    return;
  }

  double bwb[4];
  if(!_calculate_bogus_daylight_wb(self, bwb))
  {
    // found camera matrix and used it to calculate bogus daylight wb
    for_four_channels(c)
      coeffs[c] = bwb[c];
    return;
  }

  // no cam matrix??? try presets:
  for(int i = 0; i < dt_wb_presets_count(); i++)
  {
    const dt_wb_data *wbp = dt_wb_preset(i);

    if(!strcmp(wbp->make, img->camera_maker)
       && !strcmp(wbp->model, img->camera_model))
    {
      // just take the first preset we find for this camera
      for(int k = 0; k < 3; k++)
        coeffs[k] = wbp->channels[k];
      return;
    }
  }

  // did not find preset either?
  if(!_ignore_missing_wb(&(self->dev->image_storage)))
  {
    //  only display this if we have a sample, otherwise it is better to keep
    //  on screen the more important message about missing sample and the way
    //  to contribute.
    if(!img->camera_missing_sample)
      dt_control_log(_("failed to read camera white balance information from `%s'!"),
                     img->filename);
    dt_print(DT_DEBUG_ALWAYS,
             "[temperature] failed to read camera white balance information from `%s'!",
             img->filename);
  }

  // final security net: hardcoded default that fits most cams.
  coeffs[0] = 2.0;
  coeffs[1] = 1.0;
  coeffs[2] = 1.5;
  coeffs[3] = 1.0;
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_temperature_params_t *d = self->default_params;
  dt_iop_temperature_params_t *p = self->params;

  d->preset = DT_IOP_TEMP_AS_SHOT;

  // defaults for new edits: reference the raw in D65 and let color
  // calibration perform the chromatic adaptation
  d->use_cat = TRUE;
  d->cat_illuminant = DT_ILLUMINANT_FROM_WB;
  d->cat_illum_fluo = DT_ILLUMINANT_FLUO_F3;
  d->cat_illum_led = DT_ILLUMINANT_LED_B5;
  d->cat_adaptation = DT_ADAPTATION_CAT16;
  d->cat_x = 0.3127f;
  d->cat_y = 0.3290f;
  d->cat_temperature = 6504.0f;

  const gboolean scene_referred = dt_is_scene_referred();

  float *dcoeffs = (float *)d;
  for_four_channels(k)
    dcoeffs[k] = 1.0f;

  // we might be called from presets update infrastructure => there is no image
  if(!self->dev || !dt_is_valid_imgid(self->dev->image_storage.id))
    return;

  const gboolean is_raw =
    dt_image_is_matrix_correction_supported(&self->dev->image_storage);
  const gboolean true_monochrome =
    dt_image_monochrome_flags(&self->dev->image_storage) & DT_IMAGE_MONOCHROME;

  gboolean another_cat_defined = FALSE;

  if(!scene_referred)
  {
    another_cat_defined =
      dt_history_check_module_exists(self->dev->image_storage.id,
                                     "channelmixerrgb", TRUE);
  }

  const gboolean is_modern = scene_referred || another_cat_defined;

  self->default_enabled = FALSE;
  self->hide_enable_button = true_monochrome;

  // we want these data in all cases to keep them in dev->chroma
  double daylights[4] = {1.0, 1.0, 1.0, 1.0 };
  double as_shot[4] = {1.0, 1.0, 1.0, 1.0 };

  // to have at least something and definitely not crash
  _temp_array_from_params(daylights, d);

  if(!_calculate_bogus_daylight_wb(self, daylights))
  {
    // found camera matrix and used it to calculate bogus daylight wb
  }
  else
  {
    // if we didn't find anything for daylight wb, look for a wb
    // preset with appropriate name.  we're normalizing that to be D65
    for(int i = 0; i < dt_wb_presets_count(); i++)
    {
      const dt_wb_data *wbp = dt_wb_preset(i);

      if(!strcmp(wbp->make, self->dev->image_storage.camera_maker)
         && !strcmp(wbp->model, self->dev->image_storage.camera_model)
         && (!strcmp(wbp->name, "Daylight")  //??? PO
             || !strcmp(wbp->name, "DirectSunlight"))
         && wbp->tuning == 0)
      {
        for_four_channels(k)
          daylights[k] = wbp->channels[k];
        break;
      }
    }
  }

  // Store EXIF WB coeffs
  if(is_raw)
  {
    _find_coeffs(self, as_shot);
    as_shot[0] /= as_shot[1];
    as_shot[2] /= as_shot[1];
    as_shot[3] /= as_shot[1];
    as_shot[1] = 1.0;
  }

  dt_dev_chroma_t *chr = &self->dev->chroma;
  for_four_channels(k)
  {
    chr->as_shot[k] = as_shot[k];
    chr->D65coeffs[k] = daylights[k];
  }

  dt_print(DT_DEBUG_PARAMS,
    "[dt_iop_reload_defaults] scene=%s, modern=%s, CAT=%s. D65 %.3f %.3f %.3f, AS-SHOT %.3f %.3f %.3f",
    STR_YESNO(dt_is_scene_referred()),
    STR_YESNO(is_modern),
    STR_YESNO(another_cat_defined),
    daylights[0], daylights[1], daylights[2], as_shot[0], as_shot[1], as_shot[2]);

  d->preset = p->preset = DT_IOP_TEMP_AS_SHOT;

  // White balance module doesn't need to be enabled for true_monochrome raws (like
  // for leica monochrom cameras). prepare_matrices is a noop as well, as there
  // isn't a color matrix, so we can skip that as well.

  if(!true_monochrome)
  {
    if(self->gui_data)
      _prepare_matrices(self);

    /* check if file is raw / hdr */
    if(is_raw)
    {
      // raw images need wb:
      self->default_enabled = TRUE;

      // if workflow = modern, only set WB coeffs to the as-shot coefficients.
      // the hidden cat module performs the chromatic adaptation.
      double coeffs[4] = { 0 };
      if(is_modern && !_calculate_bogus_daylight_wb(self, coeffs))
      {
        for_four_channels(k)
          dcoeffs[k] = as_shot[k];
        d->preset = p->preset = DT_IOP_TEMP_AS_SHOT;
      }
      else
      {
        // do best to find starting coeffs
        _find_coeffs(self, coeffs);
        dcoeffs[0] = coeffs[0]/coeffs[1];
        dcoeffs[2] = coeffs[2]/coeffs[1];
        dcoeffs[3] = coeffs[3]/coeffs[1];
        dcoeffs[1] = 1.0f;
      }
    }
  }

  // remember daylight wb used for temperature/tint conversion,
  // assuming it corresponds to CIE daylight (D65)
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  if(g)
  {
    gtk_stack_set_visible_child_name(GTK_STACK(self->widget),
                                     self->hide_enable_button ? "disabled" : "enabled");

    dt_bauhaus_slider_set_default(g->scale_r, dcoeffs[0]);
    dt_bauhaus_slider_set_default(g->scale_g, dcoeffs[1]);
    dt_bauhaus_slider_set_default(g->scale_b, dcoeffs[2]);
    dt_bauhaus_slider_set_default(g->scale_y, dcoeffs[3]);

    for_four_channels(k)
      g->mod_coeff[k] = dcoeffs[k];

    float TempK, tint;
    _mul2temp(self, d, &TempK, &tint);

    dt_bauhaus_slider_set_default(g->scale_k, TempK);
    dt_bauhaus_slider_set_default(g->scale_tint, tint);

    dt_bauhaus_combobox_clear(g->presets);

    // old "camera". reason for change: all other RAW development
    // tools use "As Shot" or "shot"
    dt_bauhaus_combobox_add(g->presets, C_("white balance", "as shot"));

    // old "spot", reason: describes exactly what'll happen
    dt_bauhaus_combobox_add(g->presets, C_("white balance", "from image area"));
    dt_bauhaus_combobox_add(g->presets, C_("white balance", "user modified"));
    // old "camera neutral", reason: better matches intent
    dt_bauhaus_combobox_add(g->presets, C_("white balance", "camera reference"));
    dt_bauhaus_combobox_add(g->presets, C_("white balance", "as shot to reference"));

    g->preset_cnt = DT_IOP_NUM_OF_STD_TEMP_PRESETS;
    memset(g->preset_num, 0, sizeof(g->preset_num));

    _generate_preset_combo(self);

    _gui_sliders_update(self);

    dt_bauhaus_combobox_set(g->presets, p->preset);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65_late), p->preset == DT_IOP_TEMP_D65_LATE);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_asshot), p->preset == DT_IOP_TEMP_AS_SHOT);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_user), FALSE);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65), FALSE);
  }

}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 2; // basic.cl, from programs.conf
  dt_iop_temperature_global_data_t *gd = malloc(sizeof(dt_iop_temperature_global_data_t));
  self->data = gd;
  gd->kernel_whitebalance_4f = dt_opencl_create_kernel(program, "whitebalance_4f");
  gd->kernel_whitebalance_1f = dt_opencl_create_kernel(program, "whitebalance_1f");
  gd->kernel_whitebalance_1f_xtrans =
    dt_opencl_create_kernel(program, "whitebalance_1f_xtrans");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_temperature_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_whitebalance_4f);
  dt_opencl_free_kernel(gd->kernel_whitebalance_1f);
  dt_opencl_free_kernel(gd->kernel_whitebalance_1f_xtrans);
  free(self->data);
  self->data = NULL;
}

static void _temp_tint_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  dt_iop_color_picker_reset(self, TRUE);

  g->mod_temp = dt_bauhaus_slider_get(g->scale_k);
  g->mod_tint = dt_bauhaus_slider_get(g->scale_tint);

  if(p->cat_illuminant == DT_ILLUMINANT_D)
  {
    p->cat_temperature = g->mod_temp;
    g->last_daylight_temperature = g->mod_temp;
  }
  else if(p->cat_illuminant == DT_ILLUMINANT_BB)
  {
    p->cat_temperature = g->mod_temp;
    g->last_bb_temperature = g->mod_temp;
  }

  _temp2mul(self, g->mod_temp, g->mod_tint, g->mod_coeff);

  // normalize
  g->mod_coeff[0] /= g->mod_coeff[1];
  g->mod_coeff[2] /= g->mod_coeff[1];
  g->mod_coeff[3] /= g->mod_coeff[1];
  g->mod_coeff[1] = 1.0;

  // this is an internal preset update, not a request to make the CAT follow
  // the white balance again
  g->temp_tint_update = TRUE;
  dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_USER);
  g->temp_tint_update = FALSE;
  _update_preset(self, DT_IOP_TEMP_USER);
}

static void _cat_hue_chroma_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  dt_aligned_pixel_t Lch = { 0.0f };
  Lch[0] = 100.f;
  Lch[2] = deg2radf(dt_bauhaus_slider_get(g->cat_x));
  Lch[1] = dt_bauhaus_slider_get(g->cat_y);

  dt_aligned_pixel_t xyY = { 0.0f };
  dt_Lch_to_xyY(Lch, xyY);
  p->cat_illuminant = DT_ILLUMINANT_CUSTOM;
  p->cat_x = xyY[0];
  p->cat_y = xyY[1];

  _cat_xy_sliders_update(self, FALSE);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _cat_xy_sliders_update(dt_iop_module_t *self, const gboolean sync_values)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  const dt_iop_temperature_params_t *p = self->params;
  if(!g || !g->cat_x || !g->cat_y) return;

  const dt_aligned_pixel_t xyY = { p->cat_x, p->cat_y, 1.f, 0.f };
  dt_aligned_pixel_t Lch = { 0.0f };
  dt_xyY_to_Lch(xyY, Lch);

  if(sync_values)
  {
    DT_ENTER_GUI_UPDATE();
    if(Lch[1] > 0.0f)
      dt_bauhaus_slider_set(g->cat_x, rad2degf(Lch[2]));
    dt_bauhaus_slider_set(g->cat_y, Lch[1]);
    DT_LEAVE_GUI_UPDATE();
  }

  dt_bauhaus_slider_clear_stops(g->cat_x);
  dt_bauhaus_slider_clear_stops(g->cat_y);

  // both sliders show a gradient when colored sliders are enabled. Use the
  // hollow theme indicator there so the reference color stays visible, and
  // the regular filled indicator otherwise.
  const int feedback = !g->colored_sliders;
  dt_bauhaus_slider_set_feedback(g->cat_x, feedback);
  dt_bauhaus_slider_set_feedback(g->cat_y, feedback);

  if(!g->colored_sliders) return;

  // hue reference: fixed chroma, same as color calibration
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    const dt_aligned_pixel_t Lch_h = { 100.f, 50.f,
                                       deg2radf(stop * DT_IOP_WB_CAT_HUE_MAX) };
    dt_aligned_pixel_t xyY_h = { 0.0f };
    dt_Lch_to_xyY(Lch_h, xyY_h);
    dt_aligned_pixel_t RGB = { 0.0f };
    illuminant_xy_to_RGB(xyY_h[0], xyY_h[1], RGB);
    dt_bauhaus_slider_set_stop(g->cat_x, stop, RGB[0], RGB[1], RGB[2]);
  }

  // chroma reference: current hue, same as color calibration
  const dt_aligned_pixel_t xyY_hue = { p->cat_x, p->cat_y, 1.f, 0.f };
  dt_aligned_pixel_t Lch_hue = { 0.0f };
  dt_xyY_to_Lch(xyY_hue, Lch_hue);
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    const float chroma = stop * DT_IOP_WB_CAT_CHROMA_MAX / 2.0f;
    dt_aligned_pixel_t Lch_c = { 75.f, chroma, Lch_hue[2] };
    dt_aligned_pixel_t xyY_c = { 0.0f };
    dt_Lch_to_xyY(Lch_c, xyY_c);
    dt_aligned_pixel_t RGB = { 0.0f };
    illuminant_xy_to_RGB(xyY_c[0], xyY_c[1], RGB);
    dt_bauhaus_slider_set_stop(g->cat_y, stop, RGB[0], RGB[1], RGB[2]);
  }

  gtk_widget_queue_draw(g->cat_x);
  gtk_widget_queue_draw(g->cat_y);
}

static void _cat_set_source_from_wb(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  if(!g || !p->use_cat || p->cat_illuminant == DT_ILLUMINANT_FROM_WB)
    return;

  // switching the classic white balance controls back on must not erase the
  // last custom illuminant, so keep cat_x/cat_y untouched.
  p->cat_illuminant = DT_ILLUMINANT_FROM_WB;

  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_combobox_set_from_value(g->cat_illuminant, DT_ILLUMINANT_FROM_WB);
  DT_LEAVE_GUI_UPDATE();

  _cat_controls_update(self);
}

static void _update_temperature_label(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  const dt_iop_temperature_params_t *p = self->params;

  const char *label = N_("temperature");
  const char *tip = _("color temperature (in Kelvin)");

  if(p->use_cat)
  {
    if(p->cat_illuminant == DT_ILLUMINANT_D)
    {
      label = N_("temperature (daylight)");
      tip = _("color temperature of the daylight (D) illuminant (in Kelvin);\n"
              "it can differ from the camera as-shot CCT");
    }
    else if(p->cat_illuminant == DT_ILLUMINANT_BB)
    {
      label = N_("temperature (black body)");
      tip = _("color temperature of the Planckian (black body) illuminant (in Kelvin);\n"
              "it can differ from the camera as-shot CCT");
    }
    else if(p->cat_illuminant == DT_ILLUMINANT_FROM_WB)
    {
      label = N_("temperature");
      tip = _("color temperature of the white balance settings (in Kelvin)");
    }
  }

  dt_bauhaus_widget_set_label_text(g->scale_k, label);
  gtk_widget_set_tooltip_text(g->scale_k, tip);
}

static dt_iop_temperature_advanced_mode_t _advanced_mode(dt_iop_module_t *self)
{
  const dt_iop_temperature_params_t *p = self->params;
  if(p->use_cat) return DT_IOP_TEMP_ADVANCED_NONE;

  dt_iop_module_t *cc = _find_channelmixerrgb(self);
  if(cc && g_list_find(self->dev->iop, cc) && cc->enabled)
    return DT_IOP_TEMP_ADVANCED_WB_CC;

  return DT_IOP_TEMP_ADVANCED_WB_ONLY;
}

static void _advanced_set_widgets(dt_iop_module_t *self,
                                  const dt_iop_temperature_advanced_mode_t mode)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  if(!g || !g->advanced_wb_cc || !g->advanced_wb_only) return;

  g->advanced_update = TRUE;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->advanced_wb_cc),
                               mode == DT_IOP_TEMP_ADVANCED_WB_CC);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->advanced_wb_only),
                               mode == DT_IOP_TEMP_ADVANCED_WB_ONLY);
  g->advanced_update = FALSE;
}

static void _advanced_controls_update(dt_iop_module_t *self)
{
  if(!self->gui_data) return;
  _advanced_set_widgets(self, _advanced_mode(self));
}

static void _cat_controls_update(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  const dt_iop_temperature_params_t *p = self->params;

  if(!g) return;

  _advanced_controls_update(self);

  _update_temperature_label(self);

  if(p->cat_illuminant == DT_ILLUMINANT_D && p->cat_temperature > 0.0f)
    g->last_daylight_temperature = p->cat_temperature;
  else if(p->cat_illuminant == DT_ILLUMINANT_BB && p->cat_temperature > 0.0f)
    g->last_bb_temperature = p->cat_temperature;

  const gboolean legacy = !p->use_cat;

  // the bulb+body mode only makes sense for the legacy late-correction path
  gtk_widget_set_visible(GTK_WIDGET(g->btn_d65_late), legacy);
  if(dt_bauhaus_combobox_length(g->presets) > DT_IOP_TEMP_D65_LATE)
    dt_bauhaus_combobox_entry_set_sensitive(g->presets, DT_IOP_TEMP_D65_LATE, legacy);

  // legacy edits keep the old pipeline and must not show a CAT section here
  if(!p->use_cat)
  {
    gtk_widget_set_visible(GTK_WIDGET(g->cat_cs.expander), FALSE);
    gtk_widget_set_visible(g->temp_label_box, TRUE);
    gtk_widget_set_visible(g->scale_k, TRUE);
    gtk_widget_set_visible(g->scale_tint, TRUE);
    if(g->cs.expander)
      gtk_widget_set_visible(GTK_WIDGET(g->cs.expander), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->buttonbar), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->presets), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->finetune), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->colorpicker), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_k), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_tint), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_r), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_g), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_b), TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(g->scale_y), TRUE);
    return;
  }

  gtk_widget_set_visible(GTK_WIDGET(g->cat_cs.expander), TRUE);

  const gboolean from_wb = p->cat_illuminant == DT_ILLUMINANT_FROM_WB;
  const gboolean temp_only = p->cat_illuminant == DT_ILLUMINANT_D
                             || p->cat_illuminant == DT_ILLUMINANT_BB;
  const gboolean show_temp = from_wb || temp_only;

  gtk_widget_set_visible(g->temp_label_box, show_temp);
  gtk_widget_set_visible(g->scale_k, show_temp);
  gtk_widget_set_visible(g->scale_tint, from_wb);
  if(g->cs.expander)
    gtk_widget_set_visible(GTK_WIDGET(g->cs.expander), from_wb);

  // preset selectors stay available so selecting a classic white balance
  // setting can switch the CAT source back to from-white-balance. Manual
  // sliders still require the CAT to follow the white balance settings.
  gtk_widget_set_sensitive(GTK_WIDGET(g->buttonbar), TRUE);
  gtk_widget_set_sensitive(GTK_WIDGET(g->presets), TRUE);
  gtk_widget_set_sensitive(GTK_WIDGET(g->finetune), TRUE);
  gtk_widget_set_sensitive(GTK_WIDGET(g->colorpicker), TRUE);
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_r), from_wb);
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_g), from_wb);
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_b), from_wb);
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_y), from_wb);
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_k), from_wb || temp_only);
  // D/BB are defined by their CCT only, like in color calibration; tint is
  // a white-balance control and would not affect their chromaticity.
  gtk_widget_set_sensitive(GTK_WIDGET(g->scale_tint), from_wb);

  // CAT controls: the source and adaptation are always available
  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_illuminant), TRUE);
  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_adaptation), TRUE);

  const gboolean is_f = p->cat_illuminant == DT_ILLUMINANT_F;
  const gboolean is_led = p->cat_illuminant == DT_ILLUMINANT_LED;
  const gboolean is_custom = p->cat_illuminant == DT_ILLUMINANT_CUSTOM;

  gtk_widget_set_visible(GTK_WIDGET(g->cat_illum_fluo), is_f);
  gtk_widget_set_visible(GTK_WIDGET(g->cat_illum_led), is_led);
  gtk_widget_set_visible(GTK_WIDGET(g->cat_x), is_custom);
  gtk_widget_set_visible(GTK_WIDGET(g->cat_y), is_custom);

  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_illum_fluo), is_f);
  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_illum_led), is_led);
  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_x), is_custom);
  gtk_widget_set_sensitive(GTK_WIDGET(g->cat_y), is_custom);

  _cat_xy_sliders_update(self, TRUE);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  if(w && (w == g->cat_illuminant
           || w == g->cat_illum_fluo || w == g->cat_illum_led
           || w == g->cat_adaptation || w == g->cat_x || w == g->cat_y))
  {
    if(w == g->cat_illuminant)
    {
      const dt_illuminant_t new_source = p->cat_illuminant;
      const dt_illuminant_t prev_source =
        previous ? *(const dt_illuminant_t *)previous : new_source;

      if(new_source == DT_ILLUMINANT_D || new_source == DT_ILLUMINANT_BB)
      {
        float cct = (new_source == DT_ILLUMINANT_D)
                    ? g->last_daylight_temperature
                    : g->last_bb_temperature;

        if(cct <= 0.0f && prev_source != new_source)
        {
          float x, y;
          if(_source_illuminant_xy(&self->dev->image_storage, p, prev_source,
                                   &x, &y))
            cct = _cct_from_xy(x, y);
        }

        if(cct <= 0.0f)
          cct = p->cat_temperature > 0.0f
                ? p->cat_temperature
                : dt_bauhaus_slider_get(g->scale_k);

        if(cct > 0.0f)
        {
          p->cat_temperature = cct;
          g->mod_temp = cct;
          if(new_source == DT_ILLUMINANT_D)
            g->last_daylight_temperature = cct;
          else
            g->last_bb_temperature = cct;

          DT_ENTER_GUI_UPDATE();
          dt_bauhaus_slider_set(g->scale_k, cct);
          DT_LEAVE_GUI_UPDATE();
        }
      }
      else if(new_source == DT_ILLUMINANT_FROM_WB)
      {
        // restore the temperature of the classic white balance controls
        float tempK, tint;
        _mul2temp(self, p, &tempK, &tint);
        g->mod_temp = tempK;

        DT_ENTER_GUI_UPDATE();
        dt_bauhaus_slider_set(g->scale_k, tempK);
        DT_LEAVE_GUI_UPDATE();
      }
    }

    _cat_controls_update(self);
    return;
  }

  _temp_array_from_params(g->mod_coeff, p);

  _mul2temp(self, p, &g->mod_temp, &g->mod_tint);

  // changing another parameter must not redirect the CAT source
  g->temp_tint_update = TRUE;
  dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_USER);
  g->temp_tint_update = FALSE;
  _update_preset(self, DT_IOP_TEMP_USER);
}

static gboolean _btn_toggled(GtkWidget *togglebutton,
                            GdkEventButton *event,
                            dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE(TRUE);

  dt_iop_temperature_gui_data_t *g = (dt_iop_temperature_gui_data_t*)self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  const int preset = togglebutton == g->btn_asshot ? DT_IOP_TEMP_AS_SHOT :
                     togglebutton == g->btn_d65 ? DT_IOP_TEMP_D65 :
                     togglebutton == g->btn_d65_late ? DT_IOP_TEMP_D65_LATE :
                     togglebutton == g->btn_user ? DT_IOP_TEMP_USER : 0;

  // a classic white balance button must be able to bring the CAT source back
  // to white balance, even when its preset is already selected
  const gboolean needs_cat_sync =
    p->use_cat && p->cat_illuminant != DT_ILLUMINANT_FROM_WB;

  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(togglebutton)))
  {
    if(needs_cat_sync || dt_bauhaus_combobox_get(g->presets) != preset)
      dt_bauhaus_combobox_set(g->presets, preset);
  }
  else if(dt_bauhaus_combobox_get(g->presets) == preset)
  {
    if(needs_cat_sync)
      dt_bauhaus_combobox_set(g->presets, preset);
    else
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(togglebutton), TRUE);
  }

  _update_preset(self, dt_bauhaus_combobox_get(g->presets));

  const dt_dev_chroma_t *chr = &self->dev->chroma;
  dt_print_pipe(DT_DEBUG_PIPE,
    "toggled preset", NULL, self, DT_DEVICE_NONE, NULL, NULL,
    "preset='%s': D65 %.3f %.3f %.3f, AS-SHOT %.3f %.3f %.3f",
    _preset_to_str(preset),
    chr->D65coeffs[0], chr->D65coeffs[1], chr->D65coeffs[2], chr->as_shot[0], chr->as_shot[1], chr->as_shot[2]);
  return TRUE;
}

static void _preset_tune_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  const int pos = dt_bauhaus_combobox_get(g->presets);
  const int tune = dt_bauhaus_slider_get(g->finetune);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_asshot),
                               pos == DT_IOP_TEMP_AS_SHOT);
  if(pos != DT_IOP_TEMP_SPOT)
    dt_iop_color_picker_reset(self, TRUE);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_user),
                               pos == DT_IOP_TEMP_USER);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65),
                               pos == DT_IOP_TEMP_D65);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65_late),
                               pos == DT_IOP_TEMP_D65_LATE);

  gboolean show_finetune = FALSE;
  dt_dev_chroma_t *chr = &self->dev->chroma;

  _update_preset(self, pos);

  if(pos == DT_IOP_TEMP_UNKNOWN) // just un-setting.
    return;

  // any classic white balance setting chosen by the user takes precedence
  // over a fixed CAT source, while keeping the last custom illuminant in
  // params. Internal temp/tint or parameter updates must not redirect it.
  if(!g->temp_tint_update)
    _cat_set_source_from_wb(self);

  switch(pos)
  {
    case DT_IOP_TEMP_AS_SHOT: // as shot wb
      _temp_params_from_array(p, chr->as_shot);
      break;
    case DT_IOP_TEMP_SPOT: // from image area wb, expose callback will set p->rgbg2.
      if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->colorpicker)))
      {
        gboolean ret_val;
        g_signal_emit_by_name(G_OBJECT(g->colorpicker), "button-press-event",
                              NULL, &ret_val);
      }
      break;
    case DT_IOP_TEMP_USER: // directly changing one of the coeff
                           // sliders also changes the mod_coeff so it
                           // can be read here
      _temp_params_from_array(p, g->mod_coeff);
      break;
    case DT_IOP_TEMP_D65: // camera reference d65
      _temp_params_from_array(p, chr->D65coeffs);
      break;
    case DT_IOP_TEMP_D65_LATE: // as shot wb just for now
        _temp_params_from_array(p, chr->as_shot);
      break;
    default: // camera WB presets
    {
      gboolean found = FALSE;
      dt_iop_temperature_preset_data_t *preset = dt_bauhaus_combobox_get_data(g->presets);
      if(preset)
      {
        // look through all variants of this preset, with different tuning
        for(int i = preset->min_ft_pos;
            (i < (preset->max_ft_pos + 1)) // we can limit search spread
              // thanks to knowing where to
              // look!
              && !strcmp(dt_wb_preset(i)->make, self->dev->image_storage.camera_maker)
              && !strcmp(dt_wb_preset(i)->model, self->dev->image_storage.camera_model)
              && !strcmp(dt_wb_preset(i)->name, dt_wb_preset(preset->no_ft_pos)->name);
            i++)
        {
          if(dt_wb_preset(i)->tuning == tune)
          {
            // got exact match!
            _temp_params_from_array(p, dt_wb_preset(i)->channels);
            found = TRUE;
            break;
          }
        }
      }

      if(!found)
      {
        // ok, we haven't found exact match, need to interpolate

        // let's find 2 most closest tunings with needed_tuning in-between
        int min_id = INT_MIN, max_id = INT_MIN;

        // look through all variants of this preset, with different
        // tuning, starting from second entry (if any)
        int i = preset->min_ft_pos + 1;
        while((i < preset->max_ft_pos+1)
              && !strcmp(dt_wb_preset(i)->make, self->dev->image_storage.camera_maker)
              && !strcmp(dt_wb_preset(i)->model, self->dev->image_storage.camera_model)
              && !strcmp(dt_wb_preset(i)->name, dt_wb_preset(preset->no_ft_pos)->name))
        {
          if(dt_wb_preset(i - 1)->tuning < tune
             && dt_wb_preset(i)->tuning > tune)
          {
            min_id = i - 1;
            max_id = i;
            break;
          }

          i++;
        }

        // have we found enough good data?
        if(min_id == INT_MIN || max_id == INT_MIN || min_id == max_id) break; // hysteresis

        found = TRUE;
        dt_wb_data interpolated = {.tuning = tune };
        dt_wb_preset_interpolate(dt_wb_preset(min_id),
                                 dt_wb_preset(max_id),
                                 &interpolated);
        _temp_params_from_array(p, interpolated.channels);
      }

      show_finetune = preset->min_ft_pos != preset->max_ft_pos;
      if(show_finetune)
      {
        const dt_wb_data *wb_min = dt_wb_preset(preset->min_ft_pos);
        const dt_wb_data *wb_no = dt_wb_preset(preset->no_ft_pos);
        const dt_wb_data *wb_max = dt_wb_preset(preset->max_ft_pos);

        DT_ENTER_GUI_UPDATE();
        dt_bauhaus_slider_set_hard_min(g->finetune, wb_min->tuning);
        dt_bauhaus_slider_set_hard_max(g->finetune, wb_max->tuning);
        dt_bauhaus_slider_set_default(g->finetune, wb_no->tuning);
        DT_LEAVE_GUI_UPDATE();
      }
    }
    break;
  }

  gtk_widget_set_visible(GTK_WIDGET(g->finetune), show_finetune);

  if(self->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  float TempK, tint;

  if(pos == DT_IOP_TEMP_USER)
  {
    TempK = g->mod_temp;
    tint = g->mod_tint;
  }
  else
  {
    _mul2temp(self, p, &TempK, &tint);
  }

  DT_ENTER_GUI_UPDATE();
  float *pcoeffs = (float *)p;
  dt_bauhaus_slider_set(g->scale_k, TempK);
  dt_bauhaus_slider_set(g->scale_tint, tint);
  dt_bauhaus_slider_set(g->scale_r, pcoeffs[0]);
  dt_bauhaus_slider_set(g->scale_g, pcoeffs[1]);
  dt_bauhaus_slider_set(g->scale_b, pcoeffs[2]);
  dt_bauhaus_slider_set(g->scale_y, pcoeffs[3]);
  DT_LEAVE_GUI_UPDATE();

  _color_temptint_sliders(self);
  _color_rgb_sliders(self);
  _color_finetuning_slider(self);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;
  float *pcoeffs = (float *)p;

  // capture gui color picked event.
  if(self->picked_color_max[0] < self->picked_color_min[0]) return;

  // a picked white balance also restores the CAT source to white balance
  _cat_set_source_from_wb(self);

  const float *grayrgb = self->picked_color;

  // normalize green:
  const float gnormal = grayrgb[1] > 0.001f ? 1.0f / grayrgb[1] : 1.0f;
  for_four_channels(c)
    pcoeffs[c]  = fmaxf(0.0f,fminf(8.0f,(grayrgb[c] > 0.001f ? 1.0f / grayrgb[c] : 1.0f) / gnormal));
  pcoeffs[1] = 1.0f;

  dt_bauhaus_combobox_set(g->presets, DT_IOP_TEMP_SPOT);
  _update_preset(self, DT_IOP_TEMP_SPOT);
}


static void _gui_sliders_update(dt_iop_module_t *self)
{
  const dt_image_t *img = &self->dev->image_storage;
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  if(FILTERS_ARE_CYGM(img->buf_dsc.filters))
  {
    dt_bauhaus_widget_set_label(g->scale_r, NULL, N_("green"));
    gtk_widget_set_tooltip_text(g->scale_r, _("green channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_g, NULL, N_("magenta"));
    gtk_widget_set_tooltip_text(g->scale_g, _("magenta channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_b, NULL, N_("cyan"));
    gtk_widget_set_tooltip_text(g->scale_b, _("cyan channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_y, NULL, N_("yellow"));
    gtk_widget_set_tooltip_text(g->scale_y, _("yellow channel coefficient"));

    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_b, 0);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_y, 1);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_g, 2);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_r, 3);
  }
  else
  {
    dt_bauhaus_widget_set_label(g->scale_r, NULL, N_("red"));
    gtk_widget_set_tooltip_text(g->scale_r, _("red channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_g, NULL, N_("green"));
    gtk_widget_set_tooltip_text(g->scale_g, _("green channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_b, NULL, N_("blue"));
    gtk_widget_set_tooltip_text(g->scale_b, _("blue channel coefficient"));
    dt_bauhaus_widget_set_label(g->scale_y, NULL, N_("emerald"));
    gtk_widget_set_tooltip_text(g->scale_y, _("emerald channel coefficient"));

    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_r, 0);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_g, 1);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_b, 2);
    gtk_box_reorder_child(GTK_BOX(g->cs.container), g->scale_y, 3);
  }

  gtk_widget_set_visible(GTK_WIDGET(g->scale_y), (img->flags & DT_IMAGE_4BAYER));
}

static gboolean temp_label_click(GtkWidget *label,
                                 GdkEventButton *event,
                                 dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  gchar *old_config = dt_conf_get_string("plugins/darkroom/temperature/colored_sliders");

  if(!g_strcmp0(old_config, "no color"))
  {
    dt_conf_set_string("plugins/darkroom/temperature/colored_sliders", "illuminant color");
    g->colored_sliders = TRUE;
    g->blackbody_is_confusing = FALSE;
  }
  else if(!g_strcmp0(old_config, "illuminant color"))
  {
    dt_conf_set_string("plugins/darkroom/temperature/colored_sliders", "effect emulation");
    g->colored_sliders = TRUE;
    g->blackbody_is_confusing = TRUE;
  }
  else
  {
    dt_conf_set_string("plugins/darkroom/temperature/colored_sliders", "no color");
    g->colored_sliders = FALSE;
    g->blackbody_is_confusing = FALSE;
  }

  g_free(old_config);

  _color_temptint_sliders(self);
  _color_rgb_sliders(self);
  _color_finetuning_slider(self);
  _cat_xy_sliders_update(self, FALSE);
  return TRUE;
}

static void _preference_changed(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;

  const char *config = dt_conf_get_string_const("plugins/darkroom/temperature/colored_sliders");
  g->colored_sliders = g_strcmp0(config, "no color") ? TRUE : FALSE;
  g->blackbody_is_confusing = g->colored_sliders && (g_strcmp0(config, "illuminant color") ? TRUE : FALSE);

  g->button_bar_visible = dt_conf_get_bool("plugins/darkroom/temperature/button_bar");
  gtk_widget_set_visible(g->buttonbar, g->button_bar_visible);

  _color_temptint_sliders(self);
  _color_rgb_sliders(self);
  _color_finetuning_slider(self);
  _cat_xy_sliders_update(self, FALSE);
  _cat_controls_update(self);
}

static void _wb_set_custom_illuminant(dt_develop_t *dev, float x, float y)
{
  if(!dev) return;

  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(!dt_iop_module_is(mod, "temperature")) continue;

    dt_iop_temperature_params_t *p = mod->params;
    p->use_cat = TRUE;
    p->cat_illuminant = DT_ILLUMINANT_CUSTOM;
    p->cat_x = x;
    p->cat_y = y;

    dt_iop_gui_update(mod);
    dt_dev_add_history_item(dev, mod, TRUE);
    break;
  }
}

static dt_iop_module_t *_find_channelmixerrgb(dt_iop_module_t *self)
{
  for(GList *modules = self->dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod, "channelmixerrgb")) return mod;
  }
  for(GList *modules = self->dev->alliop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod, "channelmixerrgb")) return mod;
  }
  return NULL;
}

static void _apply_advanced_mode(dt_iop_module_t *self,
                                  const dt_iop_temperature_advanced_mode_t mode,
                                  GtkWidget *target)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *p = self->params;

  if(!g) return;

  const gboolean legacy = mode != DT_IOP_TEMP_ADVANCED_NONE;
  dt_iop_module_t *cc = _find_channelmixerrgb(self);
  const gboolean cc_reset = mode == DT_IOP_TEMP_ADVANCED_WB_CC && cc != NULL;
  const gboolean cc_disable =
    mode != DT_IOP_TEMP_ADVANCED_WB_CC && cc
    && g_list_find(self->dev->iop, cc) && cc->enabled;

  DT_ENTER_GUI_UPDATE();
  if(legacy)
  {
    p->use_cat = FALSE;
    p->preset = DT_IOP_TEMP_D65_LATE;
    _temp_params_from_array(p, self->dev->chroma.as_shot);
    _update_preset(self, DT_IOP_TEMP_D65_LATE);
  }
  else
  {
    p->use_cat = TRUE;
    p->preset = DT_IOP_TEMP_AS_SHOT;
    p->cat_illuminant = DT_ILLUMINANT_FROM_WB;
    _update_preset(self, DT_IOP_TEMP_AS_SHOT);
  }
  DT_LEAVE_GUI_UPDATE();

  dt_iop_gui_update(self);

  // commit white balance first so the chroma state reflects the legacy
  // pipeline before color calibration computes its own legacy defaults
  dt_dev_add_history_item_target(self->dev, self, TRUE, target);

  if(cc_reset)
  {
    // an unused color calibration lives in alliop until it is first enabled.
    // move it into the active pipeline and build its GUI before resetting it.
    if(!g_list_find(self->dev->iop, cc))
    {
      self->dev->alliop = g_list_remove(self->dev->alliop, cc);
      self->dev->iop = g_list_insert_sorted(self->dev->iop, cc, dt_sort_iop_by_order);
      dt_ioppr_insert_module_instance(self->dev, cc);
      dt_ioppr_resync_modules_order(self->dev);

      if(!dt_iop_is_hidden(cc))
      {
        // the module may come from alliop after its GUI was destroyed: its
        // widget/gui_data pointers are then stale while still non-NULL.
        // dt_iop_gui_init() reinitializes gui_lock and rebuilds the GUI.
        dt_iop_gui_init(cc);
        dt_iop_gui_set_expander(cc);
        dt_iop_gui_set_expanded(cc, TRUE, FALSE);
        dt_iop_gui_update_blending(cc);
      }
    }

    dt_iop_reload_defaults(cc);
    dt_iop_gui_reset(cc);
    cc->enabled = TRUE;
    dt_iop_gui_update(cc);
    dt_dev_add_history_item_target(self->dev, cc, TRUE, target);
  }
  else if(cc_disable)
  {
    cc->enabled = FALSE;
    dt_iop_gui_update(cc);
    dt_dev_add_history_item_target(self->dev, cc, FALSE, target);
  }

  _advanced_controls_update(self);
}

static void _advanced_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  if(!g || g->advanced_update) return;

  dt_iop_temperature_advanced_mode_t mode;
  if(widget == g->advanced_wb_cc)
  {
    if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
      mode = DT_IOP_TEMP_ADVANCED_WB_CC;
    else if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->advanced_wb_only)))
      mode = DT_IOP_TEMP_ADVANCED_WB_ONLY;
    else
      mode = DT_IOP_TEMP_ADVANCED_NONE;
  }
  else
  {
    if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
      mode = DT_IOP_TEMP_ADVANCED_WB_ONLY;
    else if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->advanced_wb_cc)))
      mode = DT_IOP_TEMP_ADVANCED_WB_CC;
    else
      mode = DT_IOP_TEMP_ADVANCED_NONE;
  }

  _advanced_set_widgets(self, mode);

  if(mode == _advanced_mode(self)) return;

  _apply_advanced_mode(self, mode, widget);
}

static void _history_changed_callback(gpointer instance, dt_iop_module_t *self)
{
  if(!self->gui_data) return;
  _advanced_controls_update(self);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = IOP_GUI_ALLOC(temperature);
  g->temp_tint_update = FALSE;
  g->advanced_update = FALSE;
  g->last_daylight_temperature = 0.0f;
  g->last_bb_temperature = 0.0f;

  const char *config =
    dt_conf_get_string_const("plugins/darkroom/temperature/colored_sliders");
  g->colored_sliders = g_strcmp0(config, "no color"); // true if config != "no color"
  // true if config != "illuminant color"
  g->blackbody_is_confusing = g->colored_sliders && g_strcmp0(config, "illuminant color");

  const gboolean feedback = g->colored_sliders ? FALSE : TRUE;
  g->button_bar_visible = dt_conf_get_bool("plugins/darkroom/temperature/button_bar");

  g->btn_asshot = dt_iop_togglebutton_new(self, N_("settings"), N_("as shot"), NULL,
                                          G_CALLBACK(_btn_toggled), FALSE, 0, 0,
                                          dtgtk_cairo_paint_camera, NULL);
  gtk_widget_set_tooltip_text(g->btn_asshot, _("set white balance to as shot"));

  // create color picker to be able to send its signal when spot
  // selected, this module may expect data in RAW or RGB, setting the
  // color picker CST to IOP_CS_NONE will make the color picker to
  // depend on the number of color channels of the pixels. It is done
  // like this as we may not know the actual kind of data we are using
  // in the GUI (it is part of the pipeline).
  g->colorpicker = dt_color_picker_new_with_cst(self, DT_COLOR_PICKER_AREA,
                                                NULL, IOP_CS_NONE);
  dt_action_define_iop(self, N_("settings"), N_("from image area"),
                       g->colorpicker, &dt_action_def_toggle);
  dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(g->colorpicker),
                               dtgtk_cairo_paint_colorpicker, 0, NULL);
  dt_gui_add_class(g->colorpicker, "dt_transparent_background");
  gtk_widget_set_tooltip_text(g->colorpicker, _("set white balance to detected from area"));

  g->btn_user = dt_iop_togglebutton_new(self, N_("settings"), N_("user modified"), NULL,
                                        G_CALLBACK(_btn_toggled), FALSE, 0, 0,
                                        dtgtk_cairo_paint_masks_drawn, NULL);
  gtk_widget_set_tooltip_text(g->btn_user, _("set white balance to user modified"));


  g->btn_d65 = dt_iop_togglebutton_new(self, N_("settings"), N_("camera reference"), NULL,
                                       G_CALLBACK(_btn_toggled), FALSE, 0, 0,
                                       dtgtk_cairo_paint_bulb, NULL);
  gtk_widget_set_tooltip_text
    (g->btn_d65,
     _("set white balance to camera reference point\nin most cases it should be D65"));

  g->btn_d65_late = dt_iop_togglebutton_new(self,
                                            N_("settings"),
                                            N_("as shot to reference"), NULL,
                                            G_CALLBACK(_btn_toggled), FALSE, 0, 0,
                                            dtgtk_cairo_paint_bulb_mod, NULL);
  gtk_widget_set_tooltip_text
    (g->btn_d65_late,
     _("set white balance to as shot and later correct to camera reference point,\nin most cases it should be D65"));

  // put buttons at top. fill later.
  g->buttonbar = dt_gui_hbox(dt_gui_expand(g->btn_asshot),
                             dt_gui_expand(g->colorpicker),
                             dt_gui_expand(g->btn_user),
                             dt_gui_expand(g->btn_d65),
                             dt_gui_expand(g->btn_d65_late));
  dt_gui_add_class(g->buttonbar, "dt_iop_toggle");

  g->presets = dt_bauhaus_combobox_new(self);
  // relabel to settings to remove confusion between module presets
  // and white balance settings
  dt_bauhaus_widget_set_label(g->presets, N_("settings"), N_("settings"));
  gtk_widget_set_tooltip_text(g->presets, _("choose white balance setting"));

  g->finetune = dt_bauhaus_slider_new_with_range_and_feedback
    (self, -9.0, 9.0, 0, 0.0, 0, feedback);
  dt_bauhaus_widget_set_label(g->finetune, NULL, N_("finetune"));
  dt_bauhaus_slider_set_format(g->finetune, " mired");
  gtk_widget_set_tooltip_text(g->finetune, _("fine tune camera's white balance setting"));

  g->mod_temp = -FLT_MAX;
  for_four_channels(k)
    g->mod_coeff[k] = 1.0;

  g->temp_label_box = gtk_event_box_new();
  g->temp_label = dt_ui_section_label_new(C_("section", "scene illuminant temp"));
  gtk_widget_set_tooltip_text(g->temp_label, _("click to cycle color mode on sliders"));
  gtk_container_add(GTK_CONTAINER(g->temp_label_box), g->temp_label);

  g_signal_connect(G_OBJECT(g->temp_label_box), "button-release-event",
                   G_CALLBACK(temp_label_click), self);

  //Match UI order: temp first, then tint (like every other app ever)
  g->scale_k = dt_bauhaus_slider_new_with_range_and_feedback
    (self, DT_IOP_LOWEST_TEMPERATURE, DT_IOP_HIGHEST_TEMPERATURE,
     0, 5000.0, 0, feedback);
  dt_bauhaus_slider_set_format(g->scale_k, " K");
  dt_bauhaus_widget_set_label(g->scale_k, NULL, N_("temperature"));
  gtk_widget_set_tooltip_text(g->scale_k, _("color temperature (in Kelvin)"));

  g->scale_tint = dt_bauhaus_slider_new_with_range_and_feedback
    (self, DT_IOP_LOWEST_TINT, DT_IOP_HIGHEST_TINT,
                                                                0, 1.0, 3, feedback);
  dt_bauhaus_widget_set_label(g->scale_tint, NULL, N_("tint"));
  gtk_widget_set_tooltip_text
    (g->scale_tint,
     _("color tint of the image, from magenta (value < 1) to green (value > 1)"));

  GtkWidget *box_enabled = dt_gui_vbox(g->buttonbar,
                                       g->presets,
                                       g->finetune,
                                       g->temp_label_box,
                                       g->scale_k,
                                       g->scale_tint);

  dt_gui_new_collapsible_section
    (&g->advanced_cs,
     "plugins/darkroom/temperature/expand_advanced",
     _("advanced"),
     GTK_BOX(box_enabled),
     DT_ACTION(self));

  self->widget = GTK_WIDGET(g->advanced_cs.container);

  g->advanced_wb_cc =
    gtk_check_button_new_with_label(_("white balance + color calibration"));
  gtk_widget_set_tooltip_text
    (g->advanced_wb_cc,
     _("when enabled, white balance uses 'as shot to reference';\n"
       "the first color calibration instance is reset to its legacy defaults and enabled"));
  gtk_widget_set_margin_top(g->advanced_wb_cc, 10);

  g->advanced_wb_only =
    gtk_check_button_new_with_label(_("legacy : white balance only"));
  gtk_widget_set_tooltip_text
    (g->advanced_wb_only,
     _("when enabled, white balance uses 'as shot to reference';\n"
       "the first color calibration instance is disabled"));

  dt_gui_box_add(g->advanced_cs.container,
                 g->advanced_wb_cc, g->advanced_wb_only);

  g_signal_connect(G_OBJECT(g->advanced_wb_cc), "toggled",
                   G_CALLBACK(_advanced_callback), self);
  g_signal_connect(G_OBJECT(g->advanced_wb_only), "toggled",
                   G_CALLBACK(_advanced_callback), self);

  dt_gui_new_collapsible_section
    (&g->cs,
     "plugins/darkroom/temperature/expand_coefficients",
     _("channel coefficients"),
     GTK_BOX(box_enabled),
     DT_ACTION(self));

  self->widget = GTK_WIDGET(g->cs.container);

  g->scale_r = dt_bauhaus_slider_from_params(self, N_("red"));
  g->scale_g = dt_bauhaus_slider_from_params(self, N_("green"));
  g->scale_b = dt_bauhaus_slider_from_params(self, N_("blue"));
  g->scale_y = dt_bauhaus_slider_from_params(self, N_("various"));
  dt_bauhaus_slider_set_digits(g->scale_r, 3);
  dt_bauhaus_slider_set_digits(g->scale_g, 3);
  dt_bauhaus_slider_set_digits(g->scale_b, 3);
  dt_bauhaus_slider_set_digits(g->scale_y, 3);

  gtk_widget_set_no_show_all(g->scale_y, TRUE);

  // chromatic adaptation section. The widgets are created while self->widget
  // points at the collapsible container so that the *_from_params helpers add
  // them to the right place.
  dt_gui_new_collapsible_section
    (&g->cat_cs,
     "plugins/darkroom/temperature/expand_cat",
     _("chromatic adaptation"),
     GTK_BOX(box_enabled),
     DT_ACTION(self));

  self->widget = GTK_WIDGET(g->cat_cs.container);

  g->cat_illuminant = dt_bauhaus_combobox_from_params(self, "cat_illuminant");
  gtk_widget_set_tooltip_text
    (g->cat_illuminant,
     _("choose where the scene illuminant comes from.\n"
       "from white balance derives it from the settings above;\n"
       "a fixed illuminant disables the conflicting white balance controls."));

  // keep only the meaningful sources: from WB, D, A, E, F, LED, BB and custom
  const dt_illuminant_t removed_sources[] = {
    DT_ILLUMINANT_PIPE,
    DT_ILLUMINANT_CAMERA,
    DT_ILLUMINANT_DETECT_EDGES,
    DT_ILLUMINANT_DETECT_SURFACES,
    DT_ILLUMINANT_LAST
  };
  for(size_t i = 0; i < sizeof(removed_sources) / sizeof(removed_sources[0]); i++)
  {
    const int pos = dt_bauhaus_combobox_get_from_value(g->cat_illuminant,
                                                       removed_sources[i]);
    if(pos >= 0) dt_bauhaus_combobox_remove_at(g->cat_illuminant, pos);
  }

  // keep "from white balance settings" as the first source
  const int from_wb_pos =
    dt_bauhaus_combobox_get_from_value(g->cat_illuminant, DT_ILLUMINANT_FROM_WB);
  if(from_wb_pos > 0)
  {
    char *from_wb_text =
      g_strdup(dt_bauhaus_combobox_get_entry(g->cat_illuminant, from_wb_pos));
    dt_bauhaus_combobox_remove_at(g->cat_illuminant, from_wb_pos);
    dt_bauhaus_combobox_insert_full(g->cat_illuminant, from_wb_text,
                                    DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT,
                                    GINT_TO_POINTER(DT_ILLUMINANT_FROM_WB),
                                    NULL, 0);
    g_free(from_wb_text);
    dt_bauhaus_combobox_set_from_value
      (g->cat_illuminant, ((dt_iop_temperature_params_t *)self->params)->cat_illuminant);
  }

  g->cat_illum_fluo = dt_bauhaus_combobox_from_params(self, "cat_illum_fluo");
  g->cat_illum_led = dt_bauhaus_combobox_from_params(self, "cat_illum_led");
  g->cat_adaptation = dt_bauhaus_combobox_from_params(self, "cat_adaptation");
  g->cat_x = dt_bauhaus_slider_new_with_range_and_feedback
    (self, 0.0, DT_IOP_WB_CAT_HUE_MAX, 0, 0.0, 1, 0);
  dt_bauhaus_widget_set_label(g->cat_x, NULL, N_("hue"));
  dt_bauhaus_slider_set_format(g->cat_x, "°");
  g_signal_connect(G_OBJECT(g->cat_x), "value-changed",
                   G_CALLBACK(_cat_hue_chroma_callback), self);

  g->cat_y = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0, 0.0, 1);
  dt_bauhaus_widget_set_label(g->cat_y, NULL, N_("chroma"));
  dt_bauhaus_slider_set_format(g->cat_y, "%");
  dt_bauhaus_slider_set_hard_max(g->cat_y, DT_IOP_WB_CAT_CHROMA_MAX);
  g_signal_connect(G_OBJECT(g->cat_y), "value-changed",
                   G_CALLBACK(_cat_hue_chroma_callback), self);

  dt_gui_box_add(self->widget, g->cat_x, g->cat_y);

  g_signal_connect(G_OBJECT(g->scale_k), "value-changed",
                   G_CALLBACK(_temp_tint_callback), self);
  g_signal_connect(G_OBJECT(g->scale_tint), "value-changed",
                   G_CALLBACK(_temp_tint_callback), self);

  g_signal_connect(G_OBJECT(g->presets), "value-changed",
                   G_CALLBACK(_preset_tune_callback), self);
  g_signal_connect(G_OBJECT(g->finetune), "value-changed",
                   G_CALLBACK(_preset_tune_callback), self);

  // update the gui when the preferences changed (i.e. colored sliders stuff)
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_PREFERENCES_CHANGE, _preference_changed);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_HISTORY_CHANGE,
                           _history_changed_callback);

  // start building top level widget
  self->widget = gtk_stack_new();
  gtk_stack_set_homogeneous(GTK_STACK(self->widget), FALSE);

  GtkWidget *label_disabled = gtk_label_new(_("white balance disabled for camera"));
  gtk_widget_set_halign(label_disabled, GTK_ALIGN_START);
  gtk_label_set_ellipsize(GTK_LABEL(label_disabled), PANGO_ELLIPSIZE_END);

  gtk_stack_add_named(GTK_STACK(self->widget), GTK_WIDGET(box_enabled), "enabled");
  gtk_stack_add_named(GTK_STACK(self->widget), label_disabled, "disabled");

  if(self->dev)
    self->dev->proxy.wb_set_illuminant = _wb_set_custom_illuminant;
}

void gui_cleanup(dt_iop_module_t *self)
{
  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;
}

void gui_reset(dt_iop_module_t *self)
{
  dt_iop_temperature_gui_data_t *g = self->gui_data;
  dt_iop_temperature_params_t *d = self->default_params;

  const dt_iop_temperature_params_t *p = self->params;
  const int preset = d->preset = p->use_cat
                                 ? DT_IOP_TEMP_AS_SHOT
                                 : (dt_is_scene_referred() ? DT_IOP_TEMP_D65_LATE
                                                           : DT_IOP_TEMP_AS_SHOT);

  dt_iop_color_picker_reset(self, TRUE);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_asshot),
                               preset == DT_IOP_TEMP_AS_SHOT);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_user),
                               preset == DT_IOP_TEMP_USER);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65),
                               preset == DT_IOP_TEMP_D65);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_d65_late),
                               preset == DT_IOP_TEMP_D65_LATE);

  _color_finetuning_slider(self);
  _color_rgb_sliders(self);
  _color_temptint_sliders(self);
  _cat_controls_update(self);
  _update_preset(self, preset);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

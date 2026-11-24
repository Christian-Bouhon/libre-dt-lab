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

#include "common/assistant/assistant_stats.h"

#include "common/assistant/assistant_catalog.h"
#include "common/colorspaces.h"
#include "common/darktable.h"
#include "common/image.h"
#include "imageio/imageio_common.h"
#include "develop/pixelpipe.h"
#include "imageio/imageio_module.h"
#include "control/conf.h"

#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>

#define STATS_MAX_DIM 512

typedef struct
{
  dt_imageio_module_data_t parent;
  float *pixels;
  int width;
  int height;
} dt_assistant_capture_t;

static int _check_bpp(dt_imageio_module_data_t *data)
{
  (void)data;
  return 32;
}

static int _check_levels(dt_imageio_module_data_t *data)
{
  (void)data;
  return IMAGEIO_RGB | IMAGEIO_FLOAT;
}

static const char *_get_mime(dt_imageio_module_data_t *data)
{
  (void)data;
  return "memory";
}

static int _write_image(dt_imageio_module_data_t *data,
                        const char *filename,
                        const void *in_void,
                        dt_colorspaces_color_profile_type_t over_type,
                        const char *over_filename,
                        void *exif, int exif_len,
                        dt_imgid_t imgid,
                        const int num,
                        const int total,
                        dt_dev_pixelpipe_t *pipe,
                        const gboolean export_masks)
{
  (void)filename;
  (void)over_type;
  (void)over_filename;
  (void)exif;
  (void)exif_len;
  (void)imgid;
  (void)num;
  (void)total;
  (void)pipe;
  (void)export_masks;

  dt_assistant_capture_t *capture = (dt_assistant_capture_t *)data;
  const size_t bytes = (size_t)data->width * data->height * 4 * sizeof(float);
  capture->pixels = g_try_malloc(bytes);
  if(!capture->pixels) return 1;
  memcpy(capture->pixels, in_void, bytes);
  capture->width = data->width;
  capture->height = data->height;
  return 0;
}

static inline float _linear_to_srgb(const float v)
{
  if(v <= 0.0f) return 0.0f;
  if(v >= 1.0f) return 1.0f;
  return (v <= 0.0031308f) ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

static float _percentile(const uint32_t *hist, const uint32_t total, const float p)
{
  const uint64_t target = (uint64_t)(p * (float)total);
  uint64_t sum = 0;
  for(int i = 0; i < 256; i++)
  {
    sum += hist[i];
    if(sum >= target) return (float)i / 255.0f;
  }
  return 1.0f;
}

char *dt_assistant_stats_build(const dt_imgid_t imgid, char **error)
{
  if(error) *error = NULL;
  if(!dt_is_valid_imgid(imgid))
  {
    if(error) *error = g_strdup(_("no image to analyze"));
    return NULL;
  }

  dt_assistant_capture_t capture = { 0 };
  capture.parent.max_width = STATS_MAX_DIM;
  capture.parent.max_height = STATS_MAX_DIM;

  dt_imageio_module_format_t format = {
    .mime = _get_mime,
    .levels = _check_levels,
    .bpp = _check_bpp,
    .write_image = _write_image
  };

  dt_imageio_export_with_flags(imgid,
                               "unused",
                               &format,
                               (dt_imageio_module_data_t *)&capture,
                               TRUE,   // ignore_exif
                               FALSE,  // display_byteorder
                               TRUE,   // high_quality
                               FALSE,  // upscale
                               FALSE,  // is_scaling
                               1.0,    // scale_factor
                               FALSE,  // thumbnail_export
                               NULL,   // filter
                               FALSE,  // copy_metadata
                               FALSE,  // export_masks
                               DT_COLORSPACE_LIN_REC2020, NULL,
                               DT_INTENT_PERCEPTUAL,
                               NULL, NULL, 1, 1, NULL, -1);

  if(!capture.pixels || capture.width <= 0 || capture.height <= 0)
  {
    g_free(capture.pixels);
    if(error) *error = g_strdup(_("could not compute image statistics"));
    return NULL;
  }

  uint32_t hist[256] = { 0 };
  uint32_t sat_hist[256] = { 0 };
  const uint32_t total = (uint32_t)capture.width * (uint32_t)capture.height;
  double sum_luma = 0.0;
  double sum_rgb[3] = { 0.0, 0.0, 0.0 };
  double sum_sat = 0.0;
  uint32_t clip_low = 0, clip_high = 0;
  uint32_t chan_low[3] = { 0, 0, 0 }, chan_high[3] = { 0, 0, 0 };

  for(uint32_t i = 0; i < total; i++)
  {
    const float *p = capture.pixels + (size_t)i * 4;
    float rgb[3];
    for(int c = 0; c < 3; c++)
    {
      rgb[c] = _linear_to_srgb(p[c]);
      sum_rgb[c] += rgb[c];
      if(rgb[c] <= 0.005f) chan_low[c]++;
      if(rgb[c] >= 0.995f) chan_high[c]++;
    }

    const float luma = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
    const float clamped = CLAMP(luma, 0.0f, 1.0f);
    sum_luma += clamped;
    if(clamped <= 0.005f) clip_low++;
    if(clamped >= 0.995f) clip_high++;

    const float max_rgb = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
    const float min_rgb = fminf(rgb[0], fminf(rgb[1], rgb[2]));
    const float saturation = max_rgb > 1e-5f ? (max_rgb - min_rgb) / max_rgb : 0.0f;
    sum_sat += saturation;

    const int bin = (int)(clamped * 255.0f + 0.5f);
    hist[CLAMP(bin, 0, 255)]++;
    const int sat_bin = (int)(CLAMP(saturation, 0.0f, 1.0f) * 255.0f + 0.5f);
    sat_hist[CLAMP(sat_bin, 0, 255)]++;
  }

  g_free(capture.pixels);

  const float mean = (float)(sum_luma / (double)total);
  const float p1 = _percentile(hist, total, 0.01f);
  const float p5 = _percentile(hist, total, 0.05f);
  const float p25 = _percentile(hist, total, 0.25f);
  const float p50 = _percentile(hist, total, 0.50f);
  const float p75 = _percentile(hist, total, 0.75f);
  const float p95 = _percentile(hist, total, 0.95f);
  const float p99 = _percentile(hist, total, 0.99f);
  const float spread = p95 - p5;
  const float mean_r = (float)(sum_rgb[0] / (double)total);
  const float mean_g = (float)(sum_rgb[1] / (double)total);
  const float mean_b = (float)(sum_rgb[2] / (double)total);
  const float mean_sat = (float)(sum_sat / (double)total);
  const float sat_p50 = _percentile(sat_hist, total, 0.50f);
  const float sat_p95 = _percentile(sat_hist, total, 0.95f);
  const float color_cast_rb = mean_r - mean_b;
  const float color_cast_rg = mean_r - mean_g;

  const float clip_low_pct = 100.0f * (float)clip_low / (float)total;
  const float clip_high_pct = 100.0f * (float)clip_high / (float)total;

  float exposure_ev = 0.0f;
  if(p50 < 0.12f) exposure_ev += 0.5f;
  else if(p50 > 0.35f) exposure_ev -= 0.5f;
  if(clip_high_pct > 0.5f || p99 > 0.995f) exposure_ev -= 0.3f;
  exposure_ev = CLAMP(exposure_ev, -0.7f, 0.7f);

  const float configured_bias
    = CLAMP(dt_conf_get_float(DT_ASSISTANT_CONF_EXPOSURE_BIAS), -2.0f, 2.0f);
  const gboolean highlights_clipping = (clip_high_pct > 0.5f || p99 > 0.995f);
  float applied_bias = configured_bias;
  if(highlights_clipping && applied_bias > 0.0f) applied_bias = 0.0f;
  exposure_ev = CLAMP(exposure_ev + applied_bias, -2.0f, 2.0f);

  JsonBuilder *builder = json_builder_new();
  json_builder_begin_object(builder);

  json_builder_set_member_name(builder, "luma_p1");
  json_builder_add_double_value(builder, p1);
  json_builder_set_member_name(builder, "luma_p5");
  json_builder_add_double_value(builder, p5);
  json_builder_set_member_name(builder, "luma_p25");
  json_builder_add_double_value(builder, p25);
  json_builder_set_member_name(builder, "luma_p50");
  json_builder_add_double_value(builder, p50);
  json_builder_set_member_name(builder, "luma_p75");
  json_builder_add_double_value(builder, p75);
  json_builder_set_member_name(builder, "luma_p95");
  json_builder_add_double_value(builder, p95);
  json_builder_set_member_name(builder, "luma_p99");
  json_builder_add_double_value(builder, p99);
  json_builder_set_member_name(builder, "luma_mean");
  json_builder_add_double_value(builder, mean);
  json_builder_set_member_name(builder, "contrast_spread");
  json_builder_add_double_value(builder, spread);
  json_builder_set_member_name(builder, "clip_low_pct");
  json_builder_add_double_value(builder, clip_low_pct);
  json_builder_set_member_name(builder, "clip_high_pct");
  json_builder_add_double_value(builder, clip_high_pct);

  json_builder_set_member_name(builder, "channel_clip");
  json_builder_begin_object(builder);
  for(int c = 0; c < 3; c++)
  {
    char low[16], high[16];
    snprintf(low, sizeof(low), "c%d_low_pct", c);
    snprintf(high, sizeof(high), "c%d_high_pct", c);
    json_builder_set_member_name(builder, low);
    json_builder_add_double_value(builder, 100.0f * (float)chan_low[c] / (float)total);
    json_builder_set_member_name(builder, high);
    json_builder_add_double_value(builder, 100.0f * (float)chan_high[c] / (float)total);
  }
  json_builder_end_object(builder);

  json_builder_set_member_name(builder, "rgb_mean");
  json_builder_begin_array(builder);
  json_builder_add_double_value(builder, mean_r);
  json_builder_add_double_value(builder, mean_g);
  json_builder_add_double_value(builder, mean_b);
  json_builder_end_array(builder);
  json_builder_set_member_name(builder, "saturation_mean");
  json_builder_add_double_value(builder, mean_sat);
  json_builder_set_member_name(builder, "saturation_p50");
  json_builder_add_double_value(builder, sat_p50);
  json_builder_set_member_name(builder, "saturation_p95");
  json_builder_add_double_value(builder, sat_p95);
  json_builder_set_member_name(builder, "color_cast_rb");
  json_builder_add_double_value(builder, color_cast_rb);
  json_builder_set_member_name(builder, "color_cast_rg");
  json_builder_add_double_value(builder, color_cast_rg);

  json_builder_set_member_name(builder, "exposure_bias_ev");
  json_builder_add_double_value(builder, configured_bias);
  json_builder_set_member_name(builder, "exposure_bias_applied_ev");
  json_builder_add_double_value(builder, applied_bias);
  json_builder_set_member_name(builder, "suggested_exposure_ev");
  json_builder_add_double_value(builder, exposure_ev);

  json_builder_set_member_name(builder, "hints");
  json_builder_begin_array(builder);
  if(clip_high_pct > 0.5f)
    json_builder_add_string_value(builder, "highlights are clipping; reduce exposure or highlight gain");
  if(highlights_clipping && configured_bias > 0.0f)
    json_builder_add_string_value(builder,
                                 "positive exposure bias skipped because highlights are clipping");
  if(clip_low_pct > 1.0f || p5 < 0.02f)
    json_builder_add_string_value(builder, "shadows are crushed; lift shadows");
  if(p50 < 0.12f)
    json_builder_add_string_value(builder, "image is dark; increase exposure");
  if(p50 > 0.35f)
    json_builder_add_string_value(builder, "image is bright; reduce exposure");
  if(spread < 0.45f)
    json_builder_add_string_value(builder, "low contrast; increase global contrast");
  if(spread > 0.85f)
    json_builder_add_string_value(builder, "high contrast; reduce global contrast");
  if(sat_p50 < 0.08f)
    json_builder_add_string_value(builder, "colors are muted; consider saturation, vibrance or color contrast");
  if(sat_p95 - sat_p50 < 0.12f)
    json_builder_add_string_value(builder, "low color separation; consider colorful contrast");
  if(color_cast_rb > 0.10f)
    json_builder_add_string_value(builder, "warm color cast; consider cooler color balance");
  else if(color_cast_rb < -0.10f)
    json_builder_add_string_value(builder, "cool color cast; consider warmer color balance");
  json_builder_end_array(builder);

  json_builder_end_object(builder);

  JsonNode *root = json_builder_get_root(builder);
  JsonGenerator *generator = json_generator_new();
  json_generator_set_root(generator, root);
  char *result = json_generator_to_data(generator, NULL);

  json_node_free(root);
  g_object_unref(generator);
  g_object_unref(builder);
  return result;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

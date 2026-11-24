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

#include "common/assistant/assistant_image.h"

#include "common/colorspaces.h"
#include "common/darktable.h"
#include "common/image.h"
#include "develop/pixelpipe.h"
#include "imageio/imageio_common.h"
#include "imageio/imageio_module.h"

#include <gdk-pixbuf/gdk-pixbuf.h>
#include <math.h>
#include <string.h>

typedef struct
{
  dt_imageio_module_data_t parent;
  float *pixels;
  int width;
  int height;
} dt_assistant_image_capture_t;

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

  dt_assistant_image_capture_t *capture = (dt_assistant_image_capture_t *)data;
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

char *dt_assistant_preview_build(const dt_imgid_t imgid,
                                 const int max_dim,
                                 char **error)
{
  if(error) *error = NULL;
  if(!dt_is_valid_imgid(imgid))
  {
    if(error) *error = g_strdup(_("no image to analyze"));
    return NULL;
  }

  const int dimension = CLAMP(max_dim, 128, 2048);
  dt_assistant_image_capture_t capture = { 0 };
  capture.parent.max_width = dimension;
  capture.parent.max_height = dimension;

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
                               DT_COLORSPACE_LIN_REC709, NULL,
                               DT_INTENT_PERCEPTUAL,
                               NULL, NULL, 1, 1, NULL, -1);

  if(!capture.pixels || capture.width <= 0 || capture.height <= 0)
  {
    g_free(capture.pixels);
    if(error) *error = g_strdup(_("could not build the preview image"));
    return NULL;
  }

  GdkPixbuf *pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8,
                                     capture.width, capture.height);
  if(!pixbuf)
  {
    g_free(capture.pixels);
    if(error) *error = g_strdup(_("could not allocate the preview image"));
    return NULL;
  }

  guchar *out = gdk_pixbuf_get_pixels(pixbuf);
  const int rowstride = gdk_pixbuf_get_rowstride(pixbuf);
  for(int y = 0; y < capture.height; y++)
  {
    guchar *row = out + (size_t)y * rowstride;
    for(int x = 0; x < capture.width; x++)
    {
      const float *pixel = capture.pixels + ((size_t)y * capture.width + x) * 4;
      for(int c = 0; c < 3; c++)
      {
        const float value = _linear_to_srgb(pixel[c]);
        row[x * 3 + c] = (guchar)roundf(CLAMP(value, 0.0f, 1.0f) * 255.0f);
      }
    }
  }
  g_free(capture.pixels);

  gchar *jpeg = NULL;
  gsize jpeg_size = 0;
  GError *gerror = NULL;
  if(!gdk_pixbuf_save_to_buffer(pixbuf, &jpeg, &jpeg_size, "jpeg", &gerror,
                                "quality", "85", NULL))
  {
    if(error)
      *error = g_strdup_printf(_("could not encode the preview image: %s"),
                               gerror ? gerror->message : _("unknown error"));
    if(gerror) g_error_free(gerror);
    g_object_unref(pixbuf);
    return NULL;
  }
  g_object_unref(pixbuf);

  char *base64 = g_base64_encode((const guchar *)jpeg, jpeg_size);
  g_free(jpeg);
  return base64;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

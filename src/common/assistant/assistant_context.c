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

#include "common/assistant/assistant_context.h"

#include "common/assistant/assistant_actions.h"
#include "common/assistant/assistant_catalog.h"
#include "common/darktable.h"
#include "common/image.h"
#include "control/conf.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "libs/workflow_helpers.h"

#include <json-glib/json-glib.h>
#include <string.h>

static const char *_tone_mapper_ops[] = {
  "agx", "filmicrgb", "sigmoid", "basecurve", "3dcf", "aces20", NULL
};

static const char *_active_tone_mapper(void)
{
  dt_develop_t *dev = darktable.develop;
  if(!dev) return NULL;
  for(GList *l = dev->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *module = l->data;
    if(!module || !module->enabled) continue;
    for(int i = 0; _tone_mapper_ops[i]; i++)
      if(!g_strcmp0(module->op, _tone_mapper_ops[i])) return module->op;
  }
  return NULL;
}

static void _add_module_values(JsonBuilder *builder,
                               const dt_assistant_catalog_module_t *catalog)
{
  json_builder_set_member_name(builder, "params");
  json_builder_begin_object(builder);
  for(guint j = 0; j < catalog->params->len; j++)
  {
    dt_assistant_catalog_param_t *param = g_ptr_array_index(catalog->params, j);
    float value = 0.0f;
    if(!dt_assistant_module_float(catalog->operation, param->name, &value))
      continue;
    json_builder_set_member_name(builder, param->name);
    json_builder_add_double_value(builder, value);
  }
  json_builder_end_object(builder);
}

char *dt_assistant_context_build(char **error)
{
  if(error) *error = NULL;

  dt_develop_t *dev = darktable.develop;
  if(!dev || !dt_is_valid_imgid(dev->image_storage.id))
  {
    if(error) *error = g_strdup(_("no image is open in the darkroom"));
    return NULL;
  }

  const dt_image_t *img = &dev->image_storage;

  JsonBuilder *builder = json_builder_new();
  json_builder_begin_object(builder);

  json_builder_set_member_name(builder, "image");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "camera_maker");
  json_builder_add_string_value(builder, img->camera_maker);
  json_builder_set_member_name(builder, "camera_model");
  json_builder_add_string_value(builder, img->camera_model);
  json_builder_set_member_name(builder, "lens");
  json_builder_add_string_value(builder, img->exif_lens);
  json_builder_set_member_name(builder, "iso");
  json_builder_add_double_value(builder, img->exif_iso);
  json_builder_set_member_name(builder, "exposure_s");
  json_builder_add_double_value(builder, img->exif_exposure);
  json_builder_set_member_name(builder, "aperture");
  json_builder_add_double_value(builder, img->exif_aperture);
  json_builder_set_member_name(builder, "focal_length_mm");
  json_builder_add_double_value(builder, img->exif_focal_length);
  json_builder_set_member_name(builder, "width");
  json_builder_add_int_value(builder, img->width);
  json_builder_set_member_name(builder, "height");
  json_builder_add_int_value(builder, img->height);
  json_builder_set_member_name(builder, "is_raw");
  json_builder_add_boolean_value(builder, dt_image_is_raw(img));
  json_builder_end_object(builder);

  const char *preferred = dt_assistant_preferred_tone_mapper();
  if(preferred && !dt_assistant_catalog_is_allowed(preferred)) preferred = NULL;
  const char *active = _active_tone_mapper();
  json_builder_set_member_name(builder, "preferred_tone_mapper");
  if(preferred) json_builder_add_string_value(builder, preferred);
  else json_builder_add_null_value(builder);
  json_builder_set_member_name(builder, "active_tone_mapper");
  if(active) json_builder_add_string_value(builder, active);
  else json_builder_add_null_value(builder);

  GPtrArray *ops = dt_assistant_allowed_ops();

  json_builder_set_member_name(builder, "enabled_modules");
  json_builder_begin_array(builder);
  for(guint i = 0; i < ops->len; i++)
  {
    const char *op = g_ptr_array_index(ops, i);
    const dt_assistant_catalog_module_t *catalog = dt_assistant_catalog_get(op);
    dt_iop_module_t *module = dt_workflow_find_module(op);
    if(!catalog || !module || !module->enabled) continue;

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "operation");
    json_builder_add_string_value(builder, catalog->operation);
    json_builder_set_member_name(builder, "name");
    json_builder_add_string_value(builder, catalog->name);
    json_builder_set_member_name(builder, "role");
    json_builder_add_string_value(builder, catalog->role ? catalog->role : "");
    _add_module_values(builder, catalog);
    json_builder_end_object(builder);
  }
  json_builder_end_array(builder);

  json_builder_set_member_name(builder, "available_modules");
  json_builder_begin_array(builder);
  for(guint i = 0; i < ops->len; i++)
  {
    const char *op = g_ptr_array_index(ops, i);
    const dt_assistant_catalog_module_t *catalog = dt_assistant_catalog_get(op);
    dt_iop_module_t *module = dt_workflow_find_module(op);
    if(!catalog || (module && module->enabled)) continue;
    if(!catalog->addable || !dt_iop_get_module_so(op)) continue;

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "operation");
    json_builder_add_string_value(builder, catalog->operation);
    json_builder_set_member_name(builder, "name");
    json_builder_add_string_value(builder, catalog->name);
    json_builder_set_member_name(builder, "role");
    json_builder_add_string_value(builder, catalog->role ? catalog->role : "");
    _add_module_values(builder, catalog);
    json_builder_end_object(builder);
  }
  json_builder_end_array(builder);

  g_ptr_array_free(ops, TRUE);

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

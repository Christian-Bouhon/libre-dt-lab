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

#include "common/assistant/assistant_catalog.h"

#include "common/darktable.h"
#include "common/file_location.h"
#include "control/conf.h"

#include <json-glib/json-glib.h>
#include <string.h>

#define CONF_ALLOWED "plugins/assistant/modules"
#define DEFAULT_ALLOWED \
  "exposure|contrast|toneequal|agx|filmicrgb|sigmoid|3dcf|aces20|" \
  "colorbalancergb|colorharmonizer|primaries|sharpen|diffuse"

static void _param_free(gpointer data)
{
  dt_assistant_catalog_param_t *param = data;
  if(!param) return;
  g_free(param->name);
  g_free(param->description);
  g_free(param->suffix);
  g_free(param);
}

static void _module_free(gpointer data)
{
  dt_assistant_catalog_module_t *module = data;
  if(!module) return;
  g_free(module->operation);
  g_free(module->name);
  g_free(module->description);
  g_free(module->role);
  if(module->params) g_ptr_array_free(module->params, TRUE);
  g_free(module);
}

static char *_json_string(JsonObject *object, const char *key)
{
  if(!json_object_has_member(object, key)) return NULL;
  JsonNode *node = json_object_get_member(object, key);
  if(!node || !JSON_NODE_HOLDS_VALUE(node)) return NULL;
  const char *value = json_node_get_string(node);
  return value ? g_strdup(value) : NULL;
}

static GPtrArray *_load_catalog(char **error)
{
  GPtrArray *modules = g_ptr_array_new_with_free_func(_module_free);

  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  char *path = g_build_filename(datadir, "ai_assistant_modules.json", NULL);
  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_file(parser, path, NULL))
  {
    if(error)
      *error = g_strdup_printf("cannot load %s", path);
    g_free(path);
    g_object_unref(parser);
    return modules;
  }
  g_free(path);

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    if(error) *error = g_strdup("catalog root is not an object");
    g_object_unref(parser);
    return modules;
  }

  JsonObject *root_obj = json_node_get_object(root);
  if(!json_object_has_member(root_obj, "modules"))
  {
    if(error) *error = g_strdup("catalog has no modules array");
    g_object_unref(parser);
    return modules;
  }
  JsonNode *modules_node = json_object_get_member(root_obj, "modules");
  if(!modules_node || !JSON_NODE_HOLDS_ARRAY(modules_node))
  {
    if(error) *error = g_strdup("catalog modules is not an array");
    g_object_unref(parser);
    return modules;
  }

  JsonArray *array = json_node_get_array(modules_node);
  for(guint i = 0; i < json_array_get_length(array); i++)
  {
    JsonNode *node = json_array_get_element(array, i);
    if(!node || !JSON_NODE_HOLDS_OBJECT(node)) continue;
    JsonObject *object = json_node_get_object(node);

    dt_assistant_catalog_module_t *module = g_new0(dt_assistant_catalog_module_t, 1);
    module->operation = _json_string(object, "operation");
    module->name = _json_string(object, "name");
    module->description = _json_string(object, "description");
    module->role = _json_string(object, "role");
    module->exclusive_role = json_object_has_member(object, "exclusive_role")
                             && json_object_get_boolean_member(object, "exclusive_role");
    module->addable = !json_object_has_member(object, "addable")
                      || json_object_get_boolean_member(object, "addable");
    module->params = g_ptr_array_new_with_free_func(_param_free);

    if(!module->operation || !module->operation[0])
    {
      _module_free(module);
      continue;
    }

    if(json_object_has_member(object, "params"))
    {
      JsonNode *params_node = json_object_get_member(object, "params");
      if(params_node && JSON_NODE_HOLDS_ARRAY(params_node))
      {
        JsonArray *params = json_node_get_array(params_node);
        for(guint j = 0; j < json_array_get_length(params); j++)
        {
          JsonNode *param_node = json_array_get_element(params, j);
          if(!param_node || !JSON_NODE_HOLDS_OBJECT(param_node)) continue;
          JsonObject *param_obj = json_node_get_object(param_node);
          dt_assistant_catalog_param_t *param = g_new0(dt_assistant_catalog_param_t, 1);
          param->name = _json_string(param_obj, "name");
          param->description = _json_string(param_obj, "description");
          param->max_delta = json_object_has_member(param_obj, "max_delta")
                             ? (float)json_object_get_double_member(param_obj, "max_delta")
                             : 0.5f;
          param->factor = 1.0f;
          param->offset = 0.0f;
          param->decimals = 0;
          param->has_display = FALSE;
          if(json_object_has_member(param_obj, "display"))
          {
            JsonNode *display_node = json_object_get_member(param_obj, "display");
            if(display_node && JSON_NODE_HOLDS_OBJECT(display_node))
            {
              JsonObject *display = json_node_get_object(display_node);
              param->has_display = TRUE;
              if(json_object_has_member(display, "factor"))
                param->factor = (float)json_object_get_double_member(display, "factor");
              if(json_object_has_member(display, "offset"))
                param->offset = (float)json_object_get_double_member(display, "offset");
              if(json_object_has_member(display, "decimals"))
                param->decimals = (int)json_object_get_int_member(display, "decimals");
              param->suffix = _json_string(display, "suffix");
              if(!param->suffix) param->suffix = g_strdup("");
            }
          }
          if(!param->name || !param->name[0] || param->max_delta <= 0.0f)
          {
            _param_free(param);
            continue;
          }
          g_ptr_array_add(module->params, param);
        }
      }
    }

    if(module->params->len > 0)
      g_ptr_array_add(modules, module);
    else
      _module_free(module);
  }

  g_object_unref(parser);
  return modules;
}

const GPtrArray *dt_assistant_catalog_modules(void)
{
  static GPtrArray *catalog = NULL;
  static gsize initialized = 0;
  if(g_once_init_enter(&initialized))
  {
    char *error = NULL;
    catalog = _load_catalog(&error);
    if(error)
    {
      dt_print(DT_DEBUG_AI, "[assistant_catalog] %s", error);
      g_free(error);
    }
    g_once_init_leave(&initialized, 1);
  }
  return catalog;
}

const dt_assistant_catalog_module_t *dt_assistant_catalog_get(const char *operation)
{
  if(!operation) return NULL;
  const GPtrArray *modules = dt_assistant_catalog_modules();
  for(guint i = 0; i < modules->len; i++)
  {
    dt_assistant_catalog_module_t *module = g_ptr_array_index(modules, i);
    if(!g_strcmp0(module->operation, operation)) return module;
  }
  return NULL;
}

const dt_assistant_catalog_param_t *dt_assistant_catalog_param(
  const dt_assistant_catalog_module_t *module,
  const char *name)
{
  if(!module || !name) return NULL;
  for(guint i = 0; i < module->params->len; i++)
  {
    dt_assistant_catalog_param_t *param = g_ptr_array_index(module->params, i);
    if(!g_strcmp0(param->name, name)) return param;
  }
  return NULL;
}

gboolean dt_assistant_catalog_is_tone_mapper(const dt_assistant_catalog_module_t *module)
{
  return module
         && (module->exclusive_role
             || !g_strcmp0(module->role, "tone_mapper"));
}

const char *dt_assistant_preferred_tone_mapper(void)
{
  const char *pref = dt_conf_get_string_const(DT_ASSISTANT_CONF_TONE_MAPPER);
  if(pref && pref[0] && g_strcmp0(pref, "auto"))
  {
    const dt_assistant_catalog_module_t *module = dt_assistant_catalog_get(pref);
    if(module && dt_assistant_catalog_is_tone_mapper(module))
      return module->operation;
  }

  const char *workflow = dt_conf_get_string_const("plugins/darkroom/workflow");
  if(!workflow) return NULL;
  if(strstr(workflow, "3DCF")) return "3dcf";
  if(strstr(workflow, "ACES")) return "aces20";
  if(strstr(workflow, "AgX")) return "agx";
  if(strstr(workflow, "filmic")) return "filmicrgb";
  if(strstr(workflow, "sigmoid")) return "sigmoid";
  if(strstr(workflow, "basecurve")) return "basecurve";
  return NULL;
}

char *dt_assistant_catalog_format_value(
  const dt_assistant_catalog_module_t *module,
  const char *param,
  float value)
{
  const dt_assistant_catalog_param_t *info
    = dt_assistant_catalog_param(module, param);
  if(!info || !info->has_display)
    return g_strdup_printf("%.3g", value);

  const float displayed = value * info->factor + info->offset;
  return g_strdup_printf("%.*f%s", info->decimals, displayed,
                         info->suffix ? info->suffix : "");
}

char *dt_assistant_catalog_default_allowed(void)
{
  return g_strdup(DEFAULT_ALLOWED);
}

char *dt_assistant_allowed_string(void)
{
  const char *value = dt_conf_get_string_const(CONF_ALLOWED);
  if(value && !g_strcmp0(value, "none")) return g_strdup("");
  if(value && value[0]) return g_strdup(value);
  return dt_assistant_catalog_default_allowed();
}

void dt_assistant_allowed_set_string(const char *list)
{
  dt_conf_set_string(CONF_ALLOWED, list ? list : "");
}

GPtrArray *dt_assistant_allowed_ops(void)
{
  char *value = dt_assistant_allowed_string();
  GPtrArray *ops = g_ptr_array_new_with_free_func(g_free);
  char **tokens = g_strsplit(value, "|", -1);
  GHashTable *seen = g_hash_table_new(g_str_hash, g_str_equal);
  for(char **token = tokens; token && *token; token++)
  {
    char *op = g_strstrip(g_strdup(*token));
    if(op[0] && !g_hash_table_contains(seen, op))
    {
      g_hash_table_add(seen, g_strdup(op));
      g_ptr_array_add(ops, op);
    }
    else
      g_free(op);
  }
  g_hash_table_destroy(seen);
  g_strfreev(tokens);
  g_free(value);
  return ops;
}

gboolean dt_assistant_catalog_is_allowed(const char *operation)
{
  if(!operation) return FALSE;
  GPtrArray *ops = dt_assistant_allowed_ops();
  gboolean allowed = FALSE;
  for(guint i = 0; i < ops->len && !allowed; i++)
    allowed = !g_strcmp0(g_ptr_array_index(ops, i), operation);
  g_ptr_array_free(ops, TRUE);
  return allowed;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

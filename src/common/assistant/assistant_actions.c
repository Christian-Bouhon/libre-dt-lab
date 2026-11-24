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

#include "common/assistant/assistant_actions.h"

#include "common/assistant/assistant_catalog.h"
#include "common/darktable.h"
#include "common/history_snapshot.h"
#include "common/iop_order.h"
#include "common/introspection.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "libs/workflow_helpers.h"

#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>

static const char *_tone_mapper_ops[] = {
  "agx", "filmicrgb", "sigmoid", "basecurve", "3dcf", "aces20", NULL
};

static gboolean _is_tone_mapper_op(const char *operation)
{
  for(int i = 0; _tone_mapper_ops[i]; i++)
    if(!g_strcmp0(_tone_mapper_ops[i], operation)) return TRUE;
  return FALSE;
}

static gboolean _field_for_op(const char *operation,
                              const char *param,
                              dt_iop_module_so_t **so_out,
                              dt_introspection_field_t **field_out)
{
  dt_iop_module_so_t *so = dt_iop_get_module_so(operation);
  if(!so || !so->get_f) return FALSE;
  dt_introspection_field_t *field = so->get_f(param);
  if(!field || field->header.type != DT_INTROSPECTION_TYPE_FLOAT) return FALSE;
  if(so_out) *so_out = so;
  if(field_out) *field_out = field;
  return TRUE;
}

static float _current_value(dt_iop_module_t *module,
                            dt_introspection_field_t *field)
{
  const void *base = NULL;
  if(module && module->enabled && module->params)
    base = module->params;
  else if(module && module->default_params)
    base = module->default_params;

  if(base)
    return *(float *)((char *)base + field->header.offset);
  return field->Float.Default;
}

gboolean dt_assistant_module_float(const char *operation,
                                   const char *param,
                                   float *value)
{
  dt_iop_module_so_t *so = NULL;
  dt_introspection_field_t *field = NULL;
  if(!_field_for_op(operation, param, &so, &field)) return FALSE;
  if(value)
    *value = _current_value(dt_workflow_find_module(operation), field);
  return TRUE;
}

char *dt_assistant_system_prompt(void)
{
  GString *prompt = g_string_new(NULL);
  g_string_append(prompt,
                  "You are a photography assistant inside darktable. "
                  "Your job is to propose a conservative first-pass set of slider "
                  "adjustments for the current image. The photographer will refine them.\n"
                  "\n"
                  "Hard rules:\n"
                  "- Use only the modules and sliders listed below.\n"
                  "- Use the exact parameter names listed. Do not invent sliders; "
                  "if a slider is not listed, omit that change. For Tone Equalizer, "
                  "use only the nine band names listed in the module list.\n"
                  "- You may propose values for modules that are already active, and for "
                  "modules that are available but not active. Proposing a change for an "
                  "inactive module will enable it.\n"
                  "- Propose at most 5 changes, one parameter per change.\n"
                  "- Activate at most 2 new modules.\n"
                  "- Never activate more than one tone mapper. If a tone mapper is already "
                  "active, adjust that one instead of activating another. If none is "
                  "active, prefer the preferred_tone_mapper from the context.\n"
                  "- Tonality is controlled by the active tone mapper. When the "
                  "image statistics or the attached preview show a tonal problem, "
                  "include a conservative change to the active tone mapper when it "
                  "is allowed; do not rely only on Exposure for tonality. 3DCF auto "
                  "is a separate control: do not trigger it, but you may still "
                  "propose 3DCF parameter changes as before.\n"
                  "- For color contrast and colorimetric separation, prefer Tone & "
                  "Texture's colorful_contrast, contrast_balance and color_balance "
                  "when the module is allowed. Do not use luminance contrast sliders "
                  "to fix a color-separation issue.\n"
                  "- For general colorimetry, prefer the selected color module. Use "
                  "Color Balance RGB for color contrast, chroma, saturation and "
                  "vibrance, and Color Harmonizer or RGB Primaries for targeted "
                  "hue/tint shifts. Keep color changes conservative and respect the "
                  "user's intent.\n"
                  "- Never touch masks, curves, blending, crop, white balance, lens "
                  "correction, denoise, or any file/export operation.\n"
                  "- Respect the maximum absolute delta listed below. The delta is "
                  "measured from the current value, not from the range limits: a "
                  "value can be inside the range and still violate the delta. For "
                  "example, a slider at 2.25 with max delta 0.5 may only be proposed "
                  "between 1.75 and 2.75.\n"
                  "- Use the image statistics and hints in the user message to justify "
                  "each change. If no adjustment is clearly useful, return an empty "
                  "changes array.\n"
                  "- The user request is authoritative: honor its intent and any "
                  "constraints it contains, even if the statistics suggest a more "
                  "generic edit.\n"
                  "- Return one JSON object only, with no markdown fences and no prose.\n"
                  "- Write summary and reason strings in the user interface language "
                  "given in the user message.\n");

  if(dt_assistant_catalog_is_allowed("exposure"))
    g_string_append(prompt,
                    "- For global exposure correction, prefer the Exposure module "
                    "(`exposure.exposure`) over a tone mapper's exposure control. "
                    "Use a tone mapper's exposure only for small refinements of "
                    "the tone mapping.\n");

  const float exposure_bias
    = CLAMP(dt_conf_get_float(DT_ASSISTANT_CONF_EXPOSURE_BIAS), -2.0f, 2.0f);
  if(fabsf(exposure_bias) > 0.01f)
    g_string_append_printf(prompt,
                           "- The photographer prefers a base exposure bias of "
                           "%+.1f EV. The image statistics already include it in "
                           "suggested_exposure_ev when highlights are safe; do not "
                           "add it a second time. If highlights clip, keep exposure "
                           "neutral or reduce it. When exposure is allowed and "
                           "highlights are not clipping, include an "
                           "`exposure.exposure` change based on "
                           "suggested_exposure_ev unless another proposed change "
                           "already corrects the overall brightness. Propose at "
                           "most 4 changes so the exposure correction fits.\n",
                           exposure_bias);

  const char *tone_texture
    = dt_conf_get_string_const(DT_ASSISTANT_CONF_TONE_TEXTURE);
  if(!g_strcmp0(tone_texture, "toneequal"))
    g_string_append(prompt,
                    "- For local tone and texture work, prefer Tone Equalizer "
                    "(`toneequal`) when it is allowed. Use `contrast` only if "
                    "`toneequal` is not allowed, or if `contrast` is already "
                    "active, or if the request is specifically about multiscale "
                    "texture.\n");
  else if(g_strcmp0(tone_texture, "auto") != 0)
    g_string_append(prompt,
                    "- For local tone and texture work, prefer Tone & Texture "
                    "(`contrast`) when it is allowed. Use `toneequal` only if "
                    "`contrast` is not allowed, or if `toneequal` is already "
                    "active, or if the request is specifically about "
                    "luminance-band equalization.\n");

  const char *tone_mapper
    = dt_conf_get_string_const(DT_ASSISTANT_CONF_TONE_MAPPER);
  if(tone_mapper && tone_mapper[0] && g_strcmp0(tone_mapper, "auto")
     && dt_assistant_catalog_is_allowed(tone_mapper))
  {
    const dt_assistant_catalog_module_t *module
      = dt_assistant_catalog_get(tone_mapper);
    if(module)
      g_string_append_printf(prompt,
                             "- The photographer's preferred tone mapper is %s "
                             "(`%s`). When no tone mapper is active, use it "
                             "instead of another tone mapper. If another tone "
                             "mapper is already active, keep it and adjust it.\n",
                             module->name, module->operation);
  }

  const char *color = dt_conf_get_string_const(DT_ASSISTANT_CONF_COLOR);
  if(color && color[0] && g_strcmp0(color, "auto")
     && dt_assistant_catalog_is_allowed(color))
  {
    const dt_assistant_catalog_module_t *module = dt_assistant_catalog_get(color);
    if(module)
      g_string_append_printf(prompt,
                             "- For color work, prefer %s (`%s`) when it is "
                             "relevant and allowed. Other color modules may "
                             "still be used if the request needs them.\n",
                             module->name, module->operation);
  }

  const char *detail = dt_conf_get_string_const(DT_ASSISTANT_CONF_DETAIL);
  if(detail && detail[0] && g_strcmp0(detail, "auto")
     && dt_assistant_catalog_is_allowed(detail))
  {
    const dt_assistant_catalog_module_t *module = dt_assistant_catalog_get(detail);
    if(module)
      g_string_append_printf(prompt,
                             "- For detail work, prefer %s (`%s`) when it is "
                             "relevant and allowed. Use the other detail module "
                             "only if the request needs it.\n",
                             module->name, module->operation);
  }

  g_string_append(prompt,
                  "\n"
                  "JSON schema:\n"
                  "{\n"
                  "  \"summary\": \"short explanation\",\n"
                  "  \"changes\": [\n"
                  "    {\n"
                  "      \"operation\": \"module op\",\n"
                  "      \"param\": \"parameter name\",\n"
                  "      \"value\": 0.0,\n"
                  "      \"reason\": \"short reason\"\n"
                  "    }\n"
                  "  ]\n"
                  "}\n"
                  "\n"
                  "Available modules (only these are allowed):\n");

  GPtrArray *ops = dt_assistant_allowed_ops();
  for(guint i = 0; i < ops->len; i++)
  {
    const char *op = g_ptr_array_index(ops, i);
    const dt_assistant_catalog_module_t *module = dt_assistant_catalog_get(op);
    if(!module || !dt_iop_get_module_so(op)) continue;

    g_string_append_printf(prompt, "- %s (%s", module->operation, module->name);
    if(module->role && module->role[0])
      g_string_append_printf(prompt, ", role: %s", module->role);
    if(module->exclusive_role)
      g_string_append(prompt, ", exclusive tone mapper");
    g_string_append(prompt, ")");
    if(module->description && module->description[0])
      g_string_append_printf(prompt, ": %s", module->description);
    g_string_append_c(prompt, '\n');

    for(guint j = 0; j < module->params->len; j++)
    {
      dt_assistant_catalog_param_t *param = g_ptr_array_index(module->params, j);
      g_string_append_printf(prompt, "  - %s (max delta %.3g", param->name,
                             param->max_delta);
      dt_iop_module_so_t *so = dt_iop_get_module_so(module->operation);
      dt_introspection_field_t *field
        = (so && so->get_f) ? so->get_f(param->name) : NULL;
      if(field && field->header.type == DT_INTROSPECTION_TYPE_FLOAT)
        g_string_append_printf(prompt, ", range [%.3g, %.3g]",
                               field->Float.Min, field->Float.Max);
      g_string_append(prompt, ")");
      if(param->description && param->description[0])
        g_string_append_printf(prompt, ": %s", param->description);
      g_string_append_c(prompt, '\n');
    }
  }
  g_ptr_array_free(ops, TRUE);

  return g_string_free(prompt, FALSE);
}

static void _change_free(gpointer data)
{
  dt_assistant_change_t *change = data;
  if(!change) return;
  g_free(change->operation);
  g_free(change->param);
  g_free(change->reason);
  g_free(change);
}

void dt_assistant_plan_free(dt_assistant_plan_t *plan)
{
  if(!plan) return;
  g_free(plan->summary);
  if(plan->changes) g_ptr_array_free(plan->changes, TRUE);
  g_free(plan);
}

dt_assistant_plan_t *dt_assistant_plan_parse(const char *text, char **error)
{
  if(error) *error = NULL;
  if(!text || !text[0])
  {
    if(error) *error = g_strdup(_("assistant returned an empty response"));
    return NULL;
  }

  const char *start = strchr(text, '{');
  const char *end = strrchr(text, '}');
  if(!start || !end || end < start)
  {
    if(error) *error = g_strdup(_("assistant did not return a JSON object"));
    return NULL;
  }

  char *json = g_strndup(start, (gsize)(end - start + 1));
  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, json, -1, NULL))
  {
    g_free(json);
    g_object_unref(parser);
    if(error) *error = g_strdup(_("assistant returned invalid JSON"));
    return NULL;
  }
  g_free(json);

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    g_object_unref(parser);
    if(error) *error = g_strdup(_("assistant returned an unexpected response"));
    return NULL;
  }

  JsonObject *obj = json_node_get_object(root);
  dt_assistant_plan_t *plan = g_new0(dt_assistant_plan_t, 1);
  plan->changes = g_ptr_array_new_with_free_func(_change_free);

  if(json_object_has_member(obj, "summary"))
  {
    const char *summary = json_object_get_string_member(obj, "summary");
    if(summary) plan->summary = g_strdup(summary);
  }

  if(!json_object_has_member(obj, "changes")
     || !JSON_NODE_HOLDS_ARRAY(json_object_get_member(obj, "changes")))
  {
    g_object_unref(parser);
    dt_assistant_plan_free(plan);
    if(error) *error = g_strdup(_("assistant response has no changes array"));
    return NULL;
  }

  JsonArray *changes = json_object_get_array_member(obj, "changes");
  const guint length = json_array_get_length(changes);
  for(guint i = 0; i < length; i++)
  {
    JsonNode *node = json_array_get_element(changes, i);
    if(!node || !JSON_NODE_HOLDS_OBJECT(node))
    {
      g_object_unref(parser);
      dt_assistant_plan_free(plan);
      if(error) *error = g_strdup(_("assistant returned an invalid change"));
      return NULL;
    }

    JsonObject *change_obj = json_node_get_object(node);
    if(!json_object_has_member(change_obj, "operation"))
    {
      g_object_unref(parser);
      dt_assistant_plan_free(plan);
      if(error) *error = g_strdup(_("assistant returned an incomplete change"));
      return NULL;
    }

    const char *operation
      = json_object_get_string_member(change_obj, "operation");
    if(!operation)
    {
      g_object_unref(parser);
      dt_assistant_plan_free(plan);
      if(error) *error = g_strdup(_("assistant returned an invalid change"));
      return NULL;
    }

    const char *param = NULL;
    JsonNode *value_node = NULL;
    if(json_object_has_member(change_obj, "param")
       && json_object_has_member(change_obj, "value"))
    {
      param = json_object_get_string_member(change_obj, "param");
      value_node = json_object_get_member(change_obj, "value");
    }
    else if(json_object_has_member(change_obj, "params"))
    {
      JsonNode *params_node = json_object_get_member(change_obj, "params");
      if(params_node && JSON_NODE_HOLDS_OBJECT(params_node))
      {
        JsonObject *params_obj = json_node_get_object(params_node);
        GList *members = json_object_get_members(params_obj);
        if(g_list_length(members) == 1)
        {
          param = (const char *)members->data;
          value_node = json_object_get_member(params_obj, param);
        }
        g_list_free(members);
      }
    }

    if(!param || !value_node || !JSON_NODE_HOLDS_VALUE(value_node))
    {
      g_object_unref(parser);
      dt_assistant_plan_free(plan);
      if(error) *error = g_strdup(_("assistant returned an incomplete change"));
      return NULL;
    }

    const GType value_type = json_node_get_value_type(value_node);
    if(value_type != G_TYPE_DOUBLE && value_type != G_TYPE_INT64
       && value_type != G_TYPE_INT)
    {
      g_object_unref(parser);
      dt_assistant_plan_free(plan);
      if(error) *error = g_strdup(_("assistant returned a non-numeric value"));
      return NULL;
    }

    dt_assistant_change_t *change = g_new0(dt_assistant_change_t, 1);
    change->operation = g_strdup(operation);
    change->param = g_strdup(param);
    change->proposed = (float)json_node_get_double(value_node);
    change->apply = TRUE;
    change->will_enable = FALSE;
    if(json_object_has_member(change_obj, "reason"))
    {
      const char *reason = json_object_get_string_member(change_obj, "reason");
      if(reason) change->reason = g_strdup(reason);
    }
    g_ptr_array_add(plan->changes, change);
  }

  g_object_unref(parser);
  return plan;
}

guint dt_assistant_plan_sanitize(dt_assistant_plan_t *plan, char **error)
{
  if(error) *error = NULL;
  if(!plan || !plan->changes) return 0;

  GString *dropped = g_string_new(NULL);
  guint removed = 0;
  for(gint i = (gint)plan->changes->len - 1; i >= 0; i--)
  {
    dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
    gboolean valid = FALSE;
    if(dt_assistant_catalog_is_allowed(change->operation))
    {
      const dt_assistant_catalog_module_t *module
        = dt_assistant_catalog_get(change->operation);
      const dt_assistant_catalog_param_t *param
        = dt_assistant_catalog_param(module, change->param);
      dt_iop_module_so_t *so = NULL;
      dt_introspection_field_t *field = NULL;
      valid = module && param
              && _field_for_op(change->operation, change->param, &so, &field);
    }
    if(valid) continue;

    if(dropped->len) g_string_append(dropped, "; ");
    g_string_append_printf(dropped, "`%s.%s` is not an allowed slider",
                           change->operation, change->param);
    g_ptr_array_remove_index(plan->changes, i);
    removed++;
  }

  if(removed)
  {
    if(error) *error = g_string_free(dropped, FALSE);
    else g_string_free(dropped, TRUE);
  }
  else
    g_string_free(dropped, TRUE);
  return removed;
}

guint dt_assistant_plan_clamp_values(dt_assistant_plan_t *plan, char **note)
{
  if(note) *note = NULL;
  if(!plan || !plan->changes) return 0;

  GString *limited = g_string_new(NULL);
  guint adjusted = 0;
  for(gint i = (gint)plan->changes->len - 1; i >= 0; i--)
  {
    dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
    const dt_assistant_catalog_module_t *module
      = dt_assistant_catalog_get(change->operation);
    const dt_assistant_catalog_param_t *param
      = dt_assistant_catalog_param(module, change->param);
    dt_iop_module_so_t *so = NULL;
    dt_introspection_field_t *field = NULL;
    if(!module || !param
       || !_field_for_op(change->operation, change->param, &so, &field))
      continue;

    const float current
      = _current_value(dt_workflow_find_module(change->operation), field);
    if(!isfinite(current) || !isfinite(change->proposed))
    {
      g_ptr_array_remove_index(plan->changes, i);
      continue;
    }

    float proposed = CLAMP(change->proposed, field->Float.Min, field->Float.Max);
    proposed = CLAMP(proposed, current - param->max_delta,
                     current + param->max_delta);
    proposed = CLAMP(proposed, field->Float.Min, field->Float.Max);

    if(fabsf(proposed - change->proposed) > 1e-6f)
    {
      if(limited->len) g_string_append(limited, "; ");
      g_string_append_printf(limited,
                             "`%s.%s` limited from %.3g to %.3g "
                             "(max delta %.3g)",
                             change->operation, change->param,
                             change->proposed, proposed, param->max_delta);
      change->proposed = proposed;
      char *old_reason = change->reason;
      change->reason = old_reason
                         ? g_strdup_printf("%s (%s)", old_reason,
                                           _("safety-limited"))
                         : g_strdup(_("safety-limited"));
      g_free(old_reason);
      adjusted++;
    }

    if(fabsf(change->proposed - current) < 1e-4f)
      g_ptr_array_remove_index(plan->changes, i);
  }

  if(adjusted && note) *note = g_string_free(limited, FALSE);
  else g_string_free(limited, TRUE);
  return adjusted;
}

static gboolean _tone_mapper_enabled(const char *except_operation)
{
  dt_develop_t *dev = darktable.develop;
  if(!dev) return FALSE;
  for(GList *l = dev->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *module = l->data;
    if(!module || !module->enabled) continue;
    if(!g_strcmp0(module->op, except_operation)) continue;
    if(_is_tone_mapper_op(module->op)) return TRUE;
  }
  return FALSE;
}

gboolean dt_assistant_plan_validate(dt_assistant_plan_t *plan, char **error)
{
  if(error) *error = NULL;
  if(!plan)
  {
    if(error) *error = g_strdup(_("no plan to validate"));
    return FALSE;
  }

  if(plan->changes->len > 5)
  {
    if(error) *error = g_strdup(_("assistant proposed too many changes"));
    return FALSE;
  }

  GHashTable *seen = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  GHashTable *new_ops = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  gboolean ok = TRUE;

  for(guint i = 0; i < plan->changes->len && ok; i++)
  {
    dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
    change->will_enable = FALSE;

    if(!dt_assistant_catalog_is_allowed(change->operation))
    {
      if(error)
        *error = g_strdup_printf(_("module `%s` is not allowed in preferences"),
                                 change->operation);
      ok = FALSE;
      break;
    }

    const dt_assistant_catalog_module_t *catalog
      = dt_assistant_catalog_get(change->operation);
    const dt_assistant_catalog_param_t *param_info
      = dt_assistant_catalog_param(catalog, change->param);
    if(!catalog || !param_info)
    {
      if(error)
        *error = g_strdup_printf(_("`%s.%s` is not allowed"),
                                 change->operation, change->param);
      ok = FALSE;
      break;
    }

    char *key = g_strdup_printf("%s.%s", change->operation, change->param);
    if(g_hash_table_contains(seen, key))
    {
      if(error) *error = g_strdup_printf(_("`%s` is proposed twice"), key);
      g_free(key);
      ok = FALSE;
      break;
    }
    g_hash_table_add(seen, key);

    dt_iop_module_so_t *so = NULL;
    dt_introspection_field_t *field = NULL;
    if(!_field_for_op(change->operation, change->param, &so, &field))
    {
      if(error)
        *error = g_strdup_printf(_("module `%s` cannot expose `%s`"),
                                 change->operation, change->param);
      ok = FALSE;
      break;
    }

    dt_iop_module_t *module = dt_workflow_find_module(change->operation);
    const gboolean will_enable = !module || !module->enabled;
    if(will_enable && !catalog->addable)
    {
      if(error)
        *error = g_strdup_printf(_("module `%s` cannot be activated"),
                                 change->operation);
      ok = FALSE;
      break;
    }

    if(will_enable)
    {
      if(!g_hash_table_contains(new_ops, change->operation))
      {
        g_hash_table_add(new_ops, g_strdup(change->operation));
        if(g_hash_table_size(new_ops) > 2)
        {
          if(error) *error = g_strdup(_("assistant proposed too many new modules"));
          ok = FALSE;
          break;
        }
      }

      if(_is_tone_mapper_op(change->operation))
      {
        if(_tone_mapper_enabled(change->operation))
        {
          if(error)
            *error = g_strdup(_("another tone mapper is already active"));
          ok = FALSE;
          break;
        }

        const char *preferred = dt_assistant_preferred_tone_mapper();
        if(preferred && !dt_assistant_catalog_is_allowed(preferred)) preferred = NULL;
        if(preferred && g_strcmp0(preferred, change->operation))
        {
          if(error)
            *error = g_strdup_printf(_("preferred tone mapper is `%s`"), preferred);
          ok = FALSE;
          break;
        }
      }
    }

    const float current = _current_value(module, field);
    if(!isfinite(change->proposed))
    {
      if(error) *error = g_strdup(_("assistant proposed an invalid value"));
      ok = FALSE;
      break;
    }
    if(change->proposed < field->Float.Min || change->proposed > field->Float.Max)
    {
      if(error)
        *error = g_strdup_printf(_("`%s.%s` proposed %.3g is outside [%.3g, %.3g]"),
                                 change->operation, change->param,
                                 change->proposed,
                                 field->Float.Min, field->Float.Max);
      ok = FALSE;
      break;
    }
    if(fabsf(change->proposed - current) > param_info->max_delta + 1e-4f)
    {
      if(error)
      {
        GString *message = g_string_new(NULL);
        g_string_append_printf(
          message,
          _("`%s.%s` change is too large (current %.3g, proposed %.3g, "
            "max delta %.3g)"),
          change->operation, change->param, current, change->proposed,
          param_info->max_delta);
        const gboolean exposure_like
          = !g_strcmp0(change->param, "exposure_ev")
            || !g_strcmp0(change->param, "input_exposure");
        if(exposure_like && _is_tone_mapper_op(change->operation)
           && dt_assistant_catalog_is_allowed("exposure"))
          g_string_append(message,
                          _("; use `exposure.exposure` for a larger global "
                            "exposure change"));
        *error = g_string_free(message, FALSE);
      }
      ok = FALSE;
      break;
    }

    change->current = current;
    change->will_enable = will_enable;
  }

  g_hash_table_destroy(seen);
  g_hash_table_destroy(new_ops);
  return ok;
}

gboolean dt_assistant_plan_add_exposure_adjustment(dt_assistant_plan_t *plan,
                                                   const float exposure_ev)
{
  if(!plan || fabsf(exposure_ev) < 0.05f) return FALSE;
  if(!dt_assistant_catalog_is_allowed("exposure")) return FALSE;
  if(plan->changes->len >= 5) return FALSE;

  for(guint i = 0; i < plan->changes->len; i++)
  {
    const dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
    if(!g_strcmp0(change->param, "exposure_ev")
       || !g_strcmp0(change->param, "input_exposure"))
      return FALSE;
    if(!g_strcmp0(change->operation, "exposure")
       && !g_strcmp0(change->param, "exposure"))
      return FALSE;
  }

  const dt_assistant_catalog_module_t *catalog
    = dt_assistant_catalog_get("exposure");
  const dt_assistant_catalog_param_t *param
    = dt_assistant_catalog_param(catalog, "exposure");
  dt_iop_module_so_t *so = NULL;
  dt_introspection_field_t *field = NULL;
  if(!catalog || !param || !_field_for_op("exposure", "exposure", &so, &field))
    return FALSE;

  dt_iop_module_t *module = dt_workflow_find_module("exposure");
  const gboolean will_enable = !module || !module->enabled;
  if(will_enable)
  {
    guint new_modules = 0;
    for(guint i = 0; i < plan->changes->len; i++)
    {
      const dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
      if(change->will_enable) new_modules++;
    }
    if(new_modules >= 2) return FALSE;
  }

  const float current = _current_value(module, field);
  if(!isfinite(current)) return FALSE;
  float proposed = current + exposure_ev;
  proposed = CLAMP(proposed, current - param->max_delta,
                   current + param->max_delta);
  proposed = CLAMP(proposed, field->Float.Min, field->Float.Max);
  if(!isfinite(proposed) || fabsf(proposed - current) < 1e-4f) return FALSE;

  dt_assistant_change_t *change = g_new0(dt_assistant_change_t, 1);
  change->operation = g_strdup("exposure");
  change->param = g_strdup("exposure");
  change->current = current;
  change->proposed = proposed;
  change->reason = g_strdup(_("exposure bias preference"));
  change->apply = TRUE;
  change->will_enable = will_enable;
  g_ptr_array_add(plan->changes, change);
  return TRUE;
}

typedef struct
{
  char *operation;
  dt_imgid_t imgid;
  int snap_id;
  int history_end;
  gboolean was_loaded;
  gboolean was_in_alliop;
  gboolean was_enabled;
  int32_t params_size;
  void *params;
} dt_assistant_revert_t;

static void _revert_free(gpointer data)
{
  dt_assistant_revert_t *revert = data;
  if(!revert) return;
  g_free(revert->operation);
  g_free(revert->params);
  g_free(revert);
}

void dt_assistant_revert_free(GPtrArray *revert)
{
  if(!revert) return;
  if(revert->len > 0)
  {
    dt_assistant_revert_t *first = g_ptr_array_index(revert, 0);
    if(first && dt_is_valid_imgid(first->imgid) && first->snap_id > 0)
      dt_history_snapshot_clear(first->imgid, first->snap_id);
  }
  g_ptr_array_free(revert, TRUE);
}

typedef struct
{
  char *operation;
  dt_iop_module_t *module;
  gboolean was_loaded;
  gboolean was_in_alliop;
  gboolean was_enabled;
  int32_t params_size;
  void *params;
} _apply_module_t;

static void _apply_module_clear(_apply_module_t *item)
{
  if(!item) return;
  g_free(item->operation);
  g_free(item->params);
}

gboolean dt_assistant_apply(dt_assistant_plan_t *plan,
                            GPtrArray **revert,
                            char **error)
{
  if(error) *error = NULL;
  if(revert) *revert = NULL;
  if(!plan || !plan->changes)
  {
    if(error) *error = g_strdup(_("no plan to apply"));
    return FALSE;
  }
  if(!darktable.gui || !darktable.develop)
  {
    if(error) *error = g_strdup(_("assistant changes require the darkroom view"));
    return FALSE;
  }

  char *validate_error = NULL;
  if(!dt_assistant_plan_validate(plan, &validate_error))
  {
    if(error) *error = validate_error ? validate_error : g_strdup(_("invalid plan"));
    return FALSE;
  }
  g_free(validate_error);

  const dt_imgid_t imgid = darktable.develop->image_storage.id;
  int snap_id = 0;
  int history_end = 0;
  dt_history_snapshot_undo_create(imgid, &snap_id, &history_end);

  GArray *modules = g_array_new(FALSE, FALSE, sizeof(_apply_module_t));
  GHashTable *seen_ops = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  gboolean ok = TRUE;

  for(guint i = 0; i < plan->changes->len && ok; i++)
  {
    dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
    if(!change->apply) continue;
    if(g_hash_table_contains(seen_ops, change->operation)) continue;
    g_hash_table_add(seen_ops, g_strdup(change->operation));

    dt_iop_module_t *found = dt_iop_get_module(change->operation);
    dt_iop_module_t *existing = dt_workflow_find_module(change->operation);
    dt_iop_module_t *module = dt_workflow_find_or_load_module(change->operation);
    if(!module)
    {
      if(error)
        *error = g_strdup_printf(_("cannot load module `%s`"),
                                 change->operation);
      ok = FALSE;
      break;
    }

    _apply_module_t item = { 0 };
    item.operation = g_strdup(change->operation);
    item.module = module;
    item.was_loaded = found != NULL;
    item.was_in_alliop = existing != NULL && found == NULL;
    item.was_enabled = module->enabled;
    item.params_size = module->params_size;
    if(item.params_size > 0)
    {
      item.params = g_malloc(item.params_size);
      memcpy(item.params, module->params, item.params_size);
    }
    g_array_append_val(modules, item);
  }

  if(!ok)
  {
    if(snap_id > 0) dt_history_snapshot_clear(imgid, snap_id);
    for(guint i = 0; i < modules->len; i++)
    {
      _apply_module_t *item = &g_array_index(modules, _apply_module_t, i);
      if(!item->was_loaded) dt_dev_module_remove(darktable.develop, item->module);
      _apply_module_clear(item);
    }
    g_array_free(modules, TRUE);
    g_hash_table_destroy(seen_ops);
    return FALSE;
  }

  if(modules->len == 0)
  {
    if(snap_id > 0) dt_history_snapshot_clear(imgid, snap_id);
    g_array_free(modules, TRUE);
    g_hash_table_destroy(seen_ops);
    if(error) *error = g_strdup(_("no valid change is selected"));
    return FALSE;
  }

  GPtrArray *revert_items = g_ptr_array_new_with_free_func(_revert_free);
  dt_dev_undo_start_record(darktable.develop);

  for(guint i = 0; i < modules->len; i++)
  {
    _apply_module_t *item = &g_array_index(modules, _apply_module_t, i);
    if(!item->was_enabled)
      dt_iop_reload_defaults(item->module);

    for(guint j = 0; j < plan->changes->len; j++)
    {
      dt_assistant_change_t *change = g_ptr_array_index(plan->changes, j);
      if(!change->apply || g_strcmp0(change->operation, item->operation)) continue;

      dt_iop_module_so_t *so = NULL;
      dt_introspection_field_t *field = NULL;
      if(!_field_for_op(change->operation, change->param, &so, &field)) continue;
      float *value = (float *)((char *)item->module->params + field->header.offset);
      *value = change->proposed;
    }

    item->module->enabled = TRUE;
    dt_dev_add_history_item(darktable.develop, item->module, TRUE);
    if(item->module->gui_data) dt_iop_gui_update(item->module);

    dt_assistant_revert_t *entry = g_new0(dt_assistant_revert_t, 1);
    entry->operation = g_strdup(item->operation);
    entry->imgid = imgid;
    entry->snap_id = snap_id;
    entry->history_end = history_end;
    entry->was_loaded = item->was_loaded;
    entry->was_in_alliop = item->was_in_alliop;
    entry->was_enabled = item->was_enabled;
    entry->params_size = item->params_size;
    entry->params = item->params;
    item->params = NULL; // ownership moved to the revert entry
    g_ptr_array_add(revert_items, entry);
  }

  dt_dev_undo_end_record(darktable.develop);
  dt_control_queue_redraw_center();

  for(guint i = 0; i < modules->len; i++)
    _apply_module_clear(&g_array_index(modules, _apply_module_t, i));
  g_array_free(modules, TRUE);
  g_hash_table_destroy(seen_ops);

  if(revert) *revert = revert_items;
  else dt_assistant_revert_free(revert_items);
  return TRUE;
}

gboolean dt_assistant_revert(GPtrArray *revert, char **error)
{
  if(error) *error = NULL;
  if(!revert || revert->len == 0)
  {
    if(error) *error = g_strdup(_("nothing to revert"));
    return FALSE;
  }
  if(!darktable.gui || !darktable.develop)
  {
    if(error) *error = g_strdup(_("assistant changes require the darkroom view"));
    return FALSE;
  }

  dt_assistant_revert_t *first = g_ptr_array_index(revert, 0);
  if(!first || !dt_is_valid_imgid(first->imgid) || first->snap_id <= 0)
  {
    if(error) *error = g_strdup(_("revert snapshot is unavailable"));
    return FALSE;
  }

  const dt_imgid_t imgid = first->imgid;
  const int snap_id = first->snap_id;
  const int history_end = first->history_end;

  dt_history_snapshot_restore(imgid, snap_id, history_end);
  dt_history_snapshot_clear(imgid, snap_id);
  first->snap_id = 0;

  dt_dev_undo_start_record(darktable.develop);
  dt_dev_reload_history_items(darktable.develop);
  dt_dev_pixelpipe_rebuild(darktable.develop);
  darktable.develop->history_end = history_end;
  dt_dev_pop_history_items(darktable.develop, history_end);
  dt_ioppr_resync_modules_order(darktable.develop);
  dt_dev_modulegroups_set(darktable.develop,
                          dt_dev_modulegroups_get(darktable.develop));
  dt_image_update_final_size(imgid);
  dt_dev_write_history(darktable.develop);
  dt_dev_undo_end_record(darktable.develop);
  dt_control_queue_redraw_center();
  return TRUE;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

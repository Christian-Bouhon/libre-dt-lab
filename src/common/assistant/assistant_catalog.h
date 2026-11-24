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

#pragma once

#include <glib.h>

#define DT_ASSISTANT_CONF_TONE_TEXTURE "plugins/assistant/tone_texture"
#define DT_ASSISTANT_CONF_TONE_MAPPER  "plugins/assistant/tone_mapper"
#define DT_ASSISTANT_CONF_COLOR        "plugins/assistant/color"
#define DT_ASSISTANT_CONF_DETAIL       "plugins/assistant/detail"
#define DT_ASSISTANT_CONF_EXPOSURE_BIAS "plugins/assistant/exposure_bias"

typedef struct
{
  char *name;
  char *description;
  float max_delta;
  gboolean has_display;
  float factor;
  float offset;
  char *suffix;
  int decimals;
} dt_assistant_catalog_param_t;

typedef struct
{
  char *operation;
  char *name;
  char *description;
  char *role;
  gboolean exclusive_role;
  gboolean addable;
  GPtrArray *params;
} dt_assistant_catalog_module_t;

const GPtrArray *dt_assistant_catalog_modules(void);
const dt_assistant_catalog_module_t *dt_assistant_catalog_get(const char *operation);
const dt_assistant_catalog_param_t *dt_assistant_catalog_param(
  const dt_assistant_catalog_module_t *module,
  const char *name);
char *dt_assistant_catalog_format_value(
  const dt_assistant_catalog_module_t *module,
  const char *param,
  float value);
gboolean dt_assistant_catalog_is_tone_mapper(const dt_assistant_catalog_module_t *module);
const char *dt_assistant_preferred_tone_mapper(void);

char *dt_assistant_catalog_default_allowed(void);
char *dt_assistant_allowed_string(void);
void dt_assistant_allowed_set_string(const char *list);
GPtrArray *dt_assistant_allowed_ops(void);
gboolean dt_assistant_catalog_is_allowed(const char *operation);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

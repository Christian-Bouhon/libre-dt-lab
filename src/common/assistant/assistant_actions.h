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

typedef struct
{
  char *operation;
  char *param;
  float current;
  float proposed;
  char *reason;
  gboolean apply;
  gboolean will_enable;
} dt_assistant_change_t;

typedef struct
{
  char *summary;
  GPtrArray *changes;
} dt_assistant_plan_t;

char *dt_assistant_system_prompt(void);
gboolean dt_assistant_module_float(const char *operation,
                                   const char *param,
                                   float *value);

dt_assistant_plan_t *dt_assistant_plan_parse(const char *json, char **error);
guint dt_assistant_plan_sanitize(dt_assistant_plan_t *plan, char **error);
guint dt_assistant_plan_clamp_values(dt_assistant_plan_t *plan, char **note);
void dt_assistant_plan_free(dt_assistant_plan_t *plan);
gboolean dt_assistant_plan_validate(dt_assistant_plan_t *plan, char **error);
gboolean dt_assistant_plan_add_exposure_adjustment(dt_assistant_plan_t *plan,
                                                   float exposure_ev);

gboolean dt_assistant_apply(dt_assistant_plan_t *plan,
                            GPtrArray **revert,
                            char **error);
gboolean dt_assistant_revert(GPtrArray *revert, char **error);
void dt_assistant_revert_free(GPtrArray *revert);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

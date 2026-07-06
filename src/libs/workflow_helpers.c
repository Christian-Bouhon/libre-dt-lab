/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers
    Libre DT-lab Edition (C) 2026 Christian Bouhon.

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

#include "common/darktable.h"
#include "common/iop_order.h"
#include "control/conf.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/pixelpipe_hb.h"
#include "libs/workflow_helpers.h"

static const char *_tm_ops[] =
{
  "filmicrgb", "sigmoid", "agx", "basecurve", "3dcf", "aces20", NULL
};

static dt_iop_module_t *_find_module(const char *op)
{
  dt_iop_module_t *mod = dt_iop_get_module(op);
  if(mod) return mod;

  dt_develop_t *dev = darktable.develop;
  if(!dev) return NULL;

  for(GList *l = dev->alliop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(!strcmp(m->op, op))
      return m;
  }
  return NULL;
}

static dt_iop_module_t *_load_module(const char *op)
{
  dt_develop_t *dev = darktable.develop;
  if(!dev) return NULL;

  dt_iop_module_so_t *so = dt_iop_get_module_so(op);
  if(!so) return NULL;

  dt_iop_module_t *module = calloc(1, sizeof(dt_iop_module_t));
  if(dt_iop_load_module(module, so, dev))
  {
    free(module);
    return NULL;
  }

  module->instance = dev->iop_instance++;
  module->multi_name[0] = '\0';
  dt_iop_update_multi_priority(module, 0);

  if(!dt_iop_is_hidden(module))
    module->gui_init(module);

  dev->iop = g_list_insert_sorted(dev->iop, module, dt_sort_iop_by_order);
  dt_ioppr_insert_module_instance(dev, module);
  dt_ioppr_resync_modules_order(dev);

  if(!dt_iop_is_hidden(module))
  {
    dt_iop_gui_set_expander(module);
    dt_iop_gui_set_expanded(module, TRUE, FALSE);
    dt_iop_gui_update_blending(module);
  }

  return module;
}

void dt_workflow_selector_set(const int selected)
{
  dt_develop_t *dev = darktable.develop;
  if(!dev) return;

  const char *op = NULL;
  switch(selected)
  {
    case 0: op = NULL; break;
    case 1: op = "filmicrgb"; break;
    case 2: op = "sigmoid"; break;
    case 3: op = "agx"; break;
    case 4: op = "basecurve"; break;
    case 5: op = "3dcf"; break;
    case 6: op = "aces20"; break;
    default: return;
  }

  // disable all tone mapper modules
  for(int i = 0; _tm_ops[i]; i++)
  {
    dt_iop_module_t *mod = _find_module(_tm_ops[i]);
    if(mod)
    {
      dt_iop_gui_set_expanded(mod, FALSE, FALSE);
      if(mod->enabled)
      {
        mod->enabled = FALSE;
        dt_dev_add_history_item(dev, mod, FALSE);
      }
    }
  }

  if(selected == 0)
  {
    dt_dev_pixelpipe_rebuild(dev);
    return;
  }

  // find or load the selected module
  dt_iop_module_t *mod = _find_module(op);
  if(!mod)
  {
    mod = _load_module(op);
  }
  if(!mod) return;

  // move from alliop to iop if needed
  if(!g_list_find(dev->iop, mod))
  {
    dev->alliop = g_list_remove(dev->alliop, mod);
    dev->iop = g_list_insert_sorted(dev->iop, mod, dt_sort_iop_by_order);
    dt_ioppr_insert_module_instance(dev, mod);
    dt_ioppr_resync_modules_order(dev);

    if(!dt_iop_is_hidden(mod) && !mod->expander)
    {
      dt_iop_gui_set_expander(mod);
      dt_iop_gui_set_expanded(mod, TRUE, FALSE);
      dt_iop_gui_update_blending(mod);
    }
  }

  mod->enabled = TRUE;
  dt_dev_add_history_item(dev, mod, TRUE);
  dt_iop_gui_set_expanded(mod, TRUE, FALSE);
  dt_dev_modulegroups_switch(dev, mod);
  dt_dev_pixelpipe_rebuild(dev);
}

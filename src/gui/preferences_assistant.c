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

#include "gui/preferences_assistant.h"

#include "bauhaus/bauhaus.h"
#include "common/assistant/assistant_catalog.h"
#include "common/assistant/assistant_provider.h"
#include "common/darktable.h"
#include "control/conf.h"
#include "gui/gtk.h"

#include <gtk/gtk.h>
#include <string.h>

typedef struct
{
  GtkWidget *dialog;
  GtkWidget *provider;
  GtkWidget *provider_note;
  GtkWidget *endpoint;
  GtkWidget *model;
  GtkWidget *model_status;
  GtkWidget *detect;
  GtkWidget *allow_cloud;
  GtkWidget *key;
  GtkWidget *key_status;
  GtkWidget *store_key;
  GtkWidget *clear_key;
  GtkWidget *key_box;
  GtkWidget *send_image;
  GtkWidget *modules_view;
  GtkListStore *modules_store;
  GtkWidget *module_combo;
  GtkWidget *module_add;
  GtkWidget *module_remove;
  GtkWidget *module_reset;
  GtkWidget *modules_status;
  GtkWidget *exposure_bias;
  GtkWidget *tone_texture;
  GtkWidget *tone_mapper;
  GtkWidget *color;
  GtkWidget *detail;
  gboolean populating;
} dt_prefs_assistant_data_t;

enum
{
  MOD_NAME,
  MOD_ROLE,
  MOD_OP,
  MOD_COLS
};

static GtkEntry *_model_entry(dt_prefs_assistant_data_t *data)
{
  return GTK_ENTRY(gtk_bin_get_child(GTK_BIN(data->model)));
}

static const char *_model_text(dt_prefs_assistant_data_t *data)
{
  return gtk_entry_get_text(_model_entry(data));
}

static void _set_model_text(dt_prefs_assistant_data_t *data, const char *text)
{
  data->populating = TRUE;
  gtk_entry_set_text(_model_entry(data), text ? text : "");
  data->populating = FALSE;
}

static gboolean _models_contain(GPtrArray *models, const char *name)
{
  if(!models || !name) return FALSE;
  for(guint i = 0; i < models->len; i++)
    if(!g_strcmp0(g_ptr_array_index(models, i), name)) return TRUE;
  return FALSE;
}

static gboolean _is_ollama_cloud_model(const char *name)
{
  return name
         && (g_str_has_suffix(name, "-cloud") || g_str_has_suffix(name, ":cloud"));
}

static gboolean _store_has_op(dt_prefs_assistant_data_t *data, const char *operation)
{
  GtkTreeIter iter;
  gboolean valid = gtk_tree_model_get_iter_first(GTK_TREE_MODEL(data->modules_store), &iter);
  while(valid)
  {
    char *row_op = NULL;
    gtk_tree_model_get(GTK_TREE_MODEL(data->modules_store), &iter, MOD_OP, &row_op, -1);
    const gboolean found = !g_strcmp0(row_op, operation);
    g_free(row_op);
    if(found) return TRUE;
    valid = gtk_tree_model_iter_next(GTK_TREE_MODEL(data->modules_store), &iter);
  }
  return FALSE;
}

static const dt_assistant_catalog_module_t *_catalog_by_name(const char *name)
{
  const GPtrArray *modules = dt_assistant_catalog_modules();
  for(guint i = 0; i < modules->len; i++)
  {
    dt_assistant_catalog_module_t *module = g_ptr_array_index(modules, i);
    if(!g_strcmp0(module->name, name)) return module;
  }
  return NULL;
}

static void _refresh_available_modules(dt_prefs_assistant_data_t *data)
{
  gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(data->module_combo));
  const GPtrArray *modules = dt_assistant_catalog_modules();
  int available = 0;
  for(guint i = 0; i < modules->len; i++)
  {
    dt_assistant_catalog_module_t *module = g_ptr_array_index(modules, i);
    if(_store_has_op(data, module->operation)) continue;
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->module_combo),
                                   module->name);
    available++;
  }
  gtk_widget_set_sensitive(data->module_add, available > 0);
  if(available > 0)
    gtk_combo_box_set_active(GTK_COMBO_BOX(data->module_combo), 0);

  const int count = gtk_tree_model_iter_n_children(GTK_TREE_MODEL(data->modules_store), NULL);
  char *status = g_strdup_printf(_("%d module(s) allowed"), count);
  gtk_label_set_text(GTK_LABEL(data->modules_status), status);
  g_free(status);
}

static void _save_modules_from_store(dt_prefs_assistant_data_t *data)
{
  GString *list = g_string_new(NULL);
  GtkTreeIter iter;
  gboolean valid = gtk_tree_model_get_iter_first(GTK_TREE_MODEL(data->modules_store), &iter);
  while(valid)
  {
    char *op = NULL;
    gtk_tree_model_get(GTK_TREE_MODEL(data->modules_store), &iter, MOD_OP, &op, -1);
    if(op && op[0])
    {
      if(list->len > 0) g_string_append_c(list, '|');
      g_string_append(list, op);
    }
    g_free(op);
    valid = gtk_tree_model_iter_next(GTK_TREE_MODEL(data->modules_store), &iter);
  }
  dt_assistant_allowed_set_string(list->len > 0 ? list->str : "none");
  g_string_free(list, TRUE);
}

static void _populate_modules_store(dt_prefs_assistant_data_t *data)
{
  gtk_list_store_clear(data->modules_store);
  GPtrArray *ops = dt_assistant_allowed_ops();
  for(guint i = 0; i < ops->len; i++)
  {
    const char *op = g_ptr_array_index(ops, i);
    const dt_assistant_catalog_module_t *module = dt_assistant_catalog_get(op);
    if(!module) continue;
    GtkTreeIter iter;
    gtk_list_store_append(data->modules_store, &iter);
    gtk_list_store_set(data->modules_store, &iter,
                       MOD_NAME, module->name,
                       MOD_ROLE, module->role ? module->role : "",
                       MOD_OP, module->operation,
                       -1);
  }
  g_ptr_array_free(ops, TRUE);
  _refresh_available_modules(data);
}

static void _add_module_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = user_data;
  char *name = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(data->module_combo));
  if(!name) return;
  const dt_assistant_catalog_module_t *module = _catalog_by_name(name);
  g_free(name);
  if(!module || _store_has_op(data, module->operation)) return;

  GtkTreeIter iter;
  gtk_list_store_append(data->modules_store, &iter);
  gtk_list_store_set(data->modules_store, &iter,
                     MOD_NAME, module->name,
                     MOD_ROLE, module->role ? module->role : "",
                     MOD_OP, module->operation,
                     -1);
  _save_modules_from_store(data);
  _refresh_available_modules(data);
}

static void _remove_module_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = user_data;
  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(data->modules_view));
  GtkTreeModel *model = NULL;
  GtkTreeIter iter;
  if(gtk_tree_selection_get_selected(selection, &model, &iter))
  {
    gtk_list_store_remove(data->modules_store, &iter);
    _save_modules_from_store(data);
    _refresh_available_modules(data);
  }
}

static void _reset_modules_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = user_data;
  char *defaults = dt_assistant_catalog_default_allowed();
  dt_assistant_allowed_set_string(defaults);
  g_free(defaults);
  _populate_modules_store(data);
}

static void _module_selection_changed(GtkTreeSelection *selection, gpointer user_data)
{
  dt_prefs_assistant_data_t *data = user_data;
  gtk_widget_set_sensitive(data->module_remove,
                           gtk_tree_selection_get_selected(selection, NULL, NULL));
}

typedef struct
{
  const char *conf_key;
  const char *const *values;
  int count;
} dt_role_pref_t;

static const char *const _tone_mapper_values[] = {
  "auto", "3dcf", "aces20", "agx", "filmicrgb", "sigmoid"
};
static const char *const _color_values[] = {
  "auto", "colorbalancergb", "colorharmonizer", "primaries"
};
static const char *const _detail_values[] = {
  "auto", "diffuse", "sharpen"
};

static const dt_role_pref_t _tone_mapper_pref = {
  DT_ASSISTANT_CONF_TONE_MAPPER, _tone_mapper_values, 6
};
static const dt_role_pref_t _color_pref = {
  DT_ASSISTANT_CONF_COLOR, _color_values, 4
};
static const dt_role_pref_t _detail_pref = {
  DT_ASSISTANT_CONF_DETAIL, _detail_values, 3
};

static void _role_pref_changed(GtkWidget *widget, gpointer user_data)
{
  const dt_role_pref_t *pref = user_data;
  int index = gtk_combo_box_get_active(GTK_COMBO_BOX(widget));
  if(index < 0 || index >= pref->count) index = 0;
  dt_conf_set_string(pref->conf_key, pref->values[index]);
}

static void _role_combo_set(GtkComboBoxText *combo,
                            const char *const *values,
                            int count,
                            const char *conf_key)
{
  const char *value = dt_conf_get_string_const(conf_key);
  int index = 0;
  for(int i = 0; i < count; i++)
  {
    if(!g_strcmp0(value, values[i]))
    {
      index = i;
      break;
    }
  }
  gtk_combo_box_set_active(GTK_COMBO_BOX(combo), index);
}

static void _exposure_bias_changed(GtkSpinButton *spin, gpointer user_data)
{
  (void)user_data;
  dt_conf_set_float(DT_ASSISTANT_CONF_EXPOSURE_BIAS,
                    gtk_spin_button_get_value(spin));
}

static void _tone_texture_changed(GtkWidget *widget, gpointer user_data)
{
  (void)user_data;
  const int index = gtk_combo_box_get_active(GTK_COMBO_BOX(widget));
  const char *value = index == 1 ? "toneequal" : index == 2 ? "auto" : "contrast";
  dt_conf_set_string(DT_ASSISTANT_CONF_TONE_TEXTURE, value);
}

static dt_assistant_provider_t _selected_provider(
  dt_prefs_assistant_data_t *data)
{
  return dt_assistant_provider_from_index(
    dt_bauhaus_combobox_get(data->provider));
}

static int _provider_combo_index(dt_assistant_provider_t provider)
{
  for(int i = 0; i < dt_assistant_provider_count(); i++)
    if(dt_assistant_provider_from_index(i) == provider) return i;
  return 0;
}

static const char *_provider_setting(dt_assistant_provider_t provider,
                                     const char *name)
{
  char *key = dt_assistant_provider_conf_key(provider, name);
  const char *value = dt_conf_get_string_const(key);
  g_free(key);
  return value;
}

static void _provider_setting_set(dt_assistant_provider_t provider,
                                  const char *name,
                                  const char *value)
{
  char *key = dt_assistant_provider_conf_key(provider, name);
  dt_conf_set_string(key, value ? value : "");
  g_free(key);
}

static void _endpoint_changed(GtkEditable *editable, gpointer user_data)
{
  (void)editable;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  _provider_setting_set(_selected_provider(data), "endpoint",
                        gtk_entry_get_text(GTK_ENTRY(data->endpoint)));
  gtk_label_set_text(GTK_LABEL(data->model_status),
                     _("Endpoint changed. Click detect to list models, or type a model."));
}

static void _model_changed(GtkEditable *editable, gpointer user_data)
{
  (void)editable;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  if(data->populating) return;
  _provider_setting_set(_selected_provider(data), "model", _model_text(data));
}

static void _update_provider_note(dt_prefs_assistant_data_t *data)
{
  const dt_assistant_provider_t provider = _selected_provider(data);
  if(provider == DT_ASSISTANT_PROVIDER_OLLAMA)
    gtk_label_set_text(GTK_LABEL(data->provider_note),
                       _("Ollama runs locally by default. Cloud models are opt-in. "
                         "The optional preview image stays local unless a cloud model is selected."));
  else if(dt_assistant_provider_is_cloud(provider))
    gtk_label_set_text(
      GTK_LABEL(data->provider_note),
      dt_assistant_provider_supports_vision(provider)
        ? _("Cloud provider: text is sent; a 768 px preview image is sent only "
            "if explicitly enabled, without metadata.")
        : _("Cloud provider: text is sent to the provider; images are never sent."));
  else if(provider == DT_ASSISTANT_PROVIDER_LOCAL_OPENAI)
    gtk_label_set_text(GTK_LABEL(data->provider_note),
                       _("Local server endpoint; the optional preview image stays "
                         "on this machine if the endpoint is local."));
  else
    gtk_label_set_text(GTK_LABEL(data->provider_note),
                       _("Custom endpoint: text and the optional preview image "
                         "go to this endpoint; check that you trust it."));
}

static void _update_key_widgets(dt_prefs_assistant_data_t *data)
{
  const dt_assistant_provider_t provider = _selected_provider(data);
  const gboolean needs_key = dt_assistant_provider_needs_key(provider);
  const gboolean ollama = provider == DT_ASSISTANT_PROVIDER_OLLAMA;

  gtk_widget_set_sensitive(data->key_box, !ollama);
  gtk_widget_set_sensitive(data->store_key, !ollama);
  gtk_widget_set_sensitive(data->detect, TRUE);
  gtk_widget_set_sensitive(
    data->send_image, dt_assistant_provider_supports_vision(provider));

  if(ollama)
    gtk_label_set_text(GTK_LABEL(data->key_status),
                       _("not required for Ollama"));
  else if(dt_assistant_has_stored_key(provider))
    gtk_label_set_text(GTK_LABEL(data->key_status), _("key stored securely"));
  else if(dt_assistant_has_runtime_key(provider))
    gtk_label_set_text(GTK_LABEL(data->key_status),
                       _("key kept for this session only"));
  else if(!dt_assistant_secure_storage_available())
    gtk_label_set_text(GTK_LABEL(data->key_status),
                       _("no key stored; secure storage unavailable (session only)"));
  else if(needs_key)
    gtk_label_set_text(GTK_LABEL(data->key_status), _("no key stored"));
  else
    gtk_label_set_text(GTK_LABEL(data->key_status),
                       _("no key stored (optional for this provider)"));
}

static void _provider_changed(GtkWidget *widget, gpointer user_data)
{
  (void)widget;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  const dt_assistant_provider_t provider = _selected_provider(data);
  dt_conf_set_string(DT_ASSISTANT_CONF_PROVIDER,
                     dt_assistant_provider_to_string(provider));

  char *endpoint = NULL;
  char *model = NULL;
  dt_assistant_provider_defaults(provider, &endpoint, &model);
  const char *saved_endpoint = _provider_setting(provider, "endpoint");
  const char *saved_model = _provider_setting(provider, "model");

  if(provider == DT_ASSISTANT_PROVIDER_CUSTOM)
  {
    const char *legacy_endpoint
      = dt_conf_get_string_const(DT_ASSISTANT_CONF_ENDPOINT);
    const char *legacy_model = dt_conf_get_string_const(DT_ASSISTANT_CONF_MODEL);
    if(!saved_endpoint || !saved_endpoint[0]) saved_endpoint = legacy_endpoint;
    if(!saved_model || !saved_model[0]) saved_model = legacy_model;
  }

  data->populating = TRUE;
  gtk_entry_set_text(GTK_ENTRY(data->endpoint),
                     saved_endpoint && saved_endpoint[0] ? saved_endpoint : endpoint);
  gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(data->model));
  gtk_entry_set_text(_model_entry(data),
                     saved_model && saved_model[0] ? saved_model : model);
  gtk_entry_set_text(GTK_ENTRY(data->key), "");
  data->populating = FALSE;
  g_free(endpoint);
  g_free(model);

  gtk_entry_set_placeholder_text(
    _model_entry(data),
    provider == DT_ASSISTANT_PROVIDER_OLLAMA ? _("e.g. llama3.1:8b")
                                             : _("detect or type a model"));

  if(provider == DT_ASSISTANT_PROVIDER_OLLAMA)
    gtk_label_set_text(GTK_LABEL(data->model_status),
                       _("Click detect to list installed Ollama models."));
  else
    gtk_label_set_text(GTK_LABEL(data->model_status),
                       _("Click detect to list models for this provider, or type a model."));

  gtk_widget_set_sensitive(data->allow_cloud,
                           provider == DT_ASSISTANT_PROVIDER_OLLAMA);
  _update_provider_note(data);
  _update_key_widgets(data);
}

static void _send_image_toggled(GtkToggleButton *button, gpointer user_data)
{
  (void)user_data;
  dt_conf_set_bool(DT_ASSISTANT_CONF_SEND_IMAGE,
                   gtk_toggle_button_get_active(button));
}

static void _allow_cloud_toggled(GtkToggleButton *button, gpointer user_data)
{
  (void)user_data;
  dt_conf_set_bool(DT_ASSISTANT_CONF_ALLOW_CLOUD,
                   gtk_toggle_button_get_active(button));
}

static void _store_typed_key(dt_prefs_assistant_data_t *data,
                             dt_assistant_provider_t provider)
{
  const char *typed_key = gtk_entry_get_text(GTK_ENTRY(data->key));
  if(!typed_key || !typed_key[0]) return;

  char *error = NULL;
  dt_assistant_store_key(provider, typed_key, &error);
  gtk_entry_set_text(GTK_ENTRY(data->key), "");
  _update_key_widgets(data);
  if(error)
  {
    gtk_label_set_text(GTK_LABEL(data->key_status), error);
    g_free(error);
  }
}

static void _detect_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  const dt_assistant_provider_t provider = _selected_provider(data);
  _store_typed_key(data, provider);

  const char *endpoint = gtk_entry_get_text(GTK_ENTRY(data->endpoint));
  const gboolean include_cloud
    = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(data->allow_cloud));

  char *api_key = NULL;
  dt_assistant_config_t config = dt_assistant_config_load();
  if(config.api_key && config.api_key[0]) api_key = g_strdup(config.api_key);
  dt_assistant_config_clear(&config);

  char *error = NULL;
  GPtrArray *models
    = dt_assistant_list_models(provider, endpoint, api_key, include_cloud, &error);
  g_free(api_key);

  if(!models)
  {
    const char *message = error ? error : _("Could not list models.");
    gtk_label_set_text(GTK_LABEL(data->model_status), message);
    GtkWidget *dlg
      = gtk_message_dialog_new(GTK_WINDOW(data->dialog),
                               GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                               GTK_MESSAGE_WARNING, GTK_BUTTONS_OK,
                               "%s", message);
    gtk_dialog_run(GTK_DIALOG(dlg));
    gtk_widget_destroy(dlg);
    g_free(error);
    return;
  }

  char *current = g_strdup(_model_text(data));

  data->populating = TRUE;
  gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(data->model));
  for(guint i = 0; i < models->len; i++)
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->model),
                                   (const char *)g_ptr_array_index(models, i));
  data->populating = FALSE;

  if(models->len == 0)
  {
    gtk_label_set_text(GTK_LABEL(data->model_status),
                       _("No model returned by this provider."));
  }
  else
  {
    const gboolean current_listed
      = current && current[0] && _models_contain(models, current);
    if(current && current[0] && !current_listed)
    {
      data->populating = TRUE;
      gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->model), current);
      data->populating = FALSE;
    }

    const char *chosen = NULL;
    if(current && current[0])
      chosen = current;
    else if(provider == DT_ASSISTANT_PROVIDER_OLLAMA)
    {
      for(guint i = 0; i < models->len; i++)
      {
        const char *name = g_ptr_array_index(models, i);
        if(!_is_ollama_cloud_model(name))
        {
          chosen = name;
          break;
        }
      }
    }
    if(!chosen) chosen = (const char *)g_ptr_array_index(models, 0);
    _set_model_text(data, chosen);
    _provider_setting_set(provider, "model", chosen);

    if(current && current[0] && !current_listed)
    {
      char *message
        = g_strdup_printf(_("Detected %u model(s). Kept current model '%s' "
                            "because it is not in the provider list."),
                          models->len, chosen);
      gtk_label_set_text(GTK_LABEL(data->model_status), message);
      g_free(message);
    }
    else
    {
      char *message = g_strdup_printf(_("Detected %u model(s). Selected '%s'."),
                                      models->len, chosen);
      gtk_label_set_text(GTK_LABEL(data->model_status), message);
      g_free(message);
    }
  }

  _update_key_widgets(data);
  g_free(current);
  g_ptr_array_free(models, TRUE);
}

static void _store_key_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  const dt_assistant_provider_t provider = _selected_provider(data);
  const char *key = gtk_entry_get_text(GTK_ENTRY(data->key));

  char *error = NULL;
  dt_assistant_store_key(provider, key, &error);
  gtk_entry_set_text(GTK_ENTRY(data->key), "");
  _update_key_widgets(data);

  if(error)
  {
    gtk_label_set_text(GTK_LABEL(data->key_status), error);
    GtkWidget *dlg
      = gtk_message_dialog_new(GTK_WINDOW(data->dialog),
                               GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                               GTK_MESSAGE_WARNING, GTK_BUTTONS_OK, "%s", error);
    gtk_dialog_run(GTK_DIALOG(dlg));
    gtk_widget_destroy(dlg);
    g_free(error);
  }
}

static void _clear_key_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  const dt_assistant_provider_t provider = _selected_provider(data);
  char *error = NULL;
  dt_assistant_store_key(provider, "", &error);
  gtk_entry_set_text(GTK_ENTRY(data->key), "");
  _update_key_widgets(data);
  if(error)
  {
    gtk_label_set_text(GTK_LABEL(data->key_status), error);
    g_free(error);
  }
}

static void _verify_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_prefs_assistant_data_t *data = (dt_prefs_assistant_data_t *)user_data;
  const dt_assistant_provider_t provider = _selected_provider(data);
  _store_typed_key(data, provider);

  dt_assistant_config_t config = dt_assistant_config_load();
  char *reason = NULL;
  gboolean ready = dt_assistant_config_is_ready(&config, &reason);
  char *models_error = NULL;
  GPtrArray *models = NULL;

  if(ready)
  {
    models = dt_assistant_list_models(
      config.provider, config.endpoint, config.api_key,
      dt_conf_get_bool(DT_ASSISTANT_CONF_ALLOW_CLOUD), &models_error);
    if(!models)
    {
      ready = FALSE;
      g_free(reason);
      reason = models_error ? g_strdup(models_error) : g_strdup(_("could not list models"));
    }
  }
  if(models) g_ptr_array_free(models, TRUE);
  g_free(models_error);
  dt_assistant_config_clear(&config);

  GtkWidget *dlg
    = gtk_message_dialog_new(GTK_WINDOW(data->dialog),
                             GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                             ready ? GTK_MESSAGE_INFO : GTK_MESSAGE_WARNING,
                             GTK_BUTTONS_OK,
                             "%s",
                             ready ? _("Configuration and API key look valid.\n"
                                       "Model listing succeeded.")
                                   : (reason ? reason : _("Configuration is incomplete.")));
  gtk_dialog_run(GTK_DIALOG(dlg));
  gtk_widget_destroy(dlg);
  g_free(reason);
}

void init_tab_assistant(GtkWidget *dialog, GtkWidget *stack)
{
  dt_prefs_assistant_data_t *data = g_new0(dt_prefs_assistant_data_t, 1);
  data->dialog = dialog;

  GtkWidget *main_box = dt_gui_vbox();
  g_object_set_data_full(G_OBJECT(main_box), "assistant-prefs", data, g_free);
  GtkWidget *grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), DT_PIXEL_APPLY_DPI(4));
  gtk_grid_set_column_spacing(GTK_GRID(grid), DT_PIXEL_APPLY_DPI(8));
  dt_gui_box_add(main_box, grid);

  int row = 0;
  GtkWidget *section = gtk_label_new(_("AI assistant"));
  GtkWidget *section_box = dt_gui_hbox(section);
  gtk_widget_set_name(section_box, "pref_section");
  gtk_grid_attach(GTK_GRID(grid), section_box, 0, row++, 3, 1);

  GtkWidget *provider_label = gtk_label_new(_("API provider"));
  gtk_widget_set_halign(provider_label, GTK_ALIGN_START);
  data->provider = dt_bauhaus_combobox_new(NULL);
  for(int i = 0; i < dt_assistant_provider_count(); i++)
    dt_bauhaus_combobox_add(
      data->provider,
      _(dt_assistant_provider_label(dt_assistant_provider_from_index(i))));
  gtk_grid_attach(GTK_GRID(grid), provider_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->provider, 1, row++, 2, 1);

  data->provider_note = gtk_label_new("");
  gtk_label_set_line_wrap(GTK_LABEL(data->provider_note), TRUE);
  gtk_label_set_xalign(GTK_LABEL(data->provider_note), 0.0f);
  gtk_grid_attach(GTK_GRID(grid), data->provider_note, 1, row++, 2, 1);

  GtkWidget *endpoint_label = gtk_label_new(_("endpoint"));
  gtk_widget_set_halign(endpoint_label, GTK_ALIGN_START);
  data->endpoint = gtk_entry_new();
  gtk_grid_attach(GTK_GRID(grid), endpoint_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->endpoint, 1, row++, 2, 1);

  GtkWidget *model_label = gtk_label_new(_("model"));
  gtk_widget_set_halign(model_label, GTK_ALIGN_START);
  data->model = gtk_combo_box_text_new_with_entry();
  gtk_widget_set_hexpand(data->model, TRUE);
  gtk_entry_set_placeholder_text(_model_entry(data), _("e.g. llama3.1:8b"));
  data->detect = gtk_button_new_with_label(_("detect"));
  GtkWidget *model_box = dt_gui_hbox(data->model, data->detect);
  gtk_grid_attach(GTK_GRID(grid), model_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), model_box, 1, row++, 2, 1);

  data->model_status = gtk_label_new("");
  gtk_label_set_line_wrap(GTK_LABEL(data->model_status), TRUE);
  gtk_label_set_xalign(GTK_LABEL(data->model_status), 0.0f);
  gtk_grid_attach(GTK_GRID(grid), data->model_status, 1, row++, 2, 1);

  data->allow_cloud
    = gtk_check_button_new_with_label(_("include Ollama cloud models"));
  gtk_widget_set_tooltip_text(
    data->allow_cloud,
    _("Cloud models are listed by detect. Text, and the optional preview image "
      "if enabled, can be sent to Ollama cloud."));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->allow_cloud),
                               dt_conf_get_bool(DT_ASSISTANT_CONF_ALLOW_CLOUD));
  gtk_grid_attach(GTK_GRID(grid), data->allow_cloud, 1, row++, 2, 1);

  GtkWidget *key_label = gtk_label_new(_("API key"));
  gtk_widget_set_halign(key_label, GTK_ALIGN_START);
  data->key = gtk_entry_new();
  gtk_entry_set_visibility(GTK_ENTRY(data->key), FALSE);
  gtk_entry_set_input_purpose(GTK_ENTRY(data->key), GTK_INPUT_PURPOSE_PASSWORD);
  gtk_entry_set_placeholder_text(GTK_ENTRY(data->key), _("cloud providers only"));
  data->store_key = gtk_button_new_with_label(_("store"));
  data->clear_key = gtk_button_new_with_label(_("clear"));
  gtk_widget_set_tooltip_text(data->clear_key,
                              _("Remove the stored API key for this provider."));
  data->key_box = dt_gui_hbox(data->key, data->store_key, data->clear_key);
  gtk_grid_attach(GTK_GRID(grid), key_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->key_box, 1, row++, 2, 1);

  data->key_status = gtk_label_new("");
  gtk_widget_set_halign(data->key_status, GTK_ALIGN_START);
  gtk_label_set_xalign(GTK_LABEL(data->key_status), 0.0f);
  gtk_grid_attach(GTK_GRID(grid), data->key_status, 1, row++, 2, 1);

  GtkWidget *verify = gtk_button_new_with_label(_("verify configuration"));
  gtk_grid_attach(GTK_GRID(grid), verify, 1, row++, 2, 1);

  data->send_image
    = gtk_check_button_new_with_label(_("send preview image"));
  gtk_widget_set_tooltip_text(
    data->send_image,
    _("Attach a 768 px JPEG preview of the current darkroom render. It has "
      "no metadata. The selected model must support vision; otherwise the "
      "request falls back to text-only."));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->send_image),
                               dt_conf_get_bool(DT_ASSISTANT_CONF_SEND_IMAGE));
  g_signal_connect(data->send_image, "toggled",
                   G_CALLBACK(_send_image_toggled), data);
  gtk_grid_attach(GTK_GRID(grid), data->send_image, 1, row++, 2, 1);

  GtkWidget *eb_label = gtk_label_new(_("exposure bias (EV)"));
  gtk_widget_set_halign(eb_label, GTK_ALIGN_START);
  data->exposure_bias = gtk_spin_button_new_with_range(-2.0, 2.0, 0.1);
  gtk_spin_button_set_digits(GTK_SPIN_BUTTON(data->exposure_bias), 1);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(data->exposure_bias),
                            dt_conf_get_float(DT_ASSISTANT_CONF_EXPOSURE_BIAS));
  gtk_widget_set_tooltip_text(
    data->exposure_bias,
    _("Added to the assistant's exposure suggestion when highlights allow. "
      "0.0 is neutral; the base suggestion still reacts to the image."));
  g_signal_connect(data->exposure_bias, "value-changed",
                   G_CALLBACK(_exposure_bias_changed), data);
  gtk_grid_attach(GTK_GRID(grid), eb_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->exposure_bias, 1, row++, 2, 1);

  GtkWidget *modules_box = dt_gui_vbox();
  GtkWidget *modules_section = gtk_label_new(_("assistant modules"));
  GtkWidget *modules_section_box = dt_gui_hbox(modules_section);
  gtk_widget_set_name(modules_section_box, "pref_section");
  gtk_widget_set_halign(modules_section, GTK_ALIGN_START);
  dt_gui_box_add(modules_box, modules_section_box);

  GtkWidget *modules_note
    = gtk_label_new(_("The assistant can only use these modules. It may activate "
                      "inactive modules from this list."));
  gtk_label_set_line_wrap(GTK_LABEL(modules_note), TRUE);
  gtk_label_set_xalign(GTK_LABEL(modules_note), 0.0f);
  dt_gui_box_add(modules_box, modules_note);

  data->modules_store = gtk_list_store_new(MOD_COLS, G_TYPE_STRING, G_TYPE_STRING,
                                           G_TYPE_STRING);
  data->modules_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(data->modules_store));
  g_object_unref(data->modules_store);
  gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(data->modules_view), TRUE);
  gtk_tree_view_set_enable_search(GTK_TREE_VIEW(data->modules_view), FALSE);

  GtkCellRenderer *module_text = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *module_col
    = gtk_tree_view_column_new_with_attributes(_("module"), module_text,
                                               "text", MOD_NAME, NULL);
  gtk_tree_view_column_set_expand(module_col, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(data->modules_view), module_col);

  GtkCellRenderer *role_text = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *role_col
    = gtk_tree_view_column_new_with_attributes(_("role"), role_text,
                                               "text", MOD_ROLE, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(data->modules_view), role_col);

  GtkWidget *modules_scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(modules_scroll),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_scrolled_window_set_min_content_height(GTK_SCROLLED_WINDOW(modules_scroll),
                                              DT_PIXEL_APPLY_DPI(160));
  gtk_container_add(GTK_CONTAINER(modules_scroll), data->modules_view);
  dt_gui_box_add(modules_box, modules_scroll);

  data->module_combo = gtk_combo_box_text_new();
  gtk_widget_set_hexpand(data->module_combo, TRUE);
  data->module_add = gtk_button_new_with_label(_("add"));
  data->module_remove = gtk_button_new_with_label(_("remove"));
  data->module_reset = gtk_button_new_with_label(_("reset"));
  GtkWidget *module_controls
    = dt_gui_hbox(data->module_combo, data->module_add, data->module_remove,
                  data->module_reset);
  dt_gui_box_add(modules_box, module_controls);

  data->modules_status = gtk_label_new("");
  gtk_label_set_xalign(GTK_LABEL(data->modules_status), 0.0f);
  dt_gui_box_add(modules_box, data->modules_status);

  gtk_grid_attach(GTK_GRID(grid), modules_box, 0, row++, 3, 1);

  g_signal_connect(data->module_add, "clicked",
                   G_CALLBACK(_add_module_clicked), data);
  g_signal_connect(data->module_remove, "clicked",
                   G_CALLBACK(_remove_module_clicked), data);
  g_signal_connect(data->module_reset, "clicked",
                   G_CALLBACK(_reset_modules_clicked), data);
  GtkTreeSelection *modules_selection
    = gtk_tree_view_get_selection(GTK_TREE_VIEW(data->modules_view));
  g_signal_connect(modules_selection, "changed",
                   G_CALLBACK(_module_selection_changed), data);
  _populate_modules_store(data);
  gtk_widget_set_sensitive(data->module_remove, FALSE);

  GtkWidget *tt_label = gtk_label_new(_("tone & texture preference"));
  gtk_widget_set_halign(tt_label, GTK_ALIGN_START);
  data->tone_texture = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_texture),
                                 _("Tone & Texture"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_texture),
                                 _("Tone Equalizer"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_texture),
                                 _("Automatic"));
  const char *tt_value
    = dt_conf_get_string_const(DT_ASSISTANT_CONF_TONE_TEXTURE);
  const int tt_index = !g_strcmp0(tt_value, "toneequal") ? 1
                       : !g_strcmp0(tt_value, "auto")     ? 2
                                                          : 0;
  gtk_combo_box_set_active(GTK_COMBO_BOX(data->tone_texture), tt_index);
  g_signal_connect(data->tone_texture, "changed",
                   G_CALLBACK(_tone_texture_changed), data);
  gtk_grid_attach(GTK_GRID(grid), tt_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->tone_texture, 1, row++, 2, 1);

  GtkWidget *tm_label = gtk_label_new(_("tone mapper preference"));
  gtk_widget_set_halign(tm_label, GTK_ALIGN_START);
  data->tone_mapper = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 _("default"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 "3DCF");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 "ACES 2.0");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 "AgX");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 "Filmic RGB");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                                 "Sigmoid");
  _role_combo_set(GTK_COMBO_BOX_TEXT(data->tone_mapper),
                  _tone_mapper_values, 6, DT_ASSISTANT_CONF_TONE_MAPPER);
  g_signal_connect(data->tone_mapper, "changed",
                   G_CALLBACK(_role_pref_changed), (gpointer)&_tone_mapper_pref);
  gtk_grid_attach(GTK_GRID(grid), tm_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->tone_mapper, 1, row++, 2, 1);

  GtkWidget *color_label = gtk_label_new(_("color preference"));
  gtk_widget_set_halign(color_label, GTK_ALIGN_START);
  data->color = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->color),
                                 _("automatic"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->color),
                                 "Color Balance RGB");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->color),
                                 "Color Harmonizer");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->color),
                                 "RGB Primaries");
  _role_combo_set(GTK_COMBO_BOX_TEXT(data->color),
                  _color_values, 4, DT_ASSISTANT_CONF_COLOR);
  g_signal_connect(data->color, "changed",
                   G_CALLBACK(_role_pref_changed), (gpointer)&_color_pref);
  gtk_grid_attach(GTK_GRID(grid), color_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->color, 1, row++, 2, 1);

  GtkWidget *detail_label = gtk_label_new(_("detail preference"));
  gtk_widget_set_halign(detail_label, GTK_ALIGN_START);
  data->detail = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->detail),
                                 _("automatic"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->detail),
                                 "Diffuse or Sharpen");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(data->detail),
                                 "Sharpen");
  _role_combo_set(GTK_COMBO_BOX_TEXT(data->detail),
                  _detail_values, 3, DT_ASSISTANT_CONF_DETAIL);
  g_signal_connect(data->detail, "changed",
                   G_CALLBACK(_role_pref_changed), (gpointer)&_detail_pref);
  gtk_grid_attach(GTK_GRID(grid), detail_label, 0, row, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), data->detail, 1, row++, 2, 1);

  GtkWidget *privacy
    = gtk_label_new(_("By default no image is sent. If 'send preview image' is "
                      "enabled, a 768 px JPEG of the current render without "
                      "metadata is sent to the selected provider. A "
                      "vision-capable model is required; otherwise the request "
                      "falls back to text-only."));
  gtk_label_set_line_wrap(GTK_LABEL(privacy), TRUE);
  gtk_label_set_xalign(GTK_LABEL(privacy), 0.0f);
  gtk_grid_attach(GTK_GRID(grid), privacy, 0, row++, 3, 1);

  const char *saved_provider
    = dt_conf_get_string_const(DT_ASSISTANT_CONF_PROVIDER);
  const dt_assistant_provider_t provider
    = dt_assistant_provider_from_string(saved_provider);
  dt_bauhaus_combobox_set(data->provider, _provider_combo_index(provider));
  gtk_entry_set_placeholder_text(
    _model_entry(data),
    provider == DT_ASSISTANT_PROVIDER_OLLAMA ? _("e.g. llama3.1:8b")
                                             : _("detect or type a model"));

  char *default_endpoint = NULL;
  char *default_model = NULL;
  dt_assistant_provider_defaults(provider, &default_endpoint, &default_model);
  const char *saved_endpoint = _provider_setting(provider, "endpoint");
  const char *saved_model = _provider_setting(provider, "model");
  if(provider == DT_ASSISTANT_PROVIDER_CUSTOM)
  {
    const char *legacy_endpoint
      = dt_conf_get_string_const(DT_ASSISTANT_CONF_ENDPOINT);
    const char *legacy_model = dt_conf_get_string_const(DT_ASSISTANT_CONF_MODEL);
    if(!saved_endpoint || !saved_endpoint[0]) saved_endpoint = legacy_endpoint;
    if(!saved_model || !saved_model[0]) saved_model = legacy_model;
  }

  data->populating = TRUE;
  gtk_entry_set_text(GTK_ENTRY(data->endpoint),
                     saved_endpoint && saved_endpoint[0] ? saved_endpoint
                                                         : default_endpoint);
  gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(data->model));
  gtk_entry_set_text(_model_entry(data),
                     saved_model && saved_model[0] ? saved_model : default_model);
  data->populating = FALSE;
  g_free(default_endpoint);
  g_free(default_model);

  if(provider == DT_ASSISTANT_PROVIDER_OLLAMA)
  {
    if(saved_model && saved_model[0])
    {
      char *message
        = g_strdup_printf(_("Configured model: '%s'. Click detect to verify it is installed."),
                          saved_model);
      gtk_label_set_text(GTK_LABEL(data->model_status), message);
      g_free(message);
    }
    else
      gtk_label_set_text(GTK_LABEL(data->model_status),
                         _("Click detect to list installed Ollama models."));
  }
  else
    gtk_label_set_text(GTK_LABEL(data->model_status),
                       _("Click detect to list models, or type a model."));

  g_signal_connect(data->provider, "value-changed",
                   G_CALLBACK(_provider_changed), data);
  g_signal_connect(data->endpoint, "changed",
                   G_CALLBACK(_endpoint_changed), data);
  g_signal_connect(_model_entry(data), "changed",
                   G_CALLBACK(_model_changed), data);
  g_signal_connect(data->detect, "clicked",
                   G_CALLBACK(_detect_clicked), data);
  g_signal_connect(data->allow_cloud, "toggled",
                   G_CALLBACK(_allow_cloud_toggled), data);
  g_signal_connect(data->store_key, "clicked",
                   G_CALLBACK(_store_key_clicked), data);
  g_signal_connect(data->clear_key, "clicked",
                   G_CALLBACK(_clear_key_clicked), data);
  g_signal_connect(verify, "clicked", G_CALLBACK(_verify_clicked), data);

  gtk_widget_set_sensitive(data->allow_cloud,
                           provider == DT_ASSISTANT_PROVIDER_OLLAMA);
  _update_provider_note(data);
  _update_key_widgets(data);

  GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_container_add(GTK_CONTAINER(scroll), main_box);
  gtk_stack_add_titled(GTK_STACK(stack), scroll, "assistant",
                       _("Assistant"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

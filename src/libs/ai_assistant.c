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
#include "common/assistant/assistant_context.h"
#include "common/assistant/assistant_image.h"
#include "common/assistant/assistant_provider.h"
#include "common/assistant/assistant_stats.h"
#include "common/darktable.h"
#include "common/image.h"
#include "control/control.h"
#include "control/signal.h"
#include "develop/develop.h"
#include "gui/gtk.h"
#include "gui/preferences.h"
#include "libs/lib.h"
#include "libs/lib_api.h"

#include <gtk/gtk.h>
#include <json-glib/json-glib.h>
#include <string.h>

DT_MODULE(1)

typedef struct
{
  GtkWidget *status;
  GtkWidget *log_label;
  GString *log_text;
  GtkWidget *plan_grid;
  GPtrArray *plan_checks;
  GtkWidget *entry;
  GtkWidget *analyze_button;
  GtkWidget *send_button;
  GtkWidget *apply_button;
  GtkWidget *revert_button;
  GtkWidget *stop_button;
  GtkWidget *settings_button;
  GtkWidget *send_image;

  dt_assistant_plan_t *plan;
  GPtrArray *revert;

  GThread *worker;
  struct dt_assistant_result_t *pending;
  gint request_id;
  gboolean busy;
} dt_lib_ai_assistant_t;

typedef struct dt_assistant_result_t
{
  dt_lib_ai_assistant_t *panel;
  gint request_id;
  char *text;
  char *error;
  char *prompt;
  char *image_note;
  float exposure_bias_applied;
  float suggested_exposure_ev;
  gboolean retry;
  gint cancel;
  gint abandoned;
  gint posted;
} dt_assistant_result_t;

typedef struct
{
  dt_assistant_result_t *result;
  dt_assistant_config_t config;
  char *system_prompt;
  char *user_prompt;
  dt_imgid_t imgid;
  gboolean with_stats;
} dt_assistant_worker_t;

static gpointer _worker_thread(gpointer data);
static void _start_prompt(dt_lib_ai_assistant_t *d,
                          dt_assistant_config_t config,
                          char *system_prompt,
                          char *user_prompt,
                          gboolean with_stats,
                          gboolean retry,
                          float exposure_bias_applied,
                          float suggested_exposure_ev);

const char *name(dt_lib_module_t *self)
{
  (void)self;
  return _("AI assistant");
}

const char *description(dt_lib_module_t *self)
{
  (void)self;
  return _("Analyze the current settings and propose a first-pass set of slider changes");
}

dt_view_type_flags_t views(dt_lib_module_t *self)
{
  return DT_VIEW_DARKROOM;
}

uint32_t container(dt_lib_module_t *self)
{
  return DT_UI_CONTAINER_PANEL_RIGHT_CENTER;
}

int position(const dt_lib_module_t *self)
{
  return 0;
}

static const char *_provider_label(const dt_assistant_provider_t provider)
{
  return _(dt_assistant_provider_label(provider));
}

static void _result_free(dt_assistant_result_t *result)
{
  if(!result) return;
  g_free(result->text);
  g_free(result->error);
  g_free(result->prompt);
  g_free(result->image_note);
  g_free(result);
}

static void _set_status(dt_lib_ai_assistant_t *d, const char *text)
{
  gtk_label_set_text(GTK_LABEL(d->status), text ? text : "");
}

static void _log_clear(dt_lib_ai_assistant_t *d)
{
  if(d->log_text) g_string_truncate(d->log_text, 0);
  gtk_label_set_text(GTK_LABEL(d->log_label), "");
}

static void _log_append(dt_lib_ai_assistant_t *d, const char *text)
{
  if(!text || !text[0]) return;
  g_string_append(d->log_text, text);
  g_string_append_c(d->log_text, '\n');
  if(d->log_text->len > 4000)
    g_string_erase(d->log_text, 0, d->log_text->len - 4000);
  gtk_label_set_text(GTK_LABEL(d->log_label), d->log_text->str);
}

static char *_config_summary(void)
{
  dt_assistant_config_t config = dt_assistant_config_load();
  char *reason = NULL;
  char *summary = NULL;

  if(!dt_assistant_config_is_ready(&config, &reason))
    summary = g_strdup_printf(_("Assistant not configured: %s"),
                              reason ? reason : _("check preferences"));
  else
    summary = g_strdup_printf("%s · %s", _provider_label(config.provider),
                              config.model);

  g_free(reason);
  dt_assistant_config_clear(&config);
  return summary;
}

static void _update_buttons(dt_lib_ai_assistant_t *d)
{
  const gboolean has_image
    = darktable.develop && dt_is_valid_imgid(darktable.develop->image_storage.id);
  const gboolean has_plan = d->plan && d->plan->changes && d->plan->changes->len > 0;
  const gboolean has_revert = d->revert && d->revert->len > 0;

  gtk_widget_set_sensitive(d->analyze_button, has_image && !d->busy);
  gtk_widget_set_sensitive(d->send_button, has_image && !d->busy);
  gtk_widget_set_sensitive(d->entry, has_image && !d->busy);
  gtk_widget_set_sensitive(d->apply_button, has_image && !d->busy && has_plan);
  gtk_widget_set_sensitive(d->revert_button, has_image && !d->busy && has_revert);
  gtk_widget_set_sensitive(d->stop_button, d->busy);
  gtk_widget_set_sensitive(d->settings_button, TRUE);
}

static void _update_idle_status(dt_lib_ai_assistant_t *d)
{
  if(d->busy)
  {
    _set_status(d, _("Analyzing…"));
    return;
  }

  if(!darktable.develop || !dt_is_valid_imgid(darktable.develop->image_storage.id))
  {
    _set_status(d, _("Open an image in the darkroom."));
    return;
  }

  if(d->plan)
    _set_status(d, _("Plan ready. Review and apply."));
  else if(d->revert)
    _set_status(d, _("Changes applied. You can revert them."));
  else
  {
    char *summary = _config_summary();
    _set_status(d, summary);
    g_free(summary);
  }
}

static void _clear_plan(dt_lib_ai_assistant_t *d)
{
  if(d->plan)
  {
    dt_assistant_plan_free(d->plan);
    d->plan = NULL;
  }
  if(d->plan_grid)
  {
    GList *children = gtk_container_get_children(GTK_CONTAINER(d->plan_grid));
    for(GList *l = children; l; l = l->next)
      gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(children);
  }
  if(d->plan_checks) g_ptr_array_set_size(d->plan_checks, 0);
}

static void _abandon_pending(dt_lib_ai_assistant_t *d)
{
  if(!d->pending) return;

  dt_assistant_result_t *result = d->pending;
  d->pending = NULL;
  d->request_id++;
  g_atomic_int_set(&result->abandoned, 1);
  g_atomic_int_set(&result->cancel, 1);

  if(d->worker)
  {
    g_thread_join(d->worker);
    d->worker = NULL;
  }

  if(!g_atomic_int_get(&result->posted))
    _result_free(result);

  d->busy = FALSE;
}

static void _populate_plan(dt_lib_ai_assistant_t *d)
{
  if(!d->plan || !d->plan->changes) return;

  const char *headers[] = { _("Apply"), _("Module"), _("Slider"),
                            _("Value"), _("Why") };
  for(int c = 0; c < 5; c++)
  {
    GtkWidget *label = gtk_label_new(headers[c]);
    gtk_label_set_xalign(GTK_LABEL(label), 0.0f);
    gtk_grid_attach(GTK_GRID(d->plan_grid), label, c, 0, 1, 1);
  }

  for(guint i = 0; i < d->plan->changes->len; i++)
  {
    dt_assistant_change_t *change = g_ptr_array_index(d->plan->changes, i);
    const int row = (int)i + 1;

    GtkWidget *check = gtk_check_button_new();
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check), change->apply);
    gtk_grid_attach(GTK_GRID(d->plan_grid), check, 0, row, 1, 1);
    g_ptr_array_add(d->plan_checks, check);

    char *module_text = change->will_enable
                          ? g_strdup_printf("%s · enable", change->operation)
                          : g_strdup(change->operation);
    GtkWidget *module = gtk_label_new(module_text);
    gtk_label_set_xalign(GTK_LABEL(module), 0.0f);
    gtk_grid_attach(GTK_GRID(d->plan_grid), module, 1, row, 1, 1);
    g_free(module_text);

    GtkWidget *param = gtk_label_new(change->param);
    gtk_label_set_xalign(GTK_LABEL(param), 0.0f);
    gtk_grid_attach(GTK_GRID(d->plan_grid), param, 2, row, 1, 1);

    const dt_assistant_catalog_module_t *catalog
      = dt_assistant_catalog_get(change->operation);
    char *current_text
      = dt_assistant_catalog_format_value(catalog, change->param,
                                          change->current);
    char *proposed_text
      = dt_assistant_catalog_format_value(catalog, change->param,
                                          change->proposed);
    char *value = g_strdup_printf("%s → %s", current_text, proposed_text);
    g_free(current_text);
    g_free(proposed_text);
    GtkWidget *value_label = gtk_label_new(value);
    gtk_label_set_xalign(GTK_LABEL(value_label), 0.0f);
    gtk_grid_attach(GTK_GRID(d->plan_grid), value_label, 3, row, 1, 1);
    g_free(value);

    GtkWidget *reason = gtk_label_new(change->reason ? change->reason : "");
    gtk_label_set_xalign(GTK_LABEL(reason), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(reason), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(reason), 20);
    gtk_label_set_single_line_mode(GTK_LABEL(reason), TRUE);
    gtk_widget_set_tooltip_text(reason, change->reason ? change->reason : "");
    gtk_grid_attach(GTK_GRID(d->plan_grid), reason, 4, row, 1, 1);
  }

  gtk_widget_show_all(d->plan_grid);
}

static gboolean _retry_after_error(dt_lib_ai_assistant_t *d,
                                    dt_assistant_result_t *result,
                                    const char *message)
{
  if(result->retry || !result->prompt) return FALSE;

  dt_assistant_config_t config = dt_assistant_config_load();
  char *reason = NULL;
  if(!dt_assistant_config_is_ready(&config, &reason))
  {
    g_free(reason);
    dt_assistant_config_clear(&config);
    return FALSE;
  }
  g_free(reason);

  char *repair = g_strdup_printf(
    "%s\n\nYour previous answer was rejected by the application: %s\n"
    "Return a corrected JSON object only, with no markdown fences and no prose. "
    "Respect both the absolute ranges and the maximum deltas measured from the "
    "current value; if a change was too large, reduce it to at most the maximum "
    "delta from the current value. If the previous answer was invalid JSON, "
    "return one complete JSON object. Do not repeat the rejected value.",
    result->prompt, message ? message : _("invalid plan"));
  char *system = dt_assistant_system_prompt();
  _start_prompt(d, config, system, repair, FALSE, TRUE,
                result->exposure_bias_applied, result->suggested_exposure_ev);
  _set_status(d, _("Correcting the proposal…"));
  return TRUE;
}

static gboolean _result_idle(gpointer data)
{
  dt_assistant_result_t *result = (dt_assistant_result_t *)data;
  if(g_atomic_int_get(&result->abandoned))
  {
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  dt_lib_ai_assistant_t *d = result->panel;
  if(!d || d->pending != result || d->request_id != result->request_id)
  {
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  d->pending = NULL;
  d->busy = FALSE;
  if(d->worker)
  {
    g_thread_unref(d->worker);
    d->worker = NULL;
  }

  if(g_atomic_int_get(&result->cancel))
  {
    _set_status(d, _("Request cancelled."));
    _log_append(d, _("Request cancelled."));
    _update_buttons(d);
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  if(result->image_note)
    _log_append(d, result->image_note);

  if(result->error)
  {
    _set_status(d, _("Assistant error."));
    _log_append(d, result->error);
    _update_buttons(d);
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  char *parse_error = NULL;
  dt_assistant_plan_t *plan = dt_assistant_plan_parse(result->text, &parse_error);
  if(!plan)
  {
    if(_retry_after_error(d, result,
                          parse_error ? parse_error : _("invalid response")))
    {
      g_free(parse_error);
      _result_free(result);
      return G_SOURCE_REMOVE;
    }
    _set_status(d, _("Assistant returned an unusable response."));
    _log_append(d, parse_error ? parse_error : _("invalid response"));
    g_free(parse_error);
    _update_buttons(d);
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  char *sanitize_error = NULL;
  const guint dropped = dt_assistant_plan_sanitize(plan, &sanitize_error);
  if(dropped > 0)
  {
    char *line = g_strdup_printf(_("Ignored invalid proposal: %s"),
                                 sanitize_error ? sanitize_error
                                                : _("unknown slider"));
    _log_append(d, line);
    g_free(line);

    if(plan->changes->len == 0)
    {
      const gboolean retried
        = _retry_after_error(d, result,
                             sanitize_error ? sanitize_error
                                            : _("invalid slider"));
      g_free(sanitize_error);
      if(retried)
      {
        dt_assistant_plan_free(plan);
        _result_free(result);
        return G_SOURCE_REMOVE;
      }
    }
    else
      g_free(sanitize_error);
  }

  char *limit_note = NULL;
  if(dt_assistant_plan_clamp_values(plan, &limit_note) > 0)
  {
    char *line = g_strdup_printf(_("Safety-limited proposal: %s"), limit_note);
    _log_append(d, line);
    g_free(line);
  }
  g_free(limit_note);

  char *validate_error = NULL;
  if(!dt_assistant_plan_validate(plan, &validate_error))
  {
    if(_retry_after_error(d, result,
                          validate_error ? validate_error : _("invalid plan")))
    {
      g_free(validate_error);
      dt_assistant_plan_free(plan);
      _result_free(result);
      return G_SOURCE_REMOVE;
    }
    _set_status(d, _("Assistant proposed an invalid change."));
    _log_append(d, validate_error ? validate_error : _("invalid plan"));
    g_free(validate_error);
    dt_assistant_plan_free(plan);
    _update_buttons(d);
    _result_free(result);
    return G_SOURCE_REMOVE;
  }

  if(result->exposure_bias_applied != 0.0f)
    dt_assistant_plan_add_exposure_adjustment(plan, result->suggested_exposure_ev);

  _clear_plan(d);
  d->plan = plan;
  _populate_plan(d);

  if(plan->summary && plan->summary[0])
  {
    char *line = g_strdup_printf(_("Assistant: %s"), plan->summary);
    _log_append(d, line);
    g_free(line);
  }
  if(plan->changes->len == 0)
    _log_append(d, _("No change proposed."));

  if(plan->changes->len > 0)
  {
    guint to_enable = 0;
    for(guint i = 0; i < plan->changes->len; i++)
    {
      dt_assistant_change_t *change = g_ptr_array_index(plan->changes, i);
      if(change->will_enable) to_enable++;
    }
    char *message = g_strdup_printf(_("Plan ready: %u changes, %u module(s) to enable."),
                                    plan->changes->len, to_enable);
    _set_status(d, message);
    g_free(message);
  }
  else
    _set_status(d, _("No change proposed."));
  _update_buttons(d);
  _result_free(result);
  return G_SOURCE_REMOVE;
}

static gpointer _worker_thread(gpointer data)
{
  dt_assistant_worker_t *worker = (dt_assistant_worker_t *)data;
  dt_assistant_result_t *result = worker->result;

  if(worker->with_stats)
  {
    char *stats_error = NULL;
    char *stats = dt_assistant_stats_build(worker->imgid, &stats_error);
    if(stats)
    {
      JsonParser *stats_parser = json_parser_new();
      if(json_parser_load_from_data(stats_parser, stats, -1, NULL))
      {
        JsonNode *root = json_parser_get_root(stats_parser);
        if(root && JSON_NODE_HOLDS_OBJECT(root))
        {
          JsonObject *obj = json_node_get_object(root);
          if(json_object_has_member(obj, "exposure_bias_applied_ev"))
            result->exposure_bias_applied
              = (float)json_object_get_double_member(obj,
                                                     "exposure_bias_applied_ev");
          if(json_object_has_member(obj, "suggested_exposure_ev"))
            result->suggested_exposure_ev
              = (float)json_object_get_double_member(obj,
                                                     "suggested_exposure_ev");
        }
      }
      g_object_unref(stats_parser);

      char *prompt = g_strdup_printf("%s\n\nImage statistics (JSON):\n%s",
                                     worker->user_prompt, stats);
      g_free(worker->user_prompt);
      worker->user_prompt = prompt;
    }
    else
    {
      char *prompt = g_strdup_printf("%s\n\nImage statistics unavailable%s%s",
                                     worker->user_prompt,
                                     stats_error ? ": " : "",
                                     stats_error ? stats_error : "");
      g_free(worker->user_prompt);
      worker->user_prompt = prompt;
    }
    g_free(stats);
    g_free(stats_error);
  }

  char *base_prompt = worker->user_prompt;
  worker->user_prompt = NULL;
  char *preview = NULL;

  if(worker->config.send_image
     && dt_assistant_provider_supports_vision(worker->config.provider))
  {
    char *preview_error = NULL;
    preview = dt_assistant_preview_build(worker->imgid, 768, &preview_error);
    if(!preview)
      result->image_note = g_strdup_printf(
        _("Preview image unavailable: %s. Continuing with text only."),
        preview_error ? preview_error : _("unknown error"));
    g_free(preview_error);
  }

  char *image_prompt = NULL;
  if(preview)
    image_prompt = g_strdup_printf(
      _("%s\n\nA 768 px JPEG preview of the current darkroom render is attached. "
        "It contains no metadata. Use it to judge tonality, color, exposure and "
        "dynamic range; do not infer metadata, identity or location."),
      base_prompt);

  char *error = NULL;
  char *text = dt_assistant_chat(&worker->config, worker->system_prompt,
                                 image_prompt ? image_prompt : base_prompt,
                                 preview, &result->cancel, &error);

  if(!text && preview && dt_assistant_error_is_image_retryable(error))
  {
    result->image_note = g_strdup_printf(
      _("Preview image was rejected by the provider; retried text-only. %s"),
      error ? error : "");
    g_free(error);
    error = NULL;
    text = dt_assistant_chat(&worker->config, worker->system_prompt,
                             base_prompt, NULL, &result->cancel, &error);
  }

  g_free(image_prompt);
  g_free(preview);
  result->text = text;
  result->error = error;
  result->prompt = base_prompt;

  dt_assistant_config_clear(&worker->config);
  g_free(worker->system_prompt);
  g_free(worker->user_prompt);
  g_free(worker);

  if(g_atomic_int_get(&result->abandoned))
    return NULL;

  g_atomic_int_set(&result->posted, 1);
  g_idle_add(_result_idle, result);
  return NULL;
}

static void _start_request(dt_lib_ai_assistant_t *d,
                           const char *user_text,
                           const gboolean analyze)
{
  if(d->busy || d->pending) return;

  if(!darktable.develop || !dt_is_valid_imgid(darktable.develop->image_storage.id))
  {
    _set_status(d, _("Open an image in the darkroom."));
    return;
  }

  dt_assistant_config_t config = dt_assistant_config_load();
  char *reason = NULL;
  if(!dt_assistant_config_is_ready(&config, &reason))
  {
    char *message = g_strdup_printf(_("Assistant not configured: %s"),
                                    reason ? reason : _("check preferences"));
    _set_status(d, message);
    _log_append(d, message);
    g_free(message);
    g_free(reason);
    dt_assistant_config_clear(&config);
    return;
  }
  g_free(reason);

  char *context_error = NULL;
  char *context = dt_assistant_context_build(&context_error);
  if(!context)
  {
    _set_status(d, _("Could not read the current image settings."));
    _log_append(d, context_error ? context_error : _("unknown error"));
    g_free(context_error);
    dt_assistant_config_clear(&config);
    return;
  }

  const char *request
    = (user_text && user_text[0])
          ? user_text
          : (analyze
                 ? _("Analyze the current settings and propose up to 5 conservative "
                     "slider adjustments for a first pass. Return JSON only.")
                 : _("Propose a conservative first-pass slider adjustment. "
                     "Return JSON only."));

  const gchar *const *languages = g_get_language_names();
  const char *language = (languages && languages[0]) ? languages[0] : "en";
  char *user_prompt
    = g_strdup_printf("User interface language: %s\n\n"
                      "User request:\n%s\n\n"
                      "Image context (JSON):\n%s",
                      language, request, context);
  g_free(context);

  char *system_prompt = dt_assistant_system_prompt();
  _start_prompt(d, config, system_prompt, user_prompt, TRUE, FALSE, 0.0f, 0.0f);
}

static void _start_prompt(dt_lib_ai_assistant_t *d,
                          dt_assistant_config_t config,
                          char *system_prompt,
                          char *user_prompt,
                          const gboolean with_stats,
                          const gboolean retry,
                          const float exposure_bias_applied,
                          const float suggested_exposure_ev)
{
  if(d->busy || d->pending)
  {
    dt_assistant_config_clear(&config);
    g_free(system_prompt);
    g_free(user_prompt);
    return;
  }

  dt_assistant_result_t *result = g_new0(dt_assistant_result_t, 1);
  result->panel = d;
  result->request_id = ++d->request_id;
  result->retry = retry;
  result->exposure_bias_applied = exposure_bias_applied;
  result->suggested_exposure_ev = suggested_exposure_ev;

  dt_assistant_worker_t *worker = g_new0(dt_assistant_worker_t, 1);
  worker->result = result;
  worker->config = config;
  worker->system_prompt = system_prompt;
  worker->user_prompt = user_prompt;
  worker->imgid = darktable.develop->image_storage.id;
  worker->with_stats = with_stats;

  _clear_plan(d);
  d->busy = TRUE;
  d->pending = result;
  d->worker = g_thread_new("ai-assistant", _worker_thread, worker);
  _set_status(d, _("Analyzing…"));
  _update_buttons(d);
}

static void _send_clicked(GtkButton *button, gpointer user_data);

static void _analyze_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  /* If the photographer typed a request, Analyze must honour it. */
  const char *text = gtk_entry_get_text(GTK_ENTRY(d->entry));
  if(text && text[0])
  {
    _send_clicked(NULL, user_data);
    return;
  }
  _start_request(d, NULL, TRUE);
}

static void _send_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  char *text = g_strdup(gtk_entry_get_text(GTK_ENTRY(d->entry)));
  if(!text || !text[0])
  {
    g_free(text);
    return;
  }

  char *line = g_strdup_printf(_("You: %s"), text);
  _log_append(d, line);
  g_free(line);
  gtk_entry_set_text(GTK_ENTRY(d->entry), "");
  _start_request(d, text, FALSE);
  g_free(text);
}

static void _entry_activated(GtkEntry *entry, gpointer user_data)
{
  (void)entry;
  _send_clicked(NULL, user_data);
}

static void _apply_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  if(!d->plan) return;

  for(guint i = 0;
      i < d->plan->changes->len && i < d->plan_checks->len; i++)
  {
    dt_assistant_change_t *change = g_ptr_array_index(d->plan->changes, i);
    GtkWidget *check = g_ptr_array_index(d->plan_checks, i);
    change->apply = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(check));
  }

  GPtrArray *new_revert = NULL;
  char *error = NULL;
  if(!dt_assistant_apply(d->plan, &new_revert, &error))
  {
    _set_status(d, _("Could not apply the changes."));
    _log_append(d, error ? error : _("unknown error"));
    g_free(error);
    return;
  }
  g_free(error);

  if(d->revert) dt_assistant_revert_free(d->revert);
  d->revert = new_revert;
  _log_append(d, _("Changes applied."));
  _set_status(d, _("Changes applied. You can revert them."));
  _update_buttons(d);
}

static void _revert_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  if(!d->revert) return;

  char *error = NULL;
  if(!dt_assistant_revert(d->revert, &error))
  {
    _set_status(d, _("Could not revert the changes."));
    _log_append(d, error ? error : _("unknown error"));
    g_free(error);
    return;
  }
  g_free(error);

  dt_assistant_revert_free(d->revert);
  d->revert = NULL;
  _clear_plan(d);
  _log_append(d, _("Changes reverted."));
  _set_status(d, _("Changes reverted."));
  _update_buttons(d);
}

static void _stop_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  if(d->pending)
  {
    g_atomic_int_set(&d->pending->cancel, 1);
    _set_status(d, _("Stopping…"));
  }
}

static void _settings_clicked(GtkButton *button, gpointer user_data)
{
  (void)button;
  (void)user_data;
  dt_gui_preferences_show_page("assistant");
}

static void _image_changed_callback(gpointer instance, gpointer user_data)
{
  (void)instance;
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;

  _abandon_pending(d);
  _clear_plan(d);
  if(d->revert)
  {
    dt_assistant_revert_free(d->revert);
    d->revert = NULL;
  }
  _log_clear(d);
  _update_buttons(d);
  _update_idle_status(d);
}

static void _update_send_image_widget(dt_lib_ai_assistant_t *d)
{
  dt_assistant_config_t config = dt_assistant_config_load();
  const gboolean vision
    = dt_assistant_provider_supports_vision(config.provider);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->send_image),
                               config.send_image);
  gtk_widget_set_sensitive(d->send_image, vision);
  gtk_widget_set_tooltip_text(
    d->send_image,
    vision
      ? _("Attach a 768 px JPEG preview of the current darkroom render. "
          "It contains no metadata and is sent to the selected provider. "
          "The model must support vision or the request falls back to text-only.")
      : _("Vision support is unavailable for this provider."));
  dt_assistant_config_clear(&config);
}

static void _send_image_toggled(GtkToggleButton *button, gpointer user_data)
{
  (void)user_data;
  dt_conf_set_bool(DT_ASSISTANT_CONF_SEND_IMAGE,
                   gtk_toggle_button_get_active(button));
}

static void _send_image_prefs_changed(gpointer instance, dt_lib_module_t *self)
{
  (void)instance;
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;
  _update_send_image_widget(d);
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_ai_assistant_t *d = g_new0(dt_lib_ai_assistant_t, 1);
  self->data = d;

  GtkWidget *box = dt_gui_vbox();
  self->widget = box;

  d->status = gtk_label_new("");
  gtk_label_set_line_wrap(GTK_LABEL(d->status), TRUE);
  gtk_label_set_max_width_chars(GTK_LABEL(d->status), 40);
  gtk_label_set_xalign(GTK_LABEL(d->status), 0.0f);
  dt_gui_box_add(box, d->status);

  d->log_text = g_string_new(NULL);
  d->log_label = gtk_label_new("");
  gtk_label_set_line_wrap(GTK_LABEL(d->log_label), TRUE);
  gtk_label_set_max_width_chars(GTK_LABEL(d->log_label), 40);
  gtk_label_set_xalign(GTK_LABEL(d->log_label), 0.0f);
  gtk_label_set_yalign(GTK_LABEL(d->log_label), 0.0f);
  gtk_label_set_selectable(GTK_LABEL(d->log_label), TRUE);
  dt_gui_box_add(box, d->log_label);

  d->plan_grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(d->plan_grid), DT_PIXEL_APPLY_DPI(2));
  gtk_grid_set_column_spacing(GTK_GRID(d->plan_grid), DT_PIXEL_APPLY_DPI(6));
  dt_gui_box_add(box, d->plan_grid);
  d->plan_checks = g_ptr_array_new();

  GtkWidget *buttons = dt_gui_hbox();
  d->analyze_button = gtk_button_new_with_label(_("Analyze"));
  d->apply_button = gtk_button_new_with_label(_("Apply"));
  d->revert_button = gtk_button_new_with_label(_("Revert"));
  d->stop_button = gtk_button_new_with_label(_("Stop"));
  d->settings_button = gtk_button_new_with_label(_("Settings"));
  dt_gui_box_add(buttons, d->analyze_button, d->apply_button,
                 d->revert_button, d->stop_button);
  dt_gui_box_add(box, buttons);

  GtkWidget *chat_box = dt_gui_hbox();
  d->entry = gtk_entry_new();
  gtk_widget_set_hexpand(d->entry, TRUE);
  gtk_entry_set_placeholder_text(GTK_ENTRY(d->entry),
                                 _("Ask for a first-pass adjustment…"));
  d->send_button = gtk_button_new_with_label(_("Send"));
  dt_gui_box_add(chat_box, d->entry, d->send_button);
  dt_gui_box_add(box, chat_box);

  d->send_image = gtk_check_button_new_with_label(_("send preview image"));
  dt_gui_box_add(box, d->send_image);

  dt_gui_box_add(box, d->settings_button);

  GtkWidget *privacy
    = gtk_label_new(_("By default no image is sent. If 'send preview image' is "
                      "enabled, a 768 px JPEG of the current render without "
                      "metadata is sent to the selected provider. A vision-capable "
                      "model is required, otherwise the request falls back to "
                      "text-only."));
  gtk_label_set_line_wrap(GTK_LABEL(privacy), TRUE);
  gtk_label_set_max_width_chars(GTK_LABEL(privacy), 40);
  gtk_label_set_xalign(GTK_LABEL(privacy), 0.0f);
  dt_gui_box_add(box, privacy);

  g_signal_connect(d->analyze_button, "clicked", G_CALLBACK(_analyze_clicked), self);
  g_signal_connect(d->send_button, "clicked", G_CALLBACK(_send_clicked), self);
  g_signal_connect(d->entry, "activate", G_CALLBACK(_entry_activated), self);
  g_signal_connect(d->apply_button, "clicked", G_CALLBACK(_apply_clicked), self);
  g_signal_connect(d->revert_button, "clicked", G_CALLBACK(_revert_clicked), self);
  g_signal_connect(d->stop_button, "clicked", G_CALLBACK(_stop_clicked), self);
  g_signal_connect(d->settings_button, "clicked", G_CALLBACK(_settings_clicked), self);
  g_signal_connect(d->send_image, "toggled", G_CALLBACK(_send_image_toggled), self);

  _update_send_image_widget(d);

  gtk_widget_show_all(self->widget);
  gtk_widget_set_no_show_all(self->widget, TRUE);

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_IMAGE_CHANGED, _image_changed_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_PREFERENCES_CHANGE, _send_image_prefs_changed);

  _update_buttons(d);
  _update_idle_status(d);
}

void gui_update(dt_lib_module_t *self)
{
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;
  if(!d) return;

  _update_buttons(d);
  _update_idle_status(d);
}

void gui_cleanup(dt_lib_module_t *self)
{
  dt_lib_ai_assistant_t *d = (dt_lib_ai_assistant_t *)self->data;
  if(!d) return;

  _abandon_pending(d);
  _clear_plan(d);
  if(d->revert)
  {
    dt_assistant_revert_free(d->revert);
    d->revert = NULL;
  }

  if(d->log_text) g_string_free(d->log_text, TRUE);
  if(d->plan_checks) g_ptr_array_free(d->plan_checks, FALSE);
  g_free(d);
  self->data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

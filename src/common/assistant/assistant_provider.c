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

#include "common/assistant/assistant_provider.h"

#include "common/curl_tools.h"
#include "common/darktable.h"
#include "common/pwstorage/pwstorage.h"
#include "control/conf.h"

#include <curl/curl.h>
#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>

static const dt_assistant_provider_def_t _provider_defs[] = {
  { DT_ASSISTANT_PROVIDER_OLLAMA, "ollama", N_("Ollama (local)"),
    DT_ASSISTANT_API_OLLAMA, "http://localhost:11434/api/chat", "", FALSE, FALSE, TRUE },
  { DT_ASSISTANT_PROVIDER_DEEPSEEK, "deepseek", N_("DeepSeek"),
    DT_ASSISTANT_API_OPENAI, "https://api.deepseek.com/chat/completions",
    "deepseek-chat", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_OPENAI, "openai_gpt", N_("OpenAI"),
    DT_ASSISTANT_API_OPENAI, "https://api.openai.com/v1/chat/completions",
    "gpt-4.1-mini", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_OPENROUTER, "openrouter", N_("OpenRouter"),
    DT_ASSISTANT_API_OPENAI, "https://openrouter.ai/api/v1/chat/completions",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_GROQ, "groq", N_("Groq"),
    DT_ASSISTANT_API_OPENAI, "https://api.groq.com/openai/v1/chat/completions",
    "llama-3.3-70b-versatile", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_MISTRAL, "mistral", N_("Mistral"),
    DT_ASSISTANT_API_OPENAI, "https://api.mistral.ai/v1/chat/completions",
    "mistral-small-latest", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_XAI, "xai", N_("xAI Grok"),
    DT_ASSISTANT_API_OPENAI, "https://api.x.ai/v1/chat/completions",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_TOGETHER, "together", N_("Together AI"),
    DT_ASSISTANT_API_OPENAI, "https://api.together.xyz/v1/chat/completions",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_FIREWORKS, "fireworks", N_("Fireworks AI"),
    DT_ASSISTANT_API_OPENAI, "https://api.fireworks.ai/inference/v1/chat/completions",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_PERPLEXITY, "perplexity", N_("Perplexity"),
    DT_ASSISTANT_API_OPENAI, "https://api.perplexity.ai/chat/completions",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_LOCAL_OPENAI, "local_openai",
    N_("Local server (LM Studio / vLLM)"),
    DT_ASSISTANT_API_OPENAI, "http://localhost:1234/v1/chat/completions",
    "", FALSE, FALSE, TRUE },
  { DT_ASSISTANT_PROVIDER_GEMINI, "gemini", N_("Google Gemini"),
    DT_ASSISTANT_API_GEMINI,
    "https://generativelanguage.googleapis.com/v1beta",
    "gemini-3.8-flash", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_ANTHROPIC, "anthropic", N_("Anthropic Claude"),
    DT_ASSISTANT_API_ANTHROPIC, "https://api.anthropic.com/v1/messages",
    "", TRUE, TRUE, TRUE },
  { DT_ASSISTANT_PROVIDER_CUSTOM, "custom", N_("Custom API"),
    DT_ASSISTANT_API_OPENAI, "", "", TRUE, FALSE, TRUE },
};

static const dt_assistant_provider_def_t *_def(dt_assistant_provider_t provider)
{
  for(guint i = 0; i < G_N_ELEMENTS(_provider_defs); i++)
    if(_provider_defs[i].provider == provider) return &_provider_defs[i];
  return &_provider_defs[0];
}

int dt_assistant_provider_count(void)
{
  return (int)G_N_ELEMENTS(_provider_defs);
}

dt_assistant_provider_t dt_assistant_provider_from_index(const int index)
{
  if(index < 0 || index >= (int)G_N_ELEMENTS(_provider_defs))
    return DT_ASSISTANT_PROVIDER_OLLAMA;
  return _provider_defs[index].provider;
}

const dt_assistant_provider_def_t *dt_assistant_provider_def(
  const dt_assistant_provider_t provider)
{
  return _def(provider);
}

const char *dt_assistant_provider_id(const dt_assistant_provider_t provider)
{
  return _def(provider)->id;
}

const char *dt_assistant_provider_label(const dt_assistant_provider_t provider)
{
  return _def(provider)->label;
}

dt_assistant_api_t dt_assistant_provider_api(const dt_assistant_provider_t provider)
{
  return _def(provider)->api;
}

gboolean dt_assistant_provider_is_cloud(const dt_assistant_provider_t provider)
{
  return _def(provider)->cloud;
}

gboolean dt_assistant_provider_needs_key(const dt_assistant_provider_t provider)
{
  return _def(provider)->needs_key;
}

gboolean dt_assistant_provider_supports_vision(const dt_assistant_provider_t provider)
{
  return _def(provider)->vision;
}

char *dt_assistant_provider_conf_key(const dt_assistant_provider_t provider,
                                     const char *name)
{
  return g_strdup_printf("plugins/assistant/%s/%s",
                         dt_assistant_provider_id(provider), name);
}

dt_assistant_provider_t dt_assistant_provider_from_string(const char *value)
{
  if(value && !g_ascii_strcasecmp(value, "openai"))
    return DT_ASSISTANT_PROVIDER_CUSTOM;
  if(value)
    for(guint i = 0; i < G_N_ELEMENTS(_provider_defs); i++)
      if(!g_ascii_strcasecmp(value, _provider_defs[i].id))
        return _provider_defs[i].provider;
  return DT_ASSISTANT_PROVIDER_OLLAMA;
}

const char *dt_assistant_provider_to_string(const dt_assistant_provider_t provider)
{
  return dt_assistant_provider_id(provider);
}

void dt_assistant_provider_defaults(const dt_assistant_provider_t provider,
                                    char **endpoint,
                                    char **model)
{
  const dt_assistant_provider_def_t *def = _def(provider);
  if(endpoint) *endpoint = g_strdup(def->endpoint ? def->endpoint : "");
  if(model) *model = g_strdup(def->model ? def->model : "");
}

static GMutex _key_lock;
static gchar *_session_keys[DT_ASSISTANT_PROVIDER_COUNT];
static gboolean _session_key_set[DT_ASSISTANT_PROVIDER_COUNT];

static char *_clean_key(const char *key)
{
  char *result = g_strdup(key ? key : "");
  g_strstrip(result);
  const gsize len = strlen(result);
  if(len >= 2
     && ((result[0] == '"' && result[len - 1] == '"')
         || (result[0] == '\'' && result[len - 1] == '\'')))
  {
    result[len - 1] = '\0';
    memmove(result, result + 1, len - 1);
  }
  return result;
}

void dt_assistant_set_runtime_key(const dt_assistant_provider_t provider,
                                  const char *key)
{
  if(provider < 0 || provider >= DT_ASSISTANT_PROVIDER_COUNT) return;
  g_mutex_lock(&_key_lock);
  g_free(_session_keys[provider]);
  _session_keys[provider] = g_strdup(key);
  _session_key_set[provider] = TRUE;
  g_mutex_unlock(&_key_lock);
}

static char *_get_runtime_key(const dt_assistant_provider_t provider)
{
  if(provider < 0 || provider >= DT_ASSISTANT_PROVIDER_COUNT) return NULL;
  g_mutex_lock(&_key_lock);
  char *key = _session_key_set[provider]
                ? g_strdup(_session_keys[provider]) : NULL;
  g_mutex_unlock(&_key_lock);
  return key;
}

gboolean dt_assistant_has_runtime_key(const dt_assistant_provider_t provider)
{
  char *key = _get_runtime_key(provider);
  const gboolean has = key && key[0];
  g_free(key);
  return has;
}

gboolean dt_assistant_secure_storage_available(void)
{
  return darktable.pwstorage
         && darktable.pwstorage->pw_storage_backend != PW_STORAGE_BACKEND_NONE;
}

static char *_provider_pw_slot(const dt_assistant_provider_t provider)
{
  return g_strdup_printf("ai_assistant_%s", dt_assistant_provider_id(provider));
}

gboolean dt_assistant_store_key(const dt_assistant_provider_t provider,
                                const char *key,
                                char **error)
{
  if(error) *error = NULL;

  char *clean = _clean_key(key);
  if(strpbrk(clean, "\r\n"))
  {
    g_free(clean);
    if(error) *error = g_strdup(_("API key contains invalid characters"));
    return FALSE;
  }

  dt_assistant_set_runtime_key(provider, clean);
  if(!dt_assistant_secure_storage_available())
  {
    if(error && clean[0])
      *error = g_strdup(_("secure storage is unavailable; the key is kept for this session only"));
    g_free(clean);
    return FALSE;
  }

  char *slot = _provider_pw_slot(provider);
  GHashTable *table = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
  g_hash_table_insert(table, g_strdup(DT_ASSISTANT_PW_KEY), g_strdup(clean));
  const gboolean stored = dt_pwstorage_set(slot, table);
  g_hash_table_destroy(table);
  g_free(slot);

  if(stored && provider == DT_ASSISTANT_PROVIDER_CUSTOM && !clean[0])
  {
    table = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    g_hash_table_insert(table, g_strdup(DT_ASSISTANT_PW_KEY), g_strdup(""));
    dt_pwstorage_set(DT_ASSISTANT_PW_SLOT, table);
    g_hash_table_destroy(table);
  }
  g_free(clean);

  if(!stored && error)
    *error = g_strdup(_("secure storage is unavailable; the key is kept for this session only"));
  return stored;
}

gboolean dt_assistant_has_stored_key(const dt_assistant_provider_t provider)
{
  if(!dt_assistant_secure_storage_available()) return FALSE;
  char *slot = _provider_pw_slot(provider);
  GHashTable *table = dt_pwstorage_get(slot);
  g_free(slot);
  gboolean has_key = FALSE;
  if(table)
  {
    const char *key = g_hash_table_lookup(table, DT_ASSISTANT_PW_KEY);
    has_key = key && key[0];
    g_hash_table_destroy(table);
  }

  if(!has_key && provider == DT_ASSISTANT_PROVIDER_CUSTOM)
  {
    table = dt_pwstorage_get(DT_ASSISTANT_PW_SLOT);
    if(table)
    {
      const char *key = g_hash_table_lookup(table, DT_ASSISTANT_PW_KEY);
      has_key = key && key[0];
      g_hash_table_destroy(table);
    }
  }
  return has_key;
}

static char *_stored_key(const dt_assistant_provider_t provider,
                         const gboolean allow_legacy)
{
  char *slot = _provider_pw_slot(provider);
  GHashTable *table = dt_assistant_secure_storage_available()
                        ? dt_pwstorage_get(slot) : NULL;
  g_free(slot);
  char *key = NULL;
  if(table)
  {
    const char *stored = g_hash_table_lookup(table, DT_ASSISTANT_PW_KEY);
    if(stored && stored[0]) key = _clean_key(stored);
    g_hash_table_destroy(table);
  }

  if(!key && allow_legacy && dt_assistant_secure_storage_available())
  {
    table = dt_pwstorage_get(DT_ASSISTANT_PW_SLOT);
    if(table)
    {
      const char *stored = g_hash_table_lookup(table, DT_ASSISTANT_PW_KEY);
      if(stored && stored[0]) key = _clean_key(stored);
      g_hash_table_destroy(table);
    }
  }
  return key;
}

dt_assistant_config_t dt_assistant_config_load(void)
{
  dt_assistant_config_t config = { 0 };

  const char *provider_value
    = dt_conf_get_string_const(DT_ASSISTANT_CONF_PROVIDER);
  const gboolean legacy_openai
    = provider_value && !g_ascii_strcasecmp(provider_value, "openai");
  const gboolean legacy = !provider_value || !provider_value[0]
                          || legacy_openai
                          || !g_ascii_strcasecmp(provider_value, "ollama");
  config.provider = dt_assistant_provider_from_string(provider_value);
  config.api = dt_assistant_provider_api(config.provider);

  const dt_assistant_provider_def_t *def = _def(config.provider);
  char *endpoint_key = dt_assistant_provider_conf_key(config.provider, "endpoint");
  char *model_key = dt_assistant_provider_conf_key(config.provider, "model");
  const char *endpoint = dt_conf_get_string_const(endpoint_key);
  const char *model = dt_conf_get_string_const(model_key);
  const char *legacy_endpoint = dt_conf_get_string_const(DT_ASSISTANT_CONF_ENDPOINT);
  const char *legacy_model = dt_conf_get_string_const(DT_ASSISTANT_CONF_MODEL);

  if(legacy && (!endpoint || !endpoint[0])) endpoint = legacy_endpoint;
  if(legacy && (!model || !model[0])) model = legacy_model;

  config.endpoint = g_strdup(endpoint && endpoint[0] ? endpoint : def->endpoint);
  config.model = g_strdup(model && model[0] ? model : def->model);
  g_free(endpoint_key);
  g_free(model_key);

  int timeout = dt_conf_get_int(DT_ASSISTANT_CONF_TIMEOUT);
  if(timeout <= 0)
    timeout = config.provider == DT_ASSISTANT_PROVIDER_OLLAMA ? 600 : 180;
  config.timeout_s = timeout;
  config.send_image = dt_conf_get_bool(DT_ASSISTANT_CONF_SEND_IMAGE);

  config.api_key = _stored_key(config.provider, legacy_openai);
  char *runtime = _get_runtime_key(config.provider);
  if(runtime)
  {
    g_free(config.api_key);
    config.api_key = _clean_key(runtime);
    g_free(runtime);
  }

  return config;
}

void dt_assistant_config_clear(dt_assistant_config_t *config)
{
  if(!config) return;
  g_free(config->endpoint);
  g_free(config->model);
  g_free(config->api_key);
  memset(config, 0, sizeof(*config));
}

static gboolean _is_local_url(const char *url)
{
  return url
         && (strstr(url, "://localhost") || strstr(url, "://127.0.0.1")
             || strstr(url, "://[::1]"));
}

gboolean dt_assistant_config_is_ready(const dt_assistant_config_t *config,
                                      char **reason)
{
  if(reason) *reason = NULL;

  if(!config || !config->endpoint || !config->endpoint[0])
  {
    if(reason) *reason = g_strdup(_("endpoint is empty"));
    return FALSE;
  }
  if(!config->model || !config->model[0])
  {
    if(reason) *reason = g_strdup(_("model is empty; use detect or type a model"));
    return FALSE;
  }
  if(!g_str_has_prefix(config->endpoint, "http://")
     && !g_str_has_prefix(config->endpoint, "https://"))
  {
    if(reason) *reason = g_strdup(_("endpoint must start with http:// or https://"));
    return FALSE;
  }
  if(!g_str_has_prefix(config->endpoint, "https://") && !_is_local_url(config->endpoint))
  {
    if(reason) *reason = g_strdup(_("cloud endpoints must use https:// or localhost"));
    return FALSE;
  }
  if(config->api_key && strpbrk(config->api_key, "\r\n"))
  {
    if(reason) *reason = g_strdup(_("API key contains invalid characters"));
    return FALSE;
  }
  if(dt_assistant_provider_needs_key(config->provider)
     && (!config->api_key || !config->api_key[0]))
  {
    if(reason) *reason = g_strdup(_("API key is required for this provider"));
    return FALSE;
  }
  return TRUE;
}

static size_t _write_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
  GString *response = (GString *)userdata;
  const size_t bytes = size * nmemb;
  g_string_append_len(response, ptr, bytes);
  return bytes;
}

static int _progress_cb(void *clientp, curl_off_t dltotal, curl_off_t dlnow,
                        curl_off_t ultotal, curl_off_t ulnow)
{
  (void)dltotal;
  (void)dlnow;
  (void)ultotal;
  (void)ulnow;
  const volatile gint *cancel = (const volatile gint *)clientp;
  return cancel && g_atomic_int_get(cancel) ? 1 : 0;
}

static void _add_plan_tool_schema(JsonBuilder *builder)
{
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "object");
  json_builder_set_member_name(builder, "additionalProperties");
  json_builder_add_boolean_value(builder, FALSE);
  json_builder_set_member_name(builder, "properties");
  json_builder_begin_object(builder);

  json_builder_set_member_name(builder, "summary");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "string");
  json_builder_end_object(builder);

  json_builder_set_member_name(builder, "changes");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "array");
  json_builder_set_member_name(builder, "items");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "object");
  json_builder_set_member_name(builder, "additionalProperties");
  json_builder_add_boolean_value(builder, FALSE);
  json_builder_set_member_name(builder, "properties");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "operation");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "string");
  json_builder_end_object(builder);
  json_builder_set_member_name(builder, "param");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "string");
  json_builder_end_object(builder);
  json_builder_set_member_name(builder, "value");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "number");
  json_builder_end_object(builder);
  json_builder_set_member_name(builder, "reason");
  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "type");
  json_builder_add_string_value(builder, "string");
  json_builder_end_object(builder);
  json_builder_end_object(builder);
  json_builder_set_member_name(builder, "required");
  json_builder_begin_array(builder);
  json_builder_add_string_value(builder, "operation");
  json_builder_add_string_value(builder, "param");
  json_builder_add_string_value(builder, "value");
  json_builder_add_string_value(builder, "reason");
  json_builder_end_array(builder);
  json_builder_end_object(builder);
  json_builder_end_object(builder);

  json_builder_end_object(builder);
  json_builder_set_member_name(builder, "required");
  json_builder_begin_array(builder);
  json_builder_add_string_value(builder, "summary");
  json_builder_add_string_value(builder, "changes");
  json_builder_end_array(builder);
  json_builder_end_object(builder);
}

static void _add_messages(JsonBuilder *builder,
                          const char *system_prompt,
                          const char *user_prompt,
                          const char *image_base64,
                          const dt_assistant_api_t api)
{
  const gboolean has_image = image_base64 && image_base64[0];

  json_builder_set_member_name(builder, "messages");
  json_builder_begin_array(builder);

  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "role");
  json_builder_add_string_value(builder, "system");
  json_builder_set_member_name(builder, "content");
  json_builder_add_string_value(builder, system_prompt);
  json_builder_end_object(builder);

  json_builder_begin_object(builder);
  json_builder_set_member_name(builder, "role");
  json_builder_add_string_value(builder, "user");
  json_builder_set_member_name(builder, "content");

  if(has_image && api == DT_ASSISTANT_API_OPENAI)
  {
    json_builder_begin_array(builder);
    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "type");
    json_builder_add_string_value(builder, "text");
    json_builder_set_member_name(builder, "text");
    json_builder_add_string_value(builder, user_prompt);
    json_builder_end_object(builder);
    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "type");
    json_builder_add_string_value(builder, "image_url");
    json_builder_set_member_name(builder, "image_url");
    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "url");
    char *data_url = g_strdup_printf("data:image/jpeg;base64,%s", image_base64);
    json_builder_add_string_value(builder, data_url);
    g_free(data_url);
    json_builder_end_object(builder);
    json_builder_end_object(builder);
    json_builder_end_array(builder);
  }
  else
    json_builder_add_string_value(builder, user_prompt);

  if(has_image && api == DT_ASSISTANT_API_OLLAMA)
  {
    json_builder_set_member_name(builder, "images");
    json_builder_begin_array(builder);
    json_builder_add_string_value(builder, image_base64);
    json_builder_end_array(builder);
  }

  json_builder_end_object(builder);
  json_builder_end_array(builder);
}

static char *_build_request_body(const dt_assistant_config_t *config,
                                 const char *system_prompt,
                                 const char *user_prompt,
                                 const char *image_base64)
{
  JsonBuilder *builder = json_builder_new();
  json_builder_begin_object(builder);

  if(config->api != DT_ASSISTANT_API_GEMINI)
  {
    json_builder_set_member_name(builder, "model");
    json_builder_add_string_value(builder, config->model);
  }

  switch(config->api)
  {
    case DT_ASSISTANT_API_OLLAMA:
    {
      _add_messages(builder, system_prompt, user_prompt, image_base64, config->api);
      json_builder_set_member_name(builder, "stream");
      json_builder_add_boolean_value(builder, FALSE);
      json_builder_set_member_name(builder, "format");
      json_builder_add_string_value(builder, "json");
      json_builder_set_member_name(builder, "options");
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "temperature");
      json_builder_add_double_value(builder, 0.2);
      json_builder_set_member_name(builder, "num_ctx");
      json_builder_add_int_value(builder, 8192);
      json_builder_set_member_name(builder, "num_predict");
      json_builder_add_int_value(builder, 2048);
      json_builder_end_object(builder);
      break;
    }

    case DT_ASSISTANT_API_OPENAI:
    {
      _add_messages(builder, system_prompt, user_prompt, image_base64, config->api);
      json_builder_set_member_name(builder, "temperature");
      json_builder_add_double_value(builder, 0.2);
      json_builder_set_member_name(builder, "max_tokens");
      json_builder_add_int_value(builder, 2048);
      json_builder_set_member_name(builder, "response_format");
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "type");
      json_builder_add_string_value(builder, "json_object");
      json_builder_end_object(builder);
      break;
    }

    case DT_ASSISTANT_API_ANTHROPIC:
    {
      json_builder_set_member_name(builder, "max_tokens");
      json_builder_add_int_value(builder, 2048);
      json_builder_set_member_name(builder, "temperature");
      json_builder_add_double_value(builder, 0.2);
      json_builder_set_member_name(builder, "system");
      json_builder_add_string_value(builder, system_prompt);
      json_builder_set_member_name(builder, "messages");
      json_builder_begin_array(builder);
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "role");
      json_builder_add_string_value(builder, "user");
      json_builder_set_member_name(builder, "content");
      if(image_base64 && image_base64[0])
      {
        json_builder_begin_array(builder);
        json_builder_begin_object(builder);
        json_builder_set_member_name(builder, "type");
        json_builder_add_string_value(builder, "image");
        json_builder_set_member_name(builder, "source");
        json_builder_begin_object(builder);
        json_builder_set_member_name(builder, "type");
        json_builder_add_string_value(builder, "base64");
        json_builder_set_member_name(builder, "media_type");
        json_builder_add_string_value(builder, "image/jpeg");
        json_builder_set_member_name(builder, "data");
        json_builder_add_string_value(builder, image_base64);
        json_builder_end_object(builder);
        json_builder_end_object(builder);
        json_builder_begin_object(builder);
        json_builder_set_member_name(builder, "type");
        json_builder_add_string_value(builder, "text");
        json_builder_set_member_name(builder, "text");
        json_builder_add_string_value(builder, user_prompt);
        json_builder_end_object(builder);
        json_builder_end_array(builder);
      }
      else
        json_builder_add_string_value(builder, user_prompt);
      json_builder_end_object(builder);
      json_builder_end_array(builder);
      json_builder_set_member_name(builder, "tools");
      json_builder_begin_array(builder);
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "name");
      json_builder_add_string_value(builder, "submit_plan");
      json_builder_set_member_name(builder, "description");
      json_builder_add_string_value(builder,
                                   "Submit the final photography slider plan.");
      json_builder_set_member_name(builder, "input_schema");
      _add_plan_tool_schema(builder);
      json_builder_end_object(builder);
      json_builder_end_array(builder);
      json_builder_set_member_name(builder, "tool_choice");
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "type");
      json_builder_add_string_value(builder, "tool");
      json_builder_set_member_name(builder, "name");
      json_builder_add_string_value(builder, "submit_plan");
      json_builder_end_object(builder);
      break;
    }

    case DT_ASSISTANT_API_GEMINI:
    {
      json_builder_set_member_name(builder, "systemInstruction");
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "parts");
      json_builder_begin_array(builder);
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "text");
      json_builder_add_string_value(builder, system_prompt);
      json_builder_end_object(builder);
      json_builder_end_array(builder);
      json_builder_end_object(builder);

      json_builder_set_member_name(builder, "contents");
      json_builder_begin_array(builder);
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "role");
      json_builder_add_string_value(builder, "user");
      json_builder_set_member_name(builder, "parts");
      json_builder_begin_array(builder);
      if(image_base64 && image_base64[0])
      {
        json_builder_begin_object(builder);
        json_builder_set_member_name(builder, "inlineData");
        json_builder_begin_object(builder);
        json_builder_set_member_name(builder, "mimeType");
        json_builder_add_string_value(builder, "image/jpeg");
        json_builder_set_member_name(builder, "data");
        json_builder_add_string_value(builder, image_base64);
        json_builder_end_object(builder);
        json_builder_end_object(builder);
      }
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "text");
      json_builder_add_string_value(builder, user_prompt);
      json_builder_end_object(builder);
      json_builder_end_array(builder);
      json_builder_end_object(builder);
      json_builder_end_array(builder);

      json_builder_set_member_name(builder, "generationConfig");
      json_builder_begin_object(builder);
      json_builder_set_member_name(builder, "temperature");
      json_builder_add_double_value(builder, 0.2);
      json_builder_set_member_name(builder, "maxOutputTokens");
      json_builder_add_int_value(builder, 2048);
      json_builder_set_member_name(builder, "responseMimeType");
      json_builder_add_string_value(builder, "application/json");
      json_builder_end_object(builder);
      break;
    }
  }

  json_builder_end_object(builder);

  JsonNode *root = json_builder_get_root(builder);
  JsonGenerator *generator = json_generator_new();
  json_generator_set_root(generator, root);
  char *body = json_generator_to_data(generator, NULL);

  json_node_free(root);
  g_object_unref(generator);
  g_object_unref(builder);
  return body;
}

static char *_truncate_utf8(const char *text, const size_t max_bytes)
{
  if(!text) return g_strdup("");
  const size_t len = strlen(text);
  if(len <= max_bytes) return g_strdup(text);
  char *copy = g_strndup(text, max_bytes);
  while(copy[0] && !g_utf8_validate(copy, -1, NULL))
  {
    const size_t n = strlen(copy);
    if(n == 0) break;
    copy[n - 1] = '\0';
  }
  char *result = g_strdup_printf("%s…", copy);
  g_free(copy);
  return result;
}

static char *_error_from_body(const char *body)
{
  if(!body || !body[0]) return NULL;

  JsonParser *parser = json_parser_new();
  if(json_parser_load_from_data(parser, body, -1, NULL))
  {
    JsonNode *root = json_parser_get_root(parser);
    if(root && JSON_NODE_HOLDS_OBJECT(root))
    {
      JsonObject *obj = json_node_get_object(root);
      if(json_object_has_member(obj, "error"))
      {
        JsonNode *err_node = json_object_get_member(obj, "error");
        if(err_node && JSON_NODE_HOLDS_VALUE(err_node))
        {
          const char *message = json_node_get_string(err_node);
          if(message)
          {
            char *result = g_strdup(message);
            g_object_unref(parser);
            return result;
          }
        }
        if(err_node && JSON_NODE_HOLDS_OBJECT(err_node))
        {
          JsonObject *err_obj = json_node_get_object(err_node);
          if(json_object_has_member(err_obj, "message"))
          {
            const char *message
              = json_object_get_string_member(err_obj, "message");
            if(message)
            {
              char *result = g_strdup(message);
              g_object_unref(parser);
              return result;
            }
          }
        }
      }
    }
  }
  g_object_unref(parser);
  return _truncate_utf8(body, 500);
}

static gint _compare_model_names(gconstpointer a, gconstpointer b)
{
  const char *sa = *(const char * const *)a;
  const char *sb = *(const char * const *)b;
  return g_strcmp0(sa, sb);
}

static gboolean _is_non_chat_model(const char *name)
{
  static const char *const skip[] = {
    "embedding", "whisper", "tts", "dall-e", "moderation", "rerank",
    "transcribe", "realtime", "audio", "image", NULL
  };
  if(!name) return TRUE;
  for(int i = 0; skip[i]; i++)
    if(g_strstr_len(name, -1, skip[i])) return TRUE;
  return FALSE;
}

static char *_models_url(const char *endpoint, const char *suffix)
{
  if(!endpoint || !endpoint[0]) return g_strdup("");
  char *base = NULL;
  if(suffix && suffix[0] && g_str_has_suffix(endpoint, suffix))
    base = g_strndup(endpoint, strlen(endpoint) - strlen(suffix));
  else
  {
    base = g_strdup(endpoint);
    while(base[0] && base[strlen(base) - 1] == '/')
      base[strlen(base) - 1] = '\0';
  }
  char *url = g_strdup_printf("%s/models", base);
  g_free(base);
  return url;
}

static gboolean _perform_model_get(const char *url,
                                   struct curl_slist *headers,
                                   GString **response,
                                   long *http_code,
                                   char **error)
{
  CURL *curl = curl_easy_init();
  if(!curl)
  {
    if(error) *error = g_strdup(_("could not initialize the network client"));
    return FALSE;
  }

  dt_curl_init(curl, FALSE);
  GString *body = g_string_new(NULL);
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, body);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

  const CURLcode result = curl_easy_perform(curl);
  *http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);
  curl_easy_cleanup(curl);

  if(result != CURLE_OK)
  {
    if(error) *error = g_strdup(curl_easy_strerror(result));
    g_string_free(body, TRUE);
    return FALSE;
  }
  *response = body;
  return TRUE;
}

static GPtrArray *_parse_models_array(const char *body,
                                      const char *array_name,
                                      const gboolean gemini,
                                      char **error)
{
  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, body, -1, NULL))
  {
    if(error) *error = g_strdup(_("invalid JSON response"));
    g_object_unref(parser);
    return NULL;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    g_object_unref(parser);
    if(error) *error = g_strdup(_("unexpected model list response"));
    return NULL;
  }

  GPtrArray *models = g_ptr_array_new_with_free_func(g_free);
  GHashTable *seen = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  JsonObject *obj = json_node_get_object(root);
  if(json_object_has_member(obj, array_name))
  {
    JsonNode *node = json_object_get_member(obj, array_name);
    if(node && JSON_NODE_HOLDS_ARRAY(node))
    {
      JsonArray *array = json_node_get_array(node);
      for(guint i = 0; i < json_array_get_length(array); i++)
      {
        JsonNode *item = json_array_get_element(array, i);
        if(!item || !JSON_NODE_HOLDS_OBJECT(item)) continue;
        JsonObject *model = json_node_get_object(item);
        const char *name = NULL;
        if(json_object_has_member(model, "id"))
          name = json_object_get_string_member(model, "id");
        if((!name || !name[0]) && json_object_has_member(model, "name"))
          name = json_object_get_string_member(model, "name");
        if((!name || !name[0]) && json_object_has_member(model, "model"))
          name = json_object_get_string_member(model, "model");
        if(!name || !name[0]) continue;

        if(gemini)
        {
          if(g_str_has_prefix(name, "models/")) name += strlen("models/");
          if(json_object_has_member(model, "supportedGenerationMethods"))
          {
            JsonNode *methods_node
              = json_object_get_member(model, "supportedGenerationMethods");
            gboolean supports = FALSE;
            if(methods_node && JSON_NODE_HOLDS_ARRAY(methods_node))
            {
              JsonArray *methods = json_node_get_array(methods_node);
              for(guint j = 0; j < json_array_get_length(methods); j++)
              {
                const char *method = json_array_get_string_element(methods, j);
                if(!g_strcmp0(method, "generateContent"))
                {
                  supports = TRUE;
                  break;
                }
              }
            }
            if(!supports) continue;
          }
        }

        if(_is_non_chat_model(name) || g_hash_table_contains(seen, name))
          continue;
        g_hash_table_add(seen, g_strdup(name));
        g_ptr_array_add(models, g_strdup(name));
      }
    }
  }

  g_hash_table_destroy(seen);
  g_object_unref(parser);
  g_ptr_array_sort(models, _compare_model_names);
  return models;
}

static GPtrArray *_list_ollama_models(const char *endpoint,
                                      const gboolean include_cloud,
                                      char **error)
{
  if(!endpoint || !endpoint[0])
  {
    if(error) *error = g_strdup(_("invalid Ollama endpoint"));
    return NULL;
  }
  const char *scheme = strstr(endpoint, "://");
  const char *host = scheme ? scheme + 3 : endpoint;
  const char *slash = strchr(host, '/');
  char *base = slash ? g_strndup(endpoint, slash - endpoint) : g_strdup(endpoint);
  char *url = g_strdup_printf("%s/api/tags", base);
  g_free(base);

  GString *response = NULL;
  long http_code = 0;
  char *net_error = NULL;
  if(!_perform_model_get(url, NULL, &response, &http_code, &net_error))
  {
    if(error)
      *error = g_strdup_printf(_("Ollama is not reachable: %s"),
                               net_error ? net_error : _("network error"));
    g_free(net_error);
    g_free(url);
    return NULL;
  }
  g_free(url);

  if(http_code < 200 || http_code >= 300)
  {
    char *detail = _error_from_body(response->str);
    if(error)
      *error = g_strdup_printf(_("Ollama returned HTTP %ld: %s"),
                               http_code, detail ? detail : "");
    g_free(detail);
    g_string_free(response, TRUE);
    return NULL;
  }

  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, response->str, -1, NULL))
  {
    if(error) *error = g_strdup(_("Ollama returned invalid JSON"));
    g_object_unref(parser);
    g_string_free(response, TRUE);
    return NULL;
  }

  GPtrArray *models = g_ptr_array_new_with_free_func(g_free);
  JsonNode *root = json_parser_get_root(parser);
  if(root && JSON_NODE_HOLDS_OBJECT(root))
  {
    JsonObject *obj = json_node_get_object(root);
    if(json_object_has_member(obj, "models"))
    {
      JsonNode *node = json_object_get_member(obj, "models");
      if(node && JSON_NODE_HOLDS_ARRAY(node))
      {
        JsonArray *array = json_node_get_array(node);
        for(guint i = 0; i < json_array_get_length(array); i++)
        {
          JsonNode *item = json_array_get_element(array, i);
          if(!item || !JSON_NODE_HOLDS_OBJECT(item)) continue;
          JsonObject *model = json_node_get_object(item);
          const char *name = NULL;
          if(json_object_has_member(model, "name"))
            name = json_object_get_string_member(model, "name");
          else if(json_object_has_member(model, "model"))
            name = json_object_get_string_member(model, "model");
          if(!name || !name[0]) continue;
          const gboolean cloud = g_str_has_suffix(name, "-cloud")
                                 || g_str_has_suffix(name, ":cloud");
          if(include_cloud || !cloud)
            g_ptr_array_add(models, g_strdup(name));
        }
      }
    }
  }

  g_object_unref(parser);
  g_string_free(response, TRUE);
  g_ptr_array_sort(models, _compare_model_names);
  return models;
}

static GPtrArray *_list_cloud_models(const dt_assistant_provider_t provider,
                                     const char *endpoint,
                                     const char *api_key,
                                     char **error)
{
  const dt_assistant_api_t api = dt_assistant_provider_api(provider);
  char *url = NULL;
  if(api == DT_ASSISTANT_API_GEMINI)
    url = _models_url(endpoint, NULL);
  else if(api == DT_ASSISTANT_API_ANTHROPIC)
    url = _models_url(endpoint, "/messages");
  else
    url = _models_url(endpoint, "/chat/completions");

  if(!url || !url[0] || !g_str_has_prefix(url, "http"))
  {
    g_free(url);
    if(error) *error = g_strdup(_("invalid endpoint"));
    return NULL;
  }

  struct curl_slist *headers = NULL;
  if(api == DT_ASSISTANT_API_OPENAI)
  {
    if(api_key && api_key[0])
    {
      char *auth = g_strdup_printf("Authorization: Bearer %s", api_key);
      headers = curl_slist_append(headers, auth);
      g_free(auth);
    }
  }
  else if(api == DT_ASSISTANT_API_ANTHROPIC)
  {
    if(api_key && api_key[0])
    {
      char *auth = g_strdup_printf("x-api-key: %s", api_key);
      headers = curl_slist_append(headers, auth);
      g_free(auth);
    }
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
  }
  else if(api == DT_ASSISTANT_API_GEMINI)
  {
    if(api_key && api_key[0])
    {
      char *auth = g_strdup_printf("x-goog-api-key: %s", api_key);
      headers = curl_slist_append(headers, auth);
      g_free(auth);
    }
  }
  headers = curl_slist_append(headers, "Accept: application/json");

  GString *response = NULL;
  long http_code = 0;
  char *net_error = NULL;
  if(!_perform_model_get(url, headers, &response, &http_code, &net_error))
  {
    if(error)
      *error = g_strdup_printf(_("%s is not reachable: %s"),
                               dt_assistant_provider_label(provider),
                               net_error ? net_error : _("network error"));
    g_free(net_error);
    curl_slist_free_all(headers);
    g_free(url);
    return NULL;
  }
  curl_slist_free_all(headers);
  g_free(url);

  if(http_code < 200 || http_code >= 300)
  {
    char *detail = _error_from_body(response->str);
    if(error)
      *error = g_strdup_printf(_("%s returned HTTP %ld: %s"),
                               dt_assistant_provider_label(provider),
                               http_code, detail ? detail : "");
    g_free(detail);
    g_string_free(response, TRUE);
    return NULL;
  }

  char *parse_error = NULL;
  GPtrArray *models = NULL;
  if(api == DT_ASSISTANT_API_GEMINI)
    models = _parse_models_array(response->str, "models", TRUE, &parse_error);
  else
    models = _parse_models_array(response->str, "data", FALSE, &parse_error);

  if(!models && error)
    *error = g_strdup_printf(_("%s returned an invalid model list: %s"),
                             dt_assistant_provider_label(provider),
                             parse_error ? parse_error : _("parse error"));
  g_free(parse_error);
  g_string_free(response, TRUE);
  return models;
}

GPtrArray *dt_assistant_list_models(dt_assistant_provider_t provider,
                                    const char *endpoint,
                                    const char *api_key,
                                    gboolean include_ollama_cloud,
                                    char **error)
{
  if(error) *error = NULL;
  if(provider == DT_ASSISTANT_PROVIDER_OLLAMA)
    return _list_ollama_models(endpoint, include_ollama_cloud, error);
  return _list_cloud_models(provider, endpoint, api_key, error);
}

static char *_node_to_json(JsonNode *node)
{
  JsonGenerator *generator = json_generator_new();
  json_generator_set_root(generator, node);
  char *text = json_generator_to_data(generator, NULL);
  g_object_unref(generator);
  return text;
}

static char *_parse_response(const dt_assistant_api_t api,
                             const char *body,
                             char **error)
{
  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, body, -1, NULL))
  {
    if(error) *error = g_strdup(_("provider returned invalid JSON"));
    g_object_unref(parser);
    return NULL;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    if(error) *error = g_strdup(_("provider returned an unexpected response"));
    g_object_unref(parser);
    return NULL;
  }

  JsonObject *obj = json_node_get_object(root);
  char *result = NULL;

  if(api == DT_ASSISTANT_API_OLLAMA)
  {
    if(json_object_has_member(obj, "message"))
    {
      JsonNode *message_node = json_object_get_member(obj, "message");
      if(message_node && JSON_NODE_HOLDS_OBJECT(message_node))
      {
        JsonObject *message = json_node_get_object(message_node);
        if(json_object_has_member(message, "content"))
        {
          const char *content = json_object_get_string_member(message, "content");
          if(content) result = g_strdup(content);
        }
      }
    }
  }
  else if(api == DT_ASSISTANT_API_OPENAI)
  {
    if(json_object_has_member(obj, "choices"))
    {
      JsonNode *choices_node = json_object_get_member(obj, "choices");
      if(choices_node && JSON_NODE_HOLDS_ARRAY(choices_node))
      {
        JsonArray *choices = json_node_get_array(choices_node);
        if(json_array_get_length(choices) > 0)
        {
          JsonNode *choice_node = json_array_get_element(choices, 0);
          if(choice_node && JSON_NODE_HOLDS_OBJECT(choice_node))
          {
            JsonObject *choice = json_node_get_object(choice_node);
            if(json_object_has_member(choice, "message"))
            {
              JsonNode *message_node = json_object_get_member(choice, "message");
              if(message_node && JSON_NODE_HOLDS_OBJECT(message_node))
              {
                JsonObject *message = json_node_get_object(message_node);
                if(json_object_has_member(message, "content"))
                {
                  const char *content = json_object_get_string_member(message, "content");
                  if(content) result = g_strdup(content);
                }
              }
            }
          }
        }
      }
    }
  }
  else if(api == DT_ASSISTANT_API_ANTHROPIC)
  {
    GString *text = g_string_new(NULL);
    if(json_object_has_member(obj, "content"))
    {
      JsonNode *content_node = json_object_get_member(obj, "content");
      if(content_node && JSON_NODE_HOLDS_ARRAY(content_node))
      {
        JsonArray *content = json_node_get_array(content_node);
        for(guint i = 0; i < json_array_get_length(content); i++)
        {
          JsonNode *block_node = json_array_get_element(content, i);
          if(!block_node || !JSON_NODE_HOLDS_OBJECT(block_node)) continue;
          JsonObject *block = json_node_get_object(block_node);
          const char *type = json_object_has_member(block, "type")
                               ? json_object_get_string_member(block, "type") : NULL;
          const char *name = json_object_has_member(block, "name")
                               ? json_object_get_string_member(block, "name") : NULL;
          if(!g_strcmp0(type, "tool_use") && !g_strcmp0(name, "submit_plan")
             && json_object_has_member(block, "input"))
          {
            JsonNode *input = json_object_get_member(block, "input");
            if(input)
            {
              result = _node_to_json(input);
              break;
            }
          }
          if(!g_strcmp0(type, "text") && json_object_has_member(block, "text"))
          {
            const char *part = json_object_get_string_member(block, "text");
            if(part) g_string_append(text, part);
          }
        }
      }
    }
    if(!result && text->len) result = g_string_free(text, FALSE);
    else g_string_free(text, TRUE);
  }
  else if(api == DT_ASSISTANT_API_GEMINI)
  {
    GString *text = g_string_new(NULL);
    if(json_object_has_member(obj, "candidates"))
    {
      JsonNode *candidates_node = json_object_get_member(obj, "candidates");
      if(candidates_node && JSON_NODE_HOLDS_ARRAY(candidates_node))
      {
        JsonArray *candidates = json_node_get_array(candidates_node);
        if(json_array_get_length(candidates) > 0)
        {
          JsonNode *candidate_node = json_array_get_element(candidates, 0);
          if(candidate_node && JSON_NODE_HOLDS_OBJECT(candidate_node))
          {
            JsonObject *candidate = json_node_get_object(candidate_node);
            if(json_object_has_member(candidate, "content"))
            {
              JsonNode *content_node = json_object_get_member(candidate, "content");
              if(content_node && JSON_NODE_HOLDS_OBJECT(content_node))
              {
                JsonObject *content = json_node_get_object(content_node);
                if(json_object_has_member(content, "parts"))
                {
                  JsonNode *parts_node = json_object_get_member(content, "parts");
                  if(parts_node && JSON_NODE_HOLDS_ARRAY(parts_node))
                  {
                    JsonArray *parts = json_node_get_array(parts_node);
                    for(guint i = 0; i < json_array_get_length(parts); i++)
                    {
                      JsonNode *part_node = json_array_get_element(parts, i);
                      if(!part_node || !JSON_NODE_HOLDS_OBJECT(part_node)) continue;
                      JsonObject *part = json_node_get_object(part_node);
                      if(json_object_has_member(part, "text"))
                      {
                        const char *value = json_object_get_string_member(part, "text");
                        if(value) g_string_append(text, value);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if(text->len) result = g_string_free(text, FALSE);
    else g_string_free(text, TRUE);

    if(!result && json_object_has_member(obj, "promptFeedback"))
    {
      JsonNode *feedback_node = json_object_get_member(obj, "promptFeedback");
      if(feedback_node && JSON_NODE_HOLDS_OBJECT(feedback_node))
      {
        JsonObject *feedback = json_node_get_object(feedback_node);
        if(json_object_has_member(feedback, "blockReason"))
        {
          const char *reason = json_object_get_string_member(feedback, "blockReason");
          if(error)
            *error = g_strdup_printf(_("Gemini blocked the request: %s"),
                                     reason ? reason : _("unknown reason"));
        }
      }
    }
  }

  if(!result && error && !*error)
    *error = g_strdup(_("provider returned no usable message content"));

  g_object_unref(parser);
  return result;
}

static char *_gemini_chat_url(const char *endpoint, const char *model)
{
  char *base = g_strdup(endpoint ? endpoint : "");
  while(base[0] && base[strlen(base) - 1] == '/')
    base[strlen(base) - 1] = '\0';
  const char *model_id = model ? model : "";
  if(g_str_has_prefix(model_id, "models/")) model_id += strlen("models/");
  char *escaped = g_uri_escape_string(model_id, NULL, FALSE);
  char *url = g_strdup_printf("%s/models/%s:generateContent", base, escaped);
  g_free(escaped);
  g_free(base);
  return url;
}

gboolean dt_assistant_error_is_image_retryable(const char *error)
{
  if(!error) return FALSE;
  return strstr(error, "HTTP 400") || strstr(error, "HTTP 413")
         || strstr(error, "HTTP 415") || strstr(error, "HTTP 422")
         || strstr(error, "multimodal") || strstr(error, "image")
         || strstr(error, "vision");
}

char *dt_assistant_chat(const dt_assistant_config_t *config,
                        const char *system_prompt,
                        const char *user_prompt,
                        const char *image_base64,
                        volatile gint *cancel,
                        char **error)
{
  if(error) *error = NULL;

  char *reason = NULL;
  if(!dt_assistant_config_is_ready(config, &reason))
  {
    if(error) *error = reason ? reason : g_strdup(_("assistant is not configured"));
    else g_free(reason);
    return NULL;
  }

  char *body = _build_request_body(config, system_prompt, user_prompt,
                                   image_base64);
  if(!body)
  {
    if(error) *error = g_strdup(_("could not build the request"));
    return NULL;
  }

  CURL *curl = curl_easy_init();
  if(!curl)
  {
    g_free(body);
    if(error) *error = g_strdup(_("could not initialize the network client"));
    return NULL;
  }

  dt_curl_init(curl, FALSE);

  char *url = config->api == DT_ASSISTANT_API_GEMINI
                ? _gemini_chat_url(config->endpoint, config->model)
                : g_strdup(config->endpoint);

  GString *response = g_string_new(NULL);
  struct curl_slist *headers = NULL;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  if(config->api == DT_ASSISTANT_API_OPENAI)
  {
    if(config->api_key && config->api_key[0])
    {
      char *authorization
        = g_strdup_printf("Authorization: Bearer %s", config->api_key);
      headers = curl_slist_append(headers, authorization);
      g_free(authorization);
    }
  }
  else if(config->api == DT_ASSISTANT_API_ANTHROPIC)
  {
    if(config->api_key && config->api_key[0])
    {
      char *authorization = g_strdup_printf("x-api-key: %s", config->api_key);
      headers = curl_slist_append(headers, authorization);
      g_free(authorization);
    }
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
  }
  else if(config->api == DT_ASSISTANT_API_GEMINI)
  {
    if(config->api_key && config->api_key[0])
    {
      char *authorization = g_strdup_printf("x-goog-api-key: %s", config->api_key);
      headers = curl_slist_append(headers, authorization);
      g_free(authorization);
    }
  }

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)config->timeout_s);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, _progress_cb);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, cancel);

  const CURLcode result = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  g_free(body);
  g_free(url);

  char *text = NULL;

  if(result == CURLE_ABORTED_BY_CALLBACK && cancel && g_atomic_int_get(cancel))
  {
    if(error) *error = g_strdup(_("request cancelled"));
  }
  else if(result == CURLE_OPERATION_TIMEDOUT)
  {
    if(error)
    {
      if(config->provider == DT_ASSISTANT_PROVIDER_OLLAMA)
        *error = g_strdup_printf(_("Ollama did not answer within %d seconds. "
                                   "The model may still be loading, or it may be "
                                   "too large for this machine. Try a smaller model."),
                                 config->timeout_s);
      else
        *error = g_strdup_printf(_("the provider did not answer within %d seconds"),
                                 config->timeout_s);
    }
  }
  else if(result == CURLE_COULDNT_CONNECT
          || result == CURLE_COULDNT_RESOLVE_HOST)
  {
    if(error)
    {
      if(config->provider == DT_ASSISTANT_PROVIDER_OLLAMA)
        *error = g_strdup_printf(_("Ollama is not reachable at %s. "
                                   "Make sure it is running."),
                                 config->endpoint);
      else
        *error = g_strdup_printf(_("network error: %s"),
                                 curl_easy_strerror(result));
    }
  }
  else if(result != CURLE_OK)
  {
    if(error)
    {
      if(config->provider == DT_ASSISTANT_PROVIDER_OLLAMA)
        *error = g_strdup_printf(_("Ollama request failed at %s: %s"),
                                 config->endpoint, curl_easy_strerror(result));
      else
        *error = g_strdup_printf(_("network error: %s"),
                                 curl_easy_strerror(result));
    }
  }
  else if(http_code < 200 || http_code >= 300)
  {
    char *detail = _error_from_body(response->str);
    if(error)
    {
      if(config->provider == DT_ASSISTANT_PROVIDER_OLLAMA && http_code == 404)
        *error = g_strdup_printf(_("Ollama model '%s' is not installed. "
                                   "Open Preferences > Assistant and click Detect."),
                                 config->model);
      else if(http_code == 401 || http_code == 403)
        *error = g_strdup_printf(_("authentication failed for %s (HTTP %ld): %s. "
                                   "Check that the API key belongs to this provider "
                                   "and is the secret key, not its identifier."),
                                 dt_assistant_provider_label(config->provider),
                                 http_code, detail ? detail : "");
      else
        *error = g_strdup_printf(_("%s returned HTTP %ld: %s"),
                                 dt_assistant_provider_label(config->provider),
                                 http_code, detail ? detail : "");
    }
    g_free(detail);
  }
  else
  {
    text = _parse_response(config->api, response->str, error);
  }

  g_string_free(response, TRUE);
  return text;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

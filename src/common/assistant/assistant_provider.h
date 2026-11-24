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

#define DT_ASSISTANT_CONF_PROVIDER "plugins/assistant/provider"
#define DT_ASSISTANT_CONF_ENDPOINT "plugins/assistant/endpoint"
#define DT_ASSISTANT_CONF_MODEL    "plugins/assistant/model"
#define DT_ASSISTANT_CONF_TIMEOUT  "plugins/assistant/timeout"
#define DT_ASSISTANT_CONF_ALLOW_CLOUD "plugins/assistant/allow_cloud_models"
#define DT_ASSISTANT_CONF_SEND_IMAGE "plugins/assistant/send_image"

#define DT_ASSISTANT_PW_SLOT "ai_assistant"
#define DT_ASSISTANT_PW_KEY  "api_key"

typedef enum
{
  DT_ASSISTANT_PROVIDER_OLLAMA = 0,
  DT_ASSISTANT_PROVIDER_DEEPSEEK,
  DT_ASSISTANT_PROVIDER_OPENAI,
  DT_ASSISTANT_PROVIDER_OPENROUTER,
  DT_ASSISTANT_PROVIDER_GROQ,
  DT_ASSISTANT_PROVIDER_MISTRAL,
  DT_ASSISTANT_PROVIDER_XAI,
  DT_ASSISTANT_PROVIDER_TOGETHER,
  DT_ASSISTANT_PROVIDER_FIREWORKS,
  DT_ASSISTANT_PROVIDER_PERPLEXITY,
  DT_ASSISTANT_PROVIDER_LOCAL_OPENAI,
  DT_ASSISTANT_PROVIDER_GEMINI,
  DT_ASSISTANT_PROVIDER_ANTHROPIC,
  DT_ASSISTANT_PROVIDER_CUSTOM,
  DT_ASSISTANT_PROVIDER_COUNT
} dt_assistant_provider_t;

typedef enum
{
  DT_ASSISTANT_API_OLLAMA = 0,
  DT_ASSISTANT_API_OPENAI,
  DT_ASSISTANT_API_GEMINI,
  DT_ASSISTANT_API_ANTHROPIC,
} dt_assistant_api_t;

typedef struct
{
  dt_assistant_provider_t provider;
  const char *id;
  const char *label;
  dt_assistant_api_t api;
  const char *endpoint;
  const char *model;
  gboolean cloud;
  gboolean needs_key;
  gboolean vision;
} dt_assistant_provider_def_t;

typedef struct
{
  dt_assistant_provider_t provider;
  dt_assistant_api_t api;
  char *endpoint;
  char *model;
  char *api_key;
  int timeout_s;
  gboolean send_image;
} dt_assistant_config_t;

int dt_assistant_provider_count(void);
dt_assistant_provider_t dt_assistant_provider_from_index(int index);
const dt_assistant_provider_def_t *dt_assistant_provider_def(
  dt_assistant_provider_t provider);
const char *dt_assistant_provider_id(dt_assistant_provider_t provider);
const char *dt_assistant_provider_label(dt_assistant_provider_t provider);
dt_assistant_api_t dt_assistant_provider_api(dt_assistant_provider_t provider);
gboolean dt_assistant_provider_is_cloud(dt_assistant_provider_t provider);
gboolean dt_assistant_provider_needs_key(dt_assistant_provider_t provider);
gboolean dt_assistant_provider_supports_vision(dt_assistant_provider_t provider);
char *dt_assistant_provider_conf_key(dt_assistant_provider_t provider,
                                     const char *name);

dt_assistant_provider_t dt_assistant_provider_from_string(const char *value);
const char *dt_assistant_provider_to_string(dt_assistant_provider_t provider);

void dt_assistant_provider_defaults(dt_assistant_provider_t provider,
                                    char **endpoint,
                                    char **model);

dt_assistant_config_t dt_assistant_config_load(void);
void dt_assistant_config_clear(dt_assistant_config_t *config);
gboolean dt_assistant_config_is_ready(const dt_assistant_config_t *config,
                                      char **reason);

gboolean dt_assistant_secure_storage_available(void);
gboolean dt_assistant_store_key(dt_assistant_provider_t provider,
                                const char *key,
                                char **error);
gboolean dt_assistant_has_stored_key(dt_assistant_provider_t provider);
gboolean dt_assistant_has_runtime_key(dt_assistant_provider_t provider);
void dt_assistant_set_runtime_key(dt_assistant_provider_t provider,
                                  const char *key);

char *dt_assistant_chat(const dt_assistant_config_t *config,
                        const char *system_prompt,
                        const char *user_prompt,
                        const char *image_base64,
                        volatile gint *cancel,
                        char **error);
gboolean dt_assistant_error_is_image_retryable(const char *error);

GPtrArray *dt_assistant_list_models(dt_assistant_provider_t provider,
                                    const char *endpoint,
                                    const char *api_key,
                                    gboolean include_ollama_cloud,
                                    char **error);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

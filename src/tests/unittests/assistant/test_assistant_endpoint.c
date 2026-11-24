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

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cmocka.h>

#include "common/assistant/assistant_provider.h"

static gboolean _check_endpoint(const char *endpoint, char **reason)
{
  dt_assistant_config_t config = { 0 };
  config.provider = DT_ASSISTANT_PROVIDER_CUSTOM;
  config.endpoint = g_strdup(endpoint);
  config.model = g_strdup("test-model");
  if(reason) *reason = NULL;
  const gboolean ready = dt_assistant_config_is_ready(&config, reason);
  dt_assistant_config_clear(&config);
  return ready;
}

static gboolean _endpoint_ready(const char *endpoint)
{
  char *reason = NULL;
  const gboolean ready = _check_endpoint(endpoint, &reason);
  g_free(reason);
  return ready;
}

static void test_http_localhost(void **state)
{
  (void)state;
  assert_true(_endpoint_ready("http://localhost:11434/api/chat"));
}

static void test_http_ipv4_loopback(void **state)
{
  (void)state;
  assert_true(_endpoint_ready("http://127.0.0.1:8080/v1/chat/completions"));
  assert_true(_endpoint_ready("http://127.2.3.4/v1/chat/completions"));
}

static void test_http_ipv6_loopback(void **state)
{
  (void)state;
  assert_true(_endpoint_ready("http://[::1]:8080/v1/chat/completions"));
}

static void test_http_remote_rejected(void **state)
{
  (void)state;
  assert_false(_endpoint_ready("http://example.com/v1/chat/completions"));
}

static void test_http_remote_path_localhost_rejected(void **state)
{
  (void)state;
  assert_false(_endpoint_ready("http://example.com/http://localhost/v1/chat"));
}

static void test_http_remote_query_localhost_rejected(void **state)
{
  (void)state;
  assert_false(_endpoint_ready("http://example.com/v1/chat?target=http://localhost"));
}

static void test_malformed_rejected(void **state)
{
  (void)state;
  const char *const malformed[] = {
    "http://",
    "http://[::1",
    "not a URL",
  };

  for(size_t i = 0; i < G_N_ELEMENTS(malformed); i++)
  {
    char *reason = NULL;
    assert_false(_check_endpoint(malformed[i], &reason));
    assert_non_null(reason);
    assert_true(reason[0] != '\0');
    g_free(reason);
  }
}

static void test_https_remote_allowed(void **state)
{
  (void)state;
  assert_true(_endpoint_ready("https://example.com/v1/chat/completions"));
}

static void test_lookalike_localhost_rejected(void **state)
{
  (void)state;
  assert_false(_endpoint_ready("http://localhost.example.com/v1/chat"));
}

static void test_userinfo_does_not_hide_remote_host(void **state)
{
  (void)state;
  assert_false(_endpoint_ready("http://localhost@example.com/v1/chat"));
}

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_http_localhost),
    cmocka_unit_test(test_http_ipv4_loopback),
    cmocka_unit_test(test_http_ipv6_loopback),
    cmocka_unit_test(test_http_remote_rejected),
    cmocka_unit_test(test_http_remote_path_localhost_rejected),
    cmocka_unit_test(test_http_remote_query_localhost_rejected),
    cmocka_unit_test(test_malformed_rejected),
    cmocka_unit_test(test_https_remote_allowed),
    cmocka_unit_test(test_lookalike_localhost_rejected),
    cmocka_unit_test(test_userinfo_does_not_hide_remote_host),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}

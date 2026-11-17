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
/*
 * cmocka unit tests for the RCD green debiasing helper.
 */

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmocka.h>

#include "iop/demosaicing/rcd_debias.h"

#define E 1e-6f
#define RGGB 0x94949494u

static int is_green(const int row, const int col, const uint32_t filters)
{
  return FC(row, col, filters) == GREEN;
}

static void fill_checker(float *const green, const int width, const int height,
                         const uint32_t filters, const float g_green,
                         const float g_nongreen)
{
  for(int row = 0; row < height; row++)
    for(int col = 0; col < width; col++)
      green[(size_t)row * width + col] = is_green(row, col, filters) ? g_green : g_nongreen;
}

static void test_uniform_no_change(void **state)
{
  (void)state;
  enum { W = 80, H = 80 };
  float *const green = malloc(sizeof(float) * W * H);
  float *const mod = malloc(sizeof(float) * W * H);
  float *const level = malloc(sizeof(float) * W * H);
  assert_non_null(green);
  assert_non_null(mod);
  assert_non_null(level);

  for(int i = 0; i < W * H; i++) green[i] = 0.8f;
  rcd_debias_apply(green, mod, level, H, W, W, 0, 0, RGGB);

  for(int i = 0; i < W * H; i++)
    assert_float_equal(green[i], 0.8f, E);

  free(green);
  free(mod);
  free(level);
}

static void test_checker_removed(void **state)
{
  (void)state;
  enum { W = 80, H = 80, ROI = 8 };
  const float g_green = 0.8f;
  const float g_nongreen = 0.82f;
  float *const green = malloc(sizeof(float) * W * H);
  float *const mod = malloc(sizeof(float) * W * H);
  float *const level = malloc(sizeof(float) * W * H);
  float *const original = malloc(sizeof(float) * W * H);
  assert_non_null(green);
  assert_non_null(mod);
  assert_non_null(level);
  assert_non_null(original);

  fill_checker(green, W, H, RGGB, g_green, g_nongreen);
  memcpy(original, green, sizeof(float) * W * H);

  rcd_debias_apply(green, mod, level, H, W, W, 0, 0, RGGB);

  double sum_green = 0.0;
  double sum_nongreen = 0.0;
  int count_green = 0;
  int count_nongreen = 0;
  for(int row = ROI; row < H - ROI; row++)
  {
    for(int col = ROI; col < W - ROI; col++)
    {
      const size_t idx = (size_t)row * W + col;
      if(is_green(row, col, RGGB))
      {
        sum_green += green[idx];
        count_green++;
        assert_float_equal(green[idx], original[idx], E);
      }
      else
      {
        sum_nongreen += green[idx];
        count_nongreen++;
      }
    }
  }

  assert_true(count_green > 0);
  assert_true(count_nongreen > 0);
  const double checker_after = sum_nongreen / count_nongreen - sum_green / count_green;
  assert_true(fabs(checker_after) < 1e-4);

  free(green);
  free(mod);
  free(level);
  free(original);
}

static void test_dark_limited(void **state)
{
  (void)state;
  enum { W = 80, H = 80, ROI = 8 };
  const float g_green = 0.02f;
  const float g_nongreen = 0.022f;
  float *const green = malloc(sizeof(float) * W * H);
  float *const original = malloc(sizeof(float) * W * H);
  float *const mod = malloc(sizeof(float) * W * H);
  float *const level = malloc(sizeof(float) * W * H);
  assert_non_null(green);
  assert_non_null(original);
  assert_non_null(mod);
  assert_non_null(level);

  fill_checker(green, W, H, RGGB, g_green, g_nongreen);
  memcpy(original, green, sizeof(float) * W * H);

  rcd_debias_apply(green, mod, level, H, W, W, 0, 0, RGGB);

  float max_correction = 0.0f;
  float min_green = 1.0f;
  double sum_green = 0.0;
  double sum_nongreen = 0.0;
  int count_green = 0;
  int count_nongreen = 0;
  for(int row = ROI; row < H - ROI; row++)
  {
    for(int col = ROI; col < W - ROI; col++)
    {
      const size_t idx = (size_t)row * W + col;
      const float correction = original[idx] - green[idx];
      max_correction = fmaxf(max_correction, fabsf(correction));
      min_green = fminf(min_green, green[idx]);
      if(is_green(row, col, RGGB))
      {
        sum_green += green[idx];
        count_green++;
      }
      else
      {
        sum_nongreen += green[idx];
        count_nongreen++;
      }
    }
  }

  assert_true(count_green > 0);
  assert_true(count_nongreen > 0);
  assert_true(max_correction <= 1.5e-3f);
  assert_true(min_green >= 0.0f);

  const double checker_after = sum_nongreen / count_nongreen - sum_green / count_green;
  assert_true(fabs(checker_after) > 5e-4);

  free(green);
  free(original);
  free(mod);
  free(level);
}

static float synthetic_green(const int row, const int col, const uint32_t filters)
{
  const float base = 0.5f + 0.12f * sinf(0.13f * col) * cosf(0.11f * row)
                     + 0.04f * (float)((row * 13 + col * 7) % 11) / 11.0f;
  return base + (is_green(row, col, filters) ? -0.01f : 0.01f);
}

static void test_tiled_matches_full(void **state)
{
  (void)state;
  enum { W = 110, H = 110, TILE = 48, STRIDE = 28, BORDER = 10 };
  float *const source = malloc(sizeof(float) * W * H);
  float *const full = malloc(sizeof(float) * W * H);
  float *const tiled = malloc(sizeof(float) * W * H);
  float *const tile_green = malloc(sizeof(float) * TILE * TILE);
  float *const mod = malloc(sizeof(float) * W * H);
  float *const level = malloc(sizeof(float) * W * H);
  assert_non_null(source);
  assert_non_null(full);
  assert_non_null(tiled);
  assert_non_null(tile_green);
  assert_non_null(mod);
  assert_non_null(level);

  for(int row = 0; row < H; row++)
    for(int col = 0; col < W; col++)
      source[(size_t)row * W + col] = synthetic_green(row, col, RGGB);

  memcpy(full, source, sizeof(float) * W * H);
  rcd_debias_apply(full, mod, level, H, W, W, 0, 0, RGGB);

  memcpy(tiled, source, sizeof(float) * W * H);
  for(int row_start = 0; row_start < H; row_start += STRIDE)
  {
    const int tile_rows = (TILE < H - row_start) ? TILE : (H - row_start);
    for(int col_start = 0; col_start < W; col_start += STRIDE)
    {
      const int tile_cols = (TILE < W - col_start) ? TILE : (W - col_start);
      if(tile_rows <= 2 * BORDER || tile_cols <= 2 * BORDER) continue;

      for(int row = 0; row < tile_rows; row++)
        memcpy(tile_green + (size_t)row * TILE,
               source + (size_t)(row_start + row) * W + col_start,
               sizeof(float) * tile_cols);

      rcd_debias_apply(tile_green, mod, level, tile_rows, tile_cols, TILE,
                       row_start, col_start, RGGB);

      const int out_rows = tile_rows - 2 * BORDER;
      const int out_cols = tile_cols - 2 * BORDER;
      for(int row = 0; row < out_rows; row++)
        memcpy(tiled + (size_t)(row_start + BORDER + row) * W + col_start + BORDER,
               tile_green + (size_t)(BORDER + row) * TILE + BORDER,
               sizeof(float) * out_cols);
    }
  }

  float max_diff = 0.0f;
  for(int row = BORDER; row < H - BORDER; row++)
    for(int col = BORDER; col < W - BORDER; col++)
    {
      const size_t idx = (size_t)row * W + col;
      max_diff = fmaxf(max_diff, fabsf(full[idx] - tiled[idx]));
    }
  assert_true(max_diff < 1e-5f);

  free(source);
  free(full);
  free(tiled);
  free(tile_green);
  free(mod);
  free(level);
}

int main(int argc, char *argv[])
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_uniform_no_change),
    cmocka_unit_test(test_checker_removed),
    cmocka_unit_test(test_dark_limited),
    cmocka_unit_test(test_tiled_matches_full),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

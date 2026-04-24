/*
    This file is part of darktable,
    copyright (c) 2016-2026 darktable developers
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

    ---------------------------------------------------------------------------
    Acknowledgments and Technical References (Libre DT-lab):

    - Oklab: Björn Ottosson (2020) — "A perceptual color space for image processing"
    - ACES 1.0: Stephen Hill / Narkowicz (2016) — Filmic tone mapping fit.
    - ACES 2.0: Narkowicz & Filiberto (2021) — Rational RRT/ODT approximation.
    - OpenDRT: J. Peddie — "open-display-transform" (vector norm concepts, 
                pre-tonescale brilliance, and gamut mapping).
    - JzAzBz: Safdar et al. (2017) — used in Kinematic and Dynamic modes.
    ---------------------------------------------------------------------------

#include "color_conversion.h"
#include "rgb_norms.h"

/*
  Narkowicz (2016) rational approximation of the ACES RRT+ODT curve for sRGB output.
  Widely used in real-time rendering for its simplicity and visual quality.
  Does NOT implement the full ACES pipeline (no color space transform, no D60 whitepoint).
  Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
*/
inline float _aces_tone_map(const float x)
{
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;

  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}
/*
  Narkowicz & Filiberto (2021) rational approximation of the ACES 2.0 RRT curve.
  More precise than the basic Narkowicz 2016 fit, with a softer shoulder.
  The pre-scale factor (x * 1.680) in the caller adjusts the exposure point.
  Does NOT implement the full ACES pipeline (no color space transform, no D60 whitepoint).
  Reference: https://github.com/h3r2tic/tony-mc-mapface (Narkowicz/Filiberto fit)
*/
inline float _aces_20_tonemap(const float x)
{
  const float a = 0.0245786f;
  const float b = 0.000090537f;
  const float c = 0.983729f;
  const float d = 0.4329510f;
  const float e = 0.238081f;

  return clamp((x * (x + a) - b) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// --- CONVERSIONS OKLAB POUR OPENCL ---
inline float3 rgb_to_oklab(float3 c)
{
  float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
  float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
  float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;

  float l_ = cbrt(fmax(l, 0.0f));
  float m_ = cbrt(fmax(m, 0.0f));
  float s_ = cbrt(fmax(s, 0.0f));

  return (float3)(
      0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720403f * s_,
      1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
      0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_);
}

inline float3 oklab_to_rgb(float3 c)
{
  float l_ = c.x + 0.3963377774f * c.y + 0.2158037573f * c.z;
  float m_ = c.x - 0.1055613458f * c.y - 0.0638541728f * c.z;
  float s_ = c.x - 0.0894841775f * c.y - 1.2914855480f * c.z;

  return (float3)(
      +4.0767416621f * (l_ * l_ * l_) - 3.3077115913f * (m_ * m_ * m_) + 0.2309699292f * (s_ * s_ * s_),
      -1.2684380046f * (l_ * l_ * l_) + 2.6097574011f * (m_ * m_ * m_) - 0.3413193965f * (s_ * s_ * s_),
      -0.0041960863f * (l_ * l_ * l_) - 0.7034186147f * (m_ * m_ * m_) + 1.7076147010f * (s_ * s_ * s_));
}

/*
  Primary LUT lookup.  Measures the luminance of a given pixel using a selectable function, looks up that
  luminance in the configured basecurve, and then scales each channel by the result.

  Doing it this way avoids the color shifts documented as being possible in the legacy basecurve approach.

  Also applies a multiplier prior to lookup in order to support fusion.  The idea of doing this here is to
  emulate the original use case of enfuse, which was to fuse multiple JPEGs from a camera that was set up
  for exposure bracketing, and which may have had a camera-specific base curve applied.
*/
kernel void
basecurve_lut(read_only image2d_t in, 
              write_only image2d_t out, 
              const int width, const int height,
              const float mul, 
              read_only image2d_t table, 
              constant float *a, 
              const int preserve_colors,
              constant dt_colorspaces_iccprofile_info_cl_t *profile_info, 
              read_only image2d_t lut,
              const int use_work_profile)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 pixel = read_imagef(in, sampleri, (int2)(x, y));

  float ratio = 1.f;
  const float lum = mul * dt_rgb_norm(pixel, preserve_colors, use_work_profile, profile_info, lut);
  if(lum > 0.f)
  {
    const float curve_lum = lookup_unbounded(table, lum, a);
    ratio = mul * curve_lum / lum;
  }
  pixel.xyz *= ratio;
  pixel = fmax(pixel, 0.f);

  write_imagef (out, (int2)(x, y), pixel);
}


kernel void
basecurve_zero(write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  write_imagef (out, (int2)(x, y), (float4)0.0f);
}

/*
  Original basecurve implementation.  Applies a LUT on a per-channel basis which can cause color shifts.

  These can be undesirable (skin tone shifts), or sometimes may be desired (fiery sunset).  Continue to allow
  the "old" method but don't make it the default, both for backwards compatibility and for those who are willing
  to take the risks of "artistic" impacts on their image.
*/
kernel void
basecurve_legacy_lut(read_only image2d_t in, 
                    write_only image2d_t out, 
                    const int width, const int height,
                    const float mul, read_only image2d_t table, 
                    constant float *a)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 pixel = read_imagef(in, sampleri, (int2)(x, y));

  // apply ev multiplier and use lut or extrapolation:
  float3 f = pixel.xyz * mul;

  pixel.x = lookup_unbounded(table, f.x, a);
  pixel.y = lookup_unbounded(table, f.y, a);
  pixel.z = lookup_unbounded(table, f.z, a);
  pixel = fmax(pixel, 0.f);
  write_imagef (out, (int2)(x, y), pixel);
}

kernel void
basecurve_compute_features(read_only image2d_t in, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 value = read_imagef(in, sampleri, (int2)(x, y));

  const float ma = fmax(value.x, fmax(value.y, value.z));
  const float mi = fmin(value.x, fmin(value.y, value.z));

  const float sat = 0.1f + 0.1f * (ma - mi) / fmax(1.0e-4f, ma);
  value.w = sat;

  const float c = 0.54f;

  float v = fabs(value.x - c);
  v = fmax(fabs(value.y - c), v);
  v = fmax(fabs(value.z - c), v);

  const float var = 0.5f;
  const float e = 0.2f + dt_fast_expf(-v * v / (var * var));

  value.w *= e;

  write_imagef (out, (int2)(x, y), value);
}

constant float gw[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };

kernel void
basecurve_blur_h(read_only image2d_t in, write_only image2d_t out,
                 const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  const int rad = 2;
  constant float *w = gw + rad;

  float4 sum = (float4)0.0f;

  for (int i = -rad; i <= rad; i++)
  {
    const int xx = min(max(-x - i, x + i), width - (x + i - width + 1));
    sum += read_imagef(in, sampleri, (int2)(xx, y)) * w[i];
  }

  write_imagef (out, (int2)(x, y), sum);
}


kernel void
basecurve_blur_v(read_only image2d_t in, write_only image2d_t out,
                 const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);


  if(x >= width || y >= height) return;

  const int rad = 2;
  constant float *w = gw + rad;

  float4 sum = (float4)0.0f;

  for (int i = -rad; i <= rad; i++)
  {
    const int yy = min(max(-y - i, y + i), height - (y + i - height + 1));
    sum += read_imagef(in, sampleri, (int2)(x, yy)) * w[i];
  }

  write_imagef (out, (int2)(x, y), sum);
}

kernel void
basecurve_expand(read_only image2d_t in, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  // fill numbers in even pixels, zero odd ones
  float4 pixel = (x % 2 == 0 && y % 2 == 0) ? 4.0f * read_imagef(in, sampleri, (int2)(x / 2, y / 2)) : (float4)0.0f;

  write_imagef (out, (int2)(x, y), pixel);
}

kernel void
basecurve_reduce(read_only image2d_t in, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 pixel = read_imagef(in, sampleri, (int2)(2 * x, 2 * y));

  write_imagef (out, (int2)(x, y), pixel);
}

kernel void
basecurve_detail(read_only image2d_t in, read_only image2d_t det, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 input = read_imagef(in, sampleri, (int2)(x, y));
  float4 detail = read_imagef(det, sampleri, (int2)(x, y));

  write_imagef (out, (int2)(x, y), input - detail);
}

kernel void
basecurve_adjust_features(read_only image2d_t in, read_only image2d_t det, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 input = read_imagef(in, sampleri, (int2)(x, y));
  float4 detail = read_imagef(det, sampleri, (int2)(x, y));

  input.w *= 0.1f + sqrt(detail.x * detail.x + detail.y * detail.y + detail.z * detail.z);

  write_imagef (out, (int2)(x, y), input);
}

kernel void
basecurve_blend_gaussian(read_only image2d_t in, read_only image2d_t col, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 comb = read_imagef(in, sampleri, (int2)(x, y));
  float4 collect = read_imagef(col, sampleri, (int2)(x, y));

  comb.xyz += collect.xyz * collect.w;
  comb.w += collect.w;

  write_imagef (out, (int2)(x, y), comb);
}

kernel void
basecurve_blend_laplacian(read_only image2d_t in, read_only image2d_t col, read_only image2d_t tmp, write_only image2d_t out,
                          const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 comb = read_imagef(in, sampleri, (int2)(x, y));
  float4 collect = read_imagef(col, sampleri, (int2)(x, y));
  float4 temp = read_imagef(tmp, sampleri, (int2)(x, y));

  comb.xyz += (collect.xyz - temp.xyz) * collect.w;
  comb.w += collect.w;

  write_imagef (out, (int2)(x, y), comb);
}

kernel void
basecurve_normalize(read_only image2d_t in, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 comb = read_imagef(in, sampleri, (int2)(x, y));

  comb.xyz /= (comb.w > 1.0e-8f) ? comb.w : 1.0f;

  write_imagef (out, (int2)(x, y), comb);
}

kernel void
basecurve_reconstruct(read_only image2d_t in, read_only image2d_t tmp, write_only image2d_t out, const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 comb = read_imagef(in, sampleri, (int2)(x, y));
  float4 temp = read_imagef(tmp, sampleri, (int2)(x, y));

  comb += temp;

  write_imagef (out, (int2)(x, y), comb);
}

kernel void
basecurve_finalize(read_only image2d_t in,
                   read_only image2d_t comb, 
                   write_only image2d_t out, 
                   const int width,
                   const int height, 
                   const int workflow_mode, 
                   const float use_rolloff,
                   const float shadow_lift, 
                   const float highlight_gain,
                   const float saturation_boost,
                   const float ucs_saturation_balance, 
                   const float gamut_strength, 
                   const float highlight_corr, 
                   const int target_gamut, 
                   const float look_opacity, 
                   const float16 look_mat, 
                   const float alpha,
                   constant dt_colorspaces_iccprofile_info_cl_t *profile_info,
                   const int use_work_profile)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 pixel = read_imagef(comb, sampleri, (int2)(x, y));

  // Sanitize to avoid Inf/NaN propagation
  pixel.xyz = fmax(pixel.xyz, 0.0f);
  pixel.xyz = fmin(pixel.xyz, (float3)(1e6f));
  
  float3 pixel_in = pixel.xyz;

  // Apply Color Look - Moved outside to match C logic (Display-referred support)
  if (look_opacity > 0.0f)
  {
    float3 look_transformed;
    look_transformed.x = dot(pixel_in, (float3)(look_mat.s0, look_mat.s1, look_mat.s2));
    look_transformed.y = dot(pixel_in, (float3)(look_mat.s3, look_mat.s4, look_mat.s5));
    look_transformed.z = dot(pixel_in, (float3)(look_mat.s6, look_mat.s7, look_mat.s8));

    // Mix between original and transformed
    pixel.xyz = mix(pixel_in, look_transformed, look_opacity);
    pixel.xyz = fmax(pixel.xyz, 0.0f); 
    pixel_in = pixel.xyz; // Update input for following steps
  }

  // Define coefficients at top level for scope consistency
  const float r_coeff = (use_work_profile != 0 && profile_info != 0) ? profile_info->matrix_in[3] : 0.2627f;
  const float g_coeff = (use_work_profile != 0 && profile_info != 0) ? profile_info->matrix_in[4] : 0.6780f;
  const float b_coeff = (use_work_profile != 0 && profile_info != 0) ? profile_info->matrix_in[5] : 0.0593f;

  if(workflow_mode > 0 || shadow_lift != 1.0f || highlight_gain != 1.0f || ucs_saturation_balance != 0.0f || gamut_strength > 0.0f || highlight_corr != 0.0f)
  {
    if(highlight_gain != 1.0f && workflow_mode != 3)
      pixel.xyz *= highlight_gain;

    if(shadow_lift != 1.0f && workflow_mode != 3)
    {
      pixel.x = (pixel.x > 0.0f) ? pow(pixel.x, shadow_lift) : pixel.x;
      pixel.y = (pixel.y > 0.0f) ? pow(pixel.y, shadow_lift) : pixel.y;
      pixel.z = (pixel.z > 0.0f) ? pow(pixel.z, shadow_lift) : pixel.z;
    }
    
    float y_in = pixel.x * r_coeff + pixel.y * g_coeff + pixel.z * b_coeff;
    float y_out = y_in;

    /* Scene-referred: luminance-adaptive shoulder extension for ACES-like
       tonemapping using perceptual luminance Jz. */
    if(workflow_mode == 1 || workflow_mode == 2)
    {
      float3 xyz;
      if(use_work_profile != 0 && profile_info != 0)
      {
        xyz.x = profile_info->matrix_in[0] * pixel.x + profile_info->matrix_in[1] * pixel.y + profile_info->matrix_in[2] * pixel.z;
        xyz.y = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
        xyz.z = profile_info->matrix_in[6] * pixel.x + profile_info->matrix_in[7] * pixel.y + profile_info->matrix_in[8] * pixel.z;
      }
      else
      {
        xyz.x = 0.636958f * pixel.x + 0.144617f * pixel.y + 0.168881f * pixel.z;
        xyz.y = 0.262700f * pixel.x + 0.677998f * pixel.y + 0.059302f * pixel.z;
        xyz.z = 0.000000f * pixel.x + 0.028073f * pixel.y + 1.060985f * pixel.z;
      }

      xyz = fmax(xyz, (float3)(0.0f));

      float4 xyz_scaled = (float4)(xyz.x * 400.0f, xyz.y * 400.0f, xyz.z * 400.0f, 0.0f);
      float4 jab = XYZ_to_JzAzBz(xyz_scaled);

      const float L = clamp(jab.x, 0.0f, 1.0f);
      const float k = 1.0f + alpha * L * L;

      const float x_scaled = y_in / k;
      if(workflow_mode == 1)
        y_out = _aces_tone_map(x_scaled) * k;
      else
        y_out = _aces_20_tonemap(x_scaled * 2.0f) * k;
    }
    else if(workflow_mode == 3)
    {
      // Mode 3: ACES 2.0 Pure UCS (Oklab) - Pre-tonescale Brilliance
      float3 lab = rgb_to_oklab(pixel.xyz);

      // 1. Balance Saturation UCS
      const float L_ok = lab.x;
      const float mask_ok = 1.0f / (1.0f + dtcl_exp((L_ok - 0.5f) * 10.0f));
      const float weight_ok = 2.0f * mask_ok - 1.0f;
      // Boost saturation mostly in mid-tones (bell curve weight)
      const float mid_weight = 1.0f - weight_ok * weight_ok;
      // Vibrance logic: protect already saturated colors
      float chroma = length(lab.yz);
      const float vibrance_weight = fmax(0.0f, 1.0f - chroma * 2.5f);
      const float sat_mult = (1.0f + saturation_boost * mid_weight * vibrance_weight) * (1.0f + ucs_saturation_balance * (weight_ok * weight_ok * weight_ok));
      lab.y *= sat_mult;
      lab.z *= sat_mult;

      // 2. OpenDRT-style Vector Norm & Purity Compression
      const float L_achromatic = lab.x;
      chroma = length(lab.yz);

      // Calculate Vector Norm (total energy)
      float V_norm = native_sqrt(L_achromatic * L_achromatic + chroma * chroma);
      float purity = (V_norm > 1e-6f) ? (chroma / V_norm) : 0.0f;

      // Hyperbolic purity compression
      float purity_comp = purity / (1.0f + 0.05f * purity);

      // Prepare Norm for tonemapping
      float V_orig = fmax(0.0f, pow(V_norm, 1.10f)); /* OKLAB_BRILLIANCE_POWER = 1.10 */
      V_orig *= (1.189f + highlight_gain); // +0.25 EV exposure compensation
      V_orig = fmax(0.0f, pow(V_orig, shadow_lift + 1.0f));

      float V_new = _aces_20_tonemap(V_orig);

      // 3. Highlight Hue Sat (Saturation Gate)
      float compression = (V_norm > 1e-4f) ? (V_new / V_norm) : 1.0f;
      float gate_power = 0.5f * (1.0f - highlight_corr);
      float saturation_gate = clamp(pow(compression, gate_power), 0.0f, 1.0f);

      // Reconstruct L and Chroma
      lab.x = V_new * native_sqrt(fmax(0.0f, 1.0f - purity_comp * purity_comp));
      float chroma_scale = (V_new * purity_comp * saturation_gate) / fmax(chroma, 1e-6f);
      lab.y *= chroma_scale;
      lab.z *= chroma_scale;

      pixel.xyz = oklab_to_rgb(lab);
      y_out = lab.x;
    }

    float gain = y_out / fmax(y_in, 1e-6f);
    if(workflow_mode != 3) pixel.xyz *= gain;

    if(use_rolloff > 0.0f)
    {
      const float threshold = 0.80f;
      if(y_out > threshold)
      {
        float factor = (y_out - threshold) / (1.0f - threshold);
        factor = clamp(factor, 0.0f, 1.0f) * use_rolloff;
        /* In mode 3, blend towards 1.0 (white): L_new is bounded to [0,1] by ACES,
           so y_out cannot exceed 1.0, unlike modes 1/2 where k_scale allows it. */
        const float blend_target = (workflow_mode == 3) ? 1.0f : y_out;
        pixel.xyz = mix(pixel.xyz, (float3)(blend_target), factor);
      }
    }

    float4 jab = (float4)(0.0f);
    if(ucs_saturation_balance != 0.0f || gamut_strength > 0.0f || highlight_corr != 0.0f)
    {
      // RGB -> XYZ
      float3 xyz;
      if(use_work_profile != 0 && profile_info != 0)
      {
        xyz.x = profile_info->matrix_in[0] * pixel.x + profile_info->matrix_in[1] * pixel.y + profile_info->matrix_in[2] * pixel.z;
        xyz.y = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
        xyz.z = profile_info->matrix_in[6] * pixel.x + profile_info->matrix_in[7] * pixel.y + profile_info->matrix_in[8] * pixel.z;
      }
      else
      {
        xyz.x = 0.636958f * pixel.x + 0.144617f * pixel.y + 0.168881f * pixel.z;
        xyz.y = 0.262700f * pixel.x + 0.677998f * pixel.y + 0.059302f * pixel.z;
        xyz.z = 0.000000f * pixel.x + 0.028073f * pixel.y + 1.060985f * pixel.z;
      }

      xyz = fmax(xyz, 0.0f);

      // XYZ to JzAzBz
      float4 xyz_scaled = (float4)(xyz.x * 400.0f, xyz.y * 400.0f, xyz.z * 400.0f, 0.0f);
      jab = XYZ_to_JzAzBz(xyz_scaled);

      int modified = 0;

      if(ucs_saturation_balance != 0.0f && workflow_mode != 3)
      {
        // Chroma-based modulation for saturation balance
        const float chroma = fmax(fmax(pixel.x, pixel.y), pixel.z) - fmin(fmin(pixel.x, pixel.y), pixel.z);
        const float effective_saturation = ucs_saturation_balance * fmin(chroma * 2.0f, 1.0f);

        // Apply saturation balance
        const float Y = xyz.y;
        const float L = native_sqrt(fmax(Y, 0.0f));
        const float fulcrum = 0.65f;
        const float n = (L - fulcrum) / fulcrum;
        const float mask_shadow = 1.0f / (1.0f + dtcl_exp(n * 4.0f));
        
        float sat_adjust = effective_saturation * (2.0f * mask_shadow - 1.0f);
        sat_adjust *= fmin(L * 4.0f, 1.0f);
        const float sat_factor = (1.0f + saturation_boost) * (1.0f + sat_adjust);
        jab.y *= sat_factor;
        jab.z *= sat_factor;
        modified = 1;
      }

      if(gamut_strength > 0.0f)
      {
        const float Y = xyz.y;
        const float L = native_sqrt(fmax(Y, 0.0f));
        const float chroma_factor = 1.0f - gamut_strength * (0.2f + 0.2f * L);
        jab.y *= chroma_factor;
        jab.z *= chroma_factor;
        modified = 1;
      }

      // HIGH SENSITIVITY CORRECTION
      // Start effect at 0.20 up to 0.90. Linear transition.
      float hl_mask = clamp((jab.x - 0.20f) / 0.70f, 0.0f, 1.0f);

      if(hl_mask > 0.0f && highlight_corr != 0.0f && workflow_mode != 3)
      {
        // 1. Soft symmetric desaturation (0.75 factor)
        const float desat = 1.0f - (fabs(highlight_corr) * hl_mask * 0.75f);
        jab.y *= desat;
        jab.z *= desat;

        // 2. Controlled Hue Rotation (2.0 factor)
        const float angle = highlight_corr * hl_mask * 2.0f;
        const float ca = native_cos(angle);
        const float sa = native_sin(angle);
        const float az = jab.y;
        const float bz = jab.z;

        jab.y = az * ca - bz * sa;
        jab.z = az * sa + bz * ca;
        modified = 1;
      }

      if(jab.x > 0.95f)
      {
        const float desat = clamp((1.0f - jab.x) * 20.0f, 0.0f, 1.0f);
        jab.y *= desat;
        jab.z *= desat;
        modified = 1;
      }

      if(modified)
      {
        // JzAzBz to XYZ
        xyz = JzAzBz_2_XYZ(jab).xyz / 400.0f;

        // XYZ D65 to Working RGB (using profile_info for perfect parity with C)
        if (use_work_profile != 0 && profile_info != 0)
          pixel.xyz = (float3)(dot(xyz, (float3)(profile_info->matrix_out[0], profile_info->matrix_out[1], profile_info->matrix_out[2])),
                               dot(xyz, (float3)(profile_info->matrix_out[3], profile_info->matrix_out[4], profile_info->matrix_out[5])),
                               dot(xyz, (float3)(profile_info->matrix_out[6], profile_info->matrix_out[7], profile_info->matrix_out[8])));
        else
          pixel.xyz = XYZ_to_Rec2020(xyz);
        
        const float min_val = fmin(pixel.x, fmin(pixel.y, pixel.z));
        if(min_val < 0.0f)
        {
          const float lum = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
          if(lum > 0.0f)
          {
           const float factor = lum / (lum - min_val);
            pixel.xyz = lum + factor * (pixel.xyz - lum);
          }
        }
        pixel.xyz = clamp(pixel.xyz, 0.0f, 1.0f);
      }
    }

    if(gamut_strength > 0.0f)
    {
      float4 orig = pixel;

      float Y_rgb = 0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z;
      float lum_weight = clamp((Y_rgb - 0.5f) / (0.9f - 0.5f), 0.0f, 1.0f);
      lum_weight = lum_weight * lum_weight * (3.0f - 2.0f * lum_weight);
      float effective_strength = gamut_strength * lum_weight;

      float limit = 0.72f;
      if(target_gamut == 1) limit = 0.80f;
      else if(target_gamut == 2) limit = 1.00f;

      float threshold = limit * (1.0f - (effective_strength * 0.10f));
      float max_val = fmax(pixel.x, fmax(pixel.y, pixel.z));

      if(max_val > threshold)
      {
        const float range = limit - threshold;
        const float delta = max_val - threshold;
        const float compressed = threshold + range * delta / (delta + range);
        const float factor = compressed / max_val;

        const float range_blue = 1.1f * range;
        const float compressed_blue = threshold + range * delta / (delta + range_blue);
        const float factor_blue = compressed_blue / max_val;
        // CB. Calculate current pixel luminance (before compression)
        // CB. Uses the RGB norm coefficients defined in the file
        const float luma = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
        // CB. Compress toward luminance (desaturation)
        pixel.x = luma + (pixel.x - luma) * factor;
        pixel.y = luma + (pixel.y - luma) * factor;
        pixel.z = luma + (pixel.z - luma) * factor_blue;
      }
      pixel = mix(orig, pixel, effective_strength);
    }

    // CB. OpenDRT-style weighted red and blue correction for higher precision
    if(workflow_mode > 0 && saturation_boost != 0.0f)
    {
      // Use calculated luma as the achromatic reference point (sat_L)
      const float luma = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
      const float prot_r = fmax(0.0f, 1.0f - fabs(pixel.x - luma) * 1.5f);
      const float prot_b = fmax(0.0f, 1.0f - fabs(pixel.z - luma) * 1.5f);
      pixel.x += saturation_boost * (pixel.x - luma) * prot_r; // Apply to red channel
      pixel.z += saturation_boost * (pixel.z - luma) * prot_b;
    }

    // Final gamut check to preserve hue
    if(pixel.x < 0.0f || pixel.x > 1.0f || pixel.y < 0.0f || pixel.y > 1.0f || pixel.z < 0.0f || pixel.z > 1.0f)
    {
      const float luma = r_coeff * pixel.x + g_coeff * pixel.y + b_coeff * pixel.z;
      const float target_luma = clamp(luma, 0.0f, 1.0f);
      float t = 1.0f;
      if(pixel.x < 0.0f) t = fmin(t, target_luma / (target_luma - pixel.x));
      if(pixel.y < 0.0f) t = fmin(t, target_luma / (target_luma - pixel.y));
      if(pixel.z < 0.0f) t = fmin(t, target_luma / (target_luma - pixel.z));
      if(pixel.x > 1.0f) t = fmin(t, (1.0f - target_luma) / (pixel.x - target_luma));
      if(pixel.y > 1.0f) t = fmin(t, (1.0f - target_luma) / (pixel.y - target_luma));
      if(pixel.z > 1.0f) t = fmin(t, (1.0f - target_luma) / (pixel.z - target_luma));
      t = fmax(0.0f, t);
      pixel.xyz = target_luma + t * (pixel.xyz - target_luma);
    }
  }

  pixel.w = read_imagef(in, sampleri, (int2)(x, y)).w;

  write_imagef (out, (int2)(x, y), pixel);
}

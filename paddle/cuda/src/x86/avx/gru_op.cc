/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <immintrin.h>
#include "hl_gru_ops.cuh"

namespace paddle {

namespace forward {

template <>
void gru_resetOutput::operator()(__m256 &valueUpdateGate,
                                 __m256 &valueResetGate,
                                 __m256 &prevOut,
                                 __m256 &valueResetOutput,
                                 Active<__m256>::forward actGate) {
  valueUpdateGate = actGate(valueUpdateGate);
  valueResetGate = actGate(valueResetGate);
  valueResetOutput = _mm256_mul_ps(prevOut, valueResetGate);
}

template <>
void gru_finalOutput::operator()(__m256 &valueUpdateGate,
                                 __m256 &valueFrameState,
                                 __m256 &prevOut,
                                 __m256 &valueOutput,
                                 Active<__m256>::forward actInput) {
  valueFrameState = actInput(valueFrameState);
  valueOutput = _mm256_add_ps(
      _mm256_sub_ps(prevOut, _mm256_mul_ps(valueUpdateGate, prevOut)),
      _mm256_mul_ps(valueUpdateGate, valueFrameState));
}

}  // namespace forward

namespace backward {

template <>
void gru_stateGrad::operator()(__m256 &valueUpdateGate,
                               __m256 &gradUpdateGate,
                               __m256 &valueFrameState,
                               __m256 &gradFrameState,
                               __m256 &valuePrevOut,
                               __m256 &gradPrevOut,
                               __m256 &gradOutput,
                               Active<__m256>::backward actInput) {
  gradUpdateGate = _mm256_mul_ps(gradOutput, valueFrameState);
  gradUpdateGate =
      _mm256_sub_ps(gradUpdateGate, _mm256_mul_ps(gradOutput, valuePrevOut));
  gradPrevOut = _mm256_add_ps(
      _mm256_sub_ps(gradPrevOut, _mm256_mul_ps(gradOutput, valueUpdateGate)),
      gradOutput);
  gradFrameState =
      actInput(_mm256_mul_ps(gradOutput, valueUpdateGate), valueFrameState);
}

template <>
void gru_resetGrad::operator()(__m256 &valueUpdateGate,
                               __m256 &gradUpdateGate,
                               __m256 &valueResetGate,
                               __m256 &gradResetGate,
                               __m256 &valuePrevOut,
                               __m256 &gradPrevOut,
                               __m256 &gradResetOutput,
                               Active<__m256>::backward actGate) {
  gradResetGate = _mm256_mul_ps(gradResetOutput, valuePrevOut);
  gradPrevOut = _mm256_add_ps(gradPrevOut,
                              _mm256_mul_ps(gradResetOutput, valueResetGate));
  gradUpdateGate = actGate(gradUpdateGate, valueUpdateGate);
  gradResetGate = actGate(gradResetGate, valueResetGate);
}

}  // namespace backward
}  // namespace paddle

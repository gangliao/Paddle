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

#include "paddle/utils/CpuId.h"
#include "x86/avx/lstm.h"
#include "x86/lstm.h"

namespace paddle {

template <class Op>
void hl_cpu_lstm_forward(Op op,
                         hl_lstm_value value,
                         int frameSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate,
                         hl_activation_mode_t active_state) {
  if (HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_forward_one_sequence(
        op, value, frameSize, active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_forward_one_sequence(
        op, value, frameSize, active_node, active_gate, active_state);
  }
}

template <class Op>
void hl_cpu_lstm_backward(Op op,
                          hl_lstm_value value,
                          hl_lstm_grad grad,
                          int frameSize,
                          hl_activation_mode_t active_node,
                          hl_activation_mode_t active_gate,
                          hl_activation_mode_t active_state) {
  if (HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_backward_one_sequence(
        op, value, grad, frameSize, active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_backward_one_sequence(
        op, value, grad, frameSize, active_node, active_gate, active_state);
  }
}

}  // namespace paddle

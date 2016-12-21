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

#pragma once

#include "hl_base.h"

namespace paddle {

template <class OpResetOutput>
void hl_naive_gru_forward_reset_output(OpResetOutput opResetOutput,
                                       real *gateValue,
                                       real *resetOutputValue,
                                       real *prevOutputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_gate);
template <class OpFinalOutput>
void hl_naive_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                       real *gateValue,
                                       real *prevOutputValue,
                                       real *outputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_node);

template <class OpStateGrad>
void hl_naive_gru_backward_state_grad(OpStateGrad opStateGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *outputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_node);

template <class OpResetGrad>
void hl_naive_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *resetOutputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_gate);
}

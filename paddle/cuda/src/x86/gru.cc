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

#include "x86/gru.h"
#include "hl_gru_ops.cuh"
#include "x86/activation.h"

namespace paddle {

template <class OpResetOutput>
void hl_naive_gru_forward_reset_output(OpResetOutput opResetOutput,
                                       real *gateValue,
                                       real *resetOutputValue,
                                       real *prevOutputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_gate) {
  real rValueUpdateGate;
  real rValueResetGate;
  real rValueResetOutput;
  real rPrevOut = 0;
  real *updateGate = gateValue;
  real *resetGate = gateValue + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueResetGate = resetGate[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opResetOutput(rValueUpdateGate,
                  rValueResetGate,
                  rPrevOut,
                  rValueResetOutput,
                  cpu::forward[active_gate]);

    updateGate[i] = rValueUpdateGate;
    resetGate[i] = rValueResetGate;
    resetOutputValue[i] = rValueResetOutput;
  }
}

template <class OpFinalOutput>
void hl_naive_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                       real *gateValue,
                                       real *prevOutputValue,
                                       real *outputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_node) {
  real rValueUpdateGate;
  real rValueFrameState;
  real rPrevOut = 0;
  real rOutput;
  real *updateGate = gateValue;
  real *frameState = gateValue + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueFrameState = frameState[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opFinalOutput(rValueUpdateGate,
                  rValueFrameState,
                  rPrevOut,
                  rOutput,
                  cpu::forward[active_node]);

    frameState[i] = rValueFrameState;
    outputValue[i] = rOutput;
  }
}

template <class OpStateGrad>
void hl_naive_gru_backward_state_grad(OpStateGrad opStateGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *outputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_node) {
  real rUpdateGateValue;
  real rUpdateGateGrad;
  real rFrameStateValue;
  real rFrameStateGrad;
  real rOutGrad;
  real rPrevOutValue = 0;
  real rPrevOutGrad = 0;
  real *updateGateValue = gateValue;
  real *updateGateGrad = gateGrad;
  real *frameStateValue = gateValue + frameSize * 2;
  real *frameStateGrad = gateGrad + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rFrameStateValue = frameStateValue[i];
    rOutGrad = outputGrad[i];
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = prevOutGrad[i];
    }

    opStateGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rFrameStateValue,
                rFrameStateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rOutGrad,
                cpu::backward[active_node]);

    updateGateGrad[i] = rUpdateGateGrad;
    frameStateGrad[i] = rFrameStateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

template <class OpResetGrad>
void hl_naive_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *resetOutputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_gate) {
  real rUpdateGateValue;
  real rUpdateGateGrad;
  real rResetGateValue;
  real rResetGateGrad;
  real rResetOutputGrad = 0;
  real rPrevOutValue = 0;
  real rPrevOutGrad = 0;
  real *updateGateValue = gateValue;
  real *updateGateGrad = gateGrad;
  real *resetGateValue = gateValue + frameSize;
  real *resetGateGrad = gateGrad + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rUpdateGateGrad = updateGateGrad[i];
    rResetGateValue = resetGateValue[i];

    if (prevOutValue && prevOutGrad) {
      rResetOutputGrad = resetOutputGrad[i];
    }
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = prevOutGrad[i];
    }

    opResetGrad(rUpdateGateValue,
                rUpdateGateGrad,
                rResetGateValue,
                rResetGateGrad,
                rPrevOutValue,
                rPrevOutGrad,
                rResetOutputGrad,
                cpu::backward[active_gate]);

    updateGateGrad[i] = rUpdateGateGrad;
    resetGateGrad[i] = rResetGateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

}  // namespace paddle

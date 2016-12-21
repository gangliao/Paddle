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

#include "x86/lstm.h"
#include "hl_lstm_ops.cuh"
#include "x86/activation.h"

namespace paddle {

template <class Op>
void hl_naive_lstm_forward_one_sequence(Op op,
                                        hl_lstm_value value,
                                        int frameSize,
                                        hl_activation_mode_t active_node,
                                        hl_activation_mode_t active_gate,
                                        hl_activation_mode_t active_state) {
  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rCheckI;
  real rCheckF;
  real rCheckO;
  real rState;
  real rPrevState = 0;
  real rStateAtv;
  real rOut;

  real *valueIn = value.gateValue;
  real *valueIg = value.gateValue + frameSize;
  real *valueFg = value.gateValue + frameSize * 2;
  real *valueOg = value.gateValue + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg[i];
    rCheckF = value.checkFg[i];
    rCheckO = value.checkOg[i];

    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rPrevState,
       rState,
       rStateAtv,
       rOut,
       rCheckI,
       rCheckF,
       rCheckO,
       cpu::forward[active_node],
       cpu::forward[active_gate],
       cpu::forward[active_state]);

    valueIn[i] = rValueIn;
    valueIg[i] = rValueIg;
    valueFg[i] = rValueFg;
    valueOg[i] = rValueOg;
    value.stateValue[i] = rState;
    value.stateActiveValue[i] = rStateAtv;
    value.outputValue[i] = rOut;
  }
}

template <class Op>
void hl_naive_lstm_backward_one_sequence(Op op,
                                         hl_lstm_value value,
                                         hl_lstm_grad grad,
                                         int frameSize,
                                         hl_activation_mode_t active_node,
                                         hl_activation_mode_t active_gate,
                                         hl_activation_mode_t active_state) {
  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rGradIn;
  real rGradIg;
  real rGradFg;
  real rGradOg;
  real rPrevState = 0;
  real rPrevStateGrad;
  real rState;
  real rStateGrad;
  real rStateAtv;
  real rOutputGrad;
  real rCheckI;
  real rCheckF;
  real rCheckO;
  real rCheckIGrad;
  real rCheckFGrad;
  real rCheckOGrad;

  real *valueIn = value.gateValue;
  real *valueIg = value.gateValue + frameSize;
  real *valueFg = value.gateValue + frameSize * 2;
  real *valueOg = value.gateValue + frameSize * 3;
  real *gradIn = grad.gateGrad;
  real *gradIg = grad.gateGrad + frameSize;
  real *gradFg = grad.gateGrad + frameSize * 2;
  real *gradOg = grad.gateGrad + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg[i];
    rCheckF = value.checkFg[i];
    rCheckO = value.checkOg[i];
    rState = value.stateValue[i];
    rStateAtv = value.stateActiveValue[i];
    rOutputGrad = grad.outputGrad[i];
    rStateGrad = grad.stateGrad[i];
    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rGradIn,
       rGradIg,
       rGradFg,
       rGradOg,
       rPrevState,
       rPrevStateGrad,
       rState,
       rStateGrad,
       rStateAtv,
       rOutputGrad,
       rCheckI,
       rCheckF,
       rCheckO,
       rCheckIGrad,
       rCheckFGrad,
       rCheckOGrad,
       cpu::backward[active_node],
       cpu::backward[active_gate],
       cpu::backward[active_state]);

    gradIn[i] = rGradIn;
    gradIg[i] = rGradIg;
    gradFg[i] = rGradFg;
    gradOg[i] = rGradOg;
    grad.stateGrad[i] = rStateGrad;

    if (grad.prevStateGrad) grad.prevStateGrad[i] = rPrevStateGrad;
    if (value.prevStateValue) {
      if (grad.checkIgGrad) grad.checkIgGrad[i] += rCheckIGrad;
      if (grad.checkFgGrad) grad.checkFgGrad[i] += rCheckFGrad;
    }
    if (grad.checkOgGrad) grad.checkOgGrad[i] += rCheckOGrad;
  }
}

}  // namespace paddle

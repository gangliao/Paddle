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

#include "hl_gru_ops.cuh"

namespace paddle {

namespace forward {
/**
 * @param[in,out]   valueUpdateGate  update gate
 * @param[in,out]   valueResetGate   reset gate
 * @param[in]       prevOut          previous output
 * @param[out]      valueResetOutput intermediate value for frame state
 * @param[in]       actGate          forward function of gate
 */
template <>
void gru_resetOutput::operator()(real &valueUpdateGate,
                                 real &valueResetGate,
                                 real &prevOut,
                                 real &valueResetOutput,
                                 Active<real>::forward actGate) {
  valueUpdateGate = actGate(valueUpdateGate);
  valueResetGate = actGate(valueResetGate);
  valueResetOutput = prevOut * valueResetGate;
}

/**
 * @param[in]     valueUpdateGate   update gate
 * @param[in,out] valueFrameState   frame state ({\tilde{h}_t})
 * @param[in]     prevOut           previous output
 * @param[out]    valueOutput       output
 * @param[in]     actInput          forward function of node
 */
template <>
void gru_finalOutput::operator()(real &valueUpdateGate,
                                 real &valueFrameState,
                                 real &prevOut,
                                 real &valueOutput,
                                 Active<real>::forward actInput) {
  valueFrameState = actInput(valueFrameState);
  valueOutput = prevOut - (valueUpdateGate * prevOut) +
                (valueUpdateGate * valueFrameState);
}
}  // namespace forward

namespace backward {
/**
 * @param[in]     valueUpdateGate   update gate value
 * @param[out]    gradUpdateGate    update gate grad
 * @param[in]     valueFrameState   frame state value
 * @param[out]    gradFrameState    frame state grad
 * @param[in]     valuePrevOut      previous output value
 * @param[in,out] gradPrevOut       previous output grad
 * @param[in]     gradOutput        output grad
 * @param[in]     actInput          backward function of frame state
 */
template <>
void gru_stateGrad::operator()(real &valueUpdateGate,
                               real &gradUpdateGate,
                               real &valueFrameState,
                               real &gradFrameState,
                               real &valuePrevOut,
                               real &gradPrevOut,
                               real &gradOutput,
                               Active<real>::backward actInput) {
  gradUpdateGate = (gradOutput * valueFrameState);
  gradUpdateGate -= (gradOutput * valuePrevOut);
  gradPrevOut -= (gradOutput * valueUpdateGate);
  gradPrevOut += gradOutput;
  gradFrameState = actInput(gradOutput * valueUpdateGate, valueFrameState);
}

/**
 * @param[in]     valueUpdateGate   update gate value
 * @param[in,out] gradUpdateGate    update gate grad
 * @param[in]     valueResetGate    reset gate value
 * @param[out]    gradResetGate     reset gate grad
 * @param[in]     valuePrevOut      previous output value
 * @param[in,out] gradPrevOut       previous output grad
 * @param[in]     gradResetOutput   reset output grad (temp val)
 * @param[in]     actGate           backward function of gate
 */
template <>
void gru_resetGrad::operator()(real &valueUpdateGate,
                               real &gradUpdateGate,
                               real &valueResetGate,
                               real &gradResetGate,
                               real &valuePrevOut,
                               real &gradPrevOut,
                               real &gradResetOutput,
                               Active<real>::backward actGate) {
  gradResetGate = (gradResetOutput * valuePrevOut);
  gradPrevOut += (gradResetOutput * valueResetGate);
  gradUpdateGate = actGate(gradUpdateGate, valueUpdateGate);
  gradResetGate = actGate(gradResetGate, valueResetGate);
}
}  // namespace backward
}  // namespace paddle

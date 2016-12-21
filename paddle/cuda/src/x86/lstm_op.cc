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

#include "hl_lstm_ops.cuh"

namespace paddle {

namespace forward {

/**
 * @param   valueIn     input
 * @param   valueIg     input gate
 * @param   valueFg     forget gate
 * @param   valueOg     output gate
 * @param   prevState   previous state
 * @param   state       current state
 * @param   stateAtv    state active
 * @param   output      output
 * @param   checkI      check input gate
 * @param   checkF      check forget gate
 * @param   checkO      check output gate
 * @param   actInput    forward function of input
 * @param   actGate     forward function of gate
 * @param   actState    forward function of state
 */
template <>
void lstm::operator()(real &valueIn,
                      real &valueIg,
                      real &valueFg,
                      real &valueOg,
                      real &prevState,
                      real &state,
                      real &stateAtv,
                      real &output,
                      real &checkI,
                      real &checkF,
                      real &checkO,
                      Active<real>::forward actInput,
                      Active<real>::forward actGate,
                      Active<real>::forward actState) {
  valueIn = actInput(valueIn);
  valueIg = actGate(valueIg + prevState * checkI);
  valueFg = actGate(valueFg + prevState * checkF);
  state = valueIn * valueIg + prevState * valueFg;
  valueOg = actGate(valueOg + state * checkO);
  stateAtv = actState(state);
  output = valueOg * stateAtv;
}

}  // namespace forward

namespace backward {
/**
 * @param   valueIn         input
 * @param   valueIg         input gate
 * @param   valueFg         forget gate
 * @param   valueOg         output gate
 * @param   gradIn          input grad
 * @param   gradIg          input gate grad
 * @param   gradFg          forget gate grad
 * @param   gradOg          output gate grad
 * @param   prevState       previous state value
 * @param   prevStateGrad   previous state grad
 * @param   state           current state value
 * @param   stateGrad       current state grad
 * @param   stateAtv        state active
 * @param   outputGrad      output grad
 * @param   checkI          check input gate
 * @param   checkF          check forget gate
 * @param   checkO          check output gate
 * @param   checkIGrad      check input gate grad
 * @param   checkFGrad      check forget gate grad
 * @param   checkOGrad      check output gate grad
 * @param   actInput        backward function of input
 * @param   actGate         backward function of gate
 * @param   actState        backward function of state
 */
template <>
void lstm::operator()(real &valueIn,
                      real &valueIg,
                      real &valueFg,
                      real &valueOg,
                      real &gradIn,
                      real &gradIg,
                      real &gradFg,
                      real &gradOg,
                      real &prevState,
                      real &prevStateGrad,
                      real &state,
                      real &stateGrad,
                      real &stateAtv,
                      real &outputGrad,
                      real &checkI,
                      real &checkF,
                      real &checkO,
                      real &checkIGrad,
                      real &checkFGrad,
                      real &checkOGrad,
                      Active<real>::backward actInput,
                      Active<real>::backward actGate,
                      Active<real>::backward actState) {
  gradOg = actGate(outputGrad * stateAtv, valueOg);
  stateGrad += actState(outputGrad * valueOg, stateAtv) + gradOg * checkO;
  gradIn = actInput(stateGrad * valueIg, valueIn);
  gradIg = actGate(stateGrad * valueIn, valueIg);
  gradFg = actGate(stateGrad * prevState, valueFg);
  prevStateGrad = gradIg * checkI + gradFg * checkF + stateGrad * valueFg;
  checkIGrad = gradIg * prevState;
  checkFGrad = gradFg * prevState;
  checkOGrad = gradOg * state;
}
}  // namespace backward
}  // namespace paddle

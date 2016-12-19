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

namespace hppl {
// clang-format off

/// forward activation
real relu   (const real a);
real sigmoid(const real a);
real tanh   (const real a);
real linear (const real a);

/// backward activation
real relu   (const real a, const real b);
real sigmoid(const real a, const real b);
real tanh   (const real a, const real b);
real linear (const real a, const real b);

namespace cpu {
static Active<real>::forward  forward [] = { sigmoid, relu, tanh, linear };
static Active<real>::backward backward[] = { sigmoid, relu, tanh, linear };
}

// clang-format on
}  // namespace hppl

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

#ifdef __CUDA_ARCH__
#define DEVICE   __device__
#else
#define DEVICE
#endif

namespace paddle {
#define LGENERIC_OPERATOR(__name)    \
class __name {                       \
public:                              \
template<typename ... Args>          \
DEVICE void operator()(Args...);     \
}
namespace forward {

LGENERIC_OPERATOR(lstm);


};  // namespace forward

namespace backward {
LGENERIC_OPERATOR(lstm);
}  // namespace backward
}  // namespace paddle

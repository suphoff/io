/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_IO_IOBASE_H_
#define TENSORFLOW_IO_IOBASE_H_


#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow_io {
using namespace ::tensorflow;


template <class BaseClass>
class IOBase : public core::RefCounted {
 public:

  using WrappedClass = BaseClass;
  
  //class variantWrapper<BaseClass>;
  
  Status AsVariantTensor(Tensor* tensor);
  static Status FromVariantTensor(const Tensor& tensor, BaseClass** base);

  
  static const char* VariantTypeName();

  virtual string DebugString() = 0;

};

}  // namespace tensorflow

#endif  // TENSORFLOW_IO_IOBASE_H_

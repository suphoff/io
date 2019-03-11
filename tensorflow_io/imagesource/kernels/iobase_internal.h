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
#ifndef TENSORFLOW_IO_IOBASE_INTERNAL_H_
#define TENSORFLOW_IO_IOBASE_INTERNAL_H_

#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow_io {
using namespace ::tensorflow;
 

// A wrapper class for storing an `ImageSet` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.

template <class BaseClass>
class IOVariantWrapper {
 public:
  IOVariantWrapper() : baseObject_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit IOVariantWrapper(BaseClass* baseObject) : baseObject_(baseObject) {}

  IOVariantWrapper(const IOVariantWrapper<BaseClass>& other)
      : baseObject_(other.baseObject_) {
    if (baseObject_) baseObject_->Ref();
  }

  ~IOVariantWrapper() {
    if (baseObject_) baseObject_->Unref();
  }

  BaseClass* get() const { return baseObject_; }

  string TypeName() const { return BaseClass::VariantTypeName();}

  string DebugString() const {
    if (baseObject_) {
      return baseObject_->DebugString();
    } else {
      return "<Uninitialized IOVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "IOVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "IOVariantWrapper objects.";
    return false;
  }

 private:
  BaseClass* const baseObject_;  // Owns one reference.
};





template <class BaseClass>
Status IOBase<BaseClass>::AsVariantTensor(Tensor* tensor) {
 using Wrapper = IOVariantWrapper<BaseClass>;
  if (!(tensor->dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    Unref();
    return errors::InvalidArgument(
        "IO tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = Wrapper(reinterpret_cast<BaseClass *>(this));
  return Status::OK();
}


template <class BaseClass>
Status IOBase<BaseClass>::FromVariantTensor(const Tensor& tensor, BaseClass** out_base) {
 using Wrapper = IOVariantWrapper<BaseClass>;
  
  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "IO tensor must be a scalar of dtype DT_VARIANT.");
  }
  
  const Variant& variant = tensor.scalar<Variant>()();
  const Wrapper* wrapper = variant.get<Wrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be an IO object.");
  }
  *out_base = wrapper->get();
  if (*out_base == nullptr) {
    return errors::Internal("Read uninitialized IO variant.");
  }
  return Status::OK();
}



#define IOBASE_BOILERPLATE(BaseClass) \
  template Status IOBase<BaseClass>::AsVariantTensor(Tensor* tensor);                           \
  template Status IOBase<BaseClass>::FromVariantTensor(const Tensor& tensor, BaseClass** base); \
  template<> const char* IOBase<BaseClass>::VariantTypeName() { return "tensorflow_io::" #BaseClass "VariantWrapper";};\
  template   const char* IOBase<BaseClass>::VariantTypeName();

}  // namespace tensorflow_io


#endif  // TENSORFLOW_IO_IOBASE_INTERNAL_H_

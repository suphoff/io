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

#include "imagesource.h"

#include "iobase_internal.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow_io {
using namespace ::tensorflow;

IOBASE_BOILERPLATE(ImageSource)


//string ImageSource::DebugString() {
//  return "ImageSource: To Be Filled By O.E.M";
//}



class ImageSourceToDebugOp: public OpKernel {
 public:
  explicit ImageSourceToDebugOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& variant_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(variant_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        variant_tensor.shape().DebugString()));
    
    ImageSource* image = nullptr;
    OP_REQUIRES_OK(ctx,ImageSource::FromVariantTensor(variant_tensor, &image));
                   
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));

    output->scalar<string>()() = image->DebugString();

  }
};
    
REGISTER_KERNEL_BUILDER(Name("ImageSourceToDebug").Device(DEVICE_CPU),
                        ImageSourceToDebugOp);

}  // namespace tensorflow_io

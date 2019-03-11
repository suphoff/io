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

#include "tensorflow/core/lib/core/refcount.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"

#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/env.h"

#include "webp/encode.h"
#include "imageio/webpdec.h"
#include "imageio/metadata.h"

#include <iostream>

namespace tensorflow_io {
using namespace ::tensorflow;

namespace {

class WebPImageSource: public ImageSource {
 public:
  Status InitFromFile(Env* env, const string& filename) {

    std::cout << "Initialized from " << filename << std::endl;
    uint64 size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &size));
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
    std::unique_ptr<io::RandomAccessInputStream> stream(new io::RandomAccessInputStream(file.get()));
    string data_;
    TF_RETURN_IF_ERROR(stream->ReadNBytes(size, &data_));
    return Status::OK();
  }
          
  Status InitFromString(const string& content) {
    data_ = content;
    return Status::OK();
  }

 
  string DebugString() override { return string("WebPImageSource") + std::to_string((unsigned long long) (this)) ;}
  
 private:

  WebPDecoderConfig config;
  string data_;
};
  


class WebPImageSourceFromFileOp: public OpKernel {
 public:
  explicit WebPImageSourceFromFileOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& contents_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    const auto filename = contents_tensor.scalar<string>()();

          
    core::RefCountPtr<WebPImageSource> imageSource (new WebPImageSource());
    
    OP_REQUIRES_OK(ctx , imageSource->InitFromFile(ctx->env(), filename));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    imageSource.release()->AsVariantTensor(output);
    
  }
};
    

class WebPImageSourceFromStringOp: public OpKernel {
 public:
  explicit WebPImageSourceFromStringOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& contents_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents_tensor.shape().DebugString()));
    const auto content = contents_tensor.scalar<string>()();

          
    core::RefCountPtr<WebPImageSource> imageSource(new WebPImageSource());
    
    OP_REQUIRES_OK(ctx, imageSource->InitFromString(content));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    imageSource.release()->AsVariantTensor(output);
    
  }
};
    


REGISTER_KERNEL_BUILDER(Name("WebPImageSourceFromFile").Device(DEVICE_CPU),
                        WebPImageSourceFromFileOp);

REGISTER_KERNEL_BUILDER(Name("WebPImageSourceFromString").Device(DEVICE_CPU),
                        WebPImageSourceFromStringOp);



}  // Anonymous namespace
}  // tensorflow_io


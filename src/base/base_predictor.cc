// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base_predictor.h"
#include <stdexcept>

#include <iostream>

#include "base_batch_sampler.h"
#include "src/common/image_batch_sampler.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include "src/utils/openvino_option.h"
#include "src/utils/utility.h"

BasePredictor::BasePredictor(const std::optional<std::string> &model_dir,
                             const std::optional<std::string> &model_name,
                             const std::optional<std::string> &device,
                             const std::string &precision,
                             int cpu_threads,
                             int batch_size, const std::string sampler_type)
    : model_dir_(model_dir), batch_size_(batch_size),
      sampler_type_(sampler_type) {
  // 创建一个空的配置对象，不从文件加载
  config_ = SimpleConfig({});
  
  auto status_build = BuildBatchSampler();
  if (!status_build.ok()) {
    // INFOE("Build sampler fail: %s", status_build.ToString().c_str());
    throw std::runtime_error(status_build.ToString());
  }
  
  // 直接使用传入的模型名称，不从配置文件读取
  if (model_name.has_value()) {
    model_name_ = model_name.value();
  } else {
    throw std::runtime_error("Model name is required");
  }
  pp_option_ptr_.reset(new PaddlePredictorOption());
  auto device_result = device.value_or(std::string("cpu"));

  size_t pos = device_result.find(':');
  std::string device_type = "";
  int device_id = 0;
  if (pos != std::string::npos) {
    device_type = device_result.substr(0, pos);
    device_id = std::stoi(device_result.substr(pos + 1));
  } else {
    device_type = device_result;
    device_id = 0;
  }
  auto status_device_type = pp_option_ptr_->SetDeviceType(device_type);
  if (!status_device_type.ok()) {
    // INFOE("Failed to set device");
    throw std::runtime_error("Failed to set device");
  }
  auto status_device_id = pp_option_ptr_->SetDeviceId(device_id);
  if (!status_device_id.ok()) {
    // INFOE("Failed to set device id");
    throw std::runtime_error("Failed to set device id");
  }

  // Simplified - just use paddle mode
  auto status_paddle = pp_option_ptr_->SetRunMode("paddle");
  if (!status_paddle.ok()) {
    // INFOE("Failed to set run mode");
    throw std::runtime_error("Failed to set run mode");
  }
  
  auto status_cpu_threads = pp_option_ptr_->SetCpuThreads(cpu_threads);
  if (!status_cpu_threads.ok()) {
    // INFOE("Set cpu threads fail");
    throw std::runtime_error("Set cpu threads fail");
  }
  // INFO("Create model: %s.", model_name_.c_str());
}

std::vector<std::unique_ptr<BaseCVResult>>
BasePredictor::Predict(const std::string &input) {
  std::vector<std::string> inputs = {input};
  return Predict(inputs);
}

const PaddlePredictorOption &BasePredictor::PPOption() {
  return *pp_option_ptr_;
}

void BasePredictor::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

std::unique_ptr<OpenVinoInfer> BasePredictor::CreateStaticInfer() {
  // Convert PaddlePredictorOption to OpenVinoOption
  OpenVinoOption openvino_option;
  openvino_option.SetDeviceType(PPOption().DeviceType());
  openvino_option.SetDeviceId(PPOption().DeviceId());
  openvino_option.SetCpuThreads(PPOption().CpuThreads());
  
  return std::unique_ptr<OpenVinoInfer>(new OpenVinoInfer(
      model_name_, model_dir_.value(), MODEL_FILE_PREFIX, openvino_option));
}

Status BasePredictor::BuildBatchSampler() {
  if (SAMPLER_TYPE.count(sampler_type_) == 0) {
    return Status::InvalidArgumentError("Unsupported sampler type !");
  } else if (sampler_type_ == "image") {
    batch_sampler_ptr_ =
        std::unique_ptr<BaseBatchSampler>(new ImageBatchSampler(batch_size_));
  }
  return Status::OK();
}

const std::unordered_set<std::string> BasePredictor::SAMPLER_TYPE = {
    "image",
};

bool BasePredictor::print_flag = true;


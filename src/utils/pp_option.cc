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

#include "src/utils/pp_option.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "src/utils/status.h"

const std::string &PaddlePredictorOption::RunMode() const { return run_mode_; }

const std::string &PaddlePredictorOption::DeviceType() const {
  return device_type_;
}

int PaddlePredictorOption::DeviceId() const { return device_id_; }

int PaddlePredictorOption::CpuThreads() const { return cpu_threads_; }

const std::vector<std::string> &PaddlePredictorOption::DeletePass() const {
  return delete_pass_;
}

bool PaddlePredictorOption::EnableNewIR() const { return enable_new_ir_; }

bool PaddlePredictorOption::EnableCinn() const { return enable_cinn_; }

const std::vector<std::string> &
PaddlePredictorOption::GetSupportRunMode() const {
  return SUPPORT_RUN_MODE;
}

const std::vector<std::string> &
PaddlePredictorOption::GetSupportDevice() const {
  return SUPPORT_DEVICE;
}

Status PaddlePredictorOption::SetRunMode(const std::string &run_mode) {
  if (std::find(SUPPORT_RUN_MODE.begin(), SUPPORT_RUN_MODE.end(), run_mode) ==
      SUPPORT_RUN_MODE.end()) {
    return Status::InvalidArgumentError("Unsupported run_mode: " + run_mode);
  }
  run_mode_ = run_mode;
  return Status::OK();
}

Status
PaddlePredictorOption::SetDeviceType(const std::string &device_type) {
  if (std::find(SUPPORT_DEVICE.begin(), SUPPORT_DEVICE.end(), device_type) ==
      SUPPORT_DEVICE.end()) {
    return Status::InvalidArgumentError(
        "SetDeviceType failed! Unsupported device_type: " + device_type);
  }
  device_type_ = device_type;
  if (device_type_ == "cpu") {
    device_id_ = 0;
  }
  return Status::OK();
}

Status PaddlePredictorOption::SetDeviceId(int device_id) {
  if (device_id < 0) {
    return Status::InvalidArgumentError(
        "SetDeviceId failed! device_id must be >= 0");
  }
  device_id_ = device_id;
  return Status::OK();
}

Status PaddlePredictorOption::SetCpuThreads(int cpu_threads) {
  if (cpu_threads < 1) {
    throw std::invalid_argument(
        "SetCpuThreads failed! cpu_threads must be >= 1");
  }
  cpu_threads_ = cpu_threads;
  return Status::OK();
}

void PaddlePredictorOption::SetDeletePass(
    const std::vector<std::string> &delete_pass) {
  delete_pass_ = delete_pass;
}

void PaddlePredictorOption::SetEnableNewIR(bool enable_new_ir) {
  enable_new_ir_ = enable_new_ir;
}

void PaddlePredictorOption::SetEnableCinn(bool enable_cinn) {
  enable_cinn_ = enable_cinn;
}

std::string PaddlePredictorOption::DebugString() const {
  std::ostringstream oss;
  oss << "run_mode: " << run_mode_ << ", "
      << "device_type: " << device_type_ << ", "
      << "device_id: " << device_id_ << ", "
      << "cpu_threads: " << cpu_threads_ << ", "
      << "delete_pass: [";
  for (size_t i = 0; i < delete_pass_.size(); ++i) {
    oss << delete_pass_[i];
    if (i != delete_pass_.size() - 1)
      oss << ", ";
  }
  oss << "], "
      << "enable_new_ir: " << (enable_new_ir_ ? "true" : "false") << ", "
      << "enable_cinn: " << (enable_cinn_ ? "true" : "false");
  return oss.str();
}


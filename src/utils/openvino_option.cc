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

#include "openvino_option.h"
#include <sstream>

const std::string &OpenVinoOption::DeviceType() const {
  return device_type_;
}

int OpenVinoOption::DeviceId() const { 
  return device_id_; 
}

int OpenVinoOption::CpuThreads() const { 
  return cpu_threads_; 
}

Status OpenVinoOption::SetDeviceType(const std::string &device_type) {
  // Extract device type (remove device id if present)
  size_t pos = device_type.find(':');
  std::string type = (pos != std::string::npos) ? device_type.substr(0, pos) : device_type;
  
  bool found = false;
  for (const auto &support_device : SUPPORT_DEVICE) {
    if (support_device == type) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::ostringstream oss;
    oss << "Unsupported device type: " << type << ". Supported types: ";
    for (size_t i = 0; i < SUPPORT_DEVICE.size(); ++i) {
      oss << SUPPORT_DEVICE[i];
      if (i < SUPPORT_DEVICE.size() - 1) oss << ", ";
    }
    return Status::InvalidArgumentError(oss.str());
  }
  device_type_ = type;
  return Status::OK();
}

Status OpenVinoOption::SetDeviceId(int device_id) {
  if (device_id < 0) {
    return Status::InvalidArgumentError("Device ID must be non-negative");
  }
  device_id_ = device_id;
  return Status::OK();
}

Status OpenVinoOption::SetCpuThreads(int cpu_threads) {
  if (cpu_threads <= 0) {
    return Status::InvalidArgumentError("CPU threads must be positive");
  }
  cpu_threads_ = cpu_threads;
  return Status::OK();
}

std::string OpenVinoOption::DebugString() const {
  std::ostringstream oss;
  oss << "OpenVinoOption: device_type=" << device_type_ 
      << ", device_id=" << device_id_ 
      << ", cpu_threads=" << cpu_threads_;
  return oss.str();
}

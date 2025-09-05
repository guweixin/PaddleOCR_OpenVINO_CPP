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

#pragma once

#include <string>
#include <vector>
#include "src/utils/status.h"

#ifdef WITH_GPU
static constexpr const char *DEFAULT_DEVICE = "gpu:0";
#else
static constexpr const char *DEFAULT_DEVICE = "cpu";
#endif

class OpenVinoOption {
public:
  const std::vector<std::string> SUPPORT_DEVICE = {"gpu", "cpu"};

  const std::string &DeviceType() const;
  int DeviceId() const;
  int CpuThreads() const;
  std::string DebugString() const;

  Status SetDeviceType(const std::string &device_type);
  Status SetDeviceId(int device_id);
  Status SetCpuThreads(int cpu_threads);

private:
  std::string device_type_ = DEFAULT_DEVICE;
  int device_id_ = 0;
  int cpu_threads_ = 10;
};

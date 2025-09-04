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
#include <unordered_map>
#include <optional>
#include <variant>
#include <vector>
#include "status.h"

// Simple configuration class to replace yaml-cpp dependency
class SimpleConfig {
public:
  // VectorVariant for compatibility with existing code
  struct VectorVariant {
    std::vector<int> vec_int;
    std::vector<float> vec_float;
    std::vector<std::string> vec_string;
  };

  SimpleConfig() = default;
  SimpleConfig(const std::string &config_path);
  SimpleConfig(const std::unordered_map<std::string, std::string> &data) : data_(data) {}
  ~SimpleConfig() = default;

  // Main access methods
  StatusOr<std::string> GetString(const std::string &key, const std::string &default_value = "") const;
  StatusOr<int> GetInt(const std::string &key, int default_value = 0) const;
  StatusOr<float> GetFloat(const std::string &key, float default_value = 0.0f) const;
  StatusOr<double> GetDouble(const std::string &key) const;
  StatusOr<bool> GetBool(const std::string &key, bool default_value = false) const;

  // SubModule access (simplified)
  StatusOr<SimpleConfig> GetSubModule(const std::string &key) const;
  
  // Utility methods
  Status HasKey(const std::string &key) const;
  Status PrintAll() const;
  Status PrintWithPrefix(const std::string &prefix) const;
  Status FindPreProcessOp(const std::string &prefix) const;
  
  // Vector parsing
  static VectorVariant SmartParseVector(const std::string &input);
  
  // Character dictionary reading
  static std::vector<std::string> LoadCharacterDict(const std::string &dict_path);
  
  // For compatibility with original YamlConfig
  std::unordered_map<std::string, std::string> PreProcessOpInfo() const;
  std::unordered_map<std::string, std::string> PostProcessOpInfo() const;
  
  // Internal data access
  const std::unordered_map<std::string, std::string> &Data() const { return data_; }
  std::unordered_map<std::string, std::string> &Data() { return data_; }
  
  // Key finding for configuration override
  StatusOr<std::pair<std::string, std::string>> FindKey(const std::string &key);

private:
  std::unordered_map<std::string, std::string> data_;
  
  Status LoadFromFile(const std::string &config_path);
  Status GetConfigPaths(const std::string &model_dir);
  void Init();
  void ParseLine(const std::string &line, const std::string &prefix = "");
  std::string Trim(const std::string &str) const;
  bool IsNumber(const std::string &str) const;
  bool IsFloat(const std::string &str) const;
  bool IsBool(const std::string &str) const;
};

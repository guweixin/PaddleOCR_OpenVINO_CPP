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

#include "simple_config.h"
#include "utility.h"
#include "ilogger.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <vector>

SimpleConfig::SimpleConfig(const std::string &model_dir) {
  // Check if model_dir is a directory or a file
  std::string config_path = model_dir;
  
  // If it's a directory, look for config files in it
  if (model_dir.find(".yaml") == std::string::npos && 
      model_dir.find(".yml") == std::string::npos) {
    // Try common config file names
    std::vector<std::string> possible_configs = {
      model_dir + "/inference.yml",
      model_dir + "/infer_cfg.yml", 
      model_dir + "/config.yml",
      model_dir + "/inference.yaml",
      model_dir + "/infer_cfg.yaml", 
      model_dir + "/config.yaml"
    };
    
    bool found = false;
    for (const auto& path : possible_configs) {
      std::ifstream test_file(path);
      if (test_file.good()) {
        config_path = path;
        found = true;
        break;
      }
    }
    
    if (!found) {
      // If no config file found, create a minimal default config
      std::cout << "[DEBUG] No config file found in model directory, using defaults" << std::endl;
      Init();
      return;
    }
  }
  
  auto status = LoadFromFile(config_path);
  if (!status.ok()) {
    INFOE("Failed to load config: %s", status.ToString().c_str());
    exit(-1);
  }
  Init();
}

Status SimpleConfig::LoadFromFile(const std::string &config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    return Status::NotFoundError("Cannot open config file: " + config_path);
  }
  
  std::string line;
  std::vector<std::string> section_stack; // 使用栈来跟踪嵌套层级
  
  while (std::getline(file, line)) {
    std::string original_line = line;
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue; // Skip empty lines and comments
    }
    
    // 计算缩进级别
    size_t indent_level = 0;
    for (char c : original_line) {
      if (c == ' ') indent_level++;
      else break;
    }
    indent_level = indent_level / 2; // 假设每级缩进是2个空格
    
    // Handle section headers (ending with :)
    if (line.back() == ':' && line.find('=') == std::string::npos) {
      std::string section_name = line.substr(0, line.length() - 1);
      
      // 调整 section_stack 大小以匹配当前缩进级别
      section_stack.resize(indent_level);
      section_stack.push_back(section_name);
      
      continue;
    }
    
    // Handle key-value pairs
    if (line.find(':') != std::string::npos) {
      size_t pos = line.find(':');
      std::string key = Trim(line.substr(0, pos));
      std::string value = Trim(line.substr(pos + 1));
      
      // Remove quotes if present
      if (!value.empty() && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.length() - 2);
      }
      
      // Handle null values
      if (value == "null") {
        value = "";
      }
      
      // Build full key path
      std::string full_key = key;
      if (!section_stack.empty()) {
        std::string current_section = "";
        for (size_t i = 0; i < section_stack.size(); ++i) {
          if (!current_section.empty()) current_section += ".";
          current_section += section_stack[i];
        }
        full_key = current_section + "." + key;
      }
      
      data_[full_key] = value;
    }
  }
  
  return Status::OK();
}

void SimpleConfig::Init() {
  // Hard-code default OCR configuration values for compatibility
  if (data_.find("text_type") == data_.end()) {
    data_["text_type"] = "general";
  }
  
  // TextDetection defaults
  if (data_.find("SubModules.TextDetection.model_name") == data_.end()) {
    data_["SubModules.TextDetection.model_name"] = "PP-OCRv5_server_det";
  }
  if (data_.find("SubModules.TextDetection.limit_side_len") == data_.end()) {
    data_["SubModules.TextDetection.limit_side_len"] = "64";
  }
  if (data_.find("SubModules.TextDetection.limit_type") == data_.end()) {
    data_["SubModules.TextDetection.limit_type"] = "min";
  }
  if (data_.find("SubModules.TextDetection.max_side_limit") == data_.end()) {
    data_["SubModules.TextDetection.max_side_limit"] = "4000";
  }
  if (data_.find("SubModules.TextDetection.thresh") == data_.end()) {
    data_["SubModules.TextDetection.thresh"] = "0.3";
  }
  if (data_.find("SubModules.TextDetection.box_thresh") == data_.end()) {
    data_["SubModules.TextDetection.box_thresh"] = "0.6";
  }
  if (data_.find("SubModules.TextDetection.unclip_ratio") == data_.end()) {
    data_["SubModules.TextDetection.unclip_ratio"] = "1.5";
  }
  if (data_.find("SubModules.TextDetection.batch_size") == data_.end()) {
    data_["SubModules.TextDetection.batch_size"] = "1";
  }
  
  // TextRecognition defaults
  if (data_.find("SubModules.TextRecognition.model_name") == data_.end()) {
    data_["SubModules.TextRecognition.model_name"] = "PP-OCRv5_server_rec";
  }
  if (data_.find("SubModules.TextRecognition.batch_size") == data_.end()) {
    data_["SubModules.TextRecognition.batch_size"] = "6";
  }
  if (data_.find("SubModules.TextRecognition.score_thresh") == data_.end()) {
    data_["SubModules.TextRecognition.score_thresh"] = "0.0";
  }
}

StatusOr<std::string> SimpleConfig::GetString(const std::string &key, const std::string &default_value) const {
  std::cout << "[DEBUG] GetString called with key: '" << key << "'" << std::endl;
  
  auto it = data_.find(key);
  if (it != data_.end()) {
    std::cout << "[DEBUG] Found direct key: '" << key << "' = '" << it->second << "'" << std::endl;
    return it->second;
  }
  
  // Try with SubModules prefix
  std::string prefixed_key = "SubModules." + key;
  std::cout << "[DEBUG] Trying prefixed key: '" << prefixed_key << "'" << std::endl;
  it = data_.find(prefixed_key);
  if (it != data_.end()) {
    std::cout << "[DEBUG] Found prefixed key: '" << prefixed_key << "' = '" << it->second << "'" << std::endl;
    return it->second;
  }
  
  if (!default_value.empty()) {
    std::cout << "[DEBUG] Using default value: '" << default_value << "'" << std::endl;
    return default_value;
  }
  
  std::cout << "[DEBUG] Key not found: '" << key << "'" << std::endl;
  // Print all available keys for debugging
  std::cout << "[DEBUG] Available keys:" << std::endl;
  for (const auto &pair : data_) {
    std::cout << "  '" << pair.first << "' = '" << pair.second << "'" << std::endl;
  }
  
  return Status::NotFoundError("Key not found: " + key);
}

StatusOr<int> SimpleConfig::GetInt(const std::string &key, int default_value) const {
  auto result = GetString(key);
  if (!result.ok()) {
    return default_value;
  }
  
  try {
    return std::stoi(result.value());
  } catch (const std::exception &e) {
    return default_value;
  }
}

StatusOr<float> SimpleConfig::GetFloat(const std::string &key, float default_value) const {
  auto result = GetString(key);
  if (!result.ok()) {
    return default_value;
  }
  
  try {
    return std::stof(result.value());
  } catch (const std::exception &e) {
    return default_value;
  }
}

StatusOr<double> SimpleConfig::GetDouble(const std::string &key) const {
  auto result = GetString(key);
  if (!result.ok()) {
    return result.status();
  }
  
  try {
    return std::stod(result.value());
  } catch (const std::exception &e) {
    return Status::InvalidArgumentError("Cannot convert to double: " + result.value());
  }
}

StatusOr<bool> SimpleConfig::GetBool(const std::string &key, bool default_value) const {
  auto result = GetString(key);
  if (!result.ok()) {
    return default_value;
  }
  
  std::string value = result.value();
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  } else if (value == "false" || value == "0" || value == "no") {
    return false;
  }
  
  return default_value;
}

StatusOr<SimpleConfig> SimpleConfig::GetSubModule(const std::string &key) const {
  std::unordered_map<std::string, std::string> sub_data;
  std::string prefix = key + ".";
  
  for (const auto &pair : data_) {
    if (pair.first.find(prefix) == 0) {
      std::string sub_key = pair.first.substr(prefix.length());
      sub_data[sub_key] = pair.second;
    }
  }
  
  if (sub_data.empty()) {
    return Status::NotFoundError("SubModule not found: " + key);
  }
  
  return SimpleConfig(sub_data);
}

Status SimpleConfig::HasKey(const std::string &key) const {
  if (data_.find(key) != data_.end()) {
    return Status::OK();
  }
  
  std::string prefixed_key = "SubModules." + key;
  if (data_.find(prefixed_key) != data_.end()) {
    return Status::OK();
  }
  
  return Status::NotFoundError("Key not found: " + key);
}

Status SimpleConfig::PrintAll() const {
  for (const auto &pair : data_) {
    std::cout << pair.first << ": " << pair.second << std::endl;
  }
  return Status::OK();
}

Status SimpleConfig::PrintWithPrefix(const std::string &prefix) const {
  for (const auto &pair : data_) {
    if (pair.first.find(prefix) == 0) {
      std::cout << pair.first << ": " << pair.second << std::endl;
    }
  }
  return Status::OK();
}

Status SimpleConfig::FindPreProcessOp(const std::string &prefix) const {
  // Simplified implementation - just check if prefix exists
  for (const auto &pair : data_) {
    if (pair.first.find(prefix) == 0) {
      return Status::OK();
    }
  }
  return Status::NotFoundError("Prefix not found: " + prefix);
}

SimpleConfig::VectorVariant SimpleConfig::SmartParseVector(const std::string &input) {
  VectorVariant result;
  
  if (input.empty()) {
    return result;
  }
  
  // Remove brackets if present
  std::string clean_input = input;
  if (!clean_input.empty() && clean_input.front() == '[' && clean_input.back() == ']') {
    clean_input = clean_input.substr(1, clean_input.length() - 2);
  }
  
  // Split by comma
  std::stringstream ss(clean_input);
  std::string item;
  
  bool is_int_vec = true;
  bool is_float_vec = true;
  std::vector<std::string> tokens;
  
  while (std::getline(ss, item, ',')) {
    // Trim spaces
    size_t start = item.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    size_t end = item.find_last_not_of(" \t\r\n");
    item = item.substr(start, end - start + 1);
    
    if (!item.empty()) {
      tokens.push_back(item);
      
      // Check if it's a number
      bool is_num = true;
      bool is_flt = true;
      bool has_dot = false;
      size_t start_idx = 0;
      if (item[0] == '-' || item[0] == '+') start_idx = 1;
      
      for (size_t i = start_idx; i < item.length(); ++i) {
        if (item[i] == '.') {
          if (has_dot) { is_num = false; is_flt = false; break; }
          has_dot = true;
        } else if (!std::isdigit(item[i])) {
          is_num = false; is_flt = false; break;
        }
      }
      
      if (!is_num) {
        is_int_vec = false;
      }
      if (!is_flt) {
        is_float_vec = false;
      }
    }
  }
  
  // Convert based on detected type
  if (is_int_vec && !tokens.empty()) {
    for (const auto &token : tokens) {
      try {
        result.vec_int.push_back(std::stoi(token));
      } catch (...) {
        // Fall back to string
        result.vec_string.push_back(token);
      }
    }
  } else if (is_float_vec && !tokens.empty()) {
    for (const auto &token : tokens) {
      try {
        result.vec_float.push_back(std::stof(token));
      } catch (...) {
        // Fall back to string
        result.vec_string.push_back(token);
      }
    }
  } else {
    result.vec_string = tokens;
  }
  
  return result;
}

std::vector<std::string> SimpleConfig::LoadCharacterDict(const std::string &dict_path) {
  std::vector<std::string> character_list;
  std::ifstream file(dict_path);
  
  if (!file.is_open()) {
    std::cout << "[WARNING] Cannot open character dict file: " << dict_path << std::endl;
    std::cout << "[INFO] Using default character set" << std::endl;
    return character_list; // Return empty, will use default
  }
  
  std::string line;
  while (std::getline(file, line)) {
    // Remove whitespace and newlines
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
      size_t end = line.find_last_not_of(" \t\r\n");
      std::string character = line.substr(start, end - start + 1);
      if (!character.empty()) {
        character_list.push_back(character);
      }
    }
  }
  
  std::cout << "[INFO] Loaded " << character_list.size() << " characters from " << dict_path << std::endl;
  return character_list;
}

StatusOr<std::pair<std::string, std::string>> SimpleConfig::FindKey(const std::string &key) {
  auto it = data_.find(key);
  if (it != data_.end()) {
    return std::make_pair(it->first, it->second);
  }
  
  // Try with SubModules prefix
  std::string prefixed_key = "SubModules." + key;
  it = data_.find(prefixed_key);
  if (it != data_.end()) {
    return std::make_pair(it->first, it->second);
  }
  
  return Status::NotFoundError("Key not found: " + key);
}

std::string SimpleConfig::Trim(const std::string &str) const {
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

bool SimpleConfig::IsNumber(const std::string &str) const {
  if (str.empty()) return false;
  size_t start = 0;
  if (str[0] == '-' || str[0] == '+') start = 1;
  for (size_t i = start; i < str.length(); ++i) {
    if (!std::isdigit(str[i])) return false;
  }
  return true;
}

bool SimpleConfig::IsFloat(const std::string &str) const {
  if (str.empty()) return false;
  bool has_dot = false;
  size_t start = 0;
  if (str[0] == '-' || str[0] == '+') start = 1;
  
  for (size_t i = start; i < str.length(); ++i) {
    if (str[i] == '.') {
      if (has_dot) return false; // Multiple dots
      has_dot = true;
    } else if (!std::isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

bool SimpleConfig::IsBool(const std::string &str) const {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
  return (lower_str == "true" || lower_str == "false" || 
          lower_str == "yes" || lower_str == "no" ||
          lower_str == "1" || lower_str == "0");
}

std::unordered_map<std::string, std::string> SimpleConfig::PreProcessOpInfo() const {
  // Return hardcoded preprocessing parameters for OCR
  std::unordered_map<std::string, std::string> result;
  result["DecodeImage.img_mode"] = "BGR";
  result["DetResizeForTest.resize_long"] = "960";
  result["NormalizeImage.mean"] = "[0.485, 0.456, 0.406]";
  result["NormalizeImage.std"] = "[0.229, 0.224, 0.225]";
  result["NormalizeImage.is_scale"] = "true";
  return result;
}

std::unordered_map<std::string, std::string> SimpleConfig::PostProcessOpInfo() const {
  // Return hardcoded postprocessing parameters for OCR
  std::unordered_map<std::string, std::string> result;
  
  // DB (text detection) post-processing parameters
  result["DBPostProcess.thresh"] = "0.3";
  result["DBPostProcess.box_thresh"] = "0.6";
  result["DBPostProcess.max_candidates"] = "1000";
  result["DBPostProcess.unclip_ratio"] = "1.5";
  
  // CTC (text recognition) post-processing parameters
  result["PostProcess.character_dict"] = "./ppocr_keys_v1.txt";
  
  return result;
}

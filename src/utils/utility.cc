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

#include "src/utils/utility.h"

#include <dirent.h>
#include <sys/stat.h>

#include <regex>

#include "simple_logger.h"

Status Utility::FileExists(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return Status::OK();
  } else {
    return Status::NotFoundError("File is not exist:" + path);
  }
}
StatusOr<std::map<std::string, std::pair<std::string, std::string>>>
Utility::GetModelPaths(const std::string &model_dir,
                       const std::string &model_file_prefix) {
  std::map<std::string, std::pair<std::string, std::string>> model_paths;
  std::string model_path;

  std::string json_path =
      model_dir + PATH_SEPARATOR + model_file_prefix + ".json";
  std::string pdmodel_path =
      model_dir + PATH_SEPARATOR + model_file_prefix + ".pdmodel";
  std::string params_path =
      model_dir + PATH_SEPARATOR + model_file_prefix + ".pdiparams";
  if (FileExists(json_path).ok()) {
    model_path = json_path;
  } else if (FileExists(pdmodel_path).ok()) {
    model_path = pdmodel_path;
  } else {
    return Status::NotFoundError(FileExists(json_path).ToString() + " and " +
                               FileExists(pdmodel_path).ToString());
  }

  if (model_path.empty()) {
    return Status::NotFoundError(
        "No PaddlePaddle model file (.json or .pdmodel) found!");
  }

  if (FileExists(params_path).ok()) {
    model_paths["paddle"] = std::make_pair(model_path, params_path);
  } else {
    return Status::NotFoundError(
        "No PaddlePaddle params file (.pdiparams) found!");
  }

  return model_paths;
}

StatusOr<std::string>
Utility::FindModelPath(const std::string &model_dir,
                       const std::string &model_name) {
  char last_char = model_dir.back();
  std::string model_path;
  if (last_char == PATH_SEPARATOR)
    model_path = model_dir + model_name;
  else
    model_path = model_dir + PATH_SEPARATOR + model_name;
  auto status = FileExists(model_path);
  if (!status.ok()) {
    return status;
  }
  return model_path;
}
StatusOr<std::string>
Utility::GetDefaultConfig(std::string pipeline_name) {
  // Try relative paths first (relative to executable)
  std::vector<std::string> candidate_paths = {
    "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yaml",
    "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yml",
    ".." + std::string(1, PATH_SEPARATOR) + "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yaml",
    ".." + std::string(1, PATH_SEPARATOR) + "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yml",
    ".." + std::string(1, PATH_SEPARATOR) + "src" + std::string(1, PATH_SEPARATOR) + "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yaml",
    ".." + std::string(1, PATH_SEPARATOR) + "src" + std::string(1, PATH_SEPARATOR) + "configs" + std::string(1, PATH_SEPARATOR) + pipeline_name + ".yml"
  };
  
  std::cout << "[DEBUG] Searching for config file: " << pipeline_name << std::endl;
  
  for (const auto& path : candidate_paths) {
    std::cout << "[DEBUG] Trying config path: " << path << std::endl;
    if (FileExists(path).ok()) {
      std::cout << "[DEBUG] Found config file: " << path << std::endl;
      return path;
    }
  }
  
  std::cout << "[DEBUG] No config file found. Checked paths:" << std::endl;
  for (const auto& path : candidate_paths) {
    std::cout << "  - " << path << std::endl;
  }
  
  return Status::NotFoundError("Could not find pipeline config yaml: " + pipeline_name);
}
StatusOr<std::string>
Utility::GetConfigPaths(const std::string &model_dir,
                        const std::string &model_file_prefix) {
  std::string config_path = "";
  std::string config_path_find =
      model_dir + PATH_SEPARATOR + model_file_prefix + ".yml";
  if (FileExists(config_path_find).ok()) {
    config_path = config_path_find;
  } else {
    return FileExists(config_path_find);
  }
  return config_path;
};

void Utility::PrintShape(const cv::Mat &img) {
  for (int i = 0; i < img.dims; i++) {
    std::cout << img.size[i] << " ";
  }
  std::cout << std::endl;
}

Status Utility::MyCreateDirectory(const std::string &path) {
#ifdef _WIN32
  int ret = _mkdir(path.c_str());
#else
  int ret = mkdir(path.c_str(), 0755);
#endif
  if (ret == 0) {
    return Status::OK();
  }
  if (errno == EEXIST) {
    return Status::OK();
  }
  return Status::ErrnoToStatus(errno, "Failed to create directory: " + path);
}

Status Utility::MyCreatePath(const std::string &path) {
  std::vector<std::string> paths;
  std::string tmp;
  for (size_t i = 0; i < path.size(); ++i) {
    tmp += path[i];
    if (path[i] == PATH_SEPARATOR) {
      paths.push_back(tmp);
    }
  }
  if (!tmp.empty() && tmp.back() != PATH_SEPARATOR)
    paths.push_back(tmp);

  std::string current;
  for (size_t i = 0; i < paths.size(); ++i) {
    current += paths[i];
    Status status = MyCreateDirectory(current);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::OK();
}

Status Utility::MyCreateFile(const std::string &filepath) {
  std::ifstream infile(filepath.c_str());
  if (infile.good()) {
    return Status::OK();
  }

  std::ofstream outfile(filepath.c_str(), std::ios::out | std::ios::trunc);
  if (!outfile.is_open()) {
    return Status::InternalError("Failed to create file: " + filepath);
  }

  outfile.close();
  return Status::OK();
}

StatusOr<std::vector<cv::Mat>> Utility::SplitBatch(const cv::Mat &batch) {
  if (batch.dims < 1) {
    return Status::InvalidArgumentError(
        "Input batch must have at least 1 dimension.");
  }
  if (batch.type() != CV_32F) {
    return Status::InvalidArgumentError(
        "Input batch must have CV_32F element type.");
  }

  std::vector<cv::Mat> split_mats;
  int batch_size = batch.size[0];
  std::vector<cv::Range> myranges(batch.dims);
  for (int i = 0; i < batch_size; ++i) {
    myranges[0] = cv::Range(i, i + 1);
    for (int d = 1; d < batch.dims; ++d)
      myranges[d] = cv::Range::all();
    cv::Mat sub_mat = batch(&myranges[0]);

    split_mats.push_back(sub_mat);
  }

  return split_mats;
}

std::string Utility::GetFileExtension(const std::string &file_path) {
  size_t pos = file_path.find_last_of('.');
  if (pos == std::string::npos || pos == file_path.length() - 1) {
    return "";
  }
  return file_path.substr(pos + 1);
}

std::string Utility::ToLower(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

bool Utility::IsDirectory(const std::string &path) {
  struct stat path_stat;
  if (stat(path.c_str(), &path_stat) != 0) {
    return false;
  }
  return S_ISDIR(path_stat.st_mode);
}

void Utility::GetFilesRecursive(const std::string &dir_path,
                                std::vector<std::string> &file_list) {
  DIR *dir = opendir(dir_path.c_str());
  if (dir == NULL) {
    return;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }

    std::string full_path = "";
    if (dir_path.back() == PATH_SEPARATOR) {
      full_path = dir_path + name;
    } else {
      full_path = dir_path + PATH_SEPARATOR + name;
    }

    if (Utility::IsDirectory(full_path)) {
      Utility::GetFilesRecursive(full_path, file_list);
    } else if (IsImageFile(full_path)) {
      file_list.push_back(full_path);
    }
  }

  closedir(dir);
}

bool Utility::IsImageFile(const std::string &file_path) {
  std::string extension = GetFileExtension(file_path);
  std::string lower_ext = ToLower(extension);
  return kImgSuffixes.find(lower_ext) != kImgSuffixes.end();
}

StatusOr<cv::Mat> Utility::MyLoadImage(const std::string &file_path) {
  cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return Status::InvalidArgumentError("Failed to load image: " + file_path);
  }
  return image;
}

int Utility::MakeDir(const std::string &path) {
#ifdef _WIN32
  return _mkdir(path.c_str());
#else
  return mkdir(path.c_str(), 0755); // Linux/macOS 权限 755
#endif
}

Status Utility::CreateDirectoryRecursive(const std::string &path) {
  if (path.empty()) {
    return Status::InvalidArgumentError("Path cannot be empty");
  }

  size_t pos = 0;
  std::string dir = path;
#ifdef _WIN32
#define ACCESS _access
#define F_OK 0
#else
#define ACCESS access
#endif
  while (pos < dir.size()) {
    pos = dir.find_first_of(PATH_SEPARATOR, pos + 1);
    std::string subdir = (pos == std::string::npos) ? dir : dir.substr(0, pos);

    if (!subdir.empty() && ACCESS(subdir.c_str(), F_OK) != 0) {
      if (MakeDir(subdir) != 0) {
        return Status::InternalError("Failed to create directory: " + subdir);
      }
    }

    if (pos == std::string::npos) {
      break;
    }
  }
  return Status::OK();
}

Status Utility::CreateDirectoryForFile(const std::string &filePath) {
  size_t found = filePath.find_last_of(PATH_SEPARATOR);
  if (found != std::string::npos) {
    std::string dirPath = filePath.substr(0, found);
    if (!CreateDirectoryRecursive(dirPath).ok()) {
      return Status::InternalError("Failed to create file: " + filePath);
      ;
    }
  }
  return Status::OK();
}

StatusOr<std::string>
Utility::SmartCreateDirectoryForImage(std::string save_path,
                                      const std::string &input_path,
                                      const std::string &suffix) {
  size_t pos = save_path.find_last_of("/\\");
  std::string lastPart = save_path.substr(pos + 1);
  if (lastPart.find(".") == std::string::npos) {
    save_path += PATH_SEPARATOR;
  }
  std::string full_path = save_path;
  auto status = CreateDirectoryForFile(save_path);
  if (!status.ok()) {
    return status;
  }
  if (Utility::IsDirectory(save_path)) {
    auto file_path = input_path;
    size_t pos = file_path.find_last_of("/\\");
    std::string file_name =
        (pos == std::string::npos) ? file_path : file_path.substr(pos + 1);
    size_t dot_pos = file_name.find_last_of('.');
    if (dot_pos == std::string::npos) {
      file_name = file_name + suffix;
    } else {
      file_name.insert(dot_pos, suffix);
    }
    if (save_path.back() != PATH_SEPARATOR) {
      full_path += PATH_SEPARATOR;
    }
    full_path += file_name;
  }
  return full_path;
}

StatusOr<std::string>
Utility::SmartCreateDirectoryForJson(const std::string &save_path,
                                     const std::string &input_path,
                                     const std::string &suffix) {
  auto full_path = SmartCreateDirectoryForImage(save_path, input_path, suffix);
  if (!full_path.ok()) {
    return full_path.status();
  }
  size_t pos = full_path.value().rfind('.');
  if (pos != std::string::npos) {
    full_path.value().replace(pos, std::string::npos, ".json");
  }
  return full_path.value();
}

StatusOr<std::string>
Utility::SmartCreateDirectoryForTxt(const std::string &save_path,
                                    const std::string &input_path,
                                    const std::string &suffix) {
  auto full_path = SmartCreateDirectoryForImage(save_path, input_path, suffix);
  if (!full_path.ok()) {
    return full_path.status();
  }
  size_t pos = full_path.value().rfind('.');
  if (pos != std::string::npos) {
    full_path.value().replace(pos, std::string::npos, ".txt");
  }
  return full_path.value();
}

StatusOr<int> Utility::StringToInt(std::string s) {
  std::regex pattern("(\\d+)");
  std::smatch match;
  if (std::regex_search(s, match, pattern)) {
    int value = std::stoi(match[1]);
    return value;
  } else {
    return Status::NotFoundError("Could not find int !");
  }
}

bool Utility::StringToBool(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  assert(result == "true" || result == "false");
  if (result == "true") {
    return true;
  } else {
    return false;
  }
}

std::string Utility::VecToString(const std::vector<int> &input) {
  std::string result;
  for (auto it = input.begin(); it != input.end(); ++it) {
    if (it != input.begin())
      result += ",";
    result += std::to_string(*it);
  }
  return result;
}

StatusOr<std::tuple<std::string, std::string, std::string>>
Utility::GetOcrModelInfo(std::string lang, std::string ppocr_version) {
  // Font constants
  const static std::string PINGFANG_FONT = "PingFang-SC-Regular.ttf";
  const static std::string SIMFANG_FONT = "simfang.ttf";
  const static std::string LATIN_FONT = "latin.ttf";
  const static std::string KOREAN_FONT = "korean.ttf";
  const static std::string ARABIC_FONT = "arabic.ttf";
  const static std::string CYRILLIC_FONT = "cyrillic.ttf";
  const static std::string KANNADA_FONT = "kannada.ttf";
  const static std::string TELUGU_FONT = "telugu.ttf";
  const static std::string TAMIL_FONT = "tamil.ttf";
  const static std::string DEVANAGARI_FONT = "devanagari.ttf";

  // Supported PP-OCR versions
  const static std::unordered_set<std::string> SUPPORT_PPOCR_VERSION = {
      "PP-OCRv5", "PP-OCRv4", "PP-OCRv3"};

  // Language sets
  const static std::unordered_set<std::string> LATIN_LANGS = {
      "af", "az", "bs", "cs",       "cy",     "da",    "de", "es", "et",
      "fr", "ga", "hr", "hu",       "id",     "is",    "it", "ku", "la",
      "lt", "lv", "mi", "ms",       "mt",     "nl",    "no", "oc", "pi",
      "pl", "pt", "ro", "rs_latin", "sk",     "sl",    "sq", "sv", "sw",
      "tl", "tr", "uz", "vi",       "french", "german"};

  const static std::unordered_set<std::string> ARABIC_LANGS = {"ar", "fa", "ug",
                                                               "ur"};
  const static std::unordered_set<std::string> ESLAV_LANGS = {"ru", "be", "uk"};
  const static std::unordered_set<std::string> CYRILLIC_LANGS = {
      "ru",  "rs_cyrillic", "be",  "bg",  "uk",  "mn",  "abq", "ady",
      "kbd", "ava",         "dar", "inh", "che", "lbe", "lez", "tab"};
  const static std::unordered_set<std::string> DEVANAGARI_LANGS = {
      "hi",  "mr",  "ne",  "bh",  "mai", "ang", "bho",
      "mah", "sck", "new", "gom", "sa",  "bgc"};
  const static std::unordered_set<std::string> SPECIFIC_LANGS = {
      "ch", "en", "korean", "japan", "chinese_cht", "te", "ka", "ta"};

  // Validate input parameters
  if (!ppocr_version.empty() &&
      SUPPORT_PPOCR_VERSION.count(ppocr_version) == 0) {
    return Status::InvalidArgumentError("Unsupported ppocr_version: " +
                                      ppocr_version);
  }

  if (lang.empty())
    lang = "ch";

  // Create combined supported languages set
  const static std::unordered_set<std::string> supported_langs = []() {
    std::unordered_set<std::string> s;
    s.insert(LATIN_LANGS.begin(), LATIN_LANGS.end());
    s.insert(ARABIC_LANGS.begin(), ARABIC_LANGS.end());
    s.insert(ESLAV_LANGS.begin(), ESLAV_LANGS.end());
    s.insert(CYRILLIC_LANGS.begin(), CYRILLIC_LANGS.end());
    s.insert(DEVANAGARI_LANGS.begin(), DEVANAGARI_LANGS.end());
    s.insert(SPECIFIC_LANGS.begin(), SPECIFIC_LANGS.end());
    s.insert("ch");
    return s;
  }();

  if (supported_langs.count(lang) == 0) {
    return Status::InvalidArgumentError("Unsupported lang: " + lang);
  }

  // Determine default ppocr_version if not specified
  if (ppocr_version.empty()) {
    std::unordered_set<std::string> v5_langs = {"ch", "chinese_cht", "en",
                                                "japan", "korean"};
    v5_langs.insert(LATIN_LANGS.begin(), LATIN_LANGS.end());
    v5_langs.insert(ESLAV_LANGS.begin(), ESLAV_LANGS.end());

    if (v5_langs.count(lang)) {
      ppocr_version = "PP-OCRv5";
    } else {
      std::unordered_set<std::string> v3_langs = LATIN_LANGS;
      v3_langs.insert(ARABIC_LANGS.begin(), ARABIC_LANGS.end());
      v3_langs.insert(CYRILLIC_LANGS.begin(), CYRILLIC_LANGS.end());
      v3_langs.insert(DEVANAGARI_LANGS.begin(), DEVANAGARI_LANGS.end());
      v3_langs.insert(SPECIFIC_LANGS.begin(), SPECIFIC_LANGS.end());

      if (v3_langs.count(lang)) {
        ppocr_version = "PP-OCRv3";
      } else {
        return Status::InvalidArgumentError(
            "Invalid lang and ocr_version combination!");
      }
    }
  }

  // Initialize return values
  std::string det_model_name;
  std::string rec_model_name;
  std::string font_name = SIMFANG_FONT; // Default font

  // Model and font selection logic
  if (ppocr_version == "PP-OCRv5") {
    det_model_name = "PP-OCRv5_server_det";
    std::string rec_lang;

    if (lang == "ch" || lang == "chinese_cht" || lang == "en" ||
        lang == "japan") {
      rec_model_name = "PP-OCRv5_server_rec";
      font_name = SIMFANG_FONT;
    } else if (LATIN_LANGS.count(lang)) {
      rec_lang = "latin";
      font_name = LATIN_FONT;
    } else if (ESLAV_LANGS.count(lang)) {
      rec_lang = "eslav";
      font_name = CYRILLIC_FONT;
    } else if (lang == "korean") {
      rec_lang = "korean";
      font_name = KOREAN_FONT;
    }

    if (!rec_lang.empty()) {
      rec_model_name = rec_lang + "_PP-OCRv5_mobile_rec";
    }
  } else if (ppocr_version == "PP-OCRv4") {
    if (lang == "ch") {
      det_model_name = "PP-OCRv4_mobile_det";
      rec_model_name = "PP-OCRv4_mobile_rec";
      font_name = SIMFANG_FONT;
    } else if (lang == "en") {
      det_model_name = "PP-OCRv4_mobile_det";
      rec_model_name = "en_PP-OCRv4_mobile_rec";
      font_name = SIMFANG_FONT;
    } else {
      return Status::InvalidArgumentError(
          "PP-OCRv4 only support ch and en languages!");
    }
  } else { // PP-OCRv3
    det_model_name = "PP-OCRv3_mobile_det";
    std::string rec_lang;

    if (LATIN_LANGS.count(lang)) {
      rec_lang = "latin";
      font_name = LATIN_FONT;
    } else if (ARABIC_LANGS.count(lang)) {
      rec_lang = "arabic";
      font_name = ARABIC_FONT;
    } else if (CYRILLIC_LANGS.count(lang)) {
      rec_lang = "cyrillic";
      font_name = CYRILLIC_FONT;
    } else if (DEVANAGARI_LANGS.count(lang)) {
      rec_lang = "devanagari";
      font_name = DEVANAGARI_FONT;
    } else if (SPECIFIC_LANGS.count(lang)) {
      rec_lang = lang;
      if (lang == "ka") {
        font_name = KANNADA_FONT;
      } else if (lang == "te") {
        font_name = TELUGU_FONT;
      } else if (lang == "ta") {
        font_name = TAMIL_FONT;
      } else if (lang == "ch") {
        font_name = SIMFANG_FONT;
      }
    }

    if (rec_lang == "ch") {
      rec_model_name = "PP-OCRv3_mobile_rec";
    } else if (!rec_lang.empty()) {
      rec_model_name = rec_lang + "_PP-OCRv3_mobile_rec";
    }
  }

  if (rec_model_name.empty()) {
    return Status::InvalidArgumentError(
        "Invalid lang and ocr_version combination!");
  }

  return std::make_tuple(det_model_name, rec_model_name, font_name);
}
const std::set<std::string> Utility::kImgSuffixes = {"jpg", "png", "jpeg",
                                                     "bmp"};


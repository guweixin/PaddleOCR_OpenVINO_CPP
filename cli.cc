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

#include "src/api/pipelines/ocr.h"
#include "src/pipelines/ocr/result.h"
#include "src/utils/args.h"
#include "src/utils/simple_config.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif

static const std::unordered_set<std::string> SUPPORT_MODE_PIPELINE = {
    "ocr",
};

void PrintErrorInfo(const std::string &msg, const std::string &main_mode = "") {
  auto join_modes =
      [](const std::unordered_set<std::string> &modes) -> std::string {
    std::string result;
    for (const auto &mode : modes) {
      result += mode + ", ";
    }
    if (!result.empty()) {
      result.pop_back();
      result.pop_back();
    }
    return result;
  };

  std::string pipeline_modes = join_modes(SUPPORT_MODE_PIPELINE);

  INFOE("%s%s", msg.c_str(),
        main_mode.empty() ? "" : (": \"" + main_mode + "\"").c_str());
  INFO("==========================================");
  INFO("Supported pipeline : [%s]", pipeline_modes.c_str());
  INFO("==========================================");
}

PaddleOCRParams GetPipelineParams() {
  PaddleOCRParams ocr_params;
  if (!text_detection_model_name.empty()) {
    ocr_params.text_detection_model_name = text_detection_model_name;
  }
  if (!text_detection_model_dir.empty()) {
    ocr_params.text_detection_model_dir = text_detection_model_dir;
  }
  if (!text_recognition_model_name.empty()) {
    ocr_params.text_recognition_model_name = text_recognition_model_name;
  }
  if (!text_recognition_model_dir.empty()) {
    ocr_params.text_recognition_model_dir = text_recognition_model_dir;
  }
  if (!text_recognition_batch_size.empty()) {
    ocr_params.text_recognition_batch_size =
        std::stoi(text_recognition_batch_size);
  }
  if (!text_det_limit_side_len.empty()) {
    ocr_params.text_det_limit_side_len =
        std::stoi(text_det_limit_side_len);
  }
  if (!text_det_limit_type.empty()) {
    ocr_params.text_det_limit_type = text_det_limit_type;
  }
  if (!text_det_thresh.empty()) {
    ocr_params.text_det_thresh = std::stof(text_det_thresh);
  }
  if (!text_det_box_thresh.empty()) {
    ocr_params.text_det_box_thresh = std::stof(text_det_box_thresh);
  }
  if (!text_det_unclip_ratio.empty()) {
    ocr_params.text_det_unclip_ratio = std::stof(text_det_unclip_ratio);
  }
  if (!text_det_input_shape.empty()) {
    ocr_params.text_det_input_shape =
        SimpleConfig::SmartParseVector(text_det_input_shape).vec_int;
  }
  if (!text_rec_score_thresh.empty()) {
    ocr_params.text_rec_score_thresh = std::stof(text_rec_score_thresh);
  }
  if (!text_rec_input_shape.empty()) {
    ocr_params.text_rec_input_shape =
        SimpleConfig::SmartParseVector(text_rec_input_shape).vec_int;
  }
  if (!device.empty()) {
    ocr_params.device = device;
  }
  if (!precision.empty()) {
    ocr_params.precision = precision;
  }
  if (!cpu_threads.empty()) {
    ocr_params.cpu_threads = std::stoi(cpu_threads);
  }
  if (!thread_num.empty()) {
    ocr_params.thread_num = std::stoi(thread_num);
  }
  return ocr_params;
}


void showProgress(size_t current, size_t total)
{
  const int bar_width = 50;
  float progress = static_cast<float>(current) / total;
  int pos = static_cast<int>(bar_width * progress);

  std::cout << "\r[";
  for (int i = 0; i < bar_width; ++i)
  {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
            << current << "/" << total;
  std::cout.flush();

  if (current == total)
  {
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (input.empty()) {
    INFOE("Require input, such as ./build/ppocr ocr --input "
          "your_image_path [--param1] [--param2] [...]");
    exit(-1);
  }
  std::string main_mode = "";
  if (argc > 1) {
    main_mode = argv[1];
    if (SUPPORT_MODE_PIPELINE.count(main_mode) == 0) {
      PrintErrorInfo("ERROR: Unsupported pipeline", main_mode);
      exit(-1);
    }
  } else {
    PrintErrorInfo(
        "Must provide pipeline name, such as ./build/ppocr "
        "ocr [--param1] [--param2] [...]");
    exit(-1);
  }
  
  auto params = GetPipelineParams();
  auto start_time = std::chrono::high_resolution_clock::now();
  auto ocr_pipeline = PaddleOCR(params);
  auto end_time = std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  std::cout << "models init time: " << (init_time) << " ms" << std::endl;

  // 检查输入是文件还是目录
  std::vector<std::string> image_paths;
  struct stat st;
  if (stat(input.c_str(), &st) == 0) {
    if (st.st_mode & S_IFDIR) {
      // 是目录，获取目录中的所有图片文件
      std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
      
      #ifdef _WIN32
      WIN32_FIND_DATA findFileData;
      HANDLE hFind = FindFirstFile((input + "\\*").c_str(), &findFileData);
      if (hFind != INVALID_HANDLE_VALUE) {
        do {
          std::string filename = findFileData.cFileName;
          if (filename != "." && filename != "..") {
            std::string full_path = input + "\\" + filename;
            // 检查文件扩展名
            for (const auto& ext : extensions) {
              if (filename.size() >= ext.size() && 
                  filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
                image_paths.push_back(full_path);
                break;
              }
            }
          }
        } while (FindNextFile(hFind, &findFileData) != 0);
        FindClose(hFind);
      }
      #else
      // Linux/Unix implementation would go here
      #endif
    } else {
      // 是单个文件
      image_paths.push_back(input);
    }
  } else {
    INFOE("Input path does not exist: %s", input.c_str());
    exit(-1);
  }
  
  // 逐个处理每张图片并立即输出结果
  size_t total_items = image_paths.size();
  auto start_infer_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < total_items; ++i)
  {
    // INFO("Processing image: %s", image_path.c_str());
    // Show progress
    showProgress(i + 1, total_items);
    const auto& image_path = image_paths[i];
    auto outputs = ocr_pipeline.Predict(image_path);
    
    // show & save results
    // for (auto &output : outputs) {
    //   output->Print();
    //   output->SaveToImg(save_path);
    //   output->SaveToJson(save_path);
    //   static_cast<OCRResult*>(output.get())->SaveToTxt(save_path);
    // }
    // INFO("Completed processing: %s", image_path.c_str());
  }
  auto end_infer_time = std::chrono::high_resolution_clock::now();
  double inference_time = std::chrono::duration<double, std::milli>(end_infer_time - start_infer_time).count();
  std::cout << "models init time: " << (inference_time/ total_items) << " ms" << std::endl;

  
  return 0;
}

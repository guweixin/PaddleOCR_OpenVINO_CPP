// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <opencv2/imgcodecs.hpp>

#include <include/args.h>
#include <include/paddleocr.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

// Simple memory monitoring and progress display
struct MemoryUsage
{
  size_t current_mb;
  size_t peak_mb;
  size_t initial_mb;
};

static size_t initial_memory_mb = 0;

// Auto-detected GPU usage flag based on inference_device
bool auto_use_gpu = false;

MemoryUsage getMemoryUsage()
{
  MemoryUsage usage = {0, 0, initial_memory_mb};

#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS memInfo;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &memInfo, sizeof(memInfo)))
  {
    usage.current_mb = memInfo.WorkingSetSize / (1024 * 1024);
    usage.peak_mb = memInfo.PeakWorkingSetSize / (1024 * 1024);
  }
#else
  struct rusage rusage_info;
  if (getrusage(RUSAGE_SELF, &rusage_info) == 0)
  {
    usage.current_mb = rusage_info.ru_maxrss / 1024; // Linux: KB to MB
    usage.peak_mb = usage.current_mb;                // On Linux, ru_maxrss is already peak
  }
#endif

  return usage;
}

void initMemoryMonitor()
{
  MemoryUsage usage = getMemoryUsage();
  initial_memory_mb = usage.current_mb;
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

using namespace PaddleOCR;

void check_params()
{
  if (FLAGS_det)
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty())
    {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }

    // Check detection model file for OpenVINO
    if (FLAGS_inference_framework == "ov")
    {
      std::string det_model_path = getModelPath(FLAGS_det_model_dir, "det");
      std::ifstream file(det_model_path);
      if (!file.good())
      {
        std::cout << "Error: Detection model file '" << det_model_path
                  << "' not found" << std::endl;
        exit(1);
      }
    }
  }
  if (FLAGS_rec)
  {
    // std::cout
    //     << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
    //        "if you are using recognition model with PP-OCRv2 or an older "
    //        "version, "
    //        "please set --rec_image_shape='3,32,320"
    //     << std::endl;
    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty())
    {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }

    // Check recognition model file for OpenVINO
    if (FLAGS_inference_framework == "ov")
    {
      std::string rec_model_path = getModelPath(FLAGS_rec_model_dir, "rec");

      if (FLAGS_inference_device == "NPU")
      {
        // For NPU, check if the directory exists and contains required model files
        std::string model_small_path = rec_model_path + "/inference_640_bs1.xml";
        std::string model_big_path = rec_model_path + "/inference_1280_bs1.xml";

        std::ifstream file_small(model_small_path);
        std::ifstream file_big(model_big_path);

        if (!file_small.good() || !file_big.good())
        {
          std::cout << "Error: NPU recognition model files not found in '" << rec_model_path << "'" << std::endl;
          std::cout << "Required files: inference_640_bs1.xml, inference_1280_bs1.xml" << std::endl;
          exit(1);
        }
      }
      else
      {
        // For CPU/GPU, check single model file
        std::ifstream file(rec_model_path);
        if (!file.good())
        {
          std::cout << "Error: Recognition model file '" << rec_model_path
                    << "' not found" << std::endl;
          exit(1);
        }
      }
    }
  }
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8")
  {
    std::cout << "precision should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }

  // Check inference framework parameters
  if (FLAGS_inference_framework != "paddle" && FLAGS_inference_framework != "ov")
  {
    std::cout << "inference_framework should be 'paddle'(default) or 'ov'(OpenVINO). " << std::endl;
    exit(1);
  }

  // Check inference device parameters for OpenVINO
  if (FLAGS_inference_framework == "ov")
  {
    if (FLAGS_inference_device != "CPU" && FLAGS_inference_device != "GPU" && FLAGS_inference_device != "NPU")
    {
      std::cout << "inference_device should be 'CPU', 'GPU', or 'NPU' for OpenVINO. " << std::endl;
      exit(1);
    }
  }

  // Auto-detect GPU usage based on inference_device parameter
  auto_use_gpu = (FLAGS_inference_device == "GPU");

  // Check batch processing mode parameters (output is required)
  if (FLAGS_image_dir.empty())
  {
    std::cout << "Batch processing mode requires --image_dir parameter." << std::endl;
    exit(1);
  }
  if (FLAGS_output.empty())
  {
    std::cout << "Batch processing mode requires --output parameter." << std::endl;
    exit(1);
  }

  // Display inference framework information
  std::cout << "=== Inference Configuration ===" << std::endl;
  std::cout << "Framework: " << FLAGS_inference_framework << std::endl;
  std::cout << "GPU Usage: " << (auto_use_gpu ? "Enabled" : "Disabled") << " (auto-detected from device: " << FLAGS_inference_device << ")" << std::endl;
  if (FLAGS_inference_framework == "ov")
  {
    std::cout << "Device: " << FLAGS_inference_device << std::endl;
    if (FLAGS_det)
    {
      std::string det_model_path = getModelPath(FLAGS_det_model_dir, "det");
      std::cout << "Detection model: " << det_model_path << std::endl;
    }
    if (FLAGS_rec)
    {
      std::string rec_model_path = getModelPath(FLAGS_rec_model_dir, "rec");
      std::cout << "Recognition model: " << rec_model_path << std::endl;
    }
  }
  std::cout << "===============================" << std::endl;
}

std::string ocr_single_image(PPOCR &ocr, const std::string &image_path)
{
  auto start_time_imread = std::chrono::high_resolution_clock::now();
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  std::cout << "image_path:" << image_path << std::endl;
  auto end_time_imread = std::chrono::high_resolution_clock::now();
  double imread_time = std::chrono::duration<double, std::milli>(end_time_imread - start_time_imread).count();
  std::cout << "  " << std::endl;
  std::cout << "########### img_time: " << (imread_time) << " ms" << std::endl;

  if (!img.data)
  {
    std::cerr << "[ERROR] image read failed! image path: " << image_path << std::endl;
    return "";
  }

  std::vector<OCRPredictResult> ocr_results = ocr.ocr(img, FLAGS_det, FLAGS_rec);

  std::string result_text = "";
  for (const auto &res : ocr_results)
  {
    result_text += res.text + "\n";
  }

  return result_text;
}

void run_batch_processing_mode()
{
  std::cout << "Starting PaddleOCR batch processing mode..." << std::endl;

  // Initialize memory monitor
  initMemoryMonitor();

  // Check if image directory exists
  if (!Utility::PathExists(FLAGS_image_dir))
  {
    std::cerr << "[ERROR] Image directory not found: " << FLAGS_image_dir << std::endl;
    exit(1);
  }

  // Check if output directory exists, create if not
  if (!Utility::PathExists(FLAGS_output))
  {
    Utility::CreateDir(FLAGS_output);
  }

  // Get all image files from the directory
  std::vector<cv::String> cv_all_img_names;
  std::vector<cv::String> temp_files;

  // Search for different image formats
  cv::glob(FLAGS_image_dir + "/*.jpg", cv_all_img_names);
  cv::glob(FLAGS_image_dir + "/*.png", temp_files);
  cv_all_img_names.insert(cv_all_img_names.end(), temp_files.begin(), temp_files.end());
  temp_files.clear();

  cv::glob(FLAGS_image_dir + "/*.jpeg", temp_files);
  cv_all_img_names.insert(cv_all_img_names.end(), temp_files.begin(), temp_files.end());
  temp_files.clear();

  cv::glob(FLAGS_image_dir + "/*.JPG", temp_files);
  cv_all_img_names.insert(cv_all_img_names.end(), temp_files.begin(), temp_files.end());
  temp_files.clear();

  cv::glob(FLAGS_image_dir + "/*.PNG", temp_files);
  cv_all_img_names.insert(cv_all_img_names.end(), temp_files.begin(), temp_files.end());

  if (cv_all_img_names.empty())
  {
    std::cerr << "[ERROR] No image files found in: " << FLAGS_image_dir << std::endl;
    std::cerr << "[INFO] Supported formats: .jpg, .png, .jpeg" << std::endl;
    exit(1);
  }

  size_t total_items = cv_all_img_names.size();
  std::cout << "Found " << total_items << " image files." << std::endl;
  std::cout << "Processing..." << std::endl;

  // Initialize OCR
  PPOCR ocr;

  // Statistics
  double sum_inference_time = 0.0;
  MemoryUsage initial_memory = getMemoryUsage();
  MemoryUsage max_memory = initial_memory;
  double sum_memory_increase = 0.0;

  // Process each image
  for (size_t i = 0; i < total_items; ++i)
  {
    // Show progress
    showProgress(i + 1, total_items);

    // Measure inference time (excluding file I/O time)
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string image_text = ocr_single_image(ocr, cv_all_img_names[i]);
    auto end_time = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "image inference_time: " << (inference_time) << " ms" << std::endl;
    sum_inference_time += inference_time;

    // Measure memory (not included in inference time)
    MemoryUsage current_memory = getMemoryUsage();
    if (current_memory.current_mb > max_memory.current_mb)
    {
      max_memory = current_memory;
    }
    sum_memory_increase += (current_memory.current_mb - initial_memory.current_mb);

    // Save results (time not included in inference time)
    std::string base_name = cv_all_img_names[i].substr(cv_all_img_names[i].find_last_of("/\\") + 1);
    base_name = base_name.substr(0, base_name.find_last_of('.'));
    std::string output_file = FLAGS_output + "/" + base_name + ".txt";

    std::ofstream out_file(output_file);
    if (out_file.is_open())
    {
      out_file << image_text;
      out_file.close();
    }
  }

  // Print final results
  std::cout << std::endl;
  std::cout << "======================== Processing Results ========================" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Average inference time: " << (sum_inference_time / total_items) << " ms" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Memory usage (increase only):" << std::endl;
  std::cout << "  Average increase: " << (sum_memory_increase / total_items) << " MB" << std::endl;
  std::cout << "  Maximum increase: " << (max_memory.current_mb - initial_memory.current_mb) << " MB" << std::endl;
  std::cout << "Results saved to: " << FLAGS_output << std::endl;
  std::cout << "=================================================================" << std::endl;
}

int main(int argc, char **argv)
{
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  check_params();

  // Use batch processing mode as default
  run_batch_processing_mode();
  return 0;
}

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

#include <include/args.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/paddleocr.h>
#include <include/ocr_interface.h>
#include <include/ocr_det_openvino.h>

#include <auto_log/autolog.h>
#include <numeric>
#include <iomanip>

namespace PaddleOCR
{

  // Global variable to store current image path for debugging
  thread_local std::string g_current_image_path = "";

  // Utility function to get the correct model path for different frameworks
  std::string getModelPath(const std::string &model_dir, const std::string &model_type)
  {
    if (FLAGS_inference_framework == "ov")
    {
      // OpenVINO framework
      std::string filename;
      if (FLAGS_inference_device == "NPU")
      {
        if (model_type == "det")
        {
          filename = "inference_960.xml";
        }
        else if (model_type == "rec")
        {
          filename = "inference_480_bs1.xml";
        }
        else
        {
          filename = "inference.xml"; // Default for other models
        }
      }
      else if (FLAGS_inference_device == "CPU" || FLAGS_inference_device == "GPU")
      {
        filename = "inference.xml";
      }

      // Handle path separator properly for different OS
      std::string separator = "/";
#ifdef _WIN32
      // Check if model_dir uses Windows-style paths
      if (model_dir.find('\\') != std::string::npos)
      {
        separator = "\\";
      }
#endif
      return model_dir + separator + filename;
    }
    else
    {
      // Paddle framework uses the directory path
      return model_dir;
    }
  }

  struct PPOCR::PPOCR_PRIVATE
  {
    std::unique_ptr<DetectorInterface> detector_;
    std::unique_ptr<RecognizerInterface> recognizer_;
  };

  PPOCR::PPOCR() noexcept : pri_(new PPOCR_PRIVATE)
  {
    if (FLAGS_det)
    {
      std::string det_model_path = getModelPath(FLAGS_det_model_dir, "det");
      this->pri_->detector_ = DetectorFactory::CreateDetector(
          FLAGS_inference_framework, det_model_path, auto_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
          FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_limit_type,
          FLAGS_limit_side_len, FLAGS_det_db_thresh, FLAGS_det_db_box_thresh,
          FLAGS_det_db_unclip_ratio, FLAGS_det_db_score_mode, FLAGS_use_dilation,
          FLAGS_use_tensorrt, FLAGS_precision, FLAGS_inference_device);
    }

    if (FLAGS_rec)
    {
      std::string rec_model_path = getModelPath(FLAGS_rec_model_dir, "rec");
      this->pri_->recognizer_ = RecognizerFactory::CreateRecognizer(
          FLAGS_inference_framework, rec_model_path, auto_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
          FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_rec_char_dict_path,
          FLAGS_use_tensorrt, FLAGS_precision, FLAGS_rec_batch_num,
          FLAGS_rec_img_h, FLAGS_rec_img_w, FLAGS_inference_device);
    }
  }

  PPOCR::~PPOCR() { delete this->pri_; }

  std::vector<std::vector<OCRPredictResult>>
  PPOCR::ocr(const std::vector<cv::Mat> &img_list, bool det, bool rec) noexcept
  {
    std::vector<std::vector<OCRPredictResult>> ocr_results;

    if (!det)
    {
      std::vector<OCRPredictResult> ocr_result;
      ocr_result.resize(img_list.size());
      if (rec)
      {
        this->rec(img_list, ocr_result);
      }
      for (size_t i = 0; i < ocr_result.size(); ++i)
      {
        ocr_results.emplace_back(1, std::move(ocr_result[i]));
      }
    }
    else
    {
      for (size_t i = 0; i < img_list.size(); ++i)
      {
        std::vector<OCRPredictResult> ocr_result =
            this->ocr(img_list[i], true, rec);
        ocr_results.emplace_back(std::move(ocr_result));
      }
    }
    return ocr_results;
  }

  std::vector<OCRPredictResult> PPOCR::ocr(const cv::Mat &img, bool det, bool rec) noexcept
  {
    // auto total_start_time = std::chrono::high_resolution_clock::now();

    // auto start_time_det = std::chrono::high_resolution_clock::now();
    std::vector<OCRPredictResult> ocr_result;
    // det
    this->det(img, ocr_result);
    // auto end_time_det = std::chrono::high_resolution_clock::now();
    // double det_time = std::chrono::duration<double, std::milli>(end_time_det - start_time_det).count();

    // // crop image
    // auto crop_start_time = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> img_list;
    for (size_t j = 0; j < ocr_result.size(); ++j)
    {
      cv::Mat crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
      // cv::imwrite("test_crop" + std::to_string(j) + ".jpg", crop_img);
      img_list.emplace_back(std::move(crop_img));
    }
    // auto crop_end_time = std::chrono::high_resolution_clock::now();
    // double crop_time = std::chrono::duration<double, std::milli>(crop_end_time - crop_start_time).count();

    // rec
    // auto start_time_rec = std::chrono::high_resolution_clock::now();
    if (rec)
    {
      this->rec(img_list, ocr_result);
    }
    // auto end_time_rec = std::chrono::high_resolution_clock::now();
    // double rec_time = std::chrono::duration<double, std::milli>(end_time_rec - start_time_rec).count();

    // auto total_end_time = std::chrono::high_resolution_clock::now();
    // double total_time = std::chrono::duration<double, std::milli>(total_end_time - total_start_time).count();

    // std::cout << "\n======== OCR all time summary ========" << std::endl;
    // std::cout << "detection   : " << det_time << " ms (" << std::to_string((det_time / total_time) * 100) << "%)" << std::endl;
    // std::cout << "images  crop: " << crop_time << " ms (" << std::to_string((crop_time / total_time) * 100) << "%) (cut " << ocr_result.size() << " text area)" << std::endl;
    // std::cout << "recognation : " << rec_time << " ms (" << std::to_string((rec_time / total_time) * 100) << "%)" << std::endl;
    // std::cout << "OCR all time: " << total_time << " ms" << std::endl;
    // std::cout << "===============================\n"
    //           << std::endl;

    return ocr_result;
  }

  void PPOCR::det(const cv::Mat &img,
                  std::vector<OCRPredictResult> &ocr_results) noexcept
  {
    // auto det_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<double> det_times;

    // auto inference_start = std::chrono::high_resolution_clock::now();
    this->pri_->detector_->Run(img, boxes, det_times);
    // auto inference_end = std::chrono::high_resolution_clock::now();
    // double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

    // auto postprocess_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < boxes.size(); ++i)
    {
      OCRPredictResult res;
      res.box = std::move(boxes[i]);
      ocr_results.emplace_back(std::move(res));
    }

    // sort boex from top to bottom, from left to right
    Utility::sort_boxes(ocr_results);
    // auto postprocess_end = std::chrono::high_resolution_clock::now();
    // double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

    // auto det_end_time = std::chrono::high_resolution_clock::now();
    // double total_det_time = std::chrono::duration<double, std::milli>(det_end_time - det_start_time).count();

    // std::cout << "=== PPOCR::det time summary===" << std::endl;
    // // std::cout << "检测inference time: " << inference_time << " ms" << std::endl;
    // // std::cout << "post progress time (box转换+排序): " << postprocess_time << " ms" << std::endl;

    // // 打印detector内部的详细 time（如果有的话）
    // if (!det_times.empty())
    // {
    //   std::cout << "detecter detail time:" << std::endl;
    //   if (det_times.size() >= 11)
    //   {
    //     std::cout << "  step 0 - Image Resize: " << det_times[0] << " ms (" << std::to_string((det_times[0] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 1 - Normalization: " << det_times[1] << " ms (" << std::to_string((det_times[1] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 2 - (HWC->CHW): " << det_times[2] << " ms (" << std::to_string((det_times[2] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 3 - pure inference: " << det_times[3] << " ms (" << std::to_string((det_times[3] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 4 - Thresholding: " << det_times[4] << " ms (" << std::to_string((det_times[4] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 5 - morphological expansion: " << det_times[5] << " ms (" << std::to_string((det_times[5] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 6 - Contour extraction: " << det_times[6] << " ms (" << std::to_string((det_times[6] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  step 7 - border filter: " << det_times[7] << " ms (" << std::to_string((det_times[7] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  --- Summary ---" << std::endl;
    //     std::cout << "  pre progress  time: " << det_times[8] << " ms (" << std::to_string((det_times[8] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  inference all time: " << det_times[9] << " ms (" << std::to_string((det_times[9] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "  post progress time: " << det_times[10] << " ms (" << std::to_string((det_times[10] / total_det_time) * 100) << "%)" << std::endl;
    //     std::cout << "detecter total time: " << total_det_time << " ms" << std::endl;
    //   }
    //   // else
    //   // {
    //   //   for (size_t i = 0; i < det_times.size(); ++i)
    //   //   {
    //   //     std::cout << "  step " << i << ": " << det_times[i] << " ms" << std::endl;
    //   //   }
    //   // }
    // }
    // std::cout << "=========================" << std::endl;
  }

  void PPOCR::rec(const std::vector<cv::Mat> &img_list,
                  std::vector<OCRPredictResult> &ocr_results) noexcept
  {
    // auto rec_start_time = std::chrono::high_resolution_clock::now();

    // auto data_prepare_start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> rec_texts(img_list.size(), std::string());
    std::vector<float> rec_text_scores(img_list.size(), 0);
    std::vector<double> rec_times;
    // auto data_prepare_end = std::chrono::high_resolution_clock::now();
    // double data_prepare_time = std::chrono::duration<double, std::milli>(data_prepare_end - data_prepare_start).count();

    // auto inference_start = std::chrono::high_resolution_clock::now();
    this->pri_->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
    // auto inference_end = std::chrono::high_resolution_clock::now();
    // double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

    // auto postprocess_start = std::chrono::high_resolution_clock::now();
    // output rec results
    for (size_t i = 0; i < rec_texts.size(); ++i)
    {
      ocr_results[i].text = std::move(rec_texts[i]);
      ocr_results[i].score = rec_text_scores[i];
    }
    // auto postprocess_end = std::chrono::high_resolution_clock::now();
    // double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

    // auto rec_end_time = std::chrono::high_resolution_clock::now();
    // double total_rec_time = std::chrono::duration<double, std::milli>(rec_end_time - rec_start_time).count();

    // std::cout << "=== PPOCR::rec time summary===" << std::endl;
    // std::cout << "data prepare: " << data_prepare_time << " ms" << std::endl;
    // std::cout << "rec inference time: " << inference_time << " ms" << std::endl;
    // std::cout << "post progress time (Result assignment): " << postprocess_time << " ms" << std::endl;
    // std::cout << "rec all time: " << total_rec_time << " ms" << std::endl;
    // std::cout << "Number of pictures processed: " << img_list.size() << std::endl;

    // // 打印recognizer内部的详细 time（如果有的话）
    // if (!rec_times.empty())
    // {
    //   std::cout << "recogniter detail time:" << std::endl;
    //   if (rec_times.size() >= 8)
    //   {
    //     std::cout << "  step 0 - Image sorting: " << rec_times[0] << " ms (" << std::to_string((rec_times[0] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  step 1 - Image Resize : " << rec_times[1] << " ms (" << std::to_string((rec_times[1] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  step 2 - Normalization: " << rec_times[2] << " ms (" << std::to_string((rec_times[2] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  step 3 - (HWC->CHW): " << rec_times[3] << " ms (" << std::to_string((rec_times[3] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  step 4 - pure inference: " << rec_times[4] << " ms (" << std::to_string((rec_times[4] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  step 5 - CTC decoder: " << rec_times[5] << " ms (" << std::to_string((rec_times[5] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  --- time summary ---" << std::endl;
    //     std::cout << "  pre progress+post progress: " << rec_times[6] << " ms (" << std::to_string((rec_times[6] / rec_times[7]) * 100) << "%)" << std::endl;
    //     std::cout << "  total time : " << rec_times[7] << " ms" << std::endl;
    //   }
    //   // else
    //   // {
    //   //   for (size_t i = 0; i < rec_times.size(); ++i)
    //   //   {
    //   //     std::cout << "  step " << i << ": " << rec_times[i] << " ms" << std::endl;
    //   //   }
    //   // }
    // }
    // std::cout << "=========================" << std::endl;
  }

  // void PPOCR::reset_timer() noexcept
  // {
  //   // Timer reset function kept for interface compatibility but no longer needed
  // }

  // void PPOCR::benchmark_log(int img_num) noexcept
  // {
  //   // Benchmark logging function kept for interface compatibility but no longer needed
  // }

  // void PPOCR::detailed_benchmark_log(int img_num) noexcept
  // {
  //   // All timing statistics output removed - function kept for compatibility
  // }

} // namespace PaddleOCR

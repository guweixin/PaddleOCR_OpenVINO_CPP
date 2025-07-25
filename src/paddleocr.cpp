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
#include <include/data_saver.h>
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
    std::vector<OCRPredictResult> ocr_result;
    // det
    this->det(img, ocr_result);
    // crop image
    std::vector<cv::Mat> img_list;
    for (size_t j = 0; j < ocr_result.size(); ++j)
    {
      cv::Mat crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
      // cv::imwrite("test_crop" + std::to_string(j) + ".jpg", crop_img);
      img_list.emplace_back(std::move(crop_img));
    }
    // rec

    if (rec)
    {
      this->rec(img_list, ocr_result);
    }
    return ocr_result;
  }

  void PPOCR::det(const cv::Mat &img,
                  std::vector<OCRPredictResult> &ocr_results) noexcept
  {
    // // Print current image path for debugging
    // if (!g_current_image_path.empty())
    // {
    //   std::cout << "[DEBUG] Processing image: " << g_current_image_path << std::endl;
    // }

    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<double> det_times;

    this->pri_->detector_->Run(img, boxes, det_times);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
      OCRPredictResult res;
      res.box = std::move(boxes[i]);
      ocr_results.emplace_back(std::move(res));
    }

    // Save debug data if enabled
    if (FLAGS_save_debug_data)
    {
      static int det_image_counter = 0;

      // // Save original image
      // std::string orig_img_filename = "../../debug_data/cpp_original_img_" + std::to_string(det_image_counter) + ".npy";
      // DataSaver::SaveMatAsNpy(img, orig_img_filename);

      det_image_counter++;
    }

    // sort boex from top to bottom, from left to right
    Utility::sort_boxes(ocr_results);
    
    // Update detailed timing information if available
    if (det_times.size() >= 11)
    {
        // Update detailed detection timings
        // [resize, normalize, permute, inference, threshold, dilation, boxes_from_bitmap, filter_tag]
        for (int i = 0; i < 8; ++i)
        {
            this->time_info_det_detailed[i] += det_times[i];
        }
        // Update summary detection timings (indices 8, 9, 10)
        this->time_info_det[0] += det_times[8];  // preprocess
        this->time_info_det[1] += det_times[9];  // inference
        this->time_info_det[2] += det_times[10]; // postprocess
    }
    else
    {
        // Fallback to old format (3 summary timings only)
        this->time_info_det[0] += det_times[0];
        this->time_info_det[1] += det_times[1];
        this->time_info_det[2] += det_times[2];
    }
  }

  void PPOCR::rec(const std::vector<cv::Mat> &img_list,
                  std::vector<OCRPredictResult> &ocr_results) noexcept
  {
    std::vector<std::string> rec_texts(img_list.size(), std::string());
    std::vector<float> rec_text_scores(img_list.size(), 0);
    std::vector<double> rec_times;
    this->pri_->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);

    // output rec results
    for (size_t i = 0; i < rec_texts.size(); ++i)
    {
      ocr_results[i].text = std::move(rec_texts[i]);
      ocr_results[i].score = rec_text_scores[i];
    }

    // Save debug data if enabled
    if (FLAGS_save_debug_data)
    {
      static int rec_batch_counter = 0;

      // // Save input images
      // for (size_t i = 0; i < img_list.size(); ++i)
      // {
      //   std::string img_filename = "../../debug_data/cpp_rec_input_img_batch" +
      //                              std::to_string(rec_batch_counter) + "_" + std::to_string(i) + ".npy";
      //   DataSaver::SaveMatAsNpy(img_list[i], img_filename);
      // }

      rec_batch_counter++;
    }

    // Update detailed timing information if available
    if (rec_times.size() >= 8)
    {
        // Update detailed recognition timings
        // [resize, normalize, permute, inference, postprocess]
        for (int i = 0; i < 5; ++i)
        {
            this->time_info_rec_detailed[i] += rec_times[i];
        }
        // Update summary recognition timings (indices 5, 6, 7)
        this->time_info_rec[0] += rec_times[5];  // preprocess
        this->time_info_rec[1] += rec_times[6];  // inference
        this->time_info_rec[2] += rec_times[7];  // postprocess
    }
    else
    {
        // Fallback to old format (3 summary timings only)
        this->time_info_rec[0] += rec_times[0];
        this->time_info_rec[1] += rec_times[1];
        this->time_info_rec[2] += rec_times[2];
    }
  }

  void PPOCR::reset_timer() noexcept
  {
    this->time_info_det = {0, 0, 0};
    this->time_info_rec = {0, 0, 0};
    this->time_info_det_detailed = {0, 0, 0, 0, 0, 0, 0, 0};
    this->time_info_rec_detailed = {0, 0, 0, 0, 0};
  }

  void PPOCR::benchmark_log(int img_num) noexcept
  {
    if (this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2] >
        0)
    {
      AutoLogger autolog_det("ocr_det", auto_use_gpu, FLAGS_use_tensorrt,
                             FLAGS_enable_mkldnn, FLAGS_cpu_threads, 1, "dynamic",
                             FLAGS_precision, this->time_info_det, img_num);
      autolog_det.report();
    }
    if (this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2] >
        0)
    {
      AutoLogger autolog_rec("ocr_rec", auto_use_gpu, FLAGS_use_tensorrt,
                             FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                             FLAGS_rec_batch_num, "dynamic", FLAGS_precision,
                             this->time_info_rec, img_num);
      autolog_rec.report();
    }
  }

  void PPOCR::detailed_benchmark_log(int img_num) noexcept
  {
    std::cout << std::endl;
    std::cout << "==================== Detailed Timing Statistics ====================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Detection detailed timing
    if (std::accumulate(time_info_det_detailed.begin(), time_info_det_detailed.end(), 0.0) > 0)
    {
      std::cout << "Detection Module (per image average):" << std::endl;
      std::cout << "  Resize operation:       " << (time_info_det_detailed[0] / img_num) << " ms" << std::endl;
      std::cout << "  Normalize operation:    " << (time_info_det_detailed[1] / img_num) << " ms" << std::endl;
      std::cout << "  Permute operation:      " << (time_info_det_detailed[2] / img_num) << " ms" << std::endl;
      std::cout << "  Inference:              " << (time_info_det_detailed[3] / img_num) << " ms" << std::endl;
      std::cout << "  Threshold operation:    " << (time_info_det_detailed[4] / img_num) << " ms" << std::endl;
      std::cout << "  Dilation operation:     " << (time_info_det_detailed[5] / img_num) << " ms" << std::endl;
      std::cout << "  BoxesFromBitmap:        " << (time_info_det_detailed[6] / img_num) << " ms" << std::endl;
      std::cout << "  FilterTagDetRes:        " << (time_info_det_detailed[7] / img_num) << " ms" << std::endl;
      
      // Use the same calculation as Overall Summary for consistency
      double det_total = this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2];
      std::cout << "  Detection Total:        " << (det_total / img_num) << " ms" << std::endl;
      
      // Also show detailed breakdown total for comparison
      double detailed_total = std::accumulate(time_info_det_detailed.begin(), time_info_det_detailed.end(), 0.0);
      std::cout << "  (Detailed ops total:    " << (detailed_total / img_num) << " ms)" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Recognition detailed timing
    if (std::accumulate(time_info_rec_detailed.begin(), time_info_rec_detailed.end(), 0.0) > 0)
    {
      std::cout << "Recognition Module (per image average):" << std::endl;
      std::cout << "  Resize operation:       " << (time_info_rec_detailed[0] / img_num) << " ms" << std::endl;
      std::cout << "  Normalize operation:    " << (time_info_rec_detailed[1] / img_num) << " ms" << std::endl;
      std::cout << "  Permute operation:      " << (time_info_rec_detailed[2] / img_num) << " ms" << std::endl;
      std::cout << "  Inference:              " << (time_info_rec_detailed[3] / img_num) << " ms" << std::endl;
      std::cout << "  Postprocess:            " << (time_info_rec_detailed[4] / img_num) << " ms" << std::endl;
      
      // Use the same calculation as Overall Summary for consistency
      double rec_total = this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2];
      std::cout << "  Recognition Total:      " << (rec_total / img_num) << " ms" << std::endl;
      
      // Also show detailed breakdown total for comparison
      double detailed_total = std::accumulate(time_info_rec_detailed.begin(), time_info_rec_detailed.end(), 0.0);
      std::cout << "  (Detailed ops total:    " << (detailed_total / img_num) << " ms)" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Overall timing
    double overall_det = this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2];
    double overall_rec = this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2];
    double overall_total = overall_det + overall_rec;
    
    std::cout << "Overall Summary (per image average):" << std::endl;
    std::cout << "  Detection Total:        " << (overall_det / img_num) << " ms" << std::endl;
    std::cout << "  Recognition Total:      " << (overall_rec / img_num) << " ms" << std::endl;
    std::cout << "  End-to-End Total:       " << (overall_total / img_num) << " ms" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Print memory transfer statistics for OpenVINO modules
    if (FLAGS_inference_framework == "openvino")
    {
      // Print detection memory transfer statistics
      if (this->pri_->detector_)
      {
        auto* det_openvino = dynamic_cast<DBDetectorOpenVINO*>(this->pri_->detector_.get());
        if (det_openvino)
        {
          det_openvino->PrintFinalDetectionStats();
        }
      }
    }
  }

} // namespace PaddleOCR

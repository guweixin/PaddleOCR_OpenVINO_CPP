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

#include "pipeline.h"

#include "result.h"
#include "src/utils/args.h"
#include <opencv2/highgui.hpp>
_OCRPipeline::_OCRPipeline(const OCRPipelineParams &params)
    : BasePipeline(), params_(params) {
  
  if (params.paddlex_config.has_value()) {
    if (params.paddlex_config.value().IsStr()) {
      config_ = SimpleConfig(params.paddlex_config.value().GetStr());
    } else {
      config_ = SimpleConfig(params.paddlex_config.value().GetMap());
    }
  } else {
    auto config_path = Utility::GetDefaultConfig("OCR");
    if (!config_path.ok()) {
      INFOE("Could not find OCR pipeline config file: %s",
            config_path.status().ToString().c_str());
      exit(-1);
    }
    config_ = SimpleConfig(config_path.value());
  }

  OverrideConfig();

  auto text_type = config_.GetString("text_type");
  if (!text_type.ok()) {
    INFOE("Get text type fail : %s", text_type.status().ToString().c_str());
    exit(-1);
  }
  text_type_ = text_type.value();
  TextDetPredictorParams params_det;
  auto result_text_det_model_name =
      config_.GetString("TextDetection.model_name");
  if (!result_text_det_model_name.ok()) {
    INFOE("Could not find TextDetection model name : %s",
          result_text_det_model_name.status().ToString().c_str());
    exit(-1);
  }
  params_det.model_name = result_text_det_model_name.value();
  auto result_text_det_model_dir = config_.GetString("TextDetection.model_dir");
  if (!result_text_det_model_dir.ok()) {
    INFOE("Could not find TextDetection model dir : %s",
          result_text_det_model_dir.status().ToString().c_str());
    exit(-1);
  }
  params_det.model_dir = result_text_det_model_dir.value();
  auto result_det_input_shape = config_.GetString("TextDetection.input_shape", "");
  if (result_det_input_shape.ok() && !result_det_input_shape.value().empty()) {
    params_det.input_shape =
        config_.SmartParseVector(result_det_input_shape.value()).vec_int;
  }
  params_det.device = params_.device;
  params_det.precision = params_.precision;
  params_det.cpu_threads = params_.cpu_threads;
  params_det.batch_size = config_.GetInt("TextDetection.batch_size", 1).value();
  if (text_type_ == "general") {
    params_det.limit_side_len =
        config_.GetInt("TextDetection.limit_side_len", 960).value();
    params_det.limit_type =
        config_.GetString("TextDetection.limit_type", "max").value();
    params_det.max_side_limit =
        config_.GetInt("TextDetection.max_side_limit", 4000).value();
    params_det.thresh = config_.GetFloat("TextDetection.thresh", 0.3).value();
    params_det.box_thresh =
        config_.GetFloat("TextDetection.box_thresh", 0.6).value();
    params_det.unclip_ratio =
        config_.GetFloat("TextDetection.unclip_ratio", 2.0).value();
    sort_boxes_ = ComponentsProcessor::SortQuadBoxes;
    crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("quad"));
  } else if (text_type_ == "seal") {
    params_det.limit_side_len =
        config_.GetInt("TextDetection.limit_side_len", 736).value();
    params_det.limit_type =
        config_.GetString("TextDetection.limit_type", "min").value();
    params_det.max_side_limit =
        config_.GetInt("TextDetection.max_side_limit", 4000).value();
    params_det.thresh = config_.GetFloat("TextDetection.thresh", 0.2).value();
    params_det.box_thresh =
        config_.GetFloat("TextDetection.box_thresh", 0.6).value();
    params_det.unclip_ratio =
        config_.GetFloat("TextDetection.unclip_ratio", 0.5).value();
    sort_boxes_ = ComponentsProcessor::SortPolyBoxes;
    crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("poly"));
  } else {
    INFOE("Unsupported text type We %s", text_type.value().c_str());
    exit(-1);
  }
  
  text_det_model_ = CreateModule<TextDetPredictor>(params_det);
  
  text_det_params_.text_det_limit_side_len = params_det.limit_side_len.value();
  text_det_params_.text_det_limit_type = params_det.limit_type.value();
  text_det_params_.text_det_max_side_limit = params_det.max_side_limit.value();
  text_det_params_.text_det_thresh = params_det.thresh.value();
  text_det_params_.text_det_box_thresh = params_det.box_thresh.value();
  text_det_params_.text_det_unclip_ratio = params_det.unclip_ratio.value();


  TextRecPredictorParams params_rec;
  auto result_text_rec_model_name =
      config_.GetString("TextRecognition.model_name");
  if (!result_text_rec_model_name.ok()) {
    INFOE("Could not find TextRecognition model name : %s",
          result_text_rec_model_name.status().ToString().c_str());
    exit(-1);
  }
  params_rec.model_name = result_text_rec_model_name.value();
  auto result_text_rec_model_dir =
      config_.GetString("TextRecognition.model_dir");
  if (!result_text_rec_model_dir.ok()) {
    INFOE("Could not find TextRecognition model dir : %s",
          result_text_rec_model_dir.status().ToString().c_str());
    exit(-1);
  }
  auto result_rec_input_shape =
      config_.GetString("TextRecognition.input_shape", "");
  if (result_rec_input_shape.ok() && !result_rec_input_shape.value().empty()) {
    params_rec.input_shape =
        config_.SmartParseVector(result_rec_input_shape.value()).vec_int;
  }
  params_rec.model_dir = result_text_rec_model_dir.value();
  params_rec.lang = params_.lang;
  params_rec.ocr_version = params_.ocr_version;
  params_rec.vis_font_dir = params_.vis_font_dir;
  params_rec.device = params_.device;
  params_rec.precision = params_.precision;
  params_rec.cpu_threads = params_.cpu_threads;
  params_rec.batch_size =
      config_.GetInt("TextRecognition.batch_size", 1).value();

  text_rec_model_ = CreateModule<TextRecPredictor>(params_rec);
  text_rec_score_thresh_ =
      config_.GetFloat("TextRecognition.score_thresh", 0.0).value();

  batch_sampler_ptr_ = std::unique_ptr<BaseBatchSampler>(
      new ImageBatchSampler(1)); //** pipeline batch_size
};

StatusOr<std::vector<cv::Mat>>
_OCRPipeline::RotateImage(const std::vector<cv::Mat> &image_array_list,
                          const std::vector<int> &rotate_angle_list) {
  if (image_array_list.size() != rotate_angle_list.size()) {
    return Status::InvalidArgumentError(
        "Length of image_array_list (" +
        std::to_string(image_array_list.size()) +
        ") must match length of rotate_angle_list (" +
        std::to_string(rotate_angle_list.size()) + ")");
  }
  std::vector<cv::Mat> rotated_images;
  rotated_images.reserve(image_array_list.size());
  for (std::size_t i = 0; i < image_array_list.size(); ++i) {
    int angle_indicator = rotate_angle_list[i];
    if (angle_indicator != 0 && angle_indicator != 1) {
      return Status::InvalidArgumentError(
          "rotate_angle must be 0 or 1, now it's: " +
          std::to_string(angle_indicator));
    }
    int rotate_angle = angle_indicator * 180;
    auto result_rotated_image =
        ComponentsProcessor::RotateImage(image_array_list[i], rotate_angle);
    if (!result_rotated_image.ok()) {
      return result_rotated_image.status();
    }
    cv::Mat rotated_image = result_rotated_image.value();
    rotated_images.push_back(rotated_image);
  }
  return rotated_images;
}

std::unordered_map<std::string, bool> _OCRPipeline::GetModelSettings() const {
  std::unordered_map<std::string, bool> model_settings = {};
  return model_settings;
}

std::vector<std::unique_ptr<BaseCVResult>>
_OCRPipeline::Predict(const std::vector<std::string> &input) {
  auto model_settings = GetModelSettings();
  auto batches = batch_sampler_ptr_->Apply(input);
  auto batches_string =
      batch_sampler_ptr_->SampleFromVectorToStringVector(input);
  if (!batches.ok()) {
    INFOE("pipeline get sample fail : %s", batches.status().ToString().c_str());
    exit(-1);
  }
  if (!batches_string.ok()) {
    INFOE("pipeline get sample fail : %s",
          batches_string.status().ToString().c_str());
    exit(-1);
  }
  auto input_path = batch_sampler_ptr_->InputPath();
  int index = 0;
  std::vector<std::unique_ptr<BaseCVResult>> base_results = {};
  pipeline_result_vec_.clear();
  for (int i = 0; i < batches.value().size(); i++) {
    std::vector<cv::Mat> origin_image = {};
    origin_image.reserve(batches.value()[i].size());
    for (const auto &mat : batches.value()[i]) {
      origin_image.push_back(mat.clone());
    }
    // 直接使用原图进行处理，不使用文档预处理
    std::vector<cv::Mat> images_for_processing = {};
    for (auto &image : batches.value()[i]) {
      images_for_processing.push_back(image.clone());
    }

    // Guard: if any image in this batch is empty, skip this batch
    bool has_empty_image = false;
    for (const auto &mat : images_for_processing) {
      if (mat.empty()) {
        has_empty_image = true;
        break;
      }
    }
    if (has_empty_image) {
      INFOE("pipeline get sample fail : Input image at batch %d contains empty image, skipping batch.", i);
      // keep input_path/index alignment by advancing index by the batch size
      index += static_cast<int>(batches.value()[i].size());
      continue;
    }
    std::vector<cv::Mat> images_for_det_copy = {};
    for (auto &item : images_for_processing) {
      images_for_det_copy.push_back(item.clone());
    }
    text_det_model_->Predict(images_for_det_copy);
    std::vector<TextDetPredictorResult> det_results =
        static_cast<TextDetPredictor *>(text_det_model_.get())
            ->PredictorResult();
    std::vector<std::vector<std::vector<cv::Point2f>>> dt_polys_list = {};
    for (auto &item : det_results) {
      if (!item.dt_polys.empty()) {
        auto sort_item = sort_boxes_(item.dt_polys);
        dt_polys_list.push_back(sort_item);
      } else {
        dt_polys_list.push_back(std::vector<std::vector<cv::Point2f>>{});
      }
    }

    std::vector<int> indices = {};
    for (int j = 0; j < images_for_processing.size(); j++) {
      if (!dt_polys_list.empty() && !dt_polys_list[j].empty()) {
        indices.push_back(j);
      }
    }
    std::vector<OCRPipelineResult> results(images_for_processing.size());
    for (int k = 0; k < results.size(); k++, index++) {
      results[k].input_path = input_path[index];
      results[k].input_image = origin_image[k];
      results[k].dt_polys = dt_polys_list[k];
      results[k].model_settings = model_settings;
      results[k].text_det_params = text_det_params_;
      results[k].text_type = text_type_;
      results[k].text_rec_score_thresh = text_rec_score_thresh_;
    }
    if (!indices.empty()) {
      std::cout << "--------------------------------" << std::endl;
      std::vector<cv::Mat> all_subs_of_imgs = {};
      std::vector<cv::Mat> all_subs_of_imgs_copy = {};
      std::vector<int> chunk_indices(1, 0);
      for (auto &idx : indices) {
        auto result_all_subs_of_img = (*crop_by_polys_)(
            images_for_processing[idx], dt_polys_list[idx]);
        if (!result_all_subs_of_img.ok()) {
          INFOE("Split image fail : ",
                result_all_subs_of_img.status().ToString().c_str());
          exit(-1);
        }
        all_subs_of_imgs.insert(all_subs_of_imgs.end(),
                                result_all_subs_of_img.value().begin(),
                                result_all_subs_of_img.value().end());
        chunk_indices.emplace_back(chunk_indices.back() +
                                   result_all_subs_of_img.value().size());
      }
      for (auto &item : all_subs_of_imgs) {
        all_subs_of_imgs_copy.push_back(item.clone());
      }
      
      // 跳过文本行方向检测，直接使用原始角度
      std::vector<int> angles = std::vector<int>(all_subs_of_imgs.size(), -1);
      // 跳过文本行方向角度记录
      for (int l = 0; l < indices.size(); l++) {
        std::vector<cv::Mat> all_subs_of_img = {};
        for (int m = chunk_indices[l]; m < chunk_indices[l + 1]; m++) {
          all_subs_of_img.push_back(all_subs_of_imgs[m]);
        }
        std::vector<std::pair<std::pair<int, float>, TextRecPredictorResult>>
            sub_img_info_list = {};

        for (int m = 0; m < all_subs_of_img.size(); m++) {
          int sub_img_id = m;
          float sub_img_ratio = (float)all_subs_of_img[m].size[1] /
                                (float)all_subs_of_img[m].size[0];
          TextRecPredictorResult result;
          sub_img_info_list.push_back({{sub_img_id, sub_img_ratio}, result});
        }
        std::vector<std::pair<int, float>> sorted_subs_info = {};
        for (auto &item : sub_img_info_list) {
          sorted_subs_info.push_back(item.first);
        }
        std::sort(
            sorted_subs_info.begin(), sorted_subs_info.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });
        std::vector<cv::Mat> sorted_subs_of_img = {};
        
        // npu image padding
        for (auto &item : sorted_subs_info) {
          cv::Mat final_img;
          if (device =="npu"){
            std::cout << "----------here----------------------" << std::endl;
            cv::Mat tempimg = all_subs_of_img[item.first];
            
            int src_h = tempimg.rows;
            int src_w = tempimg.cols;
            float aspect_ratio = static_cast<float>(src_w) / static_cast<float>(src_h);       
            // Model specifications and thresholds
            const int target_h = 48;  // Standard OCR recognition height
            const int small_width = 480;   // Small model width
            const int medium_width = 800;  // Medium model width
            const int large_width = 1280;  // Large model width
            
            // Calculate thresholds: model_width / target_h
            float small_threshold = static_cast<float>(small_width) / static_cast<float>(target_h);    // 480/48 = 10.0
            float medium_threshold = static_cast<float>(medium_width) / static_cast<float>(target_h);  // 800/48 = 16.67
            
            int target_w = 0;        
            if (aspect_ratio <= small_threshold) {
              // selected_model_type = NPURecModelSize::SMALL;  // model_type = 0
              target_w = small_width;
            } else if (aspect_ratio <= medium_threshold) {
              // selected_model_type = NPURecModelSize::MEDIUM; // model_type = 1
              target_w = medium_width;
            } else {
              // selected_model_type = NPURecModelSize::LARGE;  // model_type = 2
              target_w = large_width;
            }

            std::cout << "[DEBUG] NPU image " <<  ": src(" << src_w << "x" << src_h 
                  << "), aspect_ratio=" << aspect_ratio 
                  << ", thresholds(small=" << small_threshold << ", medium=" << medium_threshold << ")"
                  // << ", selected_model=" << static_cast<int>(selected_model_type) 
                  << ", target_size(" << target_w << "x" << target_h << ")" << std::endl;
            
            float resize_ratio = std::min(target_h/src_h, target_w/src_w);
            int new_h = static_cast<int>(std::round(src_h * resize_ratio));
            int new_w = static_cast<int>(std::round(src_w * resize_ratio));
            if (new_h > target_h) new_h = target_h;
            if (new_w > target_w) new_w = target_w;
            cv::Mat resized;
            cv::resize(tempimg, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
            
            std::cout << "src h: " << std::to_string(src_h)<<", w: "<< std::to_string(src_w)<<std::endl;
            std::cout << "new h: " << std::to_string(new_h)<<", w: "<< std::to_string(new_w)<<std::endl;
            std::cout << "resize_ratio: " << std::to_string(resize_ratio)<<std::endl;
            
            cv::Mat final_img = cv::Mat::zeros(target_h, target_w, tempimg.type());
              int offset_y = 0;
              int offset_x = 0;
              cv::Rect roi(offset_x, offset_y, new_w, new_h);
              resized.copyTo(final_img(roi));

          }else{
            final_img = all_subs_of_img[item.first];
          }
          sorted_subs_of_img.push_back(final_img);
        }
        // for (auto &item : sorted_subs_info){
        //   sorted_subs_of_img.push_back(all_subs_of_img[item.first]);
        // }
        // Debug: display sorted_subs_of_img for inspection (window sized to image)
        try {
          for (int si = 0; si < static_cast<int>(sorted_subs_of_img.size()); ++si) {
            const cv::Mat &dbg_img = sorted_subs_of_img[si];
            if (!dbg_img.empty()) {
              std::string win = "sorted_sub_" + std::to_string(si);
              cv::namedWindow(win, cv::WINDOW_NORMAL);
              // set window size to match image size
              cv::resizeWindow(win, dbg_img.cols, dbg_img.rows);
              cv::imshow(win, dbg_img);
            }
          }
          // small wait to allow windows to refresh; 1 ms keeps it non-blocking
          cv::waitKey(1);
        } catch (const std::exception &e) {
          INFOE("GUI display failed: %s", e.what());
        }


        text_rec_model_->Predict(sorted_subs_of_img);
        auto text_rec_model_results =
            static_cast<TextRecPredictor *>(text_rec_model_.get())
                ->PredictorResult();
        for (int m = 0; m < text_rec_model_results.size(); m++) {
          int sub_img_id = sorted_subs_info[m].first;
          sub_img_info_list[sub_img_id].second = text_rec_model_results[m];
        }
        for (int sno = 0; sno < sub_img_info_list.size(); sno++) {
          auto rec_res = sub_img_info_list[sno].second;
          if (rec_res.rec_score >= text_rec_score_thresh_) {
            results[l].rec_texts.push_back(rec_res.rec_text);
            results[l].rec_scores.push_back(rec_res.rec_score);
            results[l].rec_polys.push_back(dt_polys_list[l][sno]);
            results[l].vis_fonts = rec_res.vis_font;
          }
        }
      }
    }
    for (auto &res : results) {
      if (text_type_ == "general") {
        res.rec_boxes =
            ComponentsProcessor::ConvertPointsToBoxes(res.rec_polys);
      }
      pipeline_result_vec_.push_back(res);
      base_results.push_back(std::unique_ptr<BaseCVResult>(new OCRResult(res)));
    }
  }
  return base_results;
}

std::vector<std::unique_ptr<BaseCVResult>>
OCRPipeline::Predict(const std::vector<std::string> &input) {
  if (thread_num_ == 1) {
    return infer_->Predict(input);
  }
  
  // 简化多线程处理，直接使用单线程版本
  return infer_->Predict(input);
}

void _OCRPipeline::OverrideConfig() {
  auto &data = config_.Data();
  
  // 只处理文本检测配置
  if (params_.text_detection_model_name.has_value()) {
    auto it = config_.FindKey("TextDetection.model_name");
    if (!it.ok()) {
      data["SubModules.TextDetection.model_name"] =
          params_.text_detection_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_detection_model_name.value();
    }
  }
  if (params_.text_detection_model_dir.has_value()) {
    auto it = config_.FindKey("TextDetection.model_dir");
    if (!it.ok()) {
      data["SubModules.TextDetection.model_dir"] =
          params_.text_detection_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_detection_model_dir.value();
    }
  }
  
  // 处理文本识别配置
  if (params_.text_recognition_model_name.has_value()) {
    auto it = config_.FindKey("TextRecognition.model_name");
    if (!it.ok()) {
      data["SubModules.TextRecognition.model_name"] =
          params_.text_recognition_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_recognition_model_name.value();
    }
  }
  if (params_.text_recognition_model_dir.has_value()) {
    auto it = config_.FindKey("TextRecognition.model_dir");
    if (!it.ok()) {
      data["SubModules.TextRecognition.model_dir"] =
          params_.text_recognition_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_recognition_model_dir.value();
    }
  }
  if (params_.text_recognition_batch_size.has_value()) {
    auto it = config_.FindKey("TextRecognition.batch_size");
    if (!it.ok()) {
      data["SubModules.TextRecognition.batch_size"] =
          std::to_string(params_.text_recognition_batch_size.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_recognition_batch_size.value());
    }
  }
}


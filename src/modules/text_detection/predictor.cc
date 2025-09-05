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

#include "predictor.h"

#include "result.h"
#include "src/common/image_batch_sampler.h"

TextDetPredictor::TextDetPredictor(const TextDetPredictorParams &params)
    : BasePredictor(params.model_dir, params.model_name, params.device,
                    params.precision, params.cpu_threads,
                    params.batch_size, "image"),
      params_(params) {
  std::cout << "[DEBUG] TextDetPredictor constructor started" << std::endl;
  auto status = Build();
  if (!status.ok()) {
    std::cout << "[DEBUG] Build failed: " << status.ToString() << std::endl;
    // INFOE("Build fail: %s", status.ToString().c_str());
    exit(-1);
  }
  std::cout << "[DEBUG] TextDetPredictor constructor completed successfully" << std::endl;
};

Status TextDetPredictor::Build() {
  std::cout << "[DEBUG] TextDetPredictor::Build() started" << std::endl;
  
  const auto &pre_tfs = config_.PreProcessOpInfo();
  std::cout << "[DEBUG] Got PreProcessOpInfo, size: " << pre_tfs.size() << std::endl;
  
  // Register<ReadImage>("Read", pre_tfs.at("DecodeImage.img_mode"));
  Register<ReadImage>("Read");
  std::cout << "[DEBUG] Registered ReadImage" << std::endl;
  DetResizeForTestParam resize_param;
  resize_param.input_shape = params_.input_shape;
  resize_param.max_side_limit = params_.max_side_limit;
  resize_param.limit_side_len = params_.limit_side_len;
  resize_param.limit_type = params_.limit_type;
  resize_param.max_side_limit = params_.max_side_limit;
  std::cout << "[DEBUG] About to get DetResizeForTest.resize_long" << std::endl;
  resize_param.resize_long =
      std::stoi(pre_tfs.at("DetResizeForTest.resize_long"));
  std::cout << "[DEBUG] Got resize_long: " << (resize_param.resize_long.has_value() ? std::to_string(resize_param.resize_long.value()) : "null") << std::endl;
  Register<DetResizeForTest>("Resize", resize_param);
  std::cout << "[DEBUG] Registered DetResizeForTest" << std::endl;
  Register<NormalizeImage>("Normalize");
  std::cout << "[DEBUG] Registered NormalizeImage" << std::endl;
  Register<ToCHWImage>("ToCHW");
  std::cout << "[DEBUG] Registered ToCHWImage" << std::endl;
  Register<ToBatch>("ToBatch");
  std::cout << "[DEBUG] Registered ToBatch" << std::endl;
  std::cout << "[DEBUG] About to CreateStaticInfer" << std::endl;
  infer_ptr_ = CreateStaticInfer();
  std::cout << "[DEBUG] CreateStaticInfer completed" << std::endl;
  std::cout << "[DEBUG] About to get PostProcessOpInfo" << std::endl;
  const auto &post_params = config_.PostProcessOpInfo();
  std::cout << "[DEBUG] Got PostProcessOpInfo, size: " << post_params.size() << std::endl;
  DBPostProcessParams db_param;
  db_param.thresh = params_.thresh.has_value()
                        ? params_.thresh
                        : std::stof(post_params.at("DBPostProcess.thresh"));
  db_param.box_thresh =
      params_.box_thresh.has_value()
          ? params_.box_thresh
          : std::stof(post_params.at("DBPostProcess.box_thresh"));
  db_param.unclip_ratio =
      params_.unclip_ratio.has_value()
          ? params_.unclip_ratio
          : std::stof(post_params.at("DBPostProcess.unclip_ratio"));
  db_param.max_candidates =
      std::stoi(post_params.at("DBPostProcess.max_candidates"));
  std::cout << "[DEBUG] About to create DBPostProcess" << std::endl;
  post_op_["DBPostProcess"] =
      std::unique_ptr<DBPostProcess>(new DBPostProcess(db_param));
  std::cout << "[DEBUG] DBPostProcess created successfully" << std::endl;
  return Status::OK();
};

std::vector<std::unique_ptr<BaseCVResult>>
TextDetPredictor::Process(std::vector<cv::Mat> &batch_data) {
  std::vector<cv::Mat> origin_image = {};
  origin_image.reserve(batch_data.size());
  for (const auto &mat : batch_data) {
    origin_image.push_back(mat.clone());
  }
  auto batch_raw_imgs = pre_op_.at("Read")->Apply(batch_data);
  if (!batch_raw_imgs.ok()) {
    INFOE(batch_raw_imgs.status().ToString().c_str());
    exit(-1);
  }
  std::vector<int> origin_shape = {batch_raw_imgs.value()[0].rows,
                                   batch_raw_imgs.value()[0].cols};

  auto batch_imgs = pre_op_.at("Resize")->Apply(batch_raw_imgs.value());
  if (!batch_imgs.ok()) {
    INFOE(batch_imgs.status().ToString().c_str());
    exit(-1);
  }
  auto batch_imgs_normalize =
      pre_op_.at("Normalize")->Apply(batch_imgs.value());
  if (!batch_imgs_normalize.ok()) {
    INFOE(batch_imgs_normalize.status().ToString().c_str());
    exit(-1);
  }

  auto batch_imgs_to_chw =
      pre_op_.at("ToCHW")->Apply(batch_imgs_normalize.value());
  if (!batch_imgs_to_chw.ok()) {
    INFOE(batch_imgs_to_chw.status().ToString().c_str());
    exit(-1);
  }
  auto batch_imgs_to_batch =
      pre_op_.at("ToBatch")->Apply(batch_imgs_to_chw.value());
  if (!batch_imgs_to_batch.ok()) {
    INFOE(batch_imgs_to_batch.status().ToString().c_str());
    exit(-1);
  }
  auto infer_result = infer_ptr_->Apply(batch_imgs_to_batch.value());
  if (!infer_result.ok()) {
    INFOE(infer_result.status().ToString().c_str());
    exit(-1);
  }
  auto db_result = post_op_.at("DBPostProcess")
                       ->Apply(infer_result.value()[0], origin_shape);

  std::cout << "[DEBUG] DBPostProcess Apply returned" << std::endl;
  if (!db_result.ok()) {
    std::cout << "[DEBUG] DBPostProcess failed: " << db_result.status().ToString() << std::endl;
    INFOE(db_result.status().ToString().c_str());
    exit(-1);
  }

  std::cout << "[DEBUG] DBPostProcess success, result size: " << db_result.value().size() << std::endl;
  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (int i = 0; i < db_result.value().size(); i++, input_index_++) {
    std::cout << "[DEBUG] Processing result " << i << ", polys: " << db_result.value()[i].first.size() 
              << ", scores: " << db_result.value()[i].second.size() << std::endl;
    TextDetPredictorResult predictor_result;
    if (!input_path_.empty()) {
      if (input_index_ == input_path_.size())
        input_index_ = 0;
      predictor_result.input_path = input_path_[input_index_];
    }
    predictor_result.input_image = origin_image[i];
    predictor_result.dt_polys = db_result.value()[i].first;
    predictor_result.dt_scores = db_result.value()[i].second;
    predictor_result_vec_.push_back(predictor_result);
    base_cv_result_ptr_vec.push_back(
        std::unique_ptr<BaseCVResult>(new TextDetResult(predictor_result)));
  }

  std::cout << "[DEBUG] TextDetPredictor::Process completed successfully with " << base_cv_result_ptr_vec.size() << " results" << std::endl;
  return base_cv_result_ptr_vec;
}


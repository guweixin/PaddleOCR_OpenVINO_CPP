
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

#include <iostream>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "src/utils/status.h"
#include "polyclipping/clipper.hpp"
#include "src/utils/func_register.h"

struct DetResizeForTestParam {
  std::optional<std::vector<int>> input_shape = std::nullopt;
  std::optional<int> max_side_limit = std::nullopt;
  std::optional<std::vector<int>> image_shape = std::nullopt;
  std::optional<bool> keep_ratio = std::nullopt;
  std::optional<int> limit_side_len = std::nullopt;
  std::optional<std::string> limit_type = std::nullopt;
  std::optional<int> resize_long = std::nullopt;
};

class DetResizeForTest : public BaseProcessor {
public:
  DetResizeForTest(const DetResizeForTestParam &params);
  StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param_ptr = nullptr) const override;

private:
  int resize_type_ = 0;
  bool keep_ratio_ = false;
  int resize_long_;
  std::vector<int> input_shape_;
  std::vector<int> image_shape_;
  int limit_side_len_;
  std::string limit_type_;
  int max_side_limit_ = 4000;

  StatusOr<cv::Mat> Resize(const cv::Mat &img, int limit_side_len,
                                 const std::string &limit_type,
                                 int max_side_limit) const;

  cv::Mat ImagePadding(const cv::Mat &img, int value = 0) const;

  StatusOr<cv::Mat> ResizeImageType0(const cv::Mat &img,
                                           int limit_side_len,
                                           const std::string &limit_type,
                                           int max_side_limit) const;
  StatusOr<cv::Mat> ResizeImageType1(const cv::Mat &img) const;
  StatusOr<cv::Mat> ResizeImageType2(const cv::Mat &img) const;
  StatusOr<cv::Mat> ResizeImageType3(const cv::Mat &img) const;
  static constexpr int INPUTSHAPE = 3;
};

struct DBPostProcessParams {
  std::optional<float> thresh = std::nullopt;
  std::optional<float> box_thresh = std::nullopt;
  std::optional<float> unclip_ratio = std::nullopt;
  int max_candidates = 1000;
  bool use_dilation = false;
  std::string score_mode = "fast";
  std::string box_type = "quad";
};

class DBPostProcess {
public:
  DBPostProcess(const DBPostProcessParams &params);

  StatusOr<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  operator()(const cv::Mat &preds, const std::vector<int> &img_shapes,
             std::optional<float> thresh = std::nullopt,
             std::optional<float> box_thresh = std::nullopt,
             std::optional<float> unclip_ratio = std::nullopt);
  StatusOr<std::vector<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>>
  Apply(const cv::Mat &preds, const std::vector<int> &img_shapes,
        std::optional<float> thresh = std::nullopt,
        std::optional<float> box_thresh = std::nullopt,
        std::optional<float> unclip_ratio = std::nullopt);

private:
  StatusOr<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  Process(const cv::Mat &pred, const std::vector<int> &img_shape, float thresh,
          float box_thresh, float unclip_ratio);

  StatusOr<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  PolygonsFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap, int dest_width,
                     int dest_height, float box_thresh, float unclip_ratio);

  StatusOr<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap, int dest_width,
                  int dest_height, float box_thresh, float unclip_ratio);

  StatusOr<std::vector<cv::Point2f>>
  Unclip(const std::vector<cv::Point2f> &box, float unclip_ratio);

  std::pair<std::vector<cv::Point2f>, float>
  GetMiniBoxes(const std::vector<cv::Point2f> &contour);

  float BoxScoreFast(const cv::Mat &bitmap,
                     const std::vector<cv::Point2f> &contour);

  float BoxScoreSlow(const cv::Mat &bitmap,
                     const std::vector<cv::Point2f> &contour);

private:
  float thresh_;
  float box_thresh_;
  int max_candidates_;
  float unclip_ratio_;
  int min_size_;
  bool use_dilation_;
  std::string score_mode_;
  std::string box_type_;
};


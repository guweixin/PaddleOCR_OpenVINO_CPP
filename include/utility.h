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

#pragma once

#include <opencv2/imgproc.hpp>

namespace PaddleOCR
{

  struct OCRPredictResult
  {
    std::vector<std::vector<int>> box;
    std::string text;
    float score = -1.0;
    float cls_score;
    int cls_label = -1;
  };

  class Utility
  {
  public:
    static std::vector<std::string> ReadDict(const std::string &path) noexcept;

    template <class ForwardIterator>
    inline static size_t argmax(ForwardIterator first,
                                ForwardIterator last) noexcept
    {
      return std::distance(first, std::max_element(first, last));
    }

    static cv::Mat
    GetRotateCropImage(const cv::Mat &srcimage,
                       const std::vector<std::vector<int>> &box) noexcept;

    static std::vector<size_t> argsort(const std::vector<float> &array) noexcept;

    static bool PathExists(const char *path) noexcept;
    static inline bool PathExists(const std::string &path) noexcept
    {
      return PathExists(path.c_str());
    }

    static void CreateDir(const char *path) noexcept;
    static inline void CreateDir(const std::string &path) noexcept
    {
      CreateDir(path.c_str());
    }

    static void sort_boxes(std::vector<OCRPredictResult> &ocr_result) noexcept;

  private:
    static bool comparison_box(const OCRPredictResult &result1,
                               const OCRPredictResult &result2) noexcept
    {
      if (result1.box[0][1] < result2.box[0][1])
      {
        return true;
      }
      else if (result1.box[0][1] == result2.box[0][1])
      {
        return result1.box[0][0] < result2.box[0][0];
      }
      else
      {
        return false;
      }
    }
  };

} // namespace PaddleOCR

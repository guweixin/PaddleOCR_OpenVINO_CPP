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

#include <include/utility.h>

namespace PaddleOCR
{

  class DBPostProcessor
  {
  public:
    void GetContourArea(const std::vector<std::vector<float>> &box,
                        float unclip_ratio, float &distance) noexcept;

    cv::RotatedRect UnClip(const std::vector<std::vector<float>> &box,
                           const float &unclip_ratio) noexcept;

    float **Mat2Vec(const cv::Mat &mat) noexcept;

    std::vector<std::vector<int>>
    OrderPointsClockwise(const std::vector<std::vector<int>> &pts) noexcept;

    std::vector<std::vector<float>> GetMiniBoxes(const cv::RotatedRect &box,
                                                 float &ssid) noexcept;

    float BoxScoreFast(const std::vector<std::vector<float>> &box_array,
                       const cv::Mat &pred) noexcept;
    float PolygonScoreAcc(const std::vector<cv::Point> &contour,
                          const cv::Mat &pred) noexcept;

    std::vector<std::vector<std::vector<int>>>
    BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                    int dest_width, int dest_height) noexcept;

    void FilterTagDetRes(std::vector<std::vector<std::vector<int>>> &boxes,
                         float ratio_h, float ratio_w,
                         const cv::Mat &srcimg) noexcept;

  private:
    static bool XsortInt(const std::vector<int> &a,
                         const std::vector<int> &b) noexcept;

    static bool XsortFp32(const std::vector<float> &a,
                          const std::vector<float> &b) noexcept;

    std::vector<std::vector<float>> Mat2Vector(const cv::Mat &mat) noexcept;

    inline int _max(int a, int b) const noexcept { return a >= b ? a : b; }

    inline int _min(int a, int b) const noexcept { return a >= b ? b : a; }

    template <class T>
    inline T clamp(T x, T min, T max) const noexcept
    {
      if (x > max)
        return max;
      if (x < min)
        return min;
      return x;
    }

    inline float clampf(float x, float min, float max) const noexcept
    {
      if (x > max)
        return max;
      if (x < min)
        return min;
      return x;
    }
  };

} // namespace PaddleOCR

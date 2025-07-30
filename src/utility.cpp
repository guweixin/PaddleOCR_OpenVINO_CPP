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

#include <dirent.h>
#include <include/utility.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <vector>

#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace PaddleOCR
{

  std::vector<std::string> Utility::ReadDict(const std::string &path) noexcept
  {
    std::vector<std::string> m_vec;
    std::ifstream in(path);
    if (in)
    {
      for (;;)
      {
        std::string line;
        if (!getline(in, line))
          break;
        m_vec.emplace_back(std::move(line));
      }
    }
    else
    {
      std::cout << "no such label file: " << path << ", exit the program..."
                << std::endl;
      exit(1);
    }

    // Note: Space character is added in ocr_rec.h after blank char
    // No need to add space here to avoid duplication

    return m_vec;
  }

  cv::Mat
  Utility::GetRotateCropImage(const cv::Mat &srcimage,
                              const std::vector<std::vector<int>> &box) noexcept
  {
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (size_t i = 0; i < points.size(); ++i)
    {
      points[i][0] -= left;
      points[i][1] -= top;
    }

    // int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
    //                               pow(points[0][1] - points[1][1], 2)));

    // int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
    //                                pow(points[0][1] - points[3][1], 2)));

    int img_crop_width = static_cast<int>(std::max(
        sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)),
        sqrt(pow(points[2][0] - points[3][0], 2) + pow(points[2][1] - points[3][1], 2))));
    int img_crop_height = static_cast<int>(std::max(
        sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)),
        sqrt(pow(points[1][0] - points[2][0], 2) + pow(points[1][1] - points[2][1], 2))));

    const cv::Point2f pts_std[4] = {
        {0., 0.},
        {(float)img_crop_width, 0.},
        {(float)img_crop_width, (float)img_crop_height},
        {0.f, (float)img_crop_height}};

    const cv::Point2f pointsf[4] = {{(float)points[0][0], (float)points[0][1]},
                                    {(float)points[1][0], (float)points[1][1]},
                                    {(float)points[2][0], (float)points[2][1]},
                                    {(float)points[3][0], (float)points[3][1]}};
    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE, cv::INTER_CUBIC);
    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5)
    {
      cv::Mat srcCopy(dst_img.rows, dst_img.cols, dst_img.depth());
      cv::transpose(dst_img, srcCopy);
      cv::flip(srcCopy, srcCopy, 0);
      return srcCopy;
    }
    else
    {
      return dst_img;
    }
  }

  std::vector<size_t> Utility::argsort(const std::vector<float> &array) noexcept
  {
    std::vector<size_t> array_index(array.size(), 0);
    for (size_t i = 0; i < array.size(); ++i)
      array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](size_t pos1, size_t pos2) noexcept
              {
                return (array[pos1] < array[pos2]);
              });

    return array_index;
  }

  bool Utility::PathExists(const char *path) noexcept
  {
#ifdef _WIN32
    struct _stat buffer;
    return (_stat(path, &buffer) == 0);
#else
    struct stat buffer;
    return (stat(path, &buffer) == 0);
#endif // !_WIN32
  }

  void Utility::CreateDir(const char *path) noexcept
  {
#ifdef _MSC_VER
    _mkdir(path);
#elif defined __MINGW32__
    mkdir(path);
#else
    mkdir(path, 0777);
#endif // !_WIN32
  }

  void Utility::sort_boxes(std::vector<OCRPredictResult> &ocr_result) noexcept
  {
    std::sort(ocr_result.begin(), ocr_result.end(), Utility::comparison_box);
    if (ocr_result.size() > 1)
    {
      for (size_t i = 0; i < ocr_result.size() - 1; ++i)
      {
        for (size_t j = i; j != size_t(-1); --j)
        {
          if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
              (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0]))
          {
            std::swap(ocr_result[i], ocr_result[i + 1]);
          }
        }
      }
    }
  }

} // namespace PaddleOCR

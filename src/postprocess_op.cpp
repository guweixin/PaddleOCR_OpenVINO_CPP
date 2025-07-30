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

#include <include/clipper.h>
#include <include/postprocess_op.h>

namespace PaddleOCR
{

  void DBPostProcessor::GetContourArea(const std::vector<std::vector<float>> &box,
                                       float unclip_ratio,
                                       float &distance) noexcept
  {
    int pts_num = 4;
    float area = 0.0f;
    float dist = 0.0f;
    for (int i = 0; i < pts_num; ++i)
    {
      area += box[i][0] * box[(i + 1) % pts_num][1] -
              box[i][1] * box[(i + 1) % pts_num][0];
      dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                        (box[i][0] - box[(i + 1) % pts_num][0]) +
                    (box[i][1] - box[(i + 1) % pts_num][1]) *
                        (box[i][1] - box[(i + 1) % pts_num][1]));
    }
    area = fabs(float(area / 2.0));

    distance = area * unclip_ratio / dist;
  }

  cv::RotatedRect
  DBPostProcessor::UnClip(const std::vector<std::vector<float>> &box,
                          const float &unclip_ratio) noexcept
  {
    float distance = 1.0;

    GetContourArea(box, unclip_ratio, distance);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p.emplace_back(int(box[0][0]), int(box[0][1]));
    p.emplace_back(int(box[1][0]), int(box[1][1]));
    p.emplace_back(int(box[2][0]), int(box[2][1]));
    p.emplace_back(int(box[3][0]), int(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    if (!offset.Execute(soln, distance))
      return cv::RotatedRect();

    std::vector<cv::Point2f> points;

    for (size_t j = 0; j < soln.size(); ++j)
    {
      for (size_t i = 0; i < soln[soln.size() - 1].size(); ++i)
      {
        points.emplace_back(soln[j][i].X, soln[j][i].Y);
      }
    }
    cv::RotatedRect res;
    if (points.size() <= 0)
    {
      res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    }
    else
    {
      res = cv::minAreaRect(points);
    }
    return res;
  }

  float **DBPostProcessor::Mat2Vec(const cv::Mat &mat) noexcept
  {
    auto **array = new float *[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
      array[i] = new float[mat.cols];
    for (int i = 0; i < mat.rows; ++i)
    {
      for (int j = 0; j < mat.cols; ++j)
      {
        array[i][j] = mat.at<float>(i, j);
      }
    }

    return array;
  }

  std::vector<std::vector<int>> DBPostProcessor::OrderPointsClockwise(
      const std::vector<std::vector<int>> &pts) noexcept
  {
    std::vector<std::vector<int>> box = pts;
    std::sort(box.begin(), box.end(), XsortInt);

    std::vector<std::vector<int>> leftmost = {box[0], box[1]};
    std::vector<std::vector<int>> rightmost = {box[2], box[3]};

    if (leftmost[0][1] > leftmost[1][1])
      std::swap(leftmost[0], leftmost[1]);

    if (rightmost[0][1] > rightmost[1][1])
      std::swap(rightmost[0], rightmost[1]);

    std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                          leftmost[1]};
    return rect;
  }

  std::vector<std::vector<float>>
  DBPostProcessor::Mat2Vector(const cv::Mat &mat) noexcept
  {
    std::vector<std::vector<float>> img_vec;

    for (int i = 0; i < mat.rows; ++i)
    {
      std::vector<float> tmp;
      for (int j = 0; j < mat.cols; ++j)
      {
        tmp.emplace_back(mat.at<float>(i, j));
      }
      img_vec.emplace_back(std::move(tmp));
    }
    return img_vec;
  }

  bool DBPostProcessor::XsortFp32(const std::vector<float> &a,
                                  const std::vector<float> &b) noexcept
  {
    if (a[0] != b[0])
      return a[0] < b[0];
    return false;
  }

  bool DBPostProcessor::XsortInt(const std::vector<int> &a,
                                 const std::vector<int> &b) noexcept
  {
    if (a[0] != b[0])
      return a[0] < b[0];
    return false;
  }

  std::vector<std::vector<float>>
  DBPostProcessor::GetMiniBoxes(const cv::RotatedRect &box,
                                float &ssid) noexcept
  {
    ssid = std::max(box.size.width, box.size.height);

    cv::Mat points;
    cv::boxPoints(box, points);

    auto array = Mat2Vector(points);
    std::sort(array.begin(), array.end(), XsortFp32);

    std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                       idx4 = array[3];
    if (array[3][1] <= array[2][1])
    {
      idx2 = array[3];
      idx3 = array[2];
    }
    else
    {
      idx2 = array[2];
      idx3 = array[3];
    }
    if (array[1][1] <= array[0][1])
    {
      idx1 = array[1];
      idx4 = array[0];
    }
    else
    {
      idx1 = array[0];
      idx4 = array[1];
    }

    array[0] = idx1;
    array[1] = idx2;
    array[2] = idx3;
    array[3] = idx4;

    return array;
  }

  float DBPostProcessor::PolygonScoreAcc(const std::vector<cv::Point> &contour,
                                         const cv::Mat &pred) noexcept
  {
    int width = pred.cols;
    int height = pred.rows;
    std::vector<float> box_x;
    std::vector<float> box_y;
    for (size_t i = 0; i < contour.size(); ++i)
    {
      box_x.emplace_back(contour[i].x);
      box_y.emplace_back(contour[i].y);
    }

    int xmin =
        clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
              width - 1);
    int xmax =
        clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
              width - 1);
    int ymin =
        clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
              height - 1);
    int ymax =
        clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
              height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point *rook_point = new cv::Point[contour.size()];

    for (size_t i = 0; i < contour.size(); ++i)
    {
      rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
    }
    const cv::Point *ppt[1] = {rook_point};
    int npt[] = {int(contour.size())};

    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
        .copyTo(croppedImg);
    float score = cv::mean(croppedImg, mask)[0];

    delete[] rook_point;
    return score;
  }

  float DBPostProcessor::BoxScoreFast(
      const std::vector<std::vector<float>> &box_array,
      const cv::Mat &pred) noexcept
  {
    const auto &array = box_array;
    int width = pred.cols;
    int height = pred.rows;

    float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
    float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

    int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
                     width - 1);
    int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
                     width - 1);
    int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
                     height - 1);
    int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
                     height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point root_point[4];
    root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
    root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
    root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
    root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
    const cv::Point *ppt[1] = {root_point};
    int npt[] = {4};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
        .copyTo(croppedImg);

    auto score = cv::mean(croppedImg, mask)[0];
    return score;
  }

  std::vector<std::vector<std::vector<int>>> DBPostProcessor::BoxesFromBitmap(
      const cv::Mat &pred, const cv::Mat &bitmap,
      int dest_width, int dest_height) noexcept
  {

    const int min_size = 3;
    const int max_candidates = 1000;
    const float box_thresh = 0.7f;          // Fixed threshold like Python version
    const float det_db_unclip_ratio = 2.0f; // Fixed unclip ratio like Python version

    int height = bitmap.rows; // Should be 544 for typical models
    int width = bitmap.cols;  // Should be 544 for typical models

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // Ensure bitmap is binary - Python uses (bitmap * 255).astype(np.uint8)
    cv::Mat bitmap_binary;
    if (bitmap.type() == CV_8UC1)
    {
      // If already uint8, multiply by 255 to ensure proper scaling like Python
      bitmap.convertTo(bitmap_binary, CV_8UC1, 255.0);
    }
    else
    {
      // Convert boolean/float bitmap to binary uint8
      bitmap.convertTo(bitmap_binary, CV_8UC1, 255.0);
    }

    cv::findContours(bitmap_binary, contours, hierarchy, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    int num_contours = std::min(static_cast<int>(contours.size()), max_candidates);

    std::vector<std::vector<std::vector<int>>> boxes;

    for (int index = 0; index < num_contours; ++index)
    {
      if (contours[index].size() <= 2)
      {
        continue;
      }

      float sside;
      cv::RotatedRect box = cv::minAreaRect(contours[index]);
      auto points = GetMiniBoxes(box, sside);

      if (sside < min_size)
      {
        continue;
      }

      // Calculate score using fast method (like Python box_score_fast)
      float score = BoxScoreFast(points, pred);

      if (score < box_thresh)
      {
        continue;
      }

      // Unclip the box (like Python unclip function)
      cv::RotatedRect unclipped_box = UnClip(points, det_db_unclip_ratio);
      if (unclipped_box.size.height < 1.001 && unclipped_box.size.width < 1.001)
      {
        continue;
      }

      // Get mini boxes from unclipped box
      auto cliparray = GetMiniBoxes(unclipped_box, sside);

      if (sside < min_size + 2)
      {
        continue;
      }

      // Convert to final coordinates exactly like Python version:
      // box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
      // box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
      std::vector<std::vector<int>> intcliparray;
      for (int num_pt = 0; num_pt < 4; ++num_pt)
      {
        // Python logic: box[:, 0] / width * dest_width
        float x_scaled = cliparray[num_pt][0] / static_cast<float>(width) * static_cast<float>(dest_width);
        float y_scaled = cliparray[num_pt][1] / static_cast<float>(height) * static_cast<float>(dest_height);

        // Python logic: np.clip(np.round(...), 0, dest_width)
        int x = static_cast<int>(std::round(x_scaled));
        int y = static_cast<int>(std::round(y_scaled));

        // Python uses np.clip(..., 0, dest_width) not dest_width-1
        x = std::max(0, std::min(x, dest_width));
        y = std::max(0, std::min(y, dest_height));

        std::vector<int> point{x, y};
        intcliparray.emplace_back(std::move(point));
      }
      boxes.emplace_back(std::move(intcliparray));
    }

    return boxes;
  }

  void DBPostProcessor::FilterTagDetRes(
      std::vector<std::vector<std::vector<int>>> &boxes, float ratio_h,
      float ratio_w, const cv::Mat &srcimg) noexcept
  {
    int oriimg_h = srcimg.rows;
    int oriimg_w = srcimg.cols;

    std::vector<std::vector<std::vector<int>>> root_points;
    for (size_t n = 0; n < boxes.size(); ++n)
    {
      boxes[n] = OrderPointsClockwise(boxes[n]);
      for (size_t m = 0; m < boxes[0].size(); ++m)
      {
        // NOTE: BoxesFromBitmap already scaled coordinates to original image size
        // No need to divide by ratio again
        // boxes[n][m][0] /= ratio_w;
        // boxes[n][m][1] /= ratio_h;

        boxes[n][m][0] = int(std::min(std::max(boxes[n][m][0], 0), oriimg_w - 1));
        boxes[n][m][1] = int(std::min(std::max(boxes[n][m][1], 0), oriimg_h - 1));
      }
    }

    for (size_t n = 0; n < boxes.size(); ++n)
    {
      int rect_width, rect_height;
      rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                            pow(boxes[n][0][1] - boxes[n][1][1], 2)));
      rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                             pow(boxes[n][0][1] - boxes[n][3][1], 2)));
      if (rect_width <= 4 || rect_height <= 4)
        continue;
      root_points.emplace_back(boxes[n]);
    }
    boxes = std::move(root_points);
  }

} // namespace PaddleOCR

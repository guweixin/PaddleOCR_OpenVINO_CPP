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

#include <include/preprocess_op.h>

namespace PaddleOCR
{

  void Permute::Run(const cv::Mat &im, float *data) noexcept
  {
    int rh = im.rows;
    int rw = im.cols;
    int rc = im.channels();
    for (int i = 0; i < rc; ++i)
    {
      cv::extractChannel(im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
    }
  }

  void PermuteBatch::Run(const std::vector<cv::Mat> &imgs, float *data) noexcept
  {
    for (size_t j = 0; j < imgs.size(); ++j)
    {
      int rh = imgs[j].rows;
      int rw = imgs[j].cols;
      int rc = imgs[j].channels();
      for (int i = 0; i < rc; ++i)
      {
        cv::extractChannel(
            imgs[j], cv::Mat(rh, rw, CV_32FC1, data + (j * rc + i) * rh * rw), i);
      }
    }
  }

  void Normalize::Run(cv::Mat &im, const std::vector<float> &mean,
                      const std::vector<float> &scale,
                      const bool is_scale) noexcept
  {
    double e = 1.0;
    if (is_scale)
    {
      e /= 255.0;
    }
    im.convertTo(im, CV_32FC3, e);
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(im, bgr_channels);
    for (size_t i = 0; i < bgr_channels.size(); ++i)
    {
      bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i],
                                (0.0 - mean[i]) * scale[i]);
    }
    cv::merge(bgr_channels, im);
  }

  void ResizeImgType0::Run(const cv::Mat &img, cv::Mat &resize_img,
                           const std::string &limit_type, int limit_side_len,
                           float &ratio_h, float &ratio_w,
                           bool use_tensorrt) noexcept
  {
    int w = img.cols;
    int h = img.rows;

    cv::Mat input_img;

    // Python version logic: check for small images and apply padding if needed
    if ((h + w) < 64)
    {
      ImagePadding padding_op;
      padding_op.Run(img, input_img, 0);
      h = input_img.rows;
      w = input_img.cols;
    }
    else
    {
      input_img = img.clone();
    }

    float ratio = 1.f;

    // Python version logic: limit the max side to limit_side_len (default 960)
    int max_wh = std::max(h, w);
    if (max_wh > limit_side_len)
    {
      if (h > w)
      {
        ratio = float(limit_side_len) / float(h);
      }
      else
      {
        ratio = float(limit_side_len) / float(w);
      }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);

    // Ensure dimensions are multiples of 32 and at least 32 (Python version logic)
    resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
    resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

    cv::resize(input_img, resize_img, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(img.rows); // Use original image dimensions for ratio
    ratio_w = float(resize_w) / float(img.cols); // Use original image dimensions for ratio
  }

  void CrnnResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                          bool use_tensorrt,
                          const std::vector<int> &rec_image_shape) noexcept
  {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    // Python version logic: dynamically calculate imgW based on max_wh_ratio
    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
      resize_w = imgW;
    else
      resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                       int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                       {255, 255, 255});
  }

  void ClsResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                         bool use_tensorrt,
                         const std::vector<int> &rec_image_shape) noexcept
  {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
      resize_w = imgW;
    else
      resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
  }

  void TableResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                           const int max_len) noexcept
  {
    int w = img.cols;
    int h = img.rows;

    int max_wh = w >= h ? w : h;
    float ratio = w >= h ? float(max_len) / float(w) : float(max_len) / float(h);

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  }

  void TablePadImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                        const int max_len) noexcept
  {
    int w = img.cols;
    int h = img.rows;
    cv::copyMakeBorder(img, resize_img, 0, max_len - h, 0, max_len - w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  }

  void Resize::Run(const cv::Mat &img, cv::Mat &resize_img, const int h,
                   const int w) noexcept
  {
    cv::resize(img, resize_img, cv::Size(w, h));
  }

  void ImagePadding::Run(const cv::Mat &img, cv::Mat &padded_img, int value) noexcept
  {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    // Create padded image with at least 32x32 dimensions (Python version logic)
    int pad_h = std::max(32, h);
    int pad_w = std::max(32, w);

    // Create padded image filled with the specified value
    padded_img = cv::Mat(pad_h, pad_w, img.type(), cv::Scalar(value, value, value));

    // Copy original image to the top-left corner of the padded image
    cv::Rect roi(0, 0, w, h);
    img.copyTo(padded_img(roi));
  }

} // namespace PaddleOCR

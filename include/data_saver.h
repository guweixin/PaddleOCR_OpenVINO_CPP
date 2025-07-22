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

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

class DataSaver
{
public:
    // Save float data as npy format
    static void SaveFloatArrayAsNpy(const std::vector<float> &data,
                                    const std::vector<size_t> &shape,
                                    const std::string &filename);

    // Save cv::Mat as npy format
    static void SaveMatAsNpy(const cv::Mat &mat, const std::string &filename);

    // Save detection input data
    static void SaveDetectionData(const std::vector<float> &input_data,
                                  const std::vector<size_t> &input_shape,
                                  const std::vector<float> &output_data,
                                  const std::vector<size_t> &output_shape,
                                  int image_idx);

    // Save recognition input data
    static void SaveRecognitionData(const std::vector<float> &input_data,
                                    const std::vector<size_t> &input_shape,
                                    const std::vector<float> &output_data,
                                    const std::vector<size_t> &output_shape,
                                    int batch_idx, int image_idx);

    // Save detection text boxes coordinates
    static void SaveDetectionBoxes(const std::vector<std::vector<std::vector<int>>> &boxes, 
                                   int image_idx);

    // Create directory if not exists
    static void CreateDirectoryIfNotExists(const std::string &dir_path);

private:
    static void WriteNpyHeader(std::ofstream &file, const std::vector<size_t> &shape, const std::string &dtype);
};

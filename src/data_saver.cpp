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

#include <include/data_saver.h>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

void DataSaver::WriteNpyHeader(std::ofstream &file, const std::vector<size_t> &shape, const std::string &dtype)
{
    // NPY format magic number
    file.write("\x93NUMPY", 6);

    // Version
    file.put(1); // major version
    file.put(0); // minor version

    // Create header string
    std::ostringstream header_stream;
    header_stream << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";

    for (size_t i = 0; i < shape.size(); ++i)
    {
        header_stream << shape[i];
        if (i < shape.size() - 1)
        {
            header_stream << ", ";
        }
        else if (shape.size() == 1)
        {
            header_stream << ","; // Add comma for 1D arrays
        }
    }
    header_stream << "), }";

    std::string header = header_stream.str();

    // Pad header to be multiple of 16 bytes (including length bytes)
    size_t header_len = header.size();
    size_t total_header_len = header_len + 2; // +2 for length bytes
    size_t padding = (16 - (total_header_len % 16)) % 16;

    for (size_t i = 0; i < padding; ++i)
    {
        header += " ";
    }
    header += "\n";

    // Write header length (little endian)
    uint16_t len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char *>(&len), 2);

    // Write header
    file.write(header.c_str(), header.size());
}

void DataSaver::CreateDirectoryIfNotExists(const std::string &dir_path)
{
#ifdef _WIN32
    // Windows
    std::string command = "if not exist \"" + dir_path + "\" mkdir \"" + dir_path + "\"";
    system(command.c_str());
#else
    // Unix/Linux
    mkdir(dir_path.c_str(), 0755);
#endif
}

void DataSaver::SaveFloatArrayAsNpy(const std::vector<float> &data,
                                    const std::vector<size_t> &shape,
                                    const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }

    WriteNpyHeader(file, shape, "<f4"); // float32 little endian

    // Write data
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    file.close();

    std::cout << "Saved " << filename << " with shape: ";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << "x";
    }
    std::cout << std::endl;
}

void DataSaver::SaveMatAsNpy(const cv::Mat &mat, const std::string &filename)
{
    std::vector<size_t> shape;
    if (mat.channels() == 1)
    {
        shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)};
    }
    else
    {
        shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols), static_cast<size_t>(mat.channels())};
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }

    if (mat.type() == CV_32F)
    {
        WriteNpyHeader(file, shape, "<f4");
        file.write(reinterpret_cast<const char *>(mat.data), mat.total() * mat.elemSize());
    }
    else if (mat.type() == CV_8UC3 || mat.type() == CV_8UC1)
    {
        WriteNpyHeader(file, shape, "|u1"); // uint8
        file.write(reinterpret_cast<const char *>(mat.data), mat.total() * mat.elemSize());
    }

    file.close();
    std::cout << "Saved Mat " << filename << " with shape: ";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << "x";
    }
    std::cout << std::endl;
}

void DataSaver::SaveDetectionData(const std::vector<float> &input_data,
                                  const std::vector<size_t> &input_shape,
                                  const std::vector<float> &output_data,
                                  const std::vector<size_t> &output_shape,
                                  int image_idx)
{
    // Create directory if not exists
    CreateDirectoryIfNotExists("../../debug_data");
    // Save input
    std::string input_filename = "../../debug_data/cpp_det_input_" + std::to_string(image_idx) + ".npy";
    SaveFloatArrayAsNpy(input_data, input_shape, input_filename);

    // Save output
    std::string output_filename = "../../debug_data/cpp_det_output_" + std::to_string(image_idx) + ".npy";
    SaveFloatArrayAsNpy(output_data, output_shape, output_filename);
}

void DataSaver::SaveRecognitionData(const std::vector<float> &input_data,
                                    const std::vector<size_t> &input_shape,
                                    const std::vector<float> &output_data,
                                    const std::vector<size_t> &output_shape,
                                    int batch_idx, int image_idx)
{
    // Create directory if not exists
    CreateDirectoryIfNotExists("../../debug_data");
    CreateDirectoryIfNotExists("./debug_data/recognition");

    // Save input
    std::string input_filename = "../../debug_data/cpp_rec_input_batch" +
                                 std::to_string(batch_idx) + "_img" + std::to_string(image_idx) + ".npy";
    SaveFloatArrayAsNpy(input_data, input_shape, input_filename);

    // Save output
    std::string output_filename = "../../debug_data/cpp_rec_output_batch" +
                                  std::to_string(batch_idx) + "_img" + std::to_string(image_idx) + ".npy";
    SaveFloatArrayAsNpy(output_data, output_shape, output_filename);
}

void DataSaver::SaveImageWithTextBoxes(const cv::Mat &original_image,
                                       const std::vector<std::vector<std::vector<int>>> &boxes,
                                       int image_idx)
{
    try
    {
        // Create debug directory if not exists
        CreateDirectoryIfNotExists("../../debug_data");

        // Create a copy of the original image to draw on
        cv::Mat image_with_boxes = original_image.clone();

        // Define colors for different boxes (BGR format)
        std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 255, 0),   // Green
            cv::Scalar(255, 0, 0),   // Blue
            cv::Scalar(0, 0, 255),   // Red
            cv::Scalar(255, 255, 0), // Cyan
            cv::Scalar(255, 0, 255), // Magenta
            cv::Scalar(0, 255, 255)  // Yellow
        };

        // Draw each text box
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const auto &box = boxes[i];
            if (box.size() != 4) continue; // Skip invalid boxes

            // Convert box coordinates to cv::Point
            std::vector<cv::Point> points;
            for (const auto &point : box)
            {
                if (point.size() >= 2)
                {
                    points.emplace_back(point[0], point[1]);
                }
            }

            if (points.size() == 4)
            {
                // Choose color based on box index
                cv::Scalar color = colors[i % colors.size()];

                // Draw the box as a filled polygon with transparency
                std::vector<cv::Point> contour = points;
                cv::Mat overlay = image_with_boxes.clone();
                cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{contour}, color);
                cv::addWeighted(image_with_boxes, 0.7, overlay, 0.3, 0, image_with_boxes);

                // Draw the box outline
                cv::polylines(image_with_boxes, std::vector<std::vector<cv::Point>>{contour}, 
                              true, color, 3, cv::LINE_AA);

                // Draw corner points
                for (const auto &point : points)
                {
                    cv::circle(image_with_boxes, point, 5, color, -1);
                }

                // Add box index text
                cv::putText(image_with_boxes, "Box" + std::to_string(i), 
                           cv::Point(points[0].x, points[0].y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
        }

        // Save the image with text boxes
        std::string output_filename = "../../debug_data/cpp_image_with_boxes_" + 
                                      std::to_string(image_idx) + ".png";
        
        bool success = cv::imwrite(output_filename, image_with_boxes);
        if (success)
        {
            std::cout << "[DEBUG] Saved image with text boxes: " << output_filename << std::endl;
        }
        else
        {
            std::cerr << "[ERROR] Failed to save image with text boxes: " << output_filename << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Exception in SaveImageWithTextBoxes: " << e.what() << std::endl;
    }
}

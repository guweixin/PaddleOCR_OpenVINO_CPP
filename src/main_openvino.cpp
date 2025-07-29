// OpenVINO-based OCR Main Program
// Implements the same functionality as Paddle version but using OpenVINO

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>

#include "include/ocr_det_openvino.h"
#include "include/ocr_rec_openvino.h"
#include "include/utility.h"
#include <openvino/openvino.hpp>

using namespace PaddleOCR;

// Command line parameters (compatible with original PaddleOCR format)
struct OCRConfig
{
    std::string image_dir = "";
    std::string output_dir = "";
    std::string det_model_dir = "";
    std::string rec_model_dir = "";
    std::string rec_char_dict_path = "";
    std::string inference_device = "GPU"; // CPU, GPU, NPU
    std::string device = "GPU"; // Alias for inference_device

    bool det = true;
    bool rec = true;

    // Detection parameters
    std::string limit_type = "max";
    int limit_side_len = 960;
    double det_db_thresh = 0.3;
    double det_db_box_thresh = 0.6;
    double det_db_unclip_ratio = 1.5;
    std::string det_db_score_mode = "slow";
    bool use_dilation = false;

    // Recognition parameters
    int rec_batch_num = 6;
    int rec_img_h = 48;
    int rec_img_w = 320;

    // Display help and exit
    bool help = false;
};

// Memory monitoring
struct MemoryUsage
{
    size_t current_mb;
    size_t peak_mb;
    size_t initial_mb;
};

static size_t initial_memory_mb = 0;

MemoryUsage getMemoryUsage()
{
    MemoryUsage usage = {0, 0, initial_memory_mb};

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memInfo;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memInfo, sizeof(memInfo)))
    {
        usage.current_mb = memInfo.WorkingSetSize / (1024 * 1024);
        usage.peak_mb = memInfo.PeakWorkingSetSize / (1024 * 1024);
    }
#else
    struct rusage rusage_info;
    if (getrusage(RUSAGE_SELF, &rusage_info) == 0)
    {
        usage.current_mb = rusage_info.ru_maxrss / 1024; // Linux: KB to MB
        usage.peak_mb = usage.current_mb;
    }
#endif

    return usage;
}

void initMemoryMonitor()
{
    MemoryUsage usage = getMemoryUsage();
    initial_memory_mb = usage.current_mb;
}

void showProgress(size_t current, size_t total)
{
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
              << current << "/" << total;
    std::cout.flush();

    if (current == total)
    {
        std::cout << std::endl;
    }
}

// Parse command line arguments
OCRConfig parseArgs(int argc, char **argv)
{
    OCRConfig config;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            config.help = true;
            return config;
        }
        else if (arg.find("--image_dir=") == 0)
        {
            config.image_dir = arg.substr(12);
        }
        else if (arg.find("--output=") == 0)
        {
            config.output_dir = arg.substr(9);
        }
        else if (arg.find("--det_model_dir=") == 0)
        {
            config.det_model_dir = arg.substr(16);
        }
        else if (arg.find("--rec_model_dir=") == 0)
        {
            config.rec_model_dir = arg.substr(16);
        }
        else if (arg.find("--rec_char_dict_path=") == 0)
        {
            config.rec_char_dict_path = arg.substr(21);
        }
        else if (arg.find("--inference_device=") == 0)
        {
            config.inference_device = arg.substr(19);
            config.device = config.inference_device; // Keep both for compatibility
        }
        else if (arg.find("--device=") == 0)
        {
            config.device = arg.substr(9);
            config.inference_device = config.device; // Keep both for compatibility
        }
        else if (arg == "--det=false")
        {
            config.det = false;
        }
        else if (arg == "--rec=false")
        {
            config.rec = false;
        }
        else if (arg.find("--det_db_thresh=") == 0)
        {
            config.det_db_thresh = std::stod(arg.substr(16));
        }
        else if (arg.find("--det_db_box_thresh=") == 0)
        {
            config.det_db_box_thresh = std::stod(arg.substr(20));
        }
        else if (arg.find("--limit_side_len=") == 0)
        {
            config.limit_side_len = std::stoi(arg.substr(17));
        }
    }

    return config;
}

void printUsage()
{
    std::cout << "OpenVINO OCR Usage:" << std::endl;
    std::cout << "  --image_dir=<path>          Input image directory" << std::endl;
    std::cout << "  --output=<path>             Output directory for results" << std::endl;
    std::cout << "  --det_model_dir=<path>      Detection model directory" << std::endl;
    std::cout << "  --rec_model_dir=<path>      Recognition model directory" << std::endl;
    std::cout << "  --rec_char_dict_path=<path> Character dictionary file" << std::endl;
    std::cout << "  --inference_device=<CPU|GPU|NPU> Inference device (default: GPU)" << std::endl;
    std::cout << "  --det=<true|false>          Enable detection (default: true)" << std::endl;
    std::cout << "  --rec=<true|false>          Enable recognition (default: true)" << std::endl;
    std::cout << "  --det_db_thresh=<float>     Detection threshold (default: 0.3)" << std::endl;
    std::cout << "  --det_db_box_thresh=<float> Box threshold (default: 0.6)" << std::endl;
    std::cout << "  --limit_side_len=<int>      Image resize limit (default: 960)" << std::endl;
    std::cout << "  --help, -h                  Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  ppocr_openvino --det_model_dir=../models/ch_PP-OCRv4_det_infer --rec_model_dir=../models/ch_PP-OCRv4_rec_infer --image_dir=../ocr_data/test_gpu --inference_device=GPU --output=../output/debug" << std::endl;
}

bool validateConfig(const OCRConfig &config)
{
    if (config.image_dir.empty())
    {
        std::cerr << "[ERROR] --image_dir parameter is required" << std::endl;
        return false;
    }

    if (config.output_dir.empty())
    {
        std::cerr << "[ERROR] --output parameter is required" << std::endl;
        return false;
    }

    if (config.det && config.det_model_dir.empty())
    {
        std::cerr << "[ERROR] --det_model_dir parameter is required when detection is enabled" << std::endl;
        return false;
    }

    if (config.rec && config.rec_model_dir.empty())
    {
        std::cerr << "[ERROR] --rec_model_dir parameter is required when recognition is enabled" << std::endl;
        return false;
    }

    // For recognition, try to find dictionary file automatically if not provided
    if (config.rec && config.rec_char_dict_path.empty())
    {
        std::cerr << "[WARNING] --rec_char_dict_path not provided, will try to find dictionary automatically" << std::endl;
    }

    if (config.inference_device != "CPU" && config.inference_device != "GPU" && config.inference_device != "NPU")
    {
        std::cerr << "[ERROR] inference_device should be 'CPU', 'GPU', or 'NPU'" << std::endl;
        return false;
    }

    return true;
}

std::string getModelPath(const std::string &model_dir, const std::string &type)
{
    std::string xml_path = model_dir + "/inference.xml";
    if (std::filesystem::exists(xml_path))
    {
        return xml_path;
    }

    xml_path = model_dir + "/" + type + ".xml";
    if (std::filesystem::exists(xml_path))
    {
        return xml_path;
    }

    return model_dir;
}

// OCR结果结构体
struct OCRResult
{
    std::vector<std::vector<int>> box;
    std::string text;
    float score;
};

std::vector<OCRResult> processImage(const std::string &image_path,
                                    DBDetectorOpenVINO *detector,
                                    CRNNRecognizerOpenVINO *recognizer,
                                    const OCRConfig &config)
{
    std::vector<OCRResult> results;

    // Read image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "[ERROR] Cannot read image: " << image_path << std::endl;
        return results;
    }

    std::vector<double> times;

    // Detection
    std::vector<std::vector<std::vector<int>>> det_boxes;
    if (config.det && detector)
    {
        detector->Run(img, det_boxes, times);
    }
    else
    {
        // If no detection, treat whole image as one text region
        det_boxes.push_back({{0, 0}, {img.cols, 0}, {img.cols, img.rows}, {0, img.rows}});
    }

    // Recognition
    if (config.rec && recognizer && !det_boxes.empty())
    {
        // Crop text regions for recognition
        std::vector<cv::Mat> img_list;
        for (const auto &box : det_boxes)
        {
            cv::Mat crop_img;
            std::vector<cv::Point2f> box_points;
            for (const auto &point : box)
            {
                box_points.push_back(cv::Point2f(point[0], point[1]));
            }

            // Get cropped image
            cv::Rect bbox = cv::boundingRect(box_points);
            bbox.x = std::max(0, bbox.x);
            bbox.y = std::max(0, bbox.y);
            bbox.width = std::min(img.cols - bbox.x, bbox.width);
            bbox.height = std::min(img.rows - bbox.y, bbox.height);

            if (bbox.width > 0 && bbox.height > 0)
            {
                crop_img = img(bbox).clone();
                img_list.push_back(crop_img);
            }
        }

        if (!img_list.empty())
        {
            std::vector<std::string> rec_texts;
            std::vector<float> rec_scores;
            std::vector<double> rec_times;

            recognizer->Run(img_list, rec_texts, rec_scores, rec_times);

            // Combine results
            for (size_t i = 0; i < det_boxes.size() && i < rec_texts.size(); i++)
            {
                OCRResult result;
                result.box = det_boxes[i];
                result.text = rec_texts[i];
                result.score = i < rec_scores.size() ? rec_scores[i] : 0.0f;
                results.push_back(result);
            }
        }
    }
    else if (!det_boxes.empty())
    {
        // Only detection, no recognition
        for (const auto &box : det_boxes)
        {
            OCRResult result;
            result.box = box;
            result.text = "";
            result.score = 1.0f;
            results.push_back(result);
        }
    }

    return results;
}

void runBatchProcessing(const OCRConfig &config)
{
    std::cout << "=== OpenVINO OCR Batch Processing ===" << std::endl;

    // Initialize memory monitor
    initMemoryMonitor();

    // Check if image directory exists
    if (!std::filesystem::exists(config.image_dir))
    {
        std::cerr << "[ERROR] Image directory not found: " << config.image_dir << std::endl;
        exit(1);
    }

    // Create output directory if not exists
    if (!std::filesystem::exists(config.output_dir))
    {
        std::filesystem::create_directories(config.output_dir);
    }

    // Get all image files
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"};

    for (const auto &entry : std::filesystem::directory_iterator(config.image_dir))
    {
        if (entry.is_regular_file())
        {
            std::string ext = entry.path().extension().string();
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end())
            {
                image_files.push_back(entry.path().string());
            }
        }
    }

    if (image_files.empty())
    {
        std::cerr << "[ERROR] No image files found in: " << config.image_dir << std::endl;
        exit(1);
    }

    std::cout << "Found " << image_files.size() << " image files." << std::endl;

    // Initialize OpenVINO models
    std::unique_ptr<DBDetectorOpenVINO> detector;
    std::unique_ptr<CRNNRecognizerOpenVINO> recognizer;

    try
    {
        if (config.det)
        {
            std::cout << "Loading detection model..." << std::endl;
            detector = std::make_unique<DBDetectorOpenVINO>(
                config.det_model_dir, config.inference_device, config.limit_type, config.limit_side_len,
                config.det_db_thresh, config.det_db_box_thresh, config.det_db_unclip_ratio,
                config.det_db_score_mode, config.use_dilation);
        }

        if (config.rec)
        {
            std::cout << "Loading recognition model..." << std::endl;
            // Try to find dictionary automatically if not provided
            std::string dict_path = config.rec_char_dict_path;
            if (dict_path.empty())
            {
                // Common dictionary file names to try
                std::vector<std::string> dict_names = {
                    "ppocr_keys_v1.txt",
                    "dict.txt", 
                    "keys.txt",
                    "../ppocr_keys_v1.txt"
                };
                
                for (const auto& name : dict_names)
                {
                    if (std::filesystem::exists(name))
                    {
                        dict_path = name;
                        std::cout << "Found dictionary file: " << dict_path << std::endl;
                        break;
                    }
                }
                
                if (dict_path.empty())
                {
                    std::cerr << "[WARNING] No dictionary file found, using default character set" << std::endl;
                    dict_path = ""; // Let the recognizer use default
                }
            }
            
            recognizer = std::make_unique<CRNNRecognizerOpenVINO>(
                config.rec_model_dir, config.inference_device, dict_path,
                config.rec_batch_num, config.rec_img_h, config.rec_img_w);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Failed to initialize models: " << e.what() << std::endl;
        exit(1);
    }

    std::cout << "Models loaded successfully!" << std::endl;
    std::cout << "Processing images..." << std::endl;

    // Statistics
    double sum_inference_time = 0.0;
    MemoryUsage initial_memory = getMemoryUsage();
    MemoryUsage max_memory = initial_memory;
    double sum_memory_increase = 0.0;

    // Process each image
    for (size_t i = 0; i < image_files.size(); ++i)
    {
        showProgress(i + 1, image_files.size());

        auto start_time = std::chrono::high_resolution_clock::now();

        // Process image
        std::vector<OCRResult> ocr_results = processImage(
            image_files[i], detector.get(), recognizer.get(), config);

        auto end_time = std::chrono::high_resolution_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        sum_inference_time += inference_time;

        // Monitor memory
        MemoryUsage current_memory = getMemoryUsage();
        if (current_memory.current_mb > max_memory.current_mb)
        {
            max_memory = current_memory;
        }
        sum_memory_increase += (current_memory.current_mb - initial_memory.current_mb);

        // Save results
        std::filesystem::path img_path(image_files[i]);
        std::string base_name = img_path.stem().string();
        std::string output_file = config.output_dir + "/" + base_name + ".txt";

        std::ofstream out_file(output_file);
        if (out_file.is_open())
        {
            for (const auto &result : ocr_results)
            {
                if (!result.text.empty())
                {
                    // Write box coordinates and text
                    out_file << "Box: ";
                    for (const auto &point : result.box)
                    {
                        out_file << "[" << point[0] << "," << point[1] << "] ";
                    }
                    out_file << "Text: " << result.text << " Score: " << result.score << std::endl;
                }
            }
            out_file.close();
        }
    }

    // Print statistics
    std::cout << std::endl;
    std::cout << "======================== Processing Results ========================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total images processed: " << image_files.size() << std::endl;
    std::cout << "Average inference time: " << (sum_inference_time / image_files.size()) << " ms" << std::endl;
    std::cout << "Memory usage:" << std::endl;
    std::cout << "  Average increase: " << (sum_memory_increase / image_files.size()) << " MB" << std::endl;
    std::cout << "  Maximum increase: " << (max_memory.current_mb - initial_memory.current_mb) << " MB" << std::endl;
    std::cout << "Results saved to: " << config.output_dir << std::endl;
    std::cout << "=================================================================" << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "=== OpenVINO OCR Application ===" << std::endl;

    // Parse command line arguments
    OCRConfig config = parseArgs(argc, argv);

    if (config.help)
    {
        printUsage();
        return 0;
    }

    // Validate configuration
    if (!validateConfig(config))
    {
        std::cout << std::endl;
        printUsage();
        return -1;
    }

    // Display configuration
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Image directory: " << config.image_dir << std::endl;
    std::cout << "Output directory: " << config.output_dir << std::endl;
    std::cout << "Inference device: " << config.inference_device << std::endl;
    std::cout << "Detection: " << (config.det ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Recognition: " << (config.rec ? "Enabled" : "Disabled") << std::endl;
    if (config.det)
    {
        std::cout << "Detection model: " << config.det_model_dir << std::endl;
    }
    if (config.rec)
    {
        std::cout << "Recognition model: " << config.rec_model_dir << std::endl;
        if (!config.rec_char_dict_path.empty())
        {
            std::cout << "Character dictionary: " << config.rec_char_dict_path << std::endl;
        }
    }
    std::cout << "====================" << std::endl;

    try
    {
        // Check OpenVINO device availability
        ov::Core core;
        auto devices = core.get_available_devices();

        bool device_found = false;
        for (const auto &device : devices)
        {
            if (device == config.inference_device)
            {
                device_found = true;
                break;
            }
        }

        if (!device_found)
        {
            std::cerr << "[WARNING] Requested device '" << config.inference_device << "' not found." << std::endl;
            std::cout << "Available devices: ";
            for (const auto &device : devices)
            {
                std::cout << device << " ";
            }
            std::cout << std::endl;
            std::cout << "Falling back to CPU..." << std::endl;
            // Don't exit, just warn and continue
        }

        // Run batch processing
        runBatchProcessing(config);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

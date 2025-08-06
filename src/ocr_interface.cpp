#include "include/ocr_interface.h"
#include "include/ocr_det_openvino.h"
#include "include/ocr_rec_openvino.h"

namespace PaddleOCR
{
    // Adapter for OpenVINO Detector
    class OpenVINODetectorAdapter : public DetectorInterface
    {
    private:
        std::unique_ptr<DBDetectorOpenVINO> detector_;

    public:
        OpenVINODetectorAdapter(std::unique_ptr<DBDetectorOpenVINO> detector)
            : detector_(std::move(detector)) {}

        void Run(const cv::Mat &img,
                 std::vector<std::vector<std::vector<int>>> &boxes,
                 std::vector<double> &times) noexcept override
        {
            cv::Mat img_copy = img.clone(); // Make a mutable copy
            detector_->Run(img_copy, boxes, times);
        }
    };

    // Adapter for OpenVINO Recognizer
    class OpenVINORecognizerAdapter : public RecognizerInterface
    {
    private:
        std::unique_ptr<CRNNRecognizerOpenVINO> recognizer_;

    public:
        OpenVINORecognizerAdapter(std::unique_ptr<CRNNRecognizerOpenVINO> recognizer)
            : recognizer_(std::move(recognizer)) {}

        void Run(const std::vector<cv::Mat> &img_list,
                 std::vector<std::string> &rec_texts,
                 std::vector<float> &rec_text_scores,
                 std::vector<double> &times) noexcept override
        {
            recognizer_->Run(img_list, rec_texts, rec_text_scores, times);
        }
    };

    // Factory implementations
    std::unique_ptr<DetectorInterface> DetectorFactory::CreateDetector(
        const std::string &model_dir,
        const int &cpu_math_library_num_threads,
        const std::string &limit_type,
        const int &limit_side_len,
        const double &det_db_thresh,
        const double &det_db_box_thresh,
        const double &det_db_unclip_ratio,
        const std::string &det_db_score_mode,
        const bool &use_dilation,
        const std::string &precision,
        const std::string &device)
    {
        auto detector = std::make_unique<DBDetectorOpenVINO>(
            model_dir, device, limit_type, limit_side_len, det_db_thresh, det_db_box_thresh,
            det_db_unclip_ratio, det_db_score_mode, use_dilation);
        return std::make_unique<OpenVINODetectorAdapter>(std::move(detector));
    }

    std::unique_ptr<RecognizerInterface> RecognizerFactory::CreateRecognizer(
        const std::string &model_dir,
        const int &cpu_math_library_num_threads,
        const std::string &label_path,
        const std::string &precision,
        const int &rec_batch_num,
        const int &rec_img_h,
        const int &rec_img_w,
        const std::string &device)
    {
        auto recognizer = std::make_unique<CRNNRecognizerOpenVINO>(
            model_dir, device, label_path, rec_batch_num, rec_img_h, rec_img_w);
        return std::make_unique<OpenVINORecognizerAdapter>(std::move(recognizer));
    }

} // namespace PaddleOCR

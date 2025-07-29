#include "include/ocr_interface.h"

#ifdef WITH_PADDLE
#include "include/ocr_det.h"
#include "include/ocr_rec.h"
#endif

#ifdef WITH_OPENVINO
#include "include/ocr_det_openvino.h"
#include "include/ocr_rec_openvino.h"
#endif

namespace PaddleOCR
{

#ifdef WITH_PADDLE
    // Adapter for Paddle Detector (only compiled when WITH_PADDLE=ON)
    class PaddleDetectorAdapter : public DetectorInterface
    {
    private:
        std::unique_ptr<DBDetector> detector_;

    public:
        PaddleDetectorAdapter(std::unique_ptr<DBDetector> detector)
            : detector_(std::move(detector)) {}

        void Run(const cv::Mat &img,
                 std::vector<std::vector<std::vector<int>>> &boxes,
                 std::vector<double> &times) noexcept override
        {
            detector_->Run(img, boxes, times);
        }
    };

    // Adapter for Paddle Recognizer (only compiled when WITH_PADDLE=ON)
    class PaddleRecognizerAdapter : public RecognizerInterface
    {
    private:
        std::unique_ptr<CRNNRecognizer> recognizer_;

    public:
        PaddleRecognizerAdapter(std::unique_ptr<CRNNRecognizer> recognizer)
            : recognizer_(std::move(recognizer)) {}

        void Run(const std::vector<cv::Mat> &img_list,
                 std::vector<std::string> &rec_texts,
                 std::vector<float> &rec_text_scores,
                 std::vector<double> &times) noexcept override
        {
            recognizer_->Run(img_list, rec_texts, rec_text_scores, times);
        }
    };
#endif // WITH_PADDLE

#ifdef WITH_OPENVINO
    // Adapter for OpenVINO Detector (only compiled when WITH_OPENVINO=ON)
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
            cv::Mat img_copy = img.clone();  // Make a mutable copy
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
#endif // WITH_OPENVINO

    // Factory implementations
    std::unique_ptr<DetectorInterface> DetectorFactory::CreateDetector(
        const std::string &framework,
        const std::string &model_dir,
        const bool &use_gpu,
        const int &gpu_id,
        const int &gpu_mem,
        const int &cpu_math_library_num_threads,
        const bool &use_mkldnn,
        const std::string &limit_type,
        const int &limit_side_len,
        const double &det_db_thresh,
        const double &det_db_box_thresh,
        const double &det_db_unclip_ratio,
        const std::string &det_db_score_mode,
        const bool &use_dilation,
        const bool &use_tensorrt,
        const std::string &precision,
        const std::string &device)
    {

#ifdef WITH_PADDLE
        if (framework == "paddle")
        {
            auto detector = std::make_unique<DBDetector>(
                model_dir, use_gpu, gpu_id, gpu_mem, cpu_math_library_num_threads,
                use_mkldnn, limit_type, limit_side_len, det_db_thresh, det_db_box_thresh,
                det_db_unclip_ratio, det_db_score_mode, use_dilation, use_tensorrt, precision);
            return std::make_unique<PaddleDetectorAdapter>(std::move(detector));
        }
        else
#endif
#ifdef WITH_OPENVINO
        if (framework == "ov")
        {
            auto detector = std::make_unique<DBDetectorOpenVINO>(
                model_dir, device, limit_type, limit_side_len, det_db_thresh, det_db_box_thresh,
                det_db_unclip_ratio, det_db_score_mode, use_dilation);
            return std::make_unique<OpenVINODetectorAdapter>(std::move(detector));
        }
        else
#endif
        {
            std::cerr << "[ERROR] Unsupported framework: " << framework << " or framework not enabled" << std::endl;
            return nullptr;
        }
    }

    std::unique_ptr<RecognizerInterface> RecognizerFactory::CreateRecognizer(
        const std::string &framework,
        const std::string &model_dir,
        const bool &use_gpu,
        const int &gpu_id,
        const int &gpu_mem,
        const int &cpu_math_library_num_threads,
        const bool &use_mkldnn,
        const std::string &label_path,
        const bool &use_tensorrt,
        const std::string &precision,
        const int &rec_batch_num,
        const int &rec_img_h,
        const int &rec_img_w,
        const std::string &device)
    {

#ifdef WITH_PADDLE
        if (framework == "paddle")
        {
            auto recognizer = std::make_unique<CRNNRecognizer>(
                model_dir, use_gpu, gpu_id, gpu_mem, cpu_math_library_num_threads,
                use_mkldnn, label_path, use_tensorrt, precision, rec_batch_num, rec_img_h, rec_img_w);
            return std::make_unique<PaddleRecognizerAdapter>(std::move(recognizer));
        }
        else
#endif
#ifdef WITH_OPENVINO
        if (framework == "ov")
        {
            auto recognizer = std::make_unique<CRNNRecognizerOpenVINO>(
                model_dir, device, label_path, rec_batch_num, rec_img_h, rec_img_w);
            return std::make_unique<OpenVINORecognizerAdapter>(std::move(recognizer));
        }
        else
#endif
        {
            std::cerr << "[ERROR] Unsupported framework: " << framework << " or framework not enabled" << std::endl;
            return nullptr;
        }
    }

} // namespace PaddleOCR

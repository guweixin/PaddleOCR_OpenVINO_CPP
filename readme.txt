cmake -G "Visual Studio 16 2019" -A x64 -DWITH_OPENVINO=ON -DOPENCV_DIR="D:/Project_source_code/opencv/opencv-4.12.0/build" -DPADDLE_LIB="D:\Project_source_code\repo_OCR\Paddle\build_openblas\paddle_inference_install_dir" ..

cmake -G "Visual Studio 16 2019" -A x64 -DWITH_OPENVINO=ON -DOPENCV_DIR="D:/Project_source_code/opencv/opencv-4.12.0/build" -DPADDLE_LIB="D:\Project_source_code\repo_OCR\Paddle\build_openblas\paddle_inference_install_dir" -DOPENVINO_DIR="D:\My_Work_progress_2\AI_framework\OpenVINO\openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64" ..
cmake --build . --config Release

ppocr.exe ocr --input=d:\My_Work_progress_2\HP_Xiaowei\20250626_paddleocr\ocr_data\hp_test\1.png  --text_detection_model_name=PP-OCRv4_mobile_det --text_detection_model_dir=..\..\..\models\PP-OCRv4_mobile_det_infer --text_recognition_model_name=PP-OCRv4_mobile_rec --text_recognition_model_dir=..\..\..\models\PP-OCRv4_mobile_rec_infer  --save_path=..\..\..\output\temp
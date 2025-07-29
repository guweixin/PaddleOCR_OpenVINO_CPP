cmake_minimum_required(VERSION 3.14)
project(ppocr CXX C)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(WITH_TENSORRT "Compile demo with TensorRT."   OFF)
option(WITH_PADDLE "Compile demo with Paddle framework support."   OFF)
option(WITH_OPENVINO "Compile demo with OpenVINO support."   ON)
option(BUILD_OPENVINO_ONLY "Build standalone OpenVINO-only executable."   OFF)

SET(PADDLE_LIB "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")
SET(CUDNN_LIB "" CACHE PATH "Location of libraries")
SET(TENSORRT_DIR "" CACHE PATH "Compile demo with TensorRT")
SET(OPENVINO_DIR "" CACHE PATH "Location of OpenVINO installation")

set(DEMO_NAME "ppocr")


macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WITH_MKL)
    ADD_DEFINITIONS(-DUSE_MKL)
endif()

if (WITH_OPENVINO)
    ADD_DEFINITIONS(-DWITH_OPENVINO)
    if(NOT DEFINED OPENVINO_DIR)
        message(FATAL_ERROR "please set OPENVINO_DIR with -DOPENVINO_DIR=/path/openvino")
    endif()
    find_package(OpenVINO REQUIRED PATHS ${OPENVINO_DIR}/cmake NO_DEFAULT_PATH)
    message(STATUS "OpenVINO found: ${OpenVINO_FOUND}")
endif()

if (WITH_PADDLE)
    ADD_DEFINITIONS(-DWITH_PADDLE)
    if(NOT DEFINED PADDLE_LIB)
      message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
    endif()
else()
    message(STATUS "Paddle framework disabled, skipping Paddle dependency checks")
endif()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()


if (WIN32)
  if (WITH_PADDLE)
    include_directories("${PADDLE_LIB}/paddle/include")
    link_directories("${PADDLE_LIB}/paddle/lib")
  endif()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)

else ()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
  if (WITH_PADDLE)
    include_directories("${PADDLE_LIB}/paddle/include")
    link_directories("${PADDLE_LIB}/paddle/lib")
  endif()
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    if(WITH_MKL)
        set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
	add_definitions(-DYAML_CPP_STATIC_DEFINE)
    endif()
    message("cmake c debug flags " ${CMAKE_C_FLAGS_DEBUG})
    message("cmake c release flags " ${CMAKE_C_FLAGS_RELEASE})
    message("cmake cxx debug flags " ${CMAKE_CXX_FLAGS_DEBUG})
    message("cmake cxx release flags " ${CMAKE_CXX_FLAGS_RELEASE})
else()
    if(WITH_MKL)
        set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 ${FLAG_OPENMP} -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    message("cmake cxx flags" ${CMAKE_CXX_FLAGS})
endif()

if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR "${CUDA_LIB}" STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    endif(NOT WIN32)
endif()

if (WITH_PADDLE)
  include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
  include_directories("${PADDLE_LIB}/third_party/install/glog/include")
  include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
  include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
  include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
  include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
  include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
  include_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/include")
  include_directories("${PADDLE_LIB}/third_party/boost")
  include_directories("${PADDLE_LIB}/third_party/eigen3")
endif()

include_directories("${CMAKE_SOURCE_DIR}/")

if (NOT WIN32)
  if (WITH_TENSORRT AND WITH_GPU)
     include_directories("${TENSORRT_DIR}/include")
     link_directories("${TENSORRT_DIR}/lib")
  endif()
endif(NOT WIN32)

if (WITH_PADDLE)
  link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
  link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
  link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
  link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
  link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
  link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
  link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
  link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
  link_directories("${PADDLE_LIB}/paddle/lib")
endif()


if (WITH_PADDLE)
  if(WITH_MKL)
    include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
    if (WIN32)
      set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib
              ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.lib)
    else ()
      set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
      execute_process(COMMAND cp -r ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
    endif ()
    set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/onednn")
    if(EXISTS ${MKLDNN_PATH})
      include_directories("${MKLDNN_PATH}/include")
      if (WIN32)
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
      else ()
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libdnnl.so.3)
      endif ()
    endif()
  else()
    # When WITH_MKL=OFF and WITH_PADDLE=ON, try to use OpenBLAS if available
    if(EXISTS "${PADDLE_LIB}/third_party/install/openblas/lib")
      if (WIN32)
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
      else ()
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif ()
    else()
      message(WARNING "OpenBLAS not found, math operations may be limited")
      set(MATH_LIB "")
    endif()
    set(MKLDNN_LIB "")
  endif()
else()
  # When WITH_PADDLE=OFF, no math library required
  set(MATH_LIB "")
  set(MKLDNN_LIB "")
  message(STATUS "Paddle disabled - no math library dependencies")
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_inference.so/a
if (WITH_PADDLE)
  if(WITH_STATIC_LIB)
    if(WIN32)
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  else()
    if(WIN32)
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
      set(DEPS
          ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
  endif(WITH_STATIC_LIB)
else()
  # No Paddle dependencies when WITH_PADDLE=OFF
  set(DEPS "")
endif()

if (NOT WIN32)
    if (WITH_PADDLE)
      set(DEPS ${DEPS}
          ${MATH_LIB} ${MKLDNN_LIB}
          glog gflags protobuf z xxhash
          )
      if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
          set(DEPS ${DEPS} snappystream)
      endif()
      if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
          set(DEPS ${DEPS} snappy)
      endif()
    endif()
else()
    if (WITH_PADDLE)
      set(DEPS ${DEPS}
          ${MATH_LIB} ${MKLDNN_LIB}
          glog gflags_static libprotobuf xxhash)
      set(DEPS ${DEPS} libcmt shlwapi)
      if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
          set(DEPS ${DEPS} snappy)
      endif()
      if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
          set(DEPS ${DEPS} snappystream)
      endif()
    endif()
endif(NOT WIN32)

if (WITH_PADDLE AND EXISTS "${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
  set(DEPS ${DEPS} yaml-cpp)
endif()

if(WITH_GPU)
  if(NOT WIN32)
    if (WITH_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(DEPS ${DEPS} ${OpenCV_LIBS})

# Add OpenVINO dependencies if enabled
if (WITH_OPENVINO)
    set(DEPS ${DEPS} openvino::runtime)
endif()

include(FetchContent)
include(external-cmake/auto-log.cmake)
include_directories(${FETCHCONTENT_BASE_DIR}/extern_autolog-src)

aux_source_directory(./src SRCS)

# Remove all source files first
set(SRCS "")

# Add only OpenVINO specific source files when WITH_OPENVINO=ON
if (WITH_OPENVINO)
    list(APPEND SRCS "./src/ocr_det_openvino.cpp")
    list(APPEND SRCS "./src/ocr_rec_openvino.cpp")
    list(APPEND SRCS "./src/utility.cpp")
    list(APPEND SRCS "./src/preprocess_op.cpp")
    list(APPEND SRCS "./src/postprocess_op.cpp")
    list(APPEND SRCS "./src/clipper.cpp")
    list(APPEND SRCS "./src/ocr_interface.cpp")
endif()

# Always add main.cpp since it contains conditional compilation
list(APPEND SRCS "./src/main.cpp")

# Only add Paddle-specific source files when WITH_PADDLE=ON
if (WITH_PADDLE)
    list(APPEND SRCS "./src/args.cpp")
    list(APPEND SRCS "./src/ocr_det.cpp")
    list(APPEND SRCS "./src/ocr_rec.cpp")
    list(APPEND SRCS "./src/paddleocr.cpp")
    list(APPEND SRCS "./src/paddlestructure.cpp")
endif()

# Always add data_saver if it exists
if(EXISTS "./src/data_saver.cpp")
    list(APPEND SRCS "./src/data_saver.cpp")
endif()

add_executable(${DEMO_NAME} ${SRCS})
target_link_libraries(${DEMO_NAME} ${DEPS})

# Add standalone OpenVINO-only executable if requested
if (BUILD_OPENVINO_ONLY AND WITH_OPENVINO)
    # Create separate source list for OpenVINO-only build
    set(OPENVINO_SRCS "")
    list(APPEND OPENVINO_SRCS "./src/main_openvino.cpp")
    list(APPEND OPENVINO_SRCS "./src/ocr_det_openvino.cpp")
    list(APPEND OPENVINO_SRCS "./src/ocr_rec_openvino.cpp")
    list(APPEND OPENVINO_SRCS "./src/utility.cpp")
    list(APPEND OPENVINO_SRCS "./src/preprocess_op.cpp")
    list(APPEND OPENVINO_SRCS "./src/postprocess_op.cpp")
    list(APPEND OPENVINO_SRCS "./src/clipper.cpp")
    
    # OpenVINO-only dependencies (no Paddle dependencies)
    set(OPENVINO_DEPS ${OpenCV_LIBS} openvino::runtime)
    
    if (NOT WIN32)
        set(OPENVINO_EXTERNAL_LIB "-ldl -lrt -lm -lpthread")
        set(OPENVINO_DEPS ${OPENVINO_DEPS} ${OPENVINO_EXTERNAL_LIB})
    endif()
    
    add_executable(ppocr_openvino ${OPENVINO_SRCS})
    target_link_libraries(ppocr_openvino ${OPENVINO_DEPS})
    
    message(STATUS "OpenVINO-only executable 'ppocr_openvino' will be built")
endif()

if (WIN32 AND WITH_MKL)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./release/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./release/mkldnn.dll
    )
endif()

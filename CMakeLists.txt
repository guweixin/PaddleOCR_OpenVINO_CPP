project(ppocr CXX)
cmake_minimum_required(VERSION 3.14)

set(DEMO_NAME "ppocr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_FREETYPE "Enable FreeType support" OFF)

SET(OPENCV_DIR "" CACHE PATH "Location of OpenCV libraries")
SET(OPENVINO_DIR "" CACHE PATH "Location of OpenVINO libraries")
SET(CUDA_LIB "" CACHE PATH "Location of CUDA libraries")
SET(CUDNN_LIB "" CACHE PATH "Location of CUDNN libraries")

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if(NOT DEFINED OPENVINO_DIR)
    message(FATAL_ERROR "please set OPENVINO_DIR with -DOPENVINO_DIR=/path/openvino")
endif()


if (WIN32)
  # OpenVINO configuration for Windows
  include_directories("${OPENVINO_DIR}/runtime/include")
  link_directories("${OPENVINO_DIR}/runtime/lib/intel64/Release")
  
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  set(OpenCV_DIR "${OPENCV_DIR}/x64/vc16/lib") 
  find_package(OpenCV REQUIRED )
  if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
      message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
  endif()  

else ()
  # OpenVINO configuration for Linux
  include_directories("${OPENVINO_DIR}/runtime/include")
  link_directories("${OPENVINO_DIR}/runtime/lib/intel64")
  
  set(OpenCV_DIR "${OPENCV_DIR}/lib64/cmake/opencv4") 
  find_package(OpenCV REQUIRED)
  if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
      message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
  endif()  
endif ()

# OpenVINO is now required by default
set(OPENVINO_LIBS openvino)

include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
    endif()
    message("cmake c debug flags " ${CMAKE_C_FLAGS_DEBUG})
    message("cmake c release flags " ${CMAKE_C_FLAGS_RELEASE})
    message("cmake cxx debug flags " ${CMAKE_CXX_FLAGS_DEBUG})
    message("cmake cxx release flags " ${CMAKE_CXX_FLAGS_RELEASE})
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    message("cmake cxx flags" ${CMAKE_CXX_FLAGS})
endif()

if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    add_definitions(-DWITH_GPU)
    endif(NOT WIN32)
endif()

# Include directories
include_directories("${CMAKE_SOURCE_DIR}/")

# Set up library dependencies
set(DEPS ${OPENVINO_LIBS})

if(WITH_GPU)
  if(NOT WIN32)
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    message($DEPS)
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party)
function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS "${filename}" AND NOT EXISTS "${decompress_dir}")
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
    message("Decompress file ${filename} ...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
  endif()
endfunction()


set(PACKAGE_LIST clipper_ver6.4.2 nlohmann)
foreach(PKG ${PACKAGE_LIST})
    set(PKG_URL "https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/${PKG}.tgz")
    set(PKG_TGZ_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PKG}.tgz")
    set(PKG_DST_PATH "${THIRD_PARTY_PATH}/${PKG}")
    download_and_decompress(${PKG_URL} ${PKG_TGZ_PATH} ${PKG_DST_PATH})
endforeach()

add_subdirectory(third_party/clipper_ver6.4.2/cpp)
include_directories(${POLYCLIPPING_INCLUDE_DIR})

set(DEPS ${DEPS} ${OpenCV_LIBS})
set(DEPS ${DEPS} polyclipping)

if(UNIX)
  find_package(Iconv REQUIRED) 
endif()

file(GLOB_RECURSE SRC_LIST "./src/*.cc")
set(SRCS cli.cc )
add_executable(${DEMO_NAME} ${SRCS} ${SRC_LIST} )
target_link_libraries(${DEMO_NAME} ${DEPS} )
cmake_minimum_required(VERSION 2.8)
project(BASIC)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE = Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_library(${PROJECT_NAME} SHARED
src/lboost.cc
src/Optimizer.cc
include/lboost.h
include/Optimizer.h)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
)




# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/bin)

# boost库序列化操作
add_executable(useboost
        Examples/boost/useboost.cc)
target_link_libraries(useboost ${PROJECT_NAME})

# yaml文件参数读取
add_executable(readyaml
        Examples/yaml/readyaml.cc)
target_link_libraries(readyaml ${PROJECT_NAME})

#eigen3 库的使用
add_executable(useeigen
        Examples/eigen/useeigen.cc)
target_link_libraries(useeigen ${PROJECT_NAME})

# c++ 调用python
set(PYTHON_INCLUDE_DIR /usr/include/python3.6)#这个好像并不影响
#python的虚拟环境，需要添加对应的.so，bash中 source /venv/bin/activate
set(PYTHON_LIBRARY /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_executable(usepython
        Examples/python/usepython.cc)
target_link_libraries(usepython ${PYTHON_LIBRARIES})
#cv::Mat的格式转化
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)#网上有人说python3.8能找到
add_executable(cvmat2py
        Examples/python/cvmat2py.cc)
target_link_libraries(cvmat2py 
${PROJECT_NAME}
${PYTHON_LIBRARIES}
Python3::NumPy)
# c++ 调用python 网络传递参数
add_executable(useNet 
Examples/python/useNet.cc)
target_link_libraries(useNet
${PYTHON_LIBRARIES}
${PROJECT_NAME}
Python3::NumPy)

# g2o 使用
add_executable(useg2o
        Examples/g2o/useg2o.cc)
target_link_libraries(useg2o ${PROJECT_NAME})


add_subdirectory(/Examples/ceres)
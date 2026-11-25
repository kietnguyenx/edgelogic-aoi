cmake_minimum_required(VERSION 3.16)
project(edgelogic_aoi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(edgelogic_aoi
    src/main.cpp
    src/aoi_engine.cpp
    src/drift_gate.cpp
    src/conformal_triage.cpp
)

target_include_directories(edgelogic_aoi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(edgelogic_aoi PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(edgelogic_aoi PRIVATE /O2 /W3)
else()
    target_compile_options(edgelogic_aoi PRIVATE -O3 -Wall -Wextra)
endif()

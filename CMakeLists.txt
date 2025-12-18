# CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 项目信息
project(Faster_HNSW)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为 Release 构建类型添加 O3 优化
# 在运行时使用 cmake .. -DCMAKE_BUILD_TYPE=Release 来启用
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 查找所有在 src 目录下的 .cpp 源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(Faster_HNSW ${SOURCES})

# 添加头文件目录
target_include_directories(Faster_HNSW PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include) 

# 为目标添加 AVX512F 编译选项
# 注意: 这需要编译器和 CPU 的支持
target_compile_options(Faster_HNSW PRIVATE -mavx512f)

# 添加 OpenMP 支持
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(Faster_HNSW PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled.")
else()
    message(WARNING "OpenMP not found. Parallel sections will run serially.")
endif()

# 打印状态信息
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Release build flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "AVX512F flag enabled for the target.")

//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

cmake_minimum_required(VERSION 3.10)
project(torch_extension LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 填入你自己的riscv交叉编译器路径
set(CMAKE_C_COMPILER /home/llj/riscv-tools/build/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_CXX_COMPILER /home/llj/riscv-tools/build/bin/riscv64-unknown-linux-gnu-g++)

# 填入你自己的riscv交叉编译环境中的python路径
set(PYTHON_INCLUDE_DIR "/home/llj/download/python3.11-include")

# 填入你自己的riscv交叉编译环境中的torch路径
set(TORCH_INCLUDE_DIRS 
    /home/llj/download/python3.11-site-packages/site-packages/torch/include 
    /home/llj/download/python3.11-site-packages/site-packages/torch/include/torch/csrc/api/include 
    /home/llj/download/python3.11-site-packages/site-packages/torch/include/TH 
    /home/llj/download/python3.11-site-packages/site-packages/torch/include/THC 
)

# 填入你自己的riscv交叉编译环境中的python、torch库路径
link_directories(
    /home/llj/download/python3.11-lib
    /home/llj/download/python3.11-site-packages/site-packages/torch/lib
)



add_compile_options(
    -Wno-unused-result
    -Wsign-compare
    -DNDEBUG
    -g
    -fwrapv
    -O3
    -Wall
    -fstack-protector-strong
    -Wformat
    -Werror=format-security
    -Wdate-time
    -D_FORIFY_SOURCE=2
    -fPIC
    -march=rv64gcv_zvfh_zfh
    -mabi=lp64d
)

add_definitions(
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DPYBIND11_COMPILER_TYPE="_gcc"
    -DPYBIND11_STDLIB="_libstdcpp"
    -DPYBIND11_BUILD_ABI="_cxxabi1011"
    -DTORCH_EXTENSION_NAME=llm_extension_cpp
)


include_directories(${PYTHON_INCLUDE_DIR})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

set(SOURCES
    src/llm_extension.cpp
    src/ThreadPool.cpp
    src/gemm.cpp
    src/hgemm.cpp
    src/gemm_ncopy8_g.c
    src/gemm_tcopy16_g.c
    src/hgemv.c
    src/gemv.c
)

# 动态库
add_library(llm_extension_cpp MODULE ${SOURCES})

# 可执行文件
# add_executable(llm_extension_cpp ${SOURCES})

#动态库
set_target_properties(llm_extension_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
    PREFIX ""
    SUFFIX ".so"
)

# 可执行文件
# set_target_properties(llm_extension_cpp PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
#     OUTPUT_NAME "llm_extension_cpp"
# )


target_link_libraries(llm_extension_cpp PRIVATE
    pthread
    python3.11
    c10
    torch_python
    torch_cpu
    torch
)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 动态库
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-o1 -Wl,-Bsymbolic-functions -Wl,--trace -fopenmp -march=rv64gcv_zvfh_zfh -mabi=lp64d")
set_target_properties(llm_extension_cpp PROPERTIES OUTPUT_NAME llm_extension_cpp)

# 可执行文件
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,-o1 -Wl,-Bsymbolic-functions -Wl,--trace -fopenmp -march=rv64gcv_zvfh_zfh -mabi=lp64d")


message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
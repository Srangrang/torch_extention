# gem5 Directory Overview

本目录包含与gem5模拟器相关的配置文件和算子验证代码。以下是目录中主要文件的说明：

## 文件说明

### 1. `rvv.py`
- **功能**: 
  - 该文件是gem5运行的配置文件。
  - 用于配置和启动与RVV（RISC-V V扩展）相关的模拟任务。
- **用途**: 可用于设置模拟参数，生成RVV架构的运行环境。

### 2. `torch_extension_cpp_thread_gemm_with_time`
- **功能**: 
  - 该文件是矩阵乘法（GEMM，General Matrix Multiplication）算子的验证代码。
  - 支持多线程执行和RVV。
- **用途**: 可用于验证矩阵乘法算子的正确性和性能。

### 3. `torch_extension_cpp_thread_gemv_without_time`
- **功能**: 
  - 该文件是向量乘矩阵（GEMV，General Matrix-Vector Multiplication）算子的验证代码。
  - 支持多线程执行和RVV。
- **用途**: 可用于验证向量乘矩阵算子的正确性和性能。

## 使用指南

### 配置和运行gem5
1. 确保安装了gem5模拟器。
2. 根据需求编辑`rvv.py`文件中的配置参数。
3. 运行命令：
   ```bash
   gem5.opt rvv.py
   ```

## 注意事项
- 确保系统支持RVV V扩展模拟。
- 多线程运行时，请根据系统的CPU核心数调整线程参数。
- 验证算子时，请提供合适的输入数据以测试算子功能和性能。

## 目录结构示例
```
.
├── rvv.py
├── torch_extension_cpp_thread_gemm_with_time
├── torch_extension_cpp_thread_gemv_without_time
└── ...
```

如果有任何问题，请参考相关文档或联系开发者。

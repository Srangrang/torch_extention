# **BLAS float16-GEMM and float16-GEMV By Pytorch-extension**

该仓库基于Pytorch扩展自定义算子，在RISCV架构下通过vector1.0 扩展实现float16矩阵向量运算,含线程池矩阵分割。

## 依赖

以下依赖仅在x86_64 ubuntu22.04交叉编译环境编译，RISCV openeuler2403下运行测试，请遵循**依赖安装**章节进行

1.python==3.11

2.numpy>=2.1.3

3.torch==2.3.0

4.rvv=1.0

5.riscv-gnu-gcc>=14.1.0

## 依赖安装

为使用该扩展，你需要确保在你自己的riscv环境中能够正确安装并运行对应版本的Python与Torch。

接下来你需要在你的交叉编译环境中配置riscv架构的上述依赖，其中Python与Torch推荐直接将riscv环境中对应的内容打包放入交叉编译环境。主要内容如下

torch-include:

    your/python/site-package/path/site-packages/torch/
    your/python/site-package/path/site-packages/torch/include/torch/csrc/api/include 
    your/python/site-package/path/site-packages/torch/include/TH 
    your/python/site-package/path/site-packages/torch/include/include/THC

torch-lib:

    your/python/site-package/path/site-packages/torch/lib/

python-include:
    
    your/python/include/path/python3.11/

python-lib:
    
    your/python/lib/path/libpython3.11.so
    your/python/lib/path/python3.11/

#### 以上内容可以结合CMakeLists文件理解，接下来需要准备交叉编译器。

本项目使用 `riscv-collab/riscv-gnu-toolchain` 交叉编译器， see [here]https://github.com/riscv-collab/riscv-gnu-toolchain 。注意：务必保证gcc版本要求，vector 1.0扩展在gcc仅在 `14.1.0` 及以上支持，并且添加编译参数 `--with-arch` 与 `--with-abi` 如下
```bash
./configure --prefix=/your/riscv --with-arch=rv64gcv --with-abi=lp64d
```
完成以上依赖的安装后便可获得基本环境，接下来进入下一章节安装本仓库

## 安装

当你准备好了基本的交叉编译环境即可遵循下面的指令安装本项目
```bash
clone torch_extension
cd torch_extension
vim CMakeLists.txt 
```
将 `CMakeLists.txt` 标注出的路径配置为你的实际环境路径，接下来
```bash
mkdir build && cd build
cmake ..
make
```
接下来你编译出来的动态库将出现在当前目录的 `build` 下。

安装完成



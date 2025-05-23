//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

#include <torch/extension.h>
#include <riscv_vector.h>
#include <vector>
#include "gemm.h"
#include "ThreadPool.h"
#include "gemv.h"

torch::Tensor gemm_16x8(torch::Tensor A, torch::Tensor B, int32_t M, int32_t N, int32_t K, torch::Tensor C){
    
    auto A_ptr_kHalf = A.data_ptr<at::Half>();
    auto B_ptr_kHalf = B.data_ptr<at::Half>();
    auto C_ptr_kHalf = C.data_ptr<at::Half>();
    size_t sizeA = A.numel();
    size_t sizeB = B.numel();
    size_t sizeC = C.numel();
    _Float16 *A_ptr = (_Float16 *)malloc(sizeA * sizeof(_Float16));
    _Float16 *B_ptr = (_Float16 *)malloc(sizeB * sizeof(_Float16));
    _Float16 *C_ptr = (_Float16 *)malloc(sizeC * sizeof(_Float16));

    std::memcpy(A_ptr, A_ptr_kHalf, A.numel() * sizeof(at::Half));
    std::memcpy(B_ptr, B_ptr_kHalf, B.numel() * sizeof(at::Half));
    std::memcpy(C_ptr, C_ptr_kHalf, C.numel() * sizeof(at::Half));
    _Float16 alpha_float16 = 1.0f;

    gemm(N, M, K, alpha_float16, B_ptr, A_ptr, C_ptr);

    std::vector<int64_t> sizes = {M, N};
    at::Tensor C_tensor = torch::from_blob(C_ptr, sizes, [](void* ptr) {
        free(ptr);
    }, torch::TensorOptions().dtype(torch::kFloat16));

    free(A_ptr);
    free(B_ptr);
    return C_tensor;
}

torch::Tensor gemv_kernel(torch::Tensor A, torch::Tensor B, int32_t K, int32_t N, torch::Tensor C){   //A*B + C
    auto A_ptr_kHalf = A.data_ptr<at::Half>();
    auto B_ptr_kHalf = B.data_ptr<at::Half>();
    auto C_ptr_kHalf = C.data_ptr<at::Half>();
    size_t sizeA = A.numel();
    size_t sizeB = B.numel();
    size_t sizeC = C.numel();
    _Float16 *A_ptr = (_Float16 *)malloc(sizeA * sizeof(_Float16));
    _Float16 *B_ptr = (_Float16 *)malloc(sizeB * sizeof(_Float16));
    _Float16 *C_ptr = (_Float16 *)malloc(sizeC * sizeof(_Float16));

    std::memcpy(A_ptr, A_ptr_kHalf, A.numel() * sizeof(at::Half));
    std::memcpy(B_ptr, B_ptr_kHalf, B.numel() * sizeof(at::Half));
    std::memcpy(C_ptr, C_ptr_kHalf, C.numel() * sizeof(at::Half));

    _Float16 alpha_float16 = 1.0f;
    gemv(N, K, 0, alpha_float16, B_ptr, N, A_ptr, 1, C_ptr, 1, &alpha_float16);

    std::vector<int64_t> sizes = {1, N};
    at::Tensor C_tensor = torch::from_blob(C_ptr, sizes, [](void* ptr) {
        free(ptr);
    }, torch::TensorOptions().dtype(torch::kFloat16));

    free(A_ptr);
    free(B_ptr);
    return C_tensor;


}

torch::Tensor multiThreadTest(torch::Tensor A, torch::Tensor B, int32_t M, int32_t N, int32_t K, torch::Tensor C){
    const unsigned int num_threads = std::thread::hardware_concurrency();
    ThreadPool pool(num_threads);

    unsigned int num_tasks = N / num_threads;
    unsigned int num_tasks_t = N % num_threads;

    std::vector<std::future<torch::Tensor>> results;
    std::vector<torch::Tensor> results_tensor;

    if(M == 1){
        int start = 0;
        for (int i = 0; i < num_threads; ++i) {
            int end = end = start + num_tasks + (num_tasks_t>i? 1: 0);

            torch::Tensor B_sliced = B.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
            torch::Tensor C_sliced = C.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
            results.emplace_back(pool.enqueue(gemv_kernel, A, B_sliced.contiguous(), K, end-start, C_sliced));

            start = end;
            
        }

        for (auto& res : results) {
            results_tensor.emplace_back(res.get());
        }
        return torch::cat(results_tensor, 1);


    }else {
        int start = 0;
        for (int i = 0; i < num_threads; ++i) {
            int end = end = start + num_tasks + (num_tasks_t>i? 1: 0);

            torch::Tensor B_sliced = B.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
            torch::Tensor C_sliced = C.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
            results.emplace_back(pool.enqueue(gemm_16x8, A, B_sliced.contiguous(), M, end-start, K, C_sliced.contiguous()));

            start = end;
            
        }
        
        for (auto& res : results) {
            results_tensor.emplace_back(res.get());
        }

        return torch::cat(results_tensor, 1);

    }
    
}


#define TORCH_EXTENSION_NAME llm_extension_cpp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("testMGEMM", &gemm_16x8, "test for A*B + C");
  m.def("testGEMV", &gemv_kernel, "test for A*B + C");
  m.def("testGEMM", &multiThreadTest, "test for multiThreadTest");
}





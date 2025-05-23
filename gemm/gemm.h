#include <riscv_vector.h>
#include <malloc.h>
#include "gemm_ncopy8_g.h"
#include "gemm_tcopy16_g.h"
#include "gemm_tcopy8_g.h"
#include "hgemm.h"

int gemm_8x8(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C);
int gemm_16x8(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C);

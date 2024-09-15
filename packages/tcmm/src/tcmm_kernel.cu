#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector>

#include "tcmm_kernel.h"

using namespace std;

static cusolverDnHandle_t g_cusolverH = NULL;
static cublasHandle_t g_cublasHandle = NULL;
static cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;


#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

cusolverDnHandle_t get_cusolver_handler() {
    if (g_cusolverH == NULL) {
        cusolver_status = cusolverDnCreate(&g_cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    }
    return g_cusolverH;
}

cublasHandle_t get_cublas_handler() {
    if (g_cublasHandle == NULL) {
        cublasErrCheck(cublasCreate(&g_cublasHandle));
    }
    return g_cublasHandle;
}


std::vector<torch::Tensor> tcmm_symeig(torch::Tensor a) {
    const auto a_shape = a.sizes();
    const int m = a_shape[0];
    const int lda = m;
    int lwork = 0;
    int *devInfo = NULL;
    cudaError_t cudaStat1 = cudaSuccess;

    auto options_float =
        torch::TensorOptions()
        .dtype(a.dtype())
        .layout(torch::kStrided)
        .device(a.device().type())
        .requires_grad(false);

    auto A = a.data_ptr<float>();
    auto V = torch::zeros({m, m}, options_float).copy_(a); // eigenvectors
    //auto V = a.copy_(a);
    auto W = torch::zeros({m}, options_float); // eigenvalues

    cusolverDnHandle_t cusolverH = get_cusolver_handler();
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    cusolver_status = cusolverDnSsyevd_bufferSize(
            cusolverH,
            jobz,
            uplo,
            m,
            V.data_ptr<float>(),
            lda,
            W.data_ptr<float>(),
            &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    float *d_work = NULL;
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnSsyevd(
            cusolverH,
            jobz,
            uplo,
            m,
            V.data_ptr<float>(),
            lda,
            W.data_ptr<float>(),
            d_work,
            lwork,
            devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    cudaFree(devInfo);
    cudaFree(d_work);
    std::vector<torch::Tensor> tuple;
    tuple.push_back(W); 
    tuple.push_back(V); 
    return tuple;
}

std::vector<torch::Tensor> tcmm_symeig_sparse(torch::Tensor a) {
    std::vector<torch::Tensor> tuple;
    tuple.push_back(a); 
    tuple.push_back(a); 
    return tuple;
}

torch::Tensor tcmm_gemm_ex(torch::Tensor a, torch::Tensor b) {
    torch::Tensor a_fp16 = at::_cast_Half(a);
    torch::Tensor b_fp16 = at::_cast_Half(b);

    const auto a_shape = a.sizes();
    const int m = a_shape[0];
    const int k = a_shape[1];
    const int n = b.sizes()[1];
    const float alpha = 1.0;
    const float beta = 0.0;
    cudaError_t cudaStat1 = cudaSuccess;

    auto options_float =
        torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(a.device().type())
        .requires_grad(false);
    auto c = torch::zeros({m, n}, options_float); 
    cublasHandle_t cublasHandle = get_cublas_handler();

    cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, 
                m, n, k, 
                &alpha,
                a_fp16.data_ptr<at::Half>(), CUDA_R_16F, k,
                b_fp16.data_ptr<at::Half>(), CUDA_R_16F, n,
                &beta, 
                c.data_ptr<float>(), CUDA_R_32F, n,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    return c;
}

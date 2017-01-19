#include "common.h"

namespace hagrid {

__host__ float profile(std::function<void()> f) {
    cudaEvent_t start_kernel, end_kernel;
    CHECK_CUDA_CALL(cudaEventCreate(&start_kernel));
    CHECK_CUDA_CALL(cudaEventCreate(&end_kernel));
    CHECK_CUDA_CALL(cudaEventRecord(start_kernel));
    f();
    CHECK_CUDA_CALL(cudaEventRecord(end_kernel));
    CHECK_CUDA_CALL(cudaEventSynchronize(end_kernel));
    float kernel_time = 0;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel));
    CHECK_CUDA_CALL(cudaEventDestroy(start_kernel));
    CHECK_CUDA_CALL(cudaEventDestroy(end_kernel));
    return kernel_time;
}

} // namespace hagrid

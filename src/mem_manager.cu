#include "mem_manager.h"
#include "common.h"

namespace hagrid {

HOST void* MemManager::alloc_no_slot(size_t size) {
    void* ptr;
    CHECK_CUDA_CALL(cudaMalloc(&ptr, size));
    return ptr;
}

HOST void MemManager::free_no_slot(void* ptr) {
    CHECK_CUDA_CALL(cudaFree(ptr));
}

HOST void MemManager::alloc_slot(Slot& slot, size_t size) {
    assert(!slot.in_use && "Buffer not deallocated properly");
    if (slot.size < size) {
        if (slot.ptr) CHECK_CUDA_CALL(cudaFree(slot.ptr));
        CHECK_CUDA_CALL(cudaMalloc(&slot.ptr, size));
        usage_ = usage_ + size - slot.size;
        peak_ = std::max(usage_, peak_);
        slot.size = size;
    }
#ifndef NDEBUG
    slot.in_use = true;
#endif
}

HOST void MemManager::free_slot(Slot& slot) {
    assert(slot.in_use);
#ifndef NDEBUG
    slot.in_use = false;
#endif
    if (!keep_) {
        usage_ = usage_ - slot.size;
        if (slot.ptr) CHECK_CUDA_CALL(cudaFree(slot.ptr));
        slot.size = 0;
        slot.ptr = nullptr;
    }
}

HOST void MemManager::copy_dev_to_dev(void* dst, const void* src, size_t bytes) {
    CHECK_CUDA_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

HOST void MemManager::copy_hst_to_dev(void* dst, const void* src, size_t bytes) {
    CHECK_CUDA_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

HOST void MemManager::copy_dev_to_hst(void* dst, const void* src, size_t bytes) {
    CHECK_CUDA_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

HOST void MemManager::zero_dev(void* ptr, size_t bytes) {
    CHECK_CUDA_CALL(cudaMemset(ptr, 0, bytes));
}

} // namespace hagrid

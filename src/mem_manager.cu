#include "mem_manager.h"
#include "common.h"

namespace hagrid {

HOST void MemManager::debug_slots() const {
    size_t total = 0;
    std::cout << "SLOTS: " << std::endl;
    for (auto& slot : slots_) {
        std::cout << "["
                  << (slot.in_use ? 'X' : ' ')
                  << "] "
                  << (double)slot.size / (1024.0 * 1024.0) << "MB" << std::endl;
        total += slot.size;
    }
    std::cout << (double)total / (1024.0 * 1024.0) << "MB total" << std::endl;
}

HOST void MemManager::alloc_slot(Slot& slot, size_t size) {
    assert(!slot.in_use && "Buffer not deallocated properly");
    if (slot.size < size) {
        if (slot.ptr) CHECK_CUDA_CALL(cudaFree(slot.ptr));
        CHECK_CUDA_CALL(cudaMalloc(&slot.ptr, size));
        usage_     = usage_ + size - slot.size;
        max_usage_ = std::max(usage_, max_usage_);
        slot.size = size;
    }
    slot.in_use = true;

    if (keep_ && usage_ == max_usage_) {
        size_t saved = 0;
        // Deallocate the least used slots
        for (auto& slot : slots_) {
            if (slot.in_use) continue;
            usage_ = usage_ - slot.size;
            if (slot.ptr) CHECK_CUDA_CALL(cudaFree(slot.ptr));
            saved += size;
            slot.size = 0;
            slot.ptr = nullptr;
            if (saved == size) break;
        }
    }
}

HOST void MemManager::free_slot(Slot& slot) {
    assert(slot.in_use);
    slot.in_use = false;
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

#ifndef MEM_MANAGER_H
#define MEM_MANAGER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <limits>
#include <unordered_map>
#include "common.h"

namespace hagrid {

/// Directions to copy memory from and to
enum class Copy {
    HST_TO_DEV,
    DEV_TO_HST,
    DEV_TO_DEV
};

/// A slot for a buffer in GPU memory
struct Slot {
    Slot()
        : ptr(nullptr)
        , size(0)
        , in_use(false)
    {}

    void* ptr;
    size_t size;
    bool in_use;
};

/// Utility class to manage memory buffers during construction
class MemManager {
public:
    /// Creates a manager object. The boolean flag controls whether
    /// buffers are kept or deallocated upon a call to free(). Keeping
    /// the buffers increases the memory usage, but speeds-up subsequent
    /// builds (often useful for dynamic scenes).
    MemManager(bool keep = false)
        : keep_(keep), usage_(0), max_usage_(0)
    {}

    /// Allocates a buffer, re-using allocated memory when possible
    template <typename T>
    HOST T* alloc(size_t n) {
        auto size = n * sizeof(T);
        auto min_diff = std::numeric_limits<size_t>::max();
        int found = -1;

        for (int i = 0, n = slots_.size(); i < n; i++) {
            auto& slot = slots_[i];
            if (!slot.in_use) {
                auto diff = std::max(size, slot.size) - std::min(size, slot.size);
                if (diff < min_diff) {
                    min_diff = diff;
                    found = i;
                }
            }
        }

        if (found < 0) {
            found = slots_.size();
            slots_.resize(found + 1);
        }

        Slot& slot = slots_[found];
        alloc_slot(slot, size);
        tracker_[slot.ptr] = found;
        return reinterpret_cast<T*>(slot.ptr);
    }

    /// Frees the contents of the given slot
    template <typename T>
    HOST void free(T* ptr) {
        if (!ptr) return;
        assert(tracker_.count(ptr));
        free_slot(slots_[tracker_[ptr]]);
        tracker_.erase(ptr);
    }

    /// Copies memory between buffers
    template <Copy type, typename T>
    HOST void copy(T* dst, const T* src, size_t n) {
        if (type == Copy::DEV_TO_DEV)      copy_dev_to_dev(dst, src, sizeof(T) * n);
        else if (type == Copy::DEV_TO_HST) copy_dev_to_hst(dst, src, sizeof(T) * n);
        else if (type == Copy::HST_TO_DEV) copy_hst_to_dev(dst, src, sizeof(T) * n);
    }

    /// Fills memory with zeros
    template <typename T>
    HOST void zero(T* ptr, size_t n) { zero_dev(ptr, n * sizeof(T)); }

    /// Fills memory with ones
    template <typename T>
    HOST void one(T* ptr, size_t n) { one_dev(ptr, n * sizeof(T)); }

    /// Displays slots and memory usage
    void debug_slots() const;

    /// Returns the current memory usage
    size_t usage() const { return usage_; }
    /// Returns the maximum memory usage
    size_t max_usage() const { return max_usage_; }

private:
    HOST void alloc_slot(Slot&, size_t);
    HOST void free_slot(Slot&);
    HOST void copy_dev_to_dev(void*, const void*, size_t);
    HOST void copy_dev_to_hst(void*, const void*, size_t);
    HOST void copy_hst_to_dev(void*, const void*, size_t);
    HOST void zero_dev(void*, size_t);
    HOST void one_dev(void*, size_t);

    std::unordered_map<void*, int> tracker_;
    std::vector<Slot> slots_;
    size_t usage_, max_usage_;
    bool keep_;
};

} // namespace hagrid

#endif

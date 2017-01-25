#ifndef MEM_MANAGER_H
#define MEM_MANAGER_H

#include <vector>
#include <iostream>
#include <cassert>
#include <limits>
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
    enum Name {
        TMP_STORAGE,
        BBOXES,
        START_EMIT,
        NEW_REF_COUNTS,
        REFS_PER_CELL,
        LOG_DIMS,
        KEPT_FLAGS,
        ARRAYS,

        // Aliases
        START_SPLIT   = START_EMIT,
        START_CELL    = START_EMIT,
        SPLIT_MASKS   = NEW_REF_COUNTS
    };

    static Name ref_array(int i)   { return Name(int(ARRAYS) + 3 * i + 0); }
    static Name cell_array(int i)  { return Name(int(ARRAYS) + 3 * i + 1); }
    static Name entry_array(int i) { return Name(int(ARRAYS) + 3 * i + 2); }

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
        : keep_(keep), usage_(0), peak_(0)
    {}

    /// Allocates a buffer
    template <typename T> HOST T* alloc(size_t n) { return reinterpret_cast<T*>(alloc_no_slot(n * sizeof(T))); }
    /// Frees a previously allocated buffer
    template <typename T> HOST void free(T* ptr)  { return free_no_slot(ptr); }

    /// Allocates a buffer in the given slot
    template <typename T>
    HOST T* alloc(Slot::Name name, size_t n) {
        slots_.resize(std::max(slots_.size(), size_t(name + 1)));
        Slot& slot = slots_[name];
        alloc_slot(slot, sizeof(T) * n);
        return reinterpret_cast<T*>(slot.ptr);
    }

    template <typename T>
    HOST T* slot_ptr(Slot::Name name) {
        return reinterpret_cast<T*>(slots_[name].ptr);
    }

    /// Frees the contents of the given slot
    HOST void free(Slot::Name name) {
        assert(name < slots_.size());
        Slot& slot = slots_[name];
        free_slot(slot);
    }

    /// Find a free slot that can hold the given number of elements of type T
    template <typename T>
    HOST Slot::Name find_slot(size_t n) {
        auto size = n * sizeof(T);
        auto min  = std::numeric_limits<size_t>::max();
        int found = -1;
        for (int i = int(Slot::ARRAYS), n = slots_.size(); i < n; i++) {
            auto& slot = slots_[i];
            if (!slot.in_use && slot.size >= size && slot.size < min) {
                found = i;
                min = slot.size;
            }
        }
        if (found < 0) found = max(int(Slot::ARRAYS), int(slots_.size()));
        auto name = Slot::Name(found);
        alloc<T>(name, n);
        return name;
    }

    /// Frees all the slots
    HOST void free_all() {
        if (keep_) {
            for (auto& slot : slots_) slot.in_use = false;
        } else {
            for (auto& slot : slots_) {
                if (slot.ptr) free_slot(slot);
            }
        }
    }

    /// Swaps two slots within the memory manager
    HOST void swap(Slot::Name name1, Slot::Name name2) {
        assert(size_t(name1) < slots_.size() && size_t(name2) < slots_.size()); 
        std::swap(slots_[name1], slots_[name2]);
    }

    template <Copy type, typename T>
    HOST void copy(T* dst, const T* src, size_t n) {
        if (type == Copy::DEV_TO_DEV)      copy_dev_to_dev(dst, src, sizeof(T) * n);
        else if (type == Copy::DEV_TO_HST) copy_dev_to_hst(dst, src, sizeof(T) * n);
        else if (type == Copy::HST_TO_DEV) copy_hst_to_dev(dst, src, sizeof(T) * n);
    }

    template <typename T>
    HOST void zero(T* ptr, size_t n) {
        zero_dev(ptr, n * sizeof(T));
    }

    /// Displays slots and memory usage
    void debug_slots() const {
        size_t total = 0;
        std::cout << "SLOTS: " << std::endl;
        for (auto& slot : slots_) {
            if (slot.in_use) std::cout << "[IN USE]";
            std::cout << (double)slot.size / (1024.0 * 1024.0) << "MB" << std::endl;
            total += slot.size;
        }
        std::cout << (double)total / (1024.0 * 1024.0) << "MB total" << std::endl;
    }

    /// Returns the current memory usage
    size_t usage() const { return usage_; }
    /// Returns the maximum memory usage
    size_t peak_usage() const { return peak_; }

private:
    HOST void* alloc_no_slot(size_t);
    HOST void free_no_slot(void*);
    HOST void alloc_slot(Slot&, size_t);
    HOST void free_slot(Slot&);
    HOST void copy_dev_to_dev(void*, const void*, size_t);
    HOST void copy_dev_to_hst(void*, const void*, size_t);
    HOST void copy_hst_to_dev(void*, const void*, size_t);
    HOST void zero_dev(void*, size_t);

    std::vector<Slot> slots_;
    size_t usage_, peak_;
    bool keep_;
};

} // namespace hagrid

#endif

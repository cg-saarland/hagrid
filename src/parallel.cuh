#ifndef PARALLEL_CUH
#define PARALLEL_CUH

#include <type_traits>
#include <cub/cub.cuh>
#include "mem_manager.h"
#include "common.h"

namespace hagrid {

/// Parallel primitives (mostly a wrapper around CUB)
class Parallel {
private:
    template <typename OutputIt>
    struct ResultType {
        typedef typename std::remove_reference<decltype(*(OutputIt()))>::type Type;
    };

public:
    Parallel(MemManager& mem)
        : mem_(mem)
    {}

    /// Creates a transformation iterator
    template <typename InputIt, typename F>
    auto transform(InputIt values, F f) -> cub::TransformInputIterator<decltype(f(*values)), F, InputIt> {
        return cub::TransformInputIterator<decltype(f(*values)), F, InputIt>(values, f);
    }

    /// Computes the exclusive sum of the given array, and returns the total
    template <typename InputIt, typename OutputIt>
    auto scan(InputIt values, int n, OutputIt result) -> typename ResultType<OutputIt>::Type {
        typedef typename ResultType<OutputIt>::Type T;
        size_t required_bytes;
        CHECK_CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, required_bytes, values, result, n));
        char* tmp_storage = mem_.alloc<char>(Slot::TMP_STORAGE, required_bytes);
        CHECK_CUDA_CALL(cub::DeviceScan::ExclusiveSum(tmp_storage, required_bytes, values, result, n));
        mem_.free(Slot::TMP_STORAGE);
        T total;
        CHECK_CUDA_CALL(cudaMemcpy(&total, result + n - 1, sizeof(T), cudaMemcpyDeviceToHost));
        return total;
    }

    /// Computes the reduction of the given operator over the given array array
    template <typename InputIt, typename OutputIt, typename F>
    auto reduce(InputIt values, int n, OutputIt result, F f, typename ResultType<OutputIt>::Type init = typename ResultType<OutputIt>::Type()) -> typename ResultType<OutputIt>::Type {
        typedef typename ResultType<OutputIt>::Type T;
        size_t required_bytes;
        CHECK_CUDA_CALL(cub::DeviceReduce::Reduce(nullptr, required_bytes, values, result, n, f, init));
        char* tmp_storage = mem_.alloc<char>(Slot::TMP_STORAGE, required_bytes);
        CHECK_CUDA_CALL(cub::DeviceReduce::Reduce(tmp_storage, required_bytes, values, result, n, f, init));
        mem_.free(Slot::TMP_STORAGE);
        T host_result;
        CHECK_CUDA_CALL(cudaMemcpy(&host_result, result, sizeof(T), cudaMemcpyDeviceToHost));
        return host_result;
    }

    /// Computes a partition of the given set according to an array of flags, returns the number of elements in first half
    template <typename InputIt, typename OutputIt, typename FlagIt>
    int partition(InputIt values, OutputIt result, int n, FlagIt flags) {
        size_t required_bytes;
        CHECK_CUDA_CALL(cub::DevicePartition::Flagged(nullptr, required_bytes, values, flags, result, (int*)nullptr, n));
        required_bytes += 4 - required_bytes % 4; // Align storage
        char* tmp_storage = mem_.alloc<char>(Slot::TMP_STORAGE, required_bytes + sizeof(int));
        int* count_ptr = reinterpret_cast<int*>(tmp_storage + required_bytes);
        CHECK_CUDA_CALL(cub::DevicePartition::Flagged(tmp_storage, required_bytes, values, flags, result, count_ptr, n));
        int count;
        CHECK_CUDA_CALL(cudaMemcpy(&count, count_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        mem_.free(Slot::TMP_STORAGE);
        return count;
    }

    /// Computes a partition of the given set according to an array of flags, returns the number of elements in first half
    template <typename Key, typename Value>
    void sort_pairs(Key* keys_in, Value* values_in, Key*& keys_out, Value*& values_out, int n) {
        size_t required_bytes;
        cub::DoubleBuffer<Key>   keys_buf(keys_in, keys_out);
        cub::DoubleBuffer<Value> values_buf(values_in, values_out);
        CHECK_CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, required_bytes, keys_buf, values_buf, n));
        char* tmp_storage = mem_.alloc<char>(Slot::TMP_STORAGE, required_bytes + sizeof(int));
        CHECK_CUDA_CALL(cub::DeviceRadixSort::SortPairs(tmp_storage, required_bytes, keys_buf, values_buf, n));
        mem_.free(Slot::TMP_STORAGE);
        keys_out   = keys_buf.Current();
        values_out = values_buf.Current();
    }

private:
    MemManager& mem_;
};

} // namespace hagrid

#endif // PARALLEL_CUH

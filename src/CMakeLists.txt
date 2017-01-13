if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPT_FLAG "-g")
else()
    set(OPT_FLAG "--use_fast_math;-O3")
endif()


cuda_compile(HAGRID_BUILD
    build.cu
    mem_manager.cu
    mem_manager.h
    parallel.cuh
    grid.h
    vec.h
    bbox.h
    common.h
    OPTIONS ${OPT_FLAG} "-std=c++11;--expt-extended-lambda;-arch=sm_52;-lineinfo;-DHOST=__host__;-DDEVICE=__device__")

add_executable(hagrid ${HAGRID_BUILD} main.cpp load_obj.cpp load_obj.h vec.h)
target_compile_definitions(hagrid PRIVATE HOST= DEVICE=)
target_link_libraries(hagrid ${CUDA_LIBRARIES})

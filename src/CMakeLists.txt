if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPT_FLAG "-g")
else()
    set(OPT_FLAG "--use_fast_math;-O3")
endif()

cuda_compile(HAGRID_BUILD
    build.cu
    merge.cu
    flatten.cu
    expand.cu
    compress.cu
    mem_manager.cu
    profile.cu
    mem_manager.h
    parallel.cuh
    build.h
    grid.h
    vec.h
    bbox.h
    prims.h
    ray.h
    common.h
    OPTIONS ${OPT_FLAG} "-std=c++11;--expt-extended-lambda;-arch=sm_52;-lineinfo;-DHOST=__host__;-DDEVICE=__device__")

add_library(hagrid_build ${HAGRID_BUILD})
set_target_properties(hagrid_build PROPERTIES LINKER_LANGUAGE CXX)

cuda_compile(HAGRID_TRAVERSE
    traverse.cu
    traverse.h
    ray.h
    prims.h
    vec.h
    grid.h
    OPTIONS ${OPT_FLAG} "-std=c++11;--expt-extended-lambda;-arch=sm_52;-lineinfo;--maxrregcount=40;-DHOST=__host__;-DDEVICE=__device__")

add_library(hagrid_traverse ${HAGRID_TRAVERSE})
set_target_properties(hagrid_traverse PROPERTIES LINKER_LANGUAGE CXX)

add_executable(hagrid main.cpp load_obj.cpp load_obj.h grid.h traverse.h build.h vec.h)
target_compile_definitions(hagrid PRIVATE HOST= DEVICE=)
target_link_libraries(hagrid ${SDL2_LIBRARY} ${CUDA_LIBRARIES} hagrid_build hagrid_traverse)

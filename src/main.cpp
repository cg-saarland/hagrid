#include <iostream>
#include <chrono>

#include "build.h"
#include "load_obj.h"
#include "mem_manager.h"

using namespace hagrid;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "error: incorrect number of arguments" << std::endl;
        return 1;
    }

    ObjLoader::File obj_file;
    ObjLoader::MaterialLib mtl_lib;
    if (!ObjLoader::load_scene(argv[1], obj_file, mtl_lib)) {
        std::cerr << "error: scene cannot be loaded (file not present or contains errors)" << std::endl;
        return 1;
    }

    std::vector<Tri> host_tris;
    for (auto& object : obj_file.objects) {
        for (auto& group : object.groups) {
            for (auto& face : group.faces) {
                auto v0 = obj_file.vertices[face.indices[0].v];
                for (int i = 0; i < face.index_count - 2; i++) {
                    auto v1 = obj_file.vertices[face.indices[i + 1].v];
                    auto v2 = obj_file.vertices[face.indices[i + 2].v];
                    auto e1 = v0 - v1;
                    auto e2 = v2 - v0;
                    auto n  = cross(e1, e2);

                    const Tri tri = {
                        v0, n.x,
                        e1, n.y,
                        e2, n.z
                    };
                    host_tris.push_back(tri);
                }
            }
        }
    }

    MemManager mem(true);
    auto tris = mem.alloc<Tri>(host_tris.size());
    mem.copy<Copy::HST_TO_DEV>(tris, host_tris.data(), host_tris.size());
    
    BuildParams params;
    params.top_density = strtof(argv[2], nullptr);
    params.snd_density = strtof(argv[3], nullptr);
    params.alpha = 0.995f;
    params.expansion = 3;
    params.level_shift = 3;
    Grid grid;

    for (int i = 0; i < 10; i++) {
        mem.free_all();
        build_grid(mem, params, tris, host_tris.size(), grid);
    }

    double total = 0;
    for (int i = 0; i < 50; i++) {
        mem.free_all();
        auto t0 = std::chrono::high_resolution_clock::now();
        build_grid(mem, params, tris, host_tris.size(), grid);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total += ms;
    }
    std::cout << total / 50 << "ms" << std::endl;

    mem.free(tris);
    return 0;
}

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "build.h"
#include "load_obj.h"
#include "mem_manager.h"

using namespace hagrid;

struct ProgramOptions {
    std::string scene_file;
    BuildParams build_params;
    int build_iter;
    int build_warmup;
    bool help;

    ProgramOptions()
        : build_params(BuildParams::static_scene())
        , build_iter(1)
        , build_warmup(0)
        , help(false)
    {}

    bool parse(int argc, char** argv);

private:
    static bool matches(const char* arg, const char* opt1, const char* opt2) {
        return !strcmp(arg, opt1) || !strcmp(arg, opt2);
    }

    static bool arg_exists(char** argv, int i, int argc) {
        if (i >= argc - 1 || argv[i + 1][0] == '-') {
            std::cerr << "argument missing for: " << argv[i] << std::endl;
            return false;
        }
        return true;
    }
};

bool ProgramOptions::parse(int argc, char** argv) {
    bool scene_parsed = false;
    for (int i = 1; i < argc; i++) {
        auto arg = argv[i];

        if (arg[0] != '-') {
            if (scene_parsed) {
                std::cerr << "cannot accept more than one model argument" << std::endl;
                return false;
            }
            scene_file = arg;
            scene_parsed = true;
            continue;
        }

        if (matches(arg, "-h", "--help")) {
            help = true;
        } else if (matches(arg, "-td", "--top-density")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_params.top_density = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-sd", "--snd-density")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_params.snd_density = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-a", "--alpha")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_params.alpha = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-e", "--expansion")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_params.expansion = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-s", "--shift")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_params.level_shift = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-nb", "--build-iter")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_iter = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-wb", "--build-warmup")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_warmup = strtol(argv[++i], nullptr, 10);
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (!scene_parsed) {
        std::cerr << "no model specified" << std::endl;
        return false;
    }

    return true;
}

static bool load_model(const std::string& file_name, std::vector<Tri>& tris) {
    ObjLoader::File obj_file;
    ObjLoader::MaterialLib mtl_lib;
    if (!ObjLoader::load_scene(file_name, obj_file, mtl_lib))
        return false;

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
                    tris.push_back(tri);
                }
            }
        }
    }

    return true;
}

static void usage() {
    std::cout << "usage: hagrid [options] file\n"
                 "options:\n"
                 "  -h      --help          Shows this message\n"
                 "  -td     --top-density   Sets the top-level density\n"
                 "  -sd     --snd-density   Sets the second-level density\n"
                 "  -a      --alpha         Sets the ratio that controls cell merging\n"
                 "  -e      --expansion     Sets the number of expansion passes\n"
                 "  -s      --shift         Sets the number of octree levels per subdivision iteration\n"
                 "  -nb     --build-iter    Sets the number of build iterations\n"
                 "  -wb     --build-warmup  Sets the number of warmup build iterations" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    ProgramOptions opts;
    if (!opts.parse(argc, argv)) return 1;

    if (opts.help) {
        usage();
        return 0;
    }

    if (!opts.build_params.valid()) {
        std::cerr << "the specified build options are invalid" << std::endl;
        return 1;
    }

    std::vector<Tri> host_tris;
    if (!load_model(opts.scene_file, host_tris)) {
        std::cerr << "scene cannot be loaded (file not present or contains errors)" << std::endl;
        return 1;
    }

    std::cout << host_tris.size() << " triangle(s)" << std::endl;

    MemManager mem(opts.build_iter + opts.build_warmup > 1);
    auto tris = mem.alloc<Tri>(host_tris.size());
    mem.copy<Copy::HST_TO_DEV>(tris, host_tris.data(), host_tris.size());

    Grid grid;
    for (int i = 0; i < opts.build_warmup; i++) {
        mem.free_all();
        build_grid(mem, opts.build_params, tris, host_tris.size(), grid);
    }

    double total = 0;
    for (int i = 0; i < opts.build_iter; i++) {
        mem.free_all();
        auto t0 = std::chrono::high_resolution_clock::now();
        build_grid(mem, opts.build_params, tris, host_tris.size(), grid);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total += ms;
    }
    auto dims = grid.top_dims << grid.shift;
    std::cout << "Grid built in " << total / opts.build_iter << " ms ("
              << dims.x << "x" << dims.y << "x" << dims.z << ", "
              << grid.num_cells << " cells, " << grid.num_refs << " references)" << std::endl;

    const size_t cells_mem = grid.num_cells * sizeof(Cell);
    const size_t entries_mem = grid.num_entries * sizeof(int);
    const size_t refs_mem = grid.num_refs * sizeof(int);
    const size_t tris_mem = host_tris.size() * sizeof(Tri);
    const size_t total_mem = cells_mem + entries_mem + refs_mem + tris_mem;
    std::cout << "Total memory: " << total_mem / double(1024 * 1024) << " MB" << std::endl;
    std::cout << "Cells: " << cells_mem / double(1024 * 1024) << " MB" << std::endl;
    std::cout << "Entries: " << entries_mem / double(1024 * 1024) << " MB" << std::endl;
    std::cout << "References: " << refs_mem / double(1024 * 1024) << " MB" << std::endl;
    std::cout << "Triangles: " << tris_mem / double(1024 * 1024) << " MB" << std::endl;
    std::cout << "Peak usage: " << mem.peak_usage() / double(1024.0 * 1024.0) << " MB" << std::endl;

    mem.free(tris);
    return 0;
}

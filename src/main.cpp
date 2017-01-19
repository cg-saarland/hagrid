#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <SDL2/SDL.h>

#include "build.h"
#include "load_obj.h"
#include "mem_manager.h"
#include "traverse.h"

using namespace hagrid;

struct Camera {
    vec3 eye;
    vec3 right;
    vec3 up;
    vec3 dir;
};

struct View {
    vec3 eye;
    vec3 forward;
    vec3 right;
    vec3 up;
    float dist;
    float rspeed;
    float tspeed;
};

inline Camera gen_camera(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio) {
    Camera cam;
    const float f = tanf(M_PI * fov / 360);
    cam.dir = normalize(center - eye);
    cam.right = normalize(cross(cam.dir, up)) * (f * ratio);
    cam.up = normalize(cross(cam.right, cam.dir)) * f;
    cam.eye = eye;
    return cam;
}

inline void gen_rays(const Camera& cam, std::vector<Ray>& rays, float clip, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            auto kx = 2 * x / float(w) - 1;
            auto ky = 1 - 2 * y / float(h);
            auto dir = cam.dir + cam.right * kx + cam.up * ky;

            auto& ray = rays[y * w + x];
            ray.org = cam.eye;
            ray.dir = dir;
            ray.tmin = 0.0f;
            ray.tmax = clip;
        }
    }
}

template <bool display_mode>
void update_surface(SDL_Surface* surf, std::vector<Hit>& hits, float clip, int w, int h) {
    for (int y = 0, my = std::min(surf->h, h); y < my; y++) {
        unsigned char* row = (unsigned char*)surf->pixels + surf->pitch * y;
        for (int x = 0, mx = std::min(surf->w, w); x < mx; x++) {
            const unsigned char color = display_mode ? hits[y * w + x].id : 255.0f * hits[y * w + x].t / clip;
            row[x * 4 + 0] = color;
            row[x * 4 + 1] = color;
            row[x * 4 + 2] = color;
            row[x * 4 + 3] = 255;
        }
    }
}

struct ProgramOptions {
    std::string scene_file;
    BuildParams build_params;
    int width, height;
    float clip, fov;
    int build_iter;
    int build_warmup;
    bool keep_alive;
    bool help;

    ProgramOptions()
        : build_params(BuildParams::static_scene())
        , width(1024)
        , height(1024)
        , clip(100)
        , fov(60)
        , build_iter(1)
        , build_warmup(0)
        , keep_alive(false)
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
        } else if (matches(arg, "-sx", "--width")) {
            if (!arg_exists(argv, i, argc)) return false;
            width = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-sy", "--height")) {
            if (!arg_exists(argv, i, argc)) return false;
            height = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-c", "--clip")) {
            if (!arg_exists(argv, i, argc)) return false;
            clip = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-f", "--fov")) {
            if (!arg_exists(argv, i, argc)) return false;
            fov = strtof(argv[++i], nullptr);
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
        } else if (matches(arg, "-k", "--keep-alive")) {
            keep_alive = true;
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

bool handle_events(View& view, bool& display_mode) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                return true;
            case SDL_MOUSEMOTION:
                if (SDL_GetMouseState(nullptr, nullptr) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
                    view.right = cross(view.forward, view.up);
                    view.forward = rotate(view.forward, view.right, -event.motion.yrel * view.rspeed);
                    view.forward = rotate(view.forward, view.up,    -event.motion.xrel * view.rspeed);
                    view.forward = normalize(view.forward);
                    view.up = normalize(cross(view.right, view.forward));
                }
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_UP:    view.eye = view.eye + view.tspeed * view.forward; break;
                    case SDLK_DOWN:  view.eye = view.eye - view.tspeed * view.forward; break;
                    case SDLK_LEFT:  view.eye = view.eye - view.tspeed * view.right;   break;
                    case SDLK_RIGHT: view.eye = view.eye + view.tspeed * view.right;   break;
                    case SDLK_KP_PLUS:  view.tspeed *= 1.1f; break;
                    case SDLK_KP_MINUS: view.tspeed /= 1.1f; break;
                    case SDLK_c:
                        {
                            auto center = view.eye + view.forward * view.dist;
                            std::cout << "Eye: " << view.eye.x << " " << view.eye.y << " " << view.eye.z << std::endl;
                            std::cout << "Center: " << center.x << " " << center.y << " " << center.z << std::endl;
                            std::cout << "Up: " << view.up.x << " " << view.up.y << " " << view.up.z << std::endl;
                        }
                        break;
                    case SDLK_m: display_mode = !display_mode; break;
                    case SDLK_ESCAPE:
                        return true;
                }
                break;
        }
    }
    return false;
}

static void usage() {
    std::cout << "usage: hagrid [options] file\n"
                 "options:\n"
                 "  -h      --help          Shows this message\n"
                 "  -sx     --width         Sets the viewport width\n"
                 "  -sy     --height        Sets the viewport height\n"
                 "  -c      --clip          Sets the clipping distance\n"
                 "  -f      --fov           Sets the field of view\n"
                 "  -td     --top-density   Sets the top-level density\n"
                 "  -sd     --snd-density   Sets the second-level density\n"
                 "  -a      --alpha         Sets the ratio that controls cell merging\n"
                 "  -e      --expansion     Sets the number of expansion passes\n"
                 "  -s      --shift         Sets the number of octree levels per subdivision iteration\n"
                 "  -nb     --build-iter    Sets the number of build iterations\n"
                 "  -wb     --build-warmup  Sets the number of warmup build iterations\n"
                 "  -k      --keep-alive    Keeps the buffers alive during construction\n" << std::endl;
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

    MemManager mem(opts.keep_alive);
    auto tris = mem.alloc<Tri>(host_tris.size());
    mem.copy<Copy::HST_TO_DEV>(tris, host_tris.data(), host_tris.size());

    Grid grid;

    // Warmup iterations
    for (int i = 0; i < opts.build_warmup; i++) {
        mem.free_all();
        build_grid(mem, opts.build_params, tris, host_tris.size(), grid);
    }

    // Benchmark construction speed
    double total_time = 0;
    for (int i = 0; i < opts.build_iter; i++) {
        mem.free_all();
        auto kernel_time = profile([&] { build_grid(mem, opts.build_params, tris, host_tris.size(), grid); });
        total_time += kernel_time;
    }
    auto dims = grid.dims << grid.shift;
    std::cout << "Grid built in " << total_time / opts.build_iter << " ms ("
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

    setup_traversal(grid);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Cannot initialize SDL." << std::endl;
        return 1;
    }

    SDL_Window* win = SDL_CreateWindow("HaGrid",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        opts.width, opts.height,
        0);

    SDL_Surface* screen = SDL_GetWindowSurface(win);

    SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);

    View view = {
        vec3(0.0f,  0.0f, -10.0f),   // Eye
        vec3(0.0f,  0.0f, 1.0f),     // Forward
        vec3(-1.0f, 0.0f, 0.0f),     // Right
        vec3(0.0f,  1.0f, 0.0f),     // Up
        100.0f, 0.005f, 1.0f         // View distance, rotation speed, translation speed
    };

    size_t num_rays = opts.width * opts.height;
    std::vector<Hit> host_hits(num_rays);
    std::vector<Ray> host_rays(num_rays);
    Ray* rays = mem.alloc<Ray>(num_rays);
    Hit* hits = mem.alloc<Hit>(num_rays);
    double kernel_time = 0;
    auto ticks = SDL_GetTicks();
    int frames = 0;
    bool display_mode = false;
    bool done = false;
    while (!done) {
        Camera cam = gen_camera(view.eye,
                                view.eye + view.forward * view.dist,
                                view.up,
                                opts.fov,
                                (float)opts.width / (float)opts.height);

        gen_rays(cam, host_rays, opts.clip, opts.width, opts.height);
        mem.copy<Copy::HST_TO_DEV>(rays, host_rays.data(), num_rays);

        kernel_time += profile([&] { traverse(grid, tris, rays, hits, num_rays); });
        frames++;

        if (SDL_GetTicks() - ticks >= 500) {
            std::ostringstream caption;
            caption << "HaGrid [" << double(frames) * double(opts.width * opts.height) / (1000 * kernel_time) << " MRays/s]";
            SDL_SetWindowTitle(win, caption.str().c_str());
            ticks = SDL_GetTicks();
            kernel_time = 0;
            frames = 0;
        }

        mem.copy<Copy::DEV_TO_HST>(host_hits.data(), hits, num_rays);
        SDL_LockSurface(screen);
        if (display_mode) update_surface<true >(screen, host_hits, opts.clip, opts.width, opts.height);
        else              update_surface<false>(screen, host_hits, opts.clip, opts.width, opts.height);
        SDL_UnlockSurface(screen);

        SDL_UpdateWindowSurface(win);
        done = handle_events(view, display_mode);
    }

    SDL_DestroyWindow(win);
    SDL_Quit();

    mem.free(rays);
    mem.free(hits);
    mem.free(tris);
    return 0;
}

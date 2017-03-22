#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <numeric>
#include <limits>

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

enum class DisplayMode {
    DEPTH,
    GRAY_SCALE,
    HEAT_MAP
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

void gradient(uint8_t* color, float k) {
    static const vec3 g[] = {
        vec3(0, 0, 255),
        vec3(0, 255, 255),
        vec3(0, 128, 0),
        vec3(255, 255, 0),
        vec3(255, 0, 0)
    };
    constexpr int n = sizeof(g) / sizeof(g[0]);
    static const float s = 1.0f / n;

    int i = min(n - 1, int(k * n));
    int j = min(n - 1, i + 1);

    float t = (k - i * s) / s;
    auto c = (1.0f - t) * g[i] + t * g[j];

    color[0] = c.z;
    color[1] = c.y;
    color[2] = c.x;
}

template <DisplayMode mode>
void update_surface(SDL_Surface* surf, std::vector<Hit>& hits, float clip, int w, int h) {
    for (int y = 0, my = std::min(surf->h, h); y < my; y++) {
        unsigned char* row = (unsigned char*)surf->pixels + surf->pitch * y;
        for (int x = 0, mx = std::min(surf->w, w); x < mx; x++) {
            if (mode == DisplayMode::DEPTH) {
                uint8_t color = 255.0f * hits[y * w + x].t / clip;
                row[x * 4 + 0] = color;
                row[x * 4 + 1] = color;
                row[x * 4 + 2] = color;
            } else if (mode == DisplayMode::GRAY_SCALE) {
                uint8_t color = std::min(255, hits[y * w + x].id);
                row[x * 4 + 0] = color;
                row[x * 4 + 1] = color;
                row[x * 4 + 2] = color;
            } else if (mode == DisplayMode::HEAT_MAP) {
                gradient(row + x * 4, std::min(100, hits[y * w + x].id) / 100.0f);
            }
            row[x * 4 + 3] = 255;
        }
    }
}

struct ProgramOptions {
    std::string scene_file;
    std::string ray_file;
    float top_density, snd_density;
    float alpha;
    int exp_iters;
    int width, height;
    float clip, fov;
    int build_iter;
    int build_warmup;
    int bench_iter;
    int bench_warmup;
    float tmin, tmax;
    bool keep_alive;
    bool help;

    ProgramOptions()
        : top_density(0.12f)
        , snd_density(2.4f)
        , alpha(0.995f)
        , exp_iters(3)
        , width(1024)
        , height(1024)
        , clip(0)
        , fov(60)
        , build_iter(1)
        , build_warmup(0)
        , bench_iter(1)
        , bench_warmup(0)
        , tmin(0)
        , tmax(std::numeric_limits<float>::max())
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
            std::cerr << "Argument missing for: " << argv[i] << std::endl;
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
                std::cerr << "Cannot accept more than one model on the command line" << std::endl;
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
            top_density = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-sd", "--snd-density")) {
            if (!arg_exists(argv, i, argc)) return false;
            snd_density = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-a", "--alpha")) {
            if (!arg_exists(argv, i, argc)) return false;
            alpha = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-e", "--expansion")) {
            if (!arg_exists(argv, i, argc)) return false;
            exp_iters = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-nb", "--build-iter")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_iter = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-wb", "--build-warmup")) {
            if (!arg_exists(argv, i, argc)) return false;
            build_warmup = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-k", "--keep-alive")) {
            keep_alive = true;
        } else if (matches(arg, "-r", "--ray-file")) {
            if (!arg_exists(argv, i, argc)) return false;
            ray_file = argv[++i];
        } else if (matches(arg, "-tmin", "--tmin")) {
            if (!arg_exists(argv, i, argc)) return false;
            tmin = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-tmax", "--tmax")) {
            if (!arg_exists(argv, i, argc)) return false;
            tmax = strtof(argv[++i], nullptr);
        } else if (matches(arg, "-n", "--bench-iter")) {
            if (!arg_exists(argv, i, argc)) return false;
            bench_iter = strtol(argv[++i], nullptr, 10);
        } else if (matches(arg, "-w", "--bench-warmup")) {
            if (!arg_exists(argv, i, argc)) return false;
            bench_warmup = strtol(argv[++i], nullptr, 10);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (!scene_parsed) {
        std::cerr << "No model specified" << std::endl;
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

static bool load_rays(const std::string& file_name, std::vector<Ray>& rays, float tmin, float tmax) {
    std::ifstream in(file_name, std::ifstream::binary);
    if (!in) return false;

    in.seekg(0, std::ifstream::end);
    int count = in.tellg() / (sizeof(float) * 6);

    rays.resize(count);
    in.seekg(0);

    for (int i = 0; i < count; i++) {
        float org_dir[6];
        in.read((char*)org_dir, sizeof(float) * 6);
        Ray& ray = rays.data()[i];

        ray.org = vec3(org_dir[0], org_dir[1], org_dir[2]);
        ray.dir = vec3(org_dir[3], org_dir[4], org_dir[5]);

        ray.tmin = tmin;
        ray.tmax = tmax;
    }

    return true;
}

bool handle_events(View& view, DisplayMode& display_mode) {
    static bool arrows[4], camera_on;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                return true;
            case SDL_MOUSEBUTTONDOWN:
                SDL_SetRelativeMouseMode(SDL_TRUE);
                camera_on = true;
                break;
            case SDL_MOUSEBUTTONUP:
                camera_on = false;
                SDL_SetRelativeMouseMode(SDL_FALSE);
                break;
            case SDL_MOUSEMOTION:
                if (camera_on) {
                    view.right = cross(view.forward, view.up);
                    view.forward = rotate(view.forward, view.right, -event.motion.yrel * view.rspeed);
                    view.forward = rotate(view.forward, view.up,    -event.motion.xrel * view.rspeed);
                    view.forward = normalize(view.forward);
                    view.up = normalize(cross(view.right, view.forward));
                }
                break;
            case SDL_KEYUP:
                switch (event.key.keysym.sym) {
                    case SDLK_UP:    arrows[0] = false; break;
                    case SDLK_DOWN:  arrows[1] = false; break;
                    case SDLK_LEFT:  arrows[2] = false; break;
                    case SDLK_RIGHT: arrows[3] = false; break;
                }
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_UP:    arrows[0] = true; break;
                    case SDLK_DOWN:  arrows[1] = true; break;
                    case SDLK_LEFT:  arrows[2] = true; break;
                    case SDLK_RIGHT: arrows[3] = true; break;
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
                    case SDLK_m:
                        if (display_mode == DisplayMode::DEPTH)
                            display_mode = DisplayMode::GRAY_SCALE;
                        else if (display_mode == DisplayMode::GRAY_SCALE)
                            display_mode = DisplayMode::HEAT_MAP;
                        else if (display_mode == DisplayMode::HEAT_MAP)
                            display_mode = DisplayMode::DEPTH;
                        break;
                    case SDLK_ESCAPE:
                        return true;
                }
                break;
        }
    }

    if (arrows[0]) view.eye = view.eye + view.tspeed * view.forward;
    if (arrows[1]) view.eye = view.eye - view.tspeed * view.forward;
    if (arrows[2]) view.eye = view.eye - view.tspeed * view.right;
    if (arrows[3]) view.eye = view.eye + view.tspeed * view.right;

    return false;
}

static void usage() {
    std::cout << "Usage: hagrid [options] file\n"
                 "Options:\n"
                 "  -h      --help          Shows this message\n"
                 "  -sx     --width         Sets the viewport width\n"
                 "  -sy     --height        Sets the viewport height\n"
                 "  -c      --clip          Sets the clipping distance\n"
                 "  -f      --fov           Sets the field of view\n"
                 " Construction parameters:\n"
                 "  -td     --top-density   Sets the top-level density\n"
                 "  -sd     --snd-density   Sets the second-level density\n"
                 "  -a      --alpha         Sets the cell merging threshold\n"
                 "  -e      --expansion     Sets the number of expansion iterations\n"
                 "  -nb     --build-iter    Sets the number of build iterations\n"
                 "  -wb     --build-warmup  Sets the number of warmup build iterations\n"
                 "  -k      --keep-alive    Keep the buffers alive during construction\n"
                 " Benchmarking:\n"
                 "  -r      --ray-file      Loads rays from a file and enters benchmark mode\n"
                 "  -tmin   --tmin          Sets the minimum distance along every ray\n"
                 "  -tmax   --tmax          Sets the maximum distance along every ray\n"
                 "  -n      --bench-iter    Sets the number of benchmarking iterations\n"
                 "  -w      --bench-warmup  Sets the number of benchmarking warmup iterations\n" << std::endl;
}

static bool benchmark(MemManager& mem,
                      const Grid& grid,
                      const Tri* tris,
                      const std::string& ray_file,
                      float tmin, float tmax,
                      int iter, int warmup) {
    std::vector<Ray> host_rays;
    if (!load_rays(ray_file, host_rays, tmin, tmax)) {
        std::cerr << "Cannot load ray file" << std::endl;
        return false;
    }

    Ray* rays = mem.alloc<Ray>(host_rays.size());
    Hit* hits = mem.alloc<Hit>(host_rays.size());
    mem.copy<Copy::HST_TO_DEV>(rays, host_rays.data(), host_rays.size());

    for (int i = 0; i < warmup; i++) {
        traverse_grid(grid, tris, rays, hits, host_rays.size());
    }

    // Benchmark traversal speed
    std::vector<double> timings;
    for (int i = 0; i < iter; i++) {
        auto kernel_time = profile([&] {
            traverse_grid(grid, tris, rays, hits, host_rays.size());
        });
        timings.emplace_back(kernel_time);
    }

    std::vector<Hit> host_hits(host_rays.size());
    mem.copy<Copy::DEV_TO_HST>(host_hits.data(), hits, host_hits.size());

    int intr = 0;
    for (int i = 0; i < host_rays.size(); i++)
        intr += (host_hits[i].id >= 0);

    std::sort(timings.begin(), timings.end());
    const double sum = std::accumulate(timings.begin(), timings.end(), 0.0f);
    const double avg = sum / timings.size();
    const double med = timings[timings.size() / 2];
    const double min = *std::min_element(timings.begin(), timings.end());
    std::cout << intr << " intersection(s)." << std::endl;
    std::cout << sum << "ms for " << iter << " iteration(s)." << std::endl;
    std::cout << host_rays.size() * iter / (1000.0 * sum) << " Mrays/sec." << std::endl;
    std::cout << "# Average: " << avg << " ms" << std::endl;
    std::cout << "# Median: " << med  << " ms" << std::endl;
    std::cout << "# Min: " << min << " ms" << std::endl;

    return true;
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

    std::vector<Tri> host_tris;
    if (!load_model(opts.scene_file, host_tris)) {
        std::cerr << "Scene cannot be loaded (file not present or contains errors)" << std::endl;
        return 1;
    }

    std::cout << host_tris.size() << " triangle(s)" << std::endl;

    MemManager mem(opts.keep_alive);
    auto tris = mem.alloc<Tri>(host_tris.size());
    mem.copy<Copy::HST_TO_DEV>(tris, host_tris.data(), host_tris.size());

    Grid grid;
    grid.entries = nullptr;
    grid.cells   = nullptr;
    grid.ref_ids = nullptr;

    // Warmup iterations
    for (int i = 0; i < opts.build_warmup; i++) {
        mem.free(grid.entries);
        mem.free(grid.cells);
        mem.free(grid.ref_ids);

        build_grid(mem, tris, host_tris.size(), grid, opts.top_density, opts.snd_density);
        merge_grid(mem, grid, opts.alpha);
        flatten_grid(mem, grid);
        expand_grid(mem, grid, tris, opts.exp_iters);
    }

    // Benchmark construction speed
    double total_time = 0;
    for (int i = 0; i < opts.build_iter; i++) {
        mem.free(grid.entries);
        mem.free(grid.cells);
        mem.free(grid.ref_ids);

        auto kernel_time = profile([&] {
            build_grid(mem, tris, host_tris.size(), grid, opts.top_density, opts.snd_density);
            merge_grid(mem, grid, opts.alpha);
            flatten_grid(mem, grid);
            expand_grid(mem, grid, tris, opts.exp_iters);
        });
        total_time += kernel_time;
    }
    auto dims = grid.dims << grid.shift;
    std::cout << "Grid built in " << total_time / opts.build_iter << " ms ("
              << dims.x << "x" << dims.y << "x" << dims.z << ", "
              << grid.num_cells << " cells, " << grid.num_refs << " references)" << std::endl;

#ifndef NDEBUG
    std::cout << std::endl;
    mem.debug_slots();
    std::cout << std::endl;
#endif

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
    std::cout << "Peak usage: " << mem.max_usage() / double(1024.0 * 1024.0) << " MB" << std::endl;

    setup_traversal(grid);

    // Compute a clipping distance from the bounding box of the scene
    auto scene_size = length(grid.bbox.extents());
    auto scene_center = grid.bbox.center();
    if (opts.clip <= 0) {
        opts.clip = scene_size;
    }

    if (opts.ray_file != "") {
        std::cout << "Entering benchmark mode" << std::endl;
        if (!benchmark(mem, grid, tris, opts.ray_file, opts.tmin, opts.tmax, opts.bench_iter, opts.bench_warmup))
            return 1;
        return 0;
    }

    std::cout << "Entering interactive mode\n"
                 "Commands:\n"
                 "  Mouse, arrow keys  Move the camera\n"
                 "  Numpad '+'/'-'     Control camera movement speed\n"
                 "  'm'                Cycle through display modes\n"
                 "  'c'                Prints the camera position" << std::endl;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Cannot initialize SDL" << std::endl;
        return 1;
    }

    SDL_Window* win = SDL_CreateWindow("HaGrid",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        opts.width, opts.height,
        0);

    SDL_Surface* screen = SDL_GetWindowSurface(win);

    SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);

    View view = {
        scene_center,                // Eye
        vec3(0.0f,  0.0f, 1.0f),     // Forward
        vec3(-1.0f, 0.0f, 0.0f),     // Right
        vec3(0.0f,  1.0f, 0.0f),     // Up
        100.0f, 0.005f,              // View distance, rotation speed
        scene_size * 0.005f          // Translation speed
    };

    size_t num_rays = opts.width * opts.height;
    std::vector<Hit> host_hits(num_rays);
    std::vector<Ray> host_rays(num_rays);
    Ray* rays = mem.alloc<Ray>(num_rays);
    Hit* hits = mem.alloc<Hit>(num_rays);
    double kernel_time = 0;
    auto ticks = SDL_GetTicks();
    int frames = 0;
    DisplayMode display_mode = DisplayMode::DEPTH;
    bool done = false;
    while (!done) {
        Camera cam = gen_camera(view.eye,
                                view.eye + view.forward * view.dist,
                                view.up,
                                opts.fov,
                                (float)opts.width / (float)opts.height);

        gen_rays(cam, host_rays, opts.clip, opts.width, opts.height);
        mem.copy<Copy::HST_TO_DEV>(rays, host_rays.data(), num_rays);

        kernel_time += profile([&] { traverse_grid(grid, tris, rays, hits, num_rays); });
        frames++;

        if (SDL_GetTicks() - ticks >= 2000) {
            std::ostringstream caption;
            caption << "HaGrid [" << double(frames) * double(opts.width * opts.height) / (1000 * kernel_time) << " MRays/s]";
            SDL_SetWindowTitle(win, caption.str().c_str());
            ticks = SDL_GetTicks();
            kernel_time = 0;
            frames = 0;
        }

        mem.copy<Copy::DEV_TO_HST>(host_hits.data(), hits, num_rays);
        SDL_LockSurface(screen);
        if (display_mode == DisplayMode::DEPTH)
            update_surface<DisplayMode::DEPTH>(screen, host_hits, opts.clip, opts.width, opts.height);
        else if (display_mode == DisplayMode::GRAY_SCALE)
            update_surface<DisplayMode::GRAY_SCALE>(screen, host_hits, opts.clip, opts.width, opts.height);
        else
            update_surface<DisplayMode::HEAT_MAP>(screen, host_hits, opts.clip, opts.width, opts.height);
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

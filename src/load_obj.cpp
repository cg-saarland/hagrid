#include <fstream>
#include <iostream> 
#include <cstring>
#include <cstdlib>

#include "load_obj.h"

namespace hagrid {

inline void error() {
    std::cerr << std::endl;
}

template <typename T, typename... Args>
inline void error(T t, Args... args) {
#ifndef NDEBUG
    std::cerr << t;
    error(args...);
#endif
}

inline void remove_eol(char* ptr) {
    int i = 0;
    while (ptr[i]) i++;
    i--;
    while (i > 0 && std::isspace(ptr[i])) {
        ptr[i] = '\0';
        i--;
    }
}

inline char* strip_text(char* ptr) {
    while (*ptr && !std::isspace(*ptr)) { ptr++; }
    return ptr;
}

inline char* strip_spaces(char* ptr) {
    while (std::isspace(*ptr)) { ptr++; }
    return ptr;
}

inline bool read_index(char** ptr, ObjLoader::Index& idx) {
    char* base = *ptr;

    // Detect end of line (negative indices are supported) 
    base = strip_spaces(base);
    if (!std::isdigit(*base) && *base != '-') return false;

    idx.v = 0;
    idx.t = 0;
    idx.n = 0;

    idx.v = std::strtol(base, &base, 10);

    base = strip_spaces(base);

    if (*base == '/') {
        base++;

        // Handle the case when there is no texture coordinate
        if (*base != '/') {
            idx.t = std::strtol(base, &base, 10);
        }

        base = strip_spaces(base);

        if (*base == '/') {
            base++;
            idx.n = std::strtol(base, &base, 10);
        }
    }

    *ptr = base;

    return true;
}

bool ObjLoader::load_obj(const std::string& path, File& file) {
    std::ifstream stream(path);
    if (!stream) return false;

    // Add an empty object to the scene
    int cur_object = 0;
    file.objects.emplace_back();

    // Add an empty group to this object
    int cur_group = 0;
    file.objects[0].groups.emplace_back();

    // Add an empty material to the scene
    int cur_mtl = 0;
    file.materials.emplace_back("");

    // Add dummy vertex, normal, and texcoord
    file.vertices.emplace_back();
    file.normals.emplace_back();
    file.texcoords.emplace_back();

    int err_count = 0;
    const int max_line = 1024;
    char line[max_line];
    while (stream.getline(line, max_line)) {
        // Strip spaces
        char* ptr = strip_spaces(line);
        const char* err_line = ptr;

        // Skip comments and empty lines
        if (*ptr == '\0' || *ptr == '#')
            continue;

        remove_eol(ptr);

        // Test each command in turn, the most frequent first
        if (*ptr == 'v') {
            switch (ptr[1]) {
                case ' ':
                case '\t':
                    {
                        vec3 v;
                        v.x = std::strtof(ptr + 1, &ptr);
                        v.y = std::strtof(ptr, &ptr);
                        v.z = std::strtof(ptr, &ptr);
                        file.vertices.push_back(v);
                    }
                    break;
                case 'n':
#ifndef SKIP_NORMALS
                    {
                        vec3 n;
                        n.x = std::strtof(ptr + 2, &ptr);
                        n.y = std::strtof(ptr, &ptr);
                        n.z = std::strtof(ptr, &ptr);
                        file.normals.push_back(n);
                    }
#endif
                    break;
                case 't':
#ifndef SKIP_TEXCOORDS
                    {
                        vec2 t;
                        t.x = std::strtof(ptr + 2, &ptr);
                        t.y = std::strtof(ptr, &ptr);
                        file.texcoords.push_back(t);
                    }
#endif
                    break;
                default:
                    error("invalid vertex");
                    err_count++;
                    break;
            }
        } else if (*ptr == 'f' && std::isspace(ptr[1])) {
            Face f;

            f.index_count = 0;
            f.material = cur_mtl;

            bool valid = true;
            ptr += 2;
            while(f.index_count < Face::max_indices) {
                Index index;
                valid = read_index(&ptr, index);

                if (valid) {
                    f.indices[f.index_count++] = index;
                } else {
                    break;
                }
            }

            if (f.index_count < 3) {
                error("invalid face");
                err_count++;
            } else {
                // Convert relative indices to absolute
                for (int i = 0; i < f.index_count; i++) {
                    f.indices[i].v = (f.indices[i].v < 0) ? file.vertices.size()  + f.indices[i].v : f.indices[i].v;
                    f.indices[i].t = (f.indices[i].t < 0) ? file.texcoords.size() + f.indices[i].t : f.indices[i].t;
                    f.indices[i].n = (f.indices[i].n < 0) ? file.normals.size()   + f.indices[i].n : f.indices[i].n;
                }

                // Check if the indices are valid or not
                valid = true;
                for (int i = 0; i < f.index_count; i++) {
                    if (f.indices[i].v <= 0 || f.indices[i].t < 0 || f.indices[i].n < 0) {
                        valid = false;
                        break;
                    }
                }

                if (valid) {
                    file.objects[cur_object].groups[cur_group].faces.push_back(f);
                } else {
                    error("invalid indices");
                    err_count++;
                }
            }
        } else if (*ptr == 'g' && std::isspace(ptr[1])) {
            file.objects[cur_object].groups.emplace_back();
            cur_group++;
        } else if (*ptr == 'o' && std::isspace(ptr[1])) {
            file.objects.emplace_back();
            cur_object++;

            file.objects[cur_object].groups.emplace_back();
            cur_group = 0;
        } else if (!std::strncmp(ptr, "usemtl", 6) && std::isspace(ptr[6])) {
            ptr += 6;

            ptr = strip_spaces(ptr);
            char* base = ptr;
            ptr = strip_text(ptr);

            const std::string mtl_name(base, ptr);

            cur_mtl = std::find(file.materials.begin(), file.materials.end(), mtl_name) - file.materials.begin();
            if (cur_mtl == (int)file.materials.size()) {
                file.materials.push_back(mtl_name);            
            }
        } else if (!std::strncmp(ptr, "mtllib", 6) && std::isspace(ptr[6])) {
            ptr += 6;

            ptr = strip_spaces(ptr);
            char* base = ptr;
            ptr = strip_text(ptr);

            const std::string lib_name(base, ptr);

            file.mtl_libs.push_back(lib_name);
        } else if (*ptr == 's' && std::isspace(ptr[1])) {
            // Ignore smooth commands
        } else {
            error("unknown command ", ptr);
            err_count++;
        }
    }

    return (err_count == 0);
}

bool ObjLoader::load_mtl(const std::string& path, MaterialLib& mtl_lib) {
    std::ifstream stream(path);
    if (!stream) return false;

    const int max_line = 1024;
    char line[max_line];
    char* err_line = line;
    int err_count = 0;

    std::string mtl_name;
    auto current_material = [&] () -> Material& {
        return mtl_lib[mtl_name];
    };

    while (stream.getline(line, max_line)) {
        // Strip spaces
        char* ptr = strip_spaces(line);
        err_line = ptr;

        // Skip comments and empty lines
        if (*ptr == '\0' || *ptr == '#')
            continue;

        remove_eol(ptr);

        if (!std::strncmp(ptr, "newmtl", 6) && std::isspace(ptr[6])) {
            ptr = strip_spaces(ptr + 7);
            char* base = ptr;
            ptr = strip_text(ptr);

            mtl_name = std::string(base, ptr);
            if (mtl_lib.find(mtl_name) != mtl_lib.end()) {
                error("material redefinition");
                err_count++;
            }
        } else if (ptr[0] == 'K') {
            if (ptr[1] == 'a' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.ka.r = std::strtof(ptr + 3, &ptr);
                mat.ka.g = std::strtof(ptr, &ptr);
                mat.ka.b = std::strtof(ptr, &ptr);
            } else if (ptr[1] == 'd' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.kd.r = std::strtof(ptr + 3, &ptr);
                mat.kd.g = std::strtof(ptr, &ptr);
                mat.kd.b = std::strtof(ptr, &ptr);
            } else if (ptr[1] == 's' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.ks.r = std::strtof(ptr + 3, &ptr);
                mat.ks.g = std::strtof(ptr, &ptr);
                mat.ks.b = std::strtof(ptr, &ptr);
            } else if (ptr[1] == 'e' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.ke.r = std::strtof(ptr + 3, &ptr);
                mat.ke.g = std::strtof(ptr, &ptr);
                mat.ke.b = std::strtof(ptr, &ptr);
            } else {
                error("invalid command");
                err_count++;
            }
        } else if (ptr[0] == 'N') {
            if (ptr[1] == 's' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.ns = std::strtof(ptr + 3, &ptr);
            } else if (ptr[1] == 'i' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.ni = std::strtof(ptr + 3, &ptr);
            } else {
                error("invalid command");
                err_count++;
            }
        } else if (ptr[0] == 'T') {
            if (ptr[1] == 'f' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.tf.r = std::strtof(ptr + 3, &ptr);
                mat.tf.g = std::strtof(ptr, &ptr);
                mat.tf.b = std::strtof(ptr, &ptr);
            } else if (ptr[1] == 'r' && std::isspace(ptr[2])) {
                auto& mat = current_material();
                mat.tr = std::strtof(ptr + 3, &ptr);
            } else {
                error("invalid command");
                err_count++;
            }
        } else if (ptr[0] == 'd' && std::isspace(ptr[1])) {
            auto& mat = current_material();
            mat.d = std::strtof(ptr + 2, &ptr);
        } else if (!std::strncmp(ptr, "illum", 5) && std::isspace(ptr[5])) {
            auto& mat = current_material();
            mat.illum = std::strtof(ptr + 6, &ptr);
        } else if (!std::strncmp(ptr, "map_Ka", 6) && std::isspace(ptr[6])) {
            auto& mat = current_material();
            mat.map_ka = std::string(strip_spaces(ptr + 7));
        } else if (!std::strncmp(ptr, "map_Kd", 6) && std::isspace(ptr[6])) {
            auto& mat = current_material();
            mat.map_kd = std::string(strip_spaces(ptr + 7));
        } else if (!std::strncmp(ptr, "map_Ks", 6) && std::isspace(ptr[6])) {
            auto& mat = current_material();
            mat.map_ks = std::string(strip_spaces(ptr + 7));
        } else if (!std::strncmp(ptr, "map_Ke", 6) && std::isspace(ptr[6])) {
            auto& mat = current_material();
            mat.map_ke = std::string(strip_spaces(ptr + 7));
        } else if (!std::strncmp(ptr, "map_bump", 8) && std::isspace(ptr[8])) {
            auto& mat = current_material();
            mat.map_bump = std::string(strip_spaces(ptr + 9));
        } else if (!std::strncmp(ptr, "bump", 4) && std::isspace(ptr[4])) {
            auto& mat = current_material();
            mat.map_bump = std::string(strip_spaces(ptr + 5));
        } else if (!std::strncmp(ptr, "map_d", 5) && std::isspace(ptr[5])) {
            auto& mat = current_material();
            mat.map_d = std::string(strip_spaces(ptr + 6));
        } else {
            error("unknown command ", ptr);
            err_count++;
        }
    }

    return (err_count == 0);
}

} // namespace hagrid

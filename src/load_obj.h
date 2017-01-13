#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "vec.h"

namespace hagrid {

class ObjLoader {
public:
    struct Index {
        int v, n, t;
    };

    struct Face {
        static constexpr int max_indices = 8;
        Index indices[max_indices];
        int index_count;
        int material;
    };

    struct Group {
        std::vector<Face> faces;
    };

    struct Object {
        std::vector<Group> groups;
    };

    struct Material {
        vec3 ka;
        vec3 kd;
        vec3 ks;
        vec3 ke;
        float ns;
        float ni;
        vec3 tf;
        float tr;
        float d;
        int illum;
        std::string map_ka;
        std::string map_kd;
        std::string map_ks;
        std::string map_ke;
        std::string map_bump;
        std::string map_d;
    };

    struct File {
        std::vector<Object>      objects;
        std::vector<vec3>        vertices;
        std::vector<vec3>        normals;
        std::vector<vec2>        texcoords;
        std::vector<std::string> materials;
        std::vector<std::string> mtl_libs;
    };

    struct Path {
        Path() {}
        Path(const char* p) : Path(std::string(p)) {}
        Path(const std::string& p)
            : path(p)
        {
            std::replace(path.begin(), path.end(), '\\', '/');
            auto pos = path.rfind('/');
            base = (pos != std::string::npos) ? path.substr(0, pos)  : ".";
            file = (pos != std::string::npos) ? path.substr(pos + 1) : path;
        }

        operator const std::string& () const {
            return path;
        }
        
        std::string path;
        std::string base;
        std::string file;
    };

    typedef std::unordered_map<std::string, Material> MaterialLib;

    static bool load_obj(const std::string&, File&);
    static bool load_mtl(const std::string&, MaterialLib&);
    static bool load_scene(const Path& path, File& file, MaterialLib& mtl_lib) {
        if (!load_obj(path, file)) return false;
        for (auto& lib : file.mtl_libs) {
            // We tolerate errors in the MTL file
            load_mtl(path.base + "/" + lib, mtl_lib);
        }
        return true;
    }
};

} // namespace hagrid

#endif // LOAD_OBJ_H

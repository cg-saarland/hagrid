#ifndef BUILD_H
#define BUILD_H

#include "mem_manager.h"
#include "prims.h"
#include "grid.h"

namespace hagrid {

/// Construction parameters
struct BuildParams {
    float top_density;          ///< Top-level density
    float snd_density;          ///< Second-level density

    /// Check the validity of the construction parameters
    bool valid() const {
        return top_density >  0 && snd_density >= 0;
    }

    /// Default construction parameters for a static scene
    static constexpr BuildParams static_scene() {
        return BuildParams{0.12f, 2.4f};
    }

    /// Default construction parameters for a dynamic scene
    static constexpr BuildParams dynamic_scene() {
        return BuildParams{0.06f, 1.2f};
    }
};

void build_grid(MemManager& mem, const BuildParams& params, const Tri* tris, int num_tris, Grid& grid);
void merge_grid(MemManager& mem, Grid& grid, float alpha);

} // namespace hagrid

#endif // BUILD_H

#ifndef BUILD_H
#define BUILD_H

#include "primitives.h"
#include "mem_manager.h"
#include "grid.h"

namespace hagrid {

/// Construction parameters
struct BuildParams {
    float top_density;          ///< Top-level density
    float snd_density;          ///< Second-level density
    float alpha;                ///< Beween [0, 1], controls the merge phase
    int expansion;              ///< Number of expansion passes
    int level_shift;            ///< Number of octree levels per subdivision iteration

    /// Check the validity of the construction parameters
    bool valid() const {
        return top_density >  0 &&
               snd_density >= 0 &&
               alpha >= 0 &&
               expansion >= 0 &&
               level_shift >= 1 && level_shift < (1 << Entry::LOG_DIM_BITS);
    }

    /// Default construction parameters for a static scene
    static constexpr BuildParams static_scene() {
        return BuildParams{0.12f, 2.4f, 0.995f, 3, 3};
    }

    /// Default construction parameters for a dynamic scene
    static constexpr BuildParams dynamic_scene() {
        return BuildParams{0.06f, 1.2f, 0.995f, 1, 3};
    }
};

void build_grid(MemManager& mem, const BuildParams& params, const Tri* tris, int num_tris, Grid& grid);

} // namespace hagrid

#endif // BUILD_H

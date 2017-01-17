#ifndef TRAVERSE_H
#define TRAVERSE_H

#include "grid.h"
#include "vec.h"
#include "prims.h"

namespace hagrid {

/// Setups the traversal constants
void setup_traversal(const Grid& grid);

/// Traverses the structure with the given set of rays
void traverse(const Grid& grid, const Tri* tris, const Ray* rays, Hit* hits, int num_rays);

} // namespace hagrid

#endif // TRAVERSE_H

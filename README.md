# Hagrid

![Screenshot](screenshot.png)

This project is an implementation of the paper: _GPU Ray Tracing using Irregular Grids_.
This is not the version that has been used in the paper.

## Changes

Some improvements have been made to the construction algorithm, which change the performance characteristics of the structure:

- The voxel map can have more than two levels.
- Construction is faster (~ +33%)
- Memory consumption is lower (~ -20%)
- Traversal is slower (~ -5%)

The traversal being slower can easily being remedied by increasing the resolution.
The improvements in build times and memory consumption more than compensate the loss in traversal performance.
Another option is to use the slower but more precise expansion algorithm, which can be enabled in [src/expand.cu](src/expand.cu) by setting the `subset_only` variable to false.

## Building

This project requires CUDA, SDL2, and CMake. Use the following commands to build the project:

    mkdir build
    cd build
    cmake-gui ..
    make -j

## License

The code is distributed under the MIT license (see [LICENSE.txt](LICENSE.txt)).

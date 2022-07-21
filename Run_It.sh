cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

mpiexec -n 6 build/Integrate
#!/bin/bash -e
echo "Configuring and building ..."
rm -rf build
mkdir build
cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
cmake .. -DCMAKE_BUILD_TYPE=Debug

make -j12

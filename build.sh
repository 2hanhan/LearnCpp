#!/bin/bash -e
echo "configuring and building Thirdparty/Dbow2"

cd Thirdparty/DBoW2
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12

cd ../../../

echo "Configuring and building ..."
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
#cmake .. -DCMAKE_BUILD_TYPE=Debug

make -j12

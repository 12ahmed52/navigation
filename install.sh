#!/bin/bash
set -e
###########################################################################
## install dependencies: BLASFEO, HPIPM, Eigen, JSON, matplotlib-cpp
###########################################################################

BASE_DIR=$(pwd)

export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Clone BLASFEO
repository_blasfeo="https://github.com/giaf/blasfeo.git"
localFolder_blasfeo="$BASE_DIR/src/third_party_libs/blasfeo"
if [ ! -d "$localFolder_blasfeo/.git" ]; then
    git clone "$repository_blasfeo" "$localFolder_blasfeo"
    cd "$localFolder_blasfeo"
    git checkout 0.1.1
fi

# Clone HPIPM
repository_hpipm="https://github.com/giaf/hpipm.git"
localFolder_hpipm="$BASE_DIR/src/third_party_libs/hpipm"
if [ ! -d "$localFolder_hpipm/.git" ]; then
    git clone "$repository_hpipm" "$localFolder_hpipm"
    cd "$localFolder_hpipm"
    git checkout 0.1.1
fi

# Clone matplotlib-cpp
repository_matplotlib="https://github.com/lava/matplotlib-cpp.git"
localFolder_matplotlib="$BASE_DIR/src/third_party_libs/matplotlib"
[ ! -d "$localFolder_matplotlib/.git" ] && git clone "$repository_matplotlib" "$localFolder_matplotlib"

# Clone Eigen
repository_eigen="https://gitlab.com/libeigen/eigen.git"
localFolder_eigen="$BASE_DIR/src/third_party_libs/Eigen"
[ ! -d "$localFolder_eigen/.git" ] && git clone "$repository_eigen" "$localFolder_eigen"

# Clone JSON
repository_json="https://github.com/nlohmann/json.git"
localFolder_json="$BASE_DIR/src/third_party_libs/Json"
[ ! -d "$localFolder_json/.git" ] && git clone "$repository_json" "$localFolder_json"

#########################################
## Build BLASFEO (shared)
#########################################
cd "$localFolder_blasfeo"
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../lib -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
make install

#########################################
## Build HPIPM (shared)
#########################################
cd "$localFolder_hpipm"
mkdir -p build lib
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../lib -DBUILD_SHARED_LIBS=ON -DBLASFEO_PATH=$localFolder_blasfeo/lib
make -j$(nproc)
make install

cd "$BASE_DIR"

mkdir -p build
cd build 
cmake ..
make 
make install

cd "$BASE_DIR"
cd src/ros_apps
catkin build

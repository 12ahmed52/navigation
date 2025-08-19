#!/bin/bash
# setup.bash for navigation_mpcc environment with relative paths

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Relative library paths
export HPIPM_PATH="$SCRIPT_DIR/../third_party_libs/hpipm/lib/lib"
export BLASFEO_PATH="$SCRIPT_DIR/../third_party_libs/blasfeo/lib/lib"
export NAV_MPCC_PATH="$SCRIPT_DIR/../install/libs/navigation_libs/navigation_mpcc/libs"

# Add all to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HPIPM_PATH:$BLASFEO_PATH:$NAV_MPCC_PATH:$LD_LIBRARY_PATH"

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Source your catkin workspace (relative to this script)
source "$SCRIPT_DIR/devel/setup.bash"

echo "navigation_mpcc environment setup completed."
echo "HPIPM_PATH   = $HPIPM_PATH"
echo "BLASFEO_PATH = $BLASFEO_PATH"
echo "NAV_MPCC_PATH= $NAV_MPCC_PATH"

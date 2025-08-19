import csv
import math
import numpy as np

def quaternion_to_yaw(x, y, z, w):
    # yaw (z-axis rotation) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def map_to_local_velocity(vx_map, vy_map, yaw):
    # Rotate velocity vector by -yaw to get local frame velocities
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    vx_local = vx_map * cos_yaw - vy_map * sin_yaw
    vy_local = vx_map * sin_yaw + vy_map * cos_yaw
    return vx_local, vy_local

def convert_csv(input_csv, output_csv):
    with open(input_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['yaw', 'vx_local', 'vy_local']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Parse quaternion components
            qx = float(row['quat_x'])
            qy = float(row['quat_y'])
            qz = float(row['quat_z'])
            qw = float(row['quat_w'])

            # Calculate yaw
            yaw = quaternion_to_yaw(qx, qy, qz, qw)

            # Parse map frame velocities
            vx_map = float(row['vx'])
            vy_map = float(row['vy'])

            # Convert velocities to local frame
            vx_local, vy_local = map_to_local_velocity(vx_map, vy_map, yaw)

            # Append new columns
            row['yaw'] = yaw
            row['vx_local'] = vx_local
            row['vy_local'] = vy_local

            writer.writerow(row)

if __name__ == '__main__':
    input_csv = ''
    output_csv = ''
    convert_csv(input_csv, output_csv)

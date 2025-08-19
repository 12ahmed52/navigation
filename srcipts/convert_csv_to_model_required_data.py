import csv

INPUT_CSV = ''
OUTPUT_CSV = ''

MAX_SPEED = 15.0  # max speed for throttle %

header = [
    "time(s)", "x(m)", "y(m)", "vx(m/s)", "vy(m/s)", "phi(rad)", "delta(rad)", "omega(rad/s)",
    "ax(m/s^2)", "deltadelta(rad/s)", "wheel_fl(kmph)", "wheel_fr(kmph)", "wheel_rl(kmph)", "wheel_rr(kmph)",
    "roll(rad)", "throttle_ped_cmd(%)", "brake_ped_cmd(kPa)", "gear"
]

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def convert_csv(input_file, output_file):
    with open(input_file, newline='') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()

        prev_steering = None
        prev_time = None
        prev_delta = None

        for row in reader:
            # time not available, set 0 or increment if you want
            time_s = 0.0

            x = float(row['x'])
            y = float(row['y'])

            vx = float(row['vx'])
            vy = float(row['vy'])

            phi = float(row['yaw'])

            delta = float(row['input_steering_command'])
            omega = float(row['omega'])

            # accel not available, set 0
            ax = 0.0

            # deltadelta (rate of change of steering)
            # Since no timestamp, or time is 0, set zero or compute if you add time logic
            deltadelta = 0.0
            # Could implement if time info available

            # wheel speeds kmph unavailable → 0
            wheel_fl = 0.0
            wheel_fr = 0.0
            wheel_rl = 0.0
            wheel_rr = 0.0

            roll = 0.0

            # throttle ped cmd (%) = input_velocity_command / 15 * 100, clamped
            input_velocity_command = float(row['input_velocity_command'])
            throttle_ped_cmd = clamp((input_velocity_command / MAX_SPEED) * 100, 0, 100)

            brake_ped_cmd = 0.0  # no brake info

            gear = 1  # default gear

            out_row = {
                "time(s)": time_s,
                "x(m)": x,
                "y(m)": y,
                "vx(m/s)": vx,
                "vy(m/s)": vy,
                "phi(rad)": phi,
                "delta(rad)": delta,
                "omega(rad/s)": omega,
                "ax(m/s^2)": ax,
                "deltadelta(rad/s)": deltadelta,
                "wheel_fl(kmph)": wheel_fl,
                "wheel_fr(kmph)": wheel_fr,
                "wheel_rl(kmph)": wheel_rl,
                "wheel_rr(kmph)": wheel_rr,
                "roll(rad)": roll,
                "throttle_ped_cmd(%)": throttle_ped_cmd,
                "brake_ped_cmd(kPa)": brake_ped_cmd,
                "gear": gear,
            }
            writer.writerow(out_row)

if __name__ == "__main__":
    convert_csv(INPUT_CSV, OUTPUT_CSV)
    print(f"Conversion done. Output saved to {OUTPUT_CSV}")

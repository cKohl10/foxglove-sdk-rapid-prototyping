import foxglove
import argparse
import logging
import numpy as np
import os
import cv2
from foxglove.schemas import FrameTransform, Vector3, Quaternion, FrameTransforms

from level_1_with_calib import Camera
from level_2 import Robot

def preload_joint_data(robot, transforms_path, num_links):

    joint_data = np.load(f"{transforms_path}/joint.npy", allow_pickle=True).item()
    gripper_data = np.load(f"{transforms_path}/gripper.npy", allow_pickle=True).item()
    # Get the first serial number's data only
    first_serial = list(joint_data.keys())[0]
    joint_data = joint_data[first_serial]
    gripper_data = gripper_data[first_serial]

    # Joint angles, {serial number: {timestamp: joint angle array}}
    for timestamp, data in joint_data.items():
        joint_angles = data[:num_links]
        joint_torques = data[num_links:] # RH20T split
        gripper_width = gripper_data[timestamp]
        gripper_width = gripper_width["gripper_info"][0] * 0.0005 #[0-110mm] scaled to the range of the gripper
        timestamp = int(timestamp * 1e6)  # convert ms to nsec
        robot.update(np.concatenate((joint_angles[:7], [gripper_width, gripper_width])), timestamp)
        robot.update_torque(joint_torques, timestamp)

    print(f"Preloaded joint data for {robot.name}")

def preload_camera_data(cam, calib_path, img_dir):
    intrinsics = np.load(f"{calib_path}/intrinsics.npy", allow_pickle=True)
    extrinsics = np.load(f"{calib_path}/extrinsics.npy", allow_pickle=True)
    P = intrinsics.item().get(cam.idx)
    T_cam_world = extrinsics.item().get(cam.idx)[0]

    print(f"\n --- Loading camera data for {cam.name} ---")
    print(f"P: {P}")
    print(f"T_cam_world: {T_cam_world}")

    # Get timestep and data of first image to calibrate camera
    img_files = sorted(os.listdir(img_dir))  # Sort filenames to get chronological order
    first_img = img_files[0]
    timestamp_ms = int(first_img.split(".")[0])
    start_timestamp_nsec = int(timestamp_ms * 1e6)
    frame = cv2.imread(os.path.join(img_dir, first_img))
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    cam.calibrate_mono_camera(P, T_cam_world, frame_width, frame_height, start_timestamp_nsec)
    cam.load_imgs(img_dir)

if __name__ == "__main__":

    robot_name = "franka"

    parser = argparse.ArgumentParser(description='Convert RH20T dataset to MCAP')
    parser.add_argument('-t', '--task_name', type=str, default=f"data/{robot_name}/task_0111", help='Task directory')
    args = parser.parse_args()

    foxglove.set_log_level(logging.INFO)

    writer = foxglove.open_mcap(args.task_name+"/data.mcap", allow_overwrite=True)

    cam_SN = np.load(f"data/{robot_name}/calib/devices.npy", allow_pickle=True)
    cams = [Camera(cam_SN[i], live=False) for i in range(len(cam_SN)-1)]

    for cam in cams:
        try:
            preload_camera_data(cam, f"data/{robot_name}/calib", f"{args.task_name}/{cam.name}/color")
        except Exception as e:
            print(f"Error preloading camera data for {cam.name}: {e}")

    robot_transform = [-0.03, -0.58, -0.021] #X, Y, Z
    robot_rotation = [-np.pi/2, np.pi, 0.0] #roll, pitch, yaw
    panda = Robot("urdf/fp3.urdf", "fp3", robot_transform, robot_rotation)
    preload_joint_data(panda, f"{args.task_name}/transformed", 7)

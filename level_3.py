import foxglove
import argparse
import logging
import cv2
import time
import os
import numpy as np
import pinocchio as pin
from transforms3d.quaternions import mat2quat
from foxglove import Channel
from foxglove.channels import RawImageChannel, CameraCalibrationChannel
from foxglove.schemas import RawImage, Timestamp, FrameTransforms, FrameTransform, Vector3, Quaternion, CameraCalibration

WORLD_FRAME_ID = "world"

class Robot:
    def __init__(self, urdf_path, name, base_translation):
        self.name = name
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        self.base_frame_id = f"{name}_base"
        self.base_translation = base_translation

        # Joint torque channel (optional)
        self.torque_channel = Channel("/joint_torques", message_encoding="json")

    def controller(self):
        """
        The controller returns a list of joint values. This can be mapped to a control interface but a simple oscillation is demoed here.
        """
        angle = np.sin(time.time()/4)
        q =np.array([angle, angle, 0.0, angle/2 - np.pi/2, 0.0, angle/2 + np.pi/2, angle, 0.0, 0.0])

        return q

    def update(self, q, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time()*1e3)
        
        # Always create the timestamp object
        timestamp = Timestamp(
            sec=int(timestamp_ms // 1e3),
            nsec=int((timestamp_ms % 1e3) * 1e6),
        )

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        transforms = []

        # Add transforms for all body frames relative to their parents
        for frame in self.model.frames:
            if frame.type != pin.FrameType.BODY:
                continue

            frame_name = frame.name
            frame_id = self.model.getFrameId(frame_name)
            parent_joint_id = frame.parent

            if parent_joint_id == 0:  # Root joint
                parent_frame_id = WORLD_FRAME_ID
                transforms.append(
                    FrameTransform(
                        timestamp=timestamp,
                        parent_frame_id=parent_frame_id,
                        child_frame_id=frame_name,
                        translation=Vector3(x=self.base_translation[0], y=self.base_translation[1], z=self.base_translation[2]),
                        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    )
                )   
                continue
                
            else:
                # Find the parent link by looking for the body frame of the parent joint
                parent_link_id = parent_joint_id - 1  # Joint IDs are 1-indexed, link IDs are 0-indexed
                parent_frame_id = f"{self.name}_link{parent_link_id}"
            
            if parent_joint_id == 0:
                # Root joint transform
                T_joint = self.model.jointPlacements[parent_joint_id]
            else:
                T_current = self.data.oMf[frame_id]
                parent_frame_name = f"{self.name}_link{parent_link_id}"
                parent_frame_id_in_model = self.model.getFrameId(parent_frame_name)
                T_parent = self.data.oMf[parent_frame_id_in_model]
                
                # Calculate relative transform from parent to current
                T_joint = T_parent.inverse() * T_current
            
            translation = T_joint.translation
            quat = pin.Quaternion(T_joint.rotation).coeffs()

            transforms.append(
                FrameTransform(
                    timestamp=timestamp,
                    parent_frame_id=parent_frame_id,
                    child_frame_id=frame_name,
                    translation=Vector3(x=float(translation[0]),
                                        y=float(translation[1]),
                                        z=float(translation[2])),
                    rotation=Quaternion(x=float(quat[0]), 
                                        y=float(quat[1]), 
                                        z=float(quat[2]), 
                                        w=float(quat[3]))
                )
            )

        foxglove.log(
            "/tf",
            FrameTransforms(transforms=transforms), 
            log_time=int(timestamp_ms*1e6)
        )

    def update_torque(self, joint_torques, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time()*1e3)
        
        torques = {
            "timestamp": timestamp_ms,
            "torques": joint_torques.tolist()
        }
        self.torque_channel.log(torques, log_time=int(timestamp_ms*1e6))

    def preload_joint_data(self, transforms_path, num_links):

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
            gripper_width = gripper_width["gripper_info"][0] / gripper_width["gripper_info"][1] #[0-1] scaled to the range of the gripper
            self.update(np.concatenate((joint_angles[:7], [gripper_width, gripper_width])), timestamp)
            self.update_torque(joint_torques, timestamp)

        print(f"Preloaded joint data for {self.name}")
                

        

class PreloadedCamera:
    def __init__(self, serial_num, calib_dir, task_dir):
        self.serial_num = serial_num
        self.name = f"cam_{serial_num}"

        # Check that camera data exists
        if not os.path.exists(f"{task_dir}/cam_{serial_num}/color"):
            print(f"Camera {serial_num} data not found in {task_dir}")
            return None

        self.channel = RawImageChannel("/"+self.name)
        self.calibration_channel = CameraCalibrationChannel("/"+self.name+"/info")
        self.calib_dir = calib_dir
        self.task_dir = task_dir
        self.img_dir = f"{task_dir}/cam_{serial_num}/color"

        # Get the default frame width and height
        self.frame_width = 640
        self.frame_height = 360

        self.calibrate()
        self.load_imgs()

        print(f"Camera {self.serial_num} loaded into MCAP")

    def load_frame(self, timestamp):
        frame = cv2.imread(f"{self.img_dir}/{timestamp}.jpg")
        if frame is None:
            return None
        
        # Convert milliseconds to seconds and nanoseconds
        sec = int(timestamp // 1e3)
        nsec = int((timestamp % 1e3) * 1e6)
        
        img_msg = RawImage(
            timestamp=Timestamp(
                sec=sec,
                nsec=nsec,
            ),
            frame_id=self.name,
            width=self.frame_width,
            height=self.frame_height,
            encoding="bgr8",
            step=self.frame_width*3,
            data=frame.tobytes(),
        )
        return img_msg

    def load_imgs(self):
        for img_file in os.listdir(self.img_dir):
            timestamp_ms = int(img_file.split(".")[0])
            try:
                img_msg = self.load_frame(timestamp_ms)
                log_time_ns = int(timestamp_ms * 1e6)
                self.channel.log(img_msg, log_time=log_time_ns) # Image message
                camera_info, camera_transform = self.get_camera_info(timestamp_ms)
                self.calibration_channel.log(camera_info, log_time=log_time_ns) #Camera info message
                foxglove.log("/tf", FrameTransforms(transforms=[camera_transform]), log_time=log_time_ns) #Camera transform message
            except Exception as e:
                print(f"Error loading frame {timestamp_ms}: {e}")

    def calibrate(self):
        intrinsics = np.load(self.calib_dir + "/intrinsics.npy", allow_pickle=True)
        extrinsics = np.load(self.calib_dir + "/extrinsics.npy", allow_pickle=True)

        # Intrinsic Calibration
        self.P = intrinsics.item().get(self.serial_num)
        self.K = self.P[:3, :3]

        # Extrinsic Calibration
        self.T_cam_world = np.array(extrinsics.item().get(self.serial_num), dtype=float)

    def get_camera_info(self, timestamp):
        camera_info = CameraCalibration(
            timestamp=Timestamp(
                sec=int(timestamp // 1e3),
                nsec=int((timestamp % 1e3) * 1e6),
            ),
            frame_id=self.name,
            width=self.frame_width,
            height=self.frame_height,
            # distortion_model="plumb_bob",
            # D=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            K=self.K.flatten(),
            P=self.P.flatten(),
        )

        if self.T_cam_world.ndim == 3:
            transform_matrix = self.T_cam_world[0]
        else:
            transform_matrix = self.T_cam_world
            
        translation = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]
        q = mat2quat(rotation_matrix)

        camera_transform = FrameTransform(
            timestamp=Timestamp(
                sec=int(timestamp // 1e3),
                nsec=int((timestamp % 1e3) * 1e6),
            ),
            parent_frame_id=WORLD_FRAME_ID,
            child_frame_id=self.name,
            translation=Vector3(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
            rotation=Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0])),
        )
        return camera_info, camera_transform

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":

    robot_name = "franka"

    parser = argparse.ArgumentParser(description='Convert RH20T dataset to MCAP')
    parser.add_argument('-t', '--task_name', type=str, default=f"data/{robot_name}/task_0111", help='Task directory')
    args = parser.parse_args()

    foxglove.set_log_level(logging.INFO)

    writer = foxglove.open_mcap(args.task_name+"/data.mcap", allow_overwrite=True)

    # cam_SN = np.load(f"data/{robot_name}/calib/devices.npy", allow_pickle=True)
    # cams = [PreloadedCamera(cam_SN[i], f"data/{robot_name}/calib", args.task_name) for i in range(len(cam_SN)-1)]

    panda = Robot("urdf/fp3.urdf", "fp3", [0.0, 0.0, 0.0])
    panda.preload_joint_data(f"{args.task_name}/transformed", 7)
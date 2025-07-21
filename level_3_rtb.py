import foxglove
import argparse
import logging
import cv2
import time
import roboticstoolbox as rtb
import numpy as np
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp, FrameTransforms, FrameTransform, Vector3, Quaternion

WORLD_FRAME_ID = "world"

def rot_matrix_to_quat(R):
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)

class Robot:
    def __init__(self, name):
        self.name = name 
        self.robot = rtb.models.Panda()
        self.q = self.robot.qr                 # start pose (6‑D vector)
        self.base_frame_id = "base"
        print(self.robot)

        self.remap = {
            "world": WORLD_FRAME_ID,
            "base_link": self.base_frame_id,
            "panda_link0": "base",
            "panda_link1": "fr3_link1",
            "panda_link2": "fr3_link2",
            "panda_link3": "fr3_link3",
            "panda_link4": "fr3_link4",
            "panda_link5": "fr3_link5",
            "panda_link6": "fr3_link6",
            "panda_link7": "fr3_link7",
            "panda_link8": "fr3_link8",
        }

    def update_joints(self, joint_values):
        self.q = joint_values
        self.robot.q = self.q
        # print(f"Updated joints: {self.q}")

        transforms = []
        transforms.append(
            FrameTransform(
                parent_frame_id=WORLD_FRAME_ID,
                child_frame_id=self.base_frame_id,
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        T_all = self.robot.fkine_all(self.q)   # list of SE3 frames
        for link, T in zip(self.robot.links, T_all):
            if link.parent is None:            # base link has no parent
                continue
            pos = T.t
            print(T.t, T.R)
            quat = rot_matrix_to_quat(T.R)     # convert rotation matrix to quaternion
            transforms.append(
                FrameTransform(
                    parent_frame_id=self.remap[link.parent.name],
                    child_frame_id=self.remap[link.name],
                    translation=Vector3(x=float(pos[0]),
                                        y=float(pos[1]),
                                        z=float(pos[2])),

                    rotation=Quaternion(x=float(quat[0]), 
                                        y=float(quat[1]), 
                                        z=float(quat[2]), 
                                        w=float(quat[3]))
                )
            )
        
        foxglove.log(
            "/tf",
            FrameTransforms(transforms=transforms)
        )

class Camera:
    def __init__(self, cam):
        self.cam = cv2.VideoCapture(cam)
        self.name = f"cam{cam}"
        self.channel = RawImageChannel("/"+self.name)

        # Get the default frame width and height
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            return None
        img_msg = RawImage(
            timestamp=Timestamp(
                sec=int(time.time()),
                nsec=int((time.time() - int(time.time())) * 1e9),
            ),
            frame_id=self.name,
            width=self.frame_width,
            height=self.frame_height,
            encoding="bgr8",
            step=self.frame_width*3,
            data=frame.tobytes(),
        )
        return img_msg

    def log_data(self, data):
        self.channel.log(data)

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stream camera data')
    parser.add_argument('-w', '--write', type=str, default=None, help='MCAP save location')
    args = parser.parse_args()

    foxglove.set_log_level(logging.INFO)
    
    server = foxglove.start_server()
    if args.write is not None:
        writer = foxglove.open_mcap(args.write, allow_overwrite=True)

    cam_indices = [0, 1, 2]  # Adjust camera indices as needed
    cams = [Camera(cam_indices[i]) for i in range(len(cam_indices))]

    fr3 = Robot("fr3")

    try:
        print("Streaming data...")
        while True:
            for cam in cams:
                img_msg = cam.get_frame()
                if img_msg is not None:
                    cam.log_data(img_msg)

            # Simulate joint updates for the robot
            joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            fr3.update_joints(joint_values)

    except KeyboardInterrupt:
        print("\nShutting down cameras...")
        cam.close()
        print("Cleanup complete.")
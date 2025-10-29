import foxglove
import argparse
import cv2
import os
from foxglove.channels import RawImageChannel, CameraCalibrationChannel
from foxglove.schemas import RawImage, CameraCalibration, FrameTransform, Vector3, Quaternion, FrameTransforms
from transforms3d.quaternions import mat2quat
import numpy as np

WORLD_FRAME_ID = "world"

class Camera:
    def __init__(self, cam_idx, live=True):
        self.idx = cam_idx
        self.name = f"cam_{cam_idx}"
        self.channel = RawImageChannel("/"+self.name)
        self.calibration_channel = CameraCalibrationChannel("/"+self.name+"/info")

        if live:
            self.cam = cv2.VideoCapture(cam_idx) # Returns None if camera is not found

    def get_img(self):
        ret, img = self.cam.read()
        if not ret: # Bad read
            return None

        return img

    def log_img(self, img, timestamp_nsec=None):
        width = img.shape[1]
        height = img.shape[0]

        img_msg = RawImage(
            frame_id=self.name,
            width=width,
            height=height,
            encoding="bgr8",
            step=width*3,
            data=img.tobytes(),
        )
        
        if timestamp_nsec is None:
            self.channel.log(img_msg) # Messages are logged with current time by default
        else:
            self.channel.log(img_msg, log_time=timestamp_nsec)

    def load_imgs(self, img_dir):
        for img_file in os.listdir(img_dir):
            timestamp_ms = int(img_file.split(".")[0])
            timestamp_nsec = int(timestamp_ms * 1e6) # Make sure all timestamps are in nanoseconds
            img = cv2.imread(os.path.join(img_dir, img_file))
            self.log_img(img, timestamp_nsec)

    def calibrate_mono_camera(self, M_in, M_ex, frame_width, frame_height, timestamp_nsec=None):
        """
        M_in: Intrinsic matrix as [K | 0]
        M_ex: Extrinsic matrix as [R_3x3, t_3x1; 0_1x3, 1]
        """
        
        K = M_in[:3, :3] # Calibration matrix
        P = np.concatenate((K, np.zeros((3, 1))), axis=1) # Projection matrix
        R_w = M_ex[:3, :3].T # Rotation matrix from world frame to camera frame
        t= M_ex[:3, 3] # Translation vector from camera frame to world frame
        c_w = -R_w @ t # Camera center in world frame
        q = mat2quat(R_w)

        camera_info = CameraCalibration(
            frame_id=self.name,
            width=frame_width,
            height=frame_height,
            P=P.flatten(),
            K=K.flatten(),
        )

        camera_transform = FrameTransform(
            parent_frame_id=WORLD_FRAME_ID,
            child_frame_id=self.name,
            translation=Vector3(x=float(c_w[0]), y=float(c_w[1]), z=float(c_w[2])),
            rotation=Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0])),
        )
        
        if timestamp_nsec is None:
            self.calibration_channel.log(camera_info)
            foxglove.log("/tf", FrameTransforms(transforms=[camera_transform]))
        else:
            self.calibration_channel.log(camera_info, log_time=timestamp_nsec)
            foxglove.log("/tf", FrameTransforms(transforms=[camera_transform]), log_time=timestamp_nsec)

    def close(self):
        self.cam.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stream camera data')
    parser.add_argument('-w', '--write', type=str, default=None, help='MCAP save location')
    args = parser.parse_args()
    
    server = foxglove.start_server()
    if args.write is not None:
        writer = foxglove.open_mcap(args.write, allow_overwrite=True)

    cam_indices = [0, 1, 2]  # Adjust camera indices as needed
    cams = [Camera(cam_indices[i]) for i in range(len(cam_indices))]

    print("Streaming data...")
    while True:
        try:
            for cam in cams:
                img = cam.get_img()
                if img is not None:
                    cam.log_img(img)

        except KeyboardInterrupt:
            print("Closing cameras")
            for cam in cams:
                cam.close()
            break
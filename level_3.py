import foxglove
import argparse
import logging
import cv2
import time
import numpy as np
import pinocchio as pin
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

    def controller(self):
        """
        The controller returns a list of joint values. This can be mapped to a control interface but a simple oscillation is demoed here.
        """
        angle = np.sin(time.time()/4)
        q =np.array([angle, angle, 0.0, angle/2 - np.pi/2, 0.0, angle/2 + np.pi/2, angle, 0.0, 0.0])

        return q

    def update(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        transforms = []

        # Add transforms for all body frames relative to their parents
        for frame in self.model.frames:
            if frame.type != pin.FrameType.BODY:
                continue

            frame_name = frame.name
            frame_id = self.model.getFrameId(frame_name)
            parent_joint_id = frame.parentJoint

            if parent_joint_id == 0:  # Root joint
                parent_frame_id = WORLD_FRAME_ID
                transforms.append(
                    FrameTransform(
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
            FrameTransforms(transforms=transforms)
        )

        

class Camera:
    def __init__(self, cam):
        self.cam = cv2.VideoCapture(cam)
        self.name = f"cam{cam}"
        self.channel = RawImageChannel("/"+self.name)
        self.calibration_channel = CameraCalibrationChannel("/"+self.name+"/info")

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

    def get_intrinsics(self, config):
        with open(config, 'r') as f:
            intrinsics = yaml.safe_load(f)
        return intrinsics

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

    panda = Robot("urdf/fp3_1.urdf", "fp3_1", [0.0, 0.0, 0.0])

    try:
        print("Streaming data...")
        while True:
            for cam in cams:
                img_msg = cam.get_frame()
                if img_msg is not None:
                    cam.log_data(img_msg)

            joint_values = np.array([0.0, 0.0, 0.0, -1.570796326794897, 0.0, 1.570796326794897, 0.7853981633974483, 0.0, 0.0])

            panda.update(joint_values)
            
            time.sleep(0.01)  # Small delay to prevent overwhelming the system

    except KeyboardInterrupt:
        print("\nShutting down cameras...")
        for cam in cams:
            cam.close()
        print("Cleanup complete.")
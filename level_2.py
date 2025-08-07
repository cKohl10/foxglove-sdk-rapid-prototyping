import foxglove
import argparse
import logging
import cv2
import time
import numpy as np
import pinocchio as pin
from transforms3d.euler import euler2quat
from foxglove import Channel
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp, FrameTransforms, FrameTransform, Vector3, Quaternion

WORLD_FRAME_ID = "world"

class Robot:
    def __init__(self, urdf_path, name, base_translation, base_rotation=None):
        self.name = name
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        self.base_frame_id = f"{name}_base"
        self.base_translation = base_translation
        self.base_rotation = base_rotation if base_rotation is not None else [-np.pi, np.pi, 0.0]

        self.torque_channel = Channel("/joint_torques", message_encoding="json")

    def update(self, q, timestamp_nsec=None):
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
                quat = euler2quat(self.base_rotation[0], self.base_rotation[1], self.base_rotation[2])
                transforms.append(
                    FrameTransform(
                        parent_frame_id=parent_frame_id,
                        child_frame_id=frame_name,
                        translation=Vector3(x=self.base_translation[0], y=self.base_translation[1], z=self.base_translation[2]),
                        rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
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
                try:
                    T_parent = self.data.oMf[parent_frame_id_in_model]
                except:
                    print(f"Parent frame {parent_frame_name} not found in model")
                    continue
                
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
            FrameTransforms(transforms=transforms),
            log_time=timestamp_nsec
        )
    
    def update_torque(self, joint_torques, timestamp_nsec=None):
        torques = {
            "data": joint_torques.tolist()
        }
        self.torque_channel.log(torques, log_time=timestamp_nsec)

def franka_controller():
    """
    The controller returns a list of joint values. This can be mapped to a control interface but a simple oscillation is demoed here.
    """
    angle = np.sin(time.time()/4)
    q =np.array([angle, angle, 0.0, angle/2 - np.pi/2, 0.0, angle/2 + np.pi/2, angle, 0.0, 0.0])

    return q

def ur5_controller():
    angle = np.sin(time.time()/4)
    q =np.array([angle, angle, 0.0, angle/2 - np.pi/2, 0.0, angle/2 + np.pi/2])

    return q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Level 2 - Using Frame Transforms')
    parser.add_argument('-w', '--write', type=str, default=None, help='MCAP save location')
    args = parser.parse_args()

    foxglove.set_log_level(logging.INFO)
    
    server = foxglove.start_server()
    if args.write is not None:
        writer = foxglove.open_mcap(args.write, allow_overwrite=True)

    franka = Robot("urdf/fp3.urdf", "fp3", [0.0, 0.0, 0.0])
    ur5 = Robot("urdf/ur5.urdf", "ur5", [0.0, 1.0, 0.0])

    while True:
        joint_values = franka_controller()
        joint_values_2 = ur5_controller()

        franka.update(joint_values)
        ur5.update(joint_values_2)
        
        time.sleep(0.01)
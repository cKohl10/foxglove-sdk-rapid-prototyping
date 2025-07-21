import foxglove
import argparse
import logging
import cv2
import time
import pinocchio as pin
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp, FrameTransforms, FrameTransform, Vector3, Quaternion

WORLD_FRAME_ID = "world"

class PandaRobot:
    def __init__(self, name):
        self.name = name
class Robot:
    def __init__(self, urdf_path, name):
        self.name = name
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()
        self.base_frame_id = f"{name}_base"

        #DEBUGGING
        for i in range(self.model.njoints):
            joint_name = self.model.names[i]
            joint_placement = self.data.oMf[i]
            print(f"Joint: {joint_name}, Placement: {joint_placement}")
            # Further processing to extract specific joint information (e.g., angles)

    def fk(self, joint_values):
        pin.forwardKinematics(self.model, self.data, joint_values)
        pin.updateFramePlacements(self.model, self.data)

        transforms = []
        transforms.append(
            FrameTransform(
                parent_frame_id=WORLD_FRAME_ID,
                child_frame_id=self.base_frame_id,
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        for j in range(1, self.model.njoints):    # loop over real joints
            # child link (BODY-frame) created by joint j
            child_fid = self.body_frame_of_joint.get(j)
            if child_fid is None:
                continue
            child_name = self.model.frames[child_fid].name

            # parent link = BODY frame of parent joint *or* world
            parent_j   = self.model.parents[j]
            parent_fid = self.body_frame_of_joint.get(parent_j)
            if parent_fid is None:    # root link
                parent_name = "world"
                T_parent_child = self.data.oMf[child_fid]
            else:
                parent_name = self.model.frames[parent_fid].name
                T_parent_child = self.data.oMf[parent_fid].inverse() * self.data.oMf[child_fid]  

            # build & send TransformStamped
            quat = pin.Quaternion(T_parent_child.rotation)  

            transforms.append(
                FrameTransform(
                    parent_frame_id=parent_name,
                    child_frame_id=child_name,
                    translation=Vector3(x=float(T_parent_child.x),
                                        y=float(T_parent_child.y),
                                        z=float(T_parent_child.z)),

                    rotation=Quaternion(x=float(quat.x), 
                                        y=float(quat.y), 
                                        z=float(quat.z), 
                                        w=float(quat.w))
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

    fr3 = Robot("urdf/fr3.urdf", "fr3")

    try:
        print("Streaming data...")
        while True:
            for cam in cams:
                img_msg = cam.get_frame()
                if img_msg is not None:
                    cam.log_data(img_msg)

            # fr3.update_joints({
            #     "fr3_joint1": 0.0,
            #     "fr3_joint2": 0.0,
            #     "fr3_joint3": 0.0,
            #     "fr3_joint4": 0.0,
            #     "fr3_joint5": 0.0,
            #     "fr3_joint6": 0.0,
            # })

    except KeyboardInterrupt:
        print("\nShutting down cameras...")
        cam.close()
        print("Cleanup complete.")
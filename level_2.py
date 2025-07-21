import foxglove
import argparse
import cv2
import time
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp

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
    
    server = foxglove.start_server()
    if args.write is not None:
        writer = foxglove.open_mcap(args.write, allow_overwrite=True)

    cam_indices = [0, 1, 2]  # Adjust camera indices as needed
    cams = [Camera(cam_indices[i]) for i in range(len(cam_indices))]

    try:
        print("Streaming data...")
        while True:
            for cam in cams:
                img_msg = cam.get_frame()
                if img_msg is not None:
                    cam.log_data(img_msg)

    except KeyboardInterrupt:
        print("\nShutting down cameras...")
        cam.close()
        print("Cleanup complete.")
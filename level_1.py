import foxglove
import argparse
import cv2
import time
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp

class Camera:
    def __init__(self, cam, live=True):
        self.name = f"cam{cam}"
        self.channel = RawImageChannel("/"+self.name)
        
        if live:
            self.cam = cv2.VideoCapture(cam)

    def get_live_img(self):
        ret, img = self.cam.read()
        if not ret:
            return None
        timestamp_sec = time.time()
        return img, timestamp_sec

    def log_rgb_img(self, img, timestamp_sec):
        width = img.shape[1]
        height = img.shape[0]

        timestamp_nsec = int((timestamp_sec - int(timestamp_sec)) * 1e9)
        
        img_msg = RawImage(
            timestamp=Timestamp(
                sec=int(timestamp_sec),
                nsec=int(timestamp_nsec),
            ),
            frame_id=self.name,
            width=width,
            height=height,
            encoding="bgr8",
            step=width*3,
            data=img.tobytes(),
        )
        self.channel.log(img_msg)

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
        for cam in cams:
            img, timestamp_sec = cam.get_live_img()
            if img is not None:
                cam.log_rgb_img(img, timestamp_sec)
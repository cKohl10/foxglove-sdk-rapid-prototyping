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


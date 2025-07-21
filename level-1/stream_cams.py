import foxglove
import argparse
from utils.camera import Camera

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
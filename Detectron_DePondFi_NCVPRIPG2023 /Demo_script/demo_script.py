import argparse
import time

from torch import cuda
from ultralytics import YOLO, checks

parser = argparse.ArgumentParser(
    description="RUN YOLOv8n inference on an input video to detect fish"
)

parser.add_argument(
    "--in_video_path",
    help="path to your input video to run YOLOv8 inference on",
    type=str,
    required=True,
)

args = parser.parse_args()

process = 'GPU' if cuda.is_available() else 'CPU'

if cuda.is_available():
    device = cuda.current_device()
    print(f'\n****** Nvidia GPU {cuda.get_device_name(device)} detected! Running inference on GPU ******\n')
    checks()
    time.sleep(3) # sleep for 3 seconds to allow the user to read the message
    start = time.time()
    model = YOLO('Model_weights/best.engine', task='detect')
    results = model.predict(
        source=f"{args.in_video_path}", save=True, save_txt=True)[0]
    end = time.time()
    total_time = end-start


else:
    print('\n****** No Nvidia GPU in system! Running inference on CPU ******\n')
    checks()
    time.sleep(2) # sleep for 2 seconds to allow the user to read the message
    start = time.time()
    model = YOLO('Model_weights/best.onnx', task='detect')
    results = model.predict(
        source=f"{args.in_video_path}", save=True, save_txt=True)[0]
    end = time.time()
    total_time = end-start


print(f"\nThe total inference time of the pipeline is: {total_time}s for 1100 frames run on {process}")
checks()
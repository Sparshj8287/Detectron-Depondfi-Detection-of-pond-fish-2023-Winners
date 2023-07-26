from sahi.utils.yolov8 import (
    download_yolov8n_model,
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import time
import argparse
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

yolov8_model_path = "Model_weights/best_YOLOv8n.pt"
download_yolov8n_model(yolov8_model_path)
model_type = "yolov8"
model_path = yolov8_model_path
model_device=0

slice_height = 213
slice_width = 213
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_video_dir = f"{args.in_video_path}"


process = 'GPU' if cuda.is_available() else 'CPU'

if cuda.is_available():
    device = cuda.current_device()
    print(f'\n****** Nvidia GPU {cuda.get_device_name(device)} detected! Running inference on GPU ******\n')
    checks()
    time.sleep(3) # sleep for 3 seconds to allow the user to read the message
    start = time.time()
    predict(
    model_type=model_type,
    model_path=model_path,
    model_device='cuda:0',
    source=source_video_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)
    end = time.time()
    total_time = end-start


else:
    print('\n****** No Nvidia GPU in system! Running inference on CPU ******\n')
    checks()
    time.sleep(2) # sleep for 2 seconds to allow the user to read the message
    start = time.time()
    predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    source=source_video_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)
    end = time.time()
    total_time = end-start


print(f"\nThe total inference time of the pipeline is: {total_time}s for 1100 frames run on {process}")
checks()


# <center>Depondfi'23 Challenge [NCVPRIPG-2023] **Winners** <center>Team-Detectron

![](fish_image.png)

<center><small>Actual prediction from our model</small></center>

## Table of Contents

- [Depondfi'23 notebooks](#1-Depondfi-notebooks)
- [Folder Directory Structure](#2-Folder-Directory-Structure)
- [Train code](#3-Train-code)
- [Data](#4-Data)
- [Demo_script](#5-Demo-script)
- [Demo_script_sahi](#6-Demo-script-sahi)
- [Inference_time](#7-inference-time)
- [Inference_time_sahi](#8-inference-time-sahi)
- [Metrics](#9-metrics)
- [Model_weights](#10-model-weights)
- [Summary](#11-summary)
- [requirements.txt](#12-requirements.txt)
- [README.md](#13-readme)
- [Usage](#14-usage)
- [System Specifications](#15-system-specifications)

## Depondfi'23 notebooks [#1-Depondfi-notebooks]

You can open the notebook in Colab (there is a button directly on said pages).

| Notebook                   | Description                                                                    |                                                                                                                                                                     |
| :------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Defondi'23 YOLOv8n Training | Training the best performing YOLOv8n model                                      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11BY4kFfrWBgMq2JUfO6yoLm01mjvfbg0?usp=sharing) |
| Defondi'23 YOLOv8n Testing  | Notebook for testing and analyzing inference time of various models we trained | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NkMJrqYZHPhVY-J5rMtKG8eZd-gBNW4o?usp=sharing) |
| Defondi'23 YOLOv8n + SAHI Testing  | Notebook for inference calculated on SAHI after YOLOv8n | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tQBHygkFfRjxM5aAI83h8GjyM6sXmPwd?usp=sharing) |

## 1) Folder Directory Structure {#2-Folder-Directory-Structure}

```

Detectron_DePondFi_NCVPRIPG2023
│
├── Train_code/
│ └── depondfi_yolov8n.ipynb
│
├── Data/
│ ├── README.md
│ └──split.ipynb
│
├── Demo_script/
│ ├── demo_script.py
│ └── test_images2video.py
│
├── Demo_script_sahi/
│ ├── demo_script.py
│ ├── Extracting_labels.ipynb
│ └── test_images2video.py
│
├── Inference_time/
│ └── Inference_time_yolov8n.txt
│
├── Inference_time_sahi/
│ └── Inference_time_yolov8n_sahi.txt
│
├── Model_weights/
│ ├── best_YOLOv8n.pt
│ ├── best.onnx
│ └── best.engine
│
├── Metrics/
│ ├── args.yaml
│ ├── confusion_matrix_normalized.png
│ ...
│ └── val_batch2_pred.jpg
│
├── Summary/
│ └── Depondfi_summary
│ 
├── requirements.txt
│
├── fish_image.png
│
└── README.md

```

## 2) Train code {#3-Train-code}

This directory contains the training code for our project, specifically the `depondfi_yolov8n.ipynb` colab Notebook.

## 3) Data {#4-Data}

This directory holds the readme of data required for training and validation our model. 

- **README.md**: Readme file providing information about the data.
- **split.ipynb**:The `split.ipynb` notebook utilizes Python code to perform the following steps:

  - **Load the dataset**: The notebook reads the images and their corresponding labels from the appropriate directories.

  - **Random Split**: Using a random splitting technique, the notebook divides the dataset into two subsets: 85% for training and 15% for validation.

  - **Save the Split Data**: The notebook saves the file names of the training and validation images into separate text files. These files will be used in the subsequent steps of the project to ensure consistent splitting between different runs.

## 4) Demo_script {#5-Demo-script}

This directory contains scripts for running a test video. It includes the following files and subdirectories:


- **demo_script.py**: The `demo_script.py` is a Python script that implements the YOLOv8n pipeline on a test video and generates an output video with object detections. This script showcases the capabilities of our model in real-world scenarios by detecting and tracking objects in a video.

  - **Prerequisites**
    Before running the `demo_script.py` script, ensure that you have the following prerequisites:

  - **Trained Model Weights**: Make sure you have the trained model weights file (best.onnx) in the ` Model_weights/` directory. This file contains the learned parameters of our YOLOv8n model.

  - **Test Video**: Prepare a test video file in a compatible format (e.g., .mp4, .avi) that you want to apply object detection to.

  - **test_images_video_frame3.mp4**: Output video generated by the `test_images2videos.py` on given 1100 test images.

- **test_images2video.py**: Python script for converting test images to a video.

## 5) Demo_script_sahi {#6-Demo-script-sahi}
This directory contains scripts for running a test video. It includes the following files and subdirectories:

- **demo_script.py**: The `demo_script.py` is a Python script that implements the YOLOv8n pipeline on a test video and generates an output video with object detections. This script showcases the capabilities of our model in real-world scenarios by detecting and tracking objects in a video.
- **Extracting_labels.ipynb**: This is notebook file where the process of extracting labels in a text file from a pickle file comes from sahi.

  - **Prerequisites**
    Before running the `demo_script.py` script, ensure that you have the following prerequisites:

  - **Trained Model Weights**: Make sure you have the trained model weights file (best.onnx) in the ` Model_weights/` directory. This file contains the learned parameters of our YOLOv8n model.

  - **Test Video**: Prepare a test video file in a compatible format (e.g., .mp4, .avi) that you want to apply object detection to.

  - **test_images_video_frame3.mp4**: Output video generated by the `test_images2videos.py` on given 1100 test images.

- **test_images2video.py**: Python script for converting test images to a video.


## 6) Inference_time {#7-inference-time}

This directory includes files related to measuring the inference time of our model. It contains:

- **Inference_time_yolov8n.txt**: Text file that records the inference time results.


## 7) Inference_time_sahi {#8-inference-time-sahi}

This directory includes files related to measuring the inference time of our model. It contains:

- **Inference_time_yolov8n_sahi.txt**: Text file that records the inference time results.

## 8) Metrics {#9-Metrics}

This directory contains images of metrics of our trained model **YOLOv8n**. 


## 9) Model_weights {#10-model-weights}

This directory stores the trained model weights. It contains the following files:


- **best_YOLOv8n.pt**: Trained model weights in pytorch format. This format is used in sahi prediction.
- **best.onnx**: Trained model weights in ONNX format. For faster inference time use this format when the inference is performed on CPU.
- **best.engine**: Trained model weights in TensorRT format. For faster inference time use this format when the inference is performed on GPU.



## 10) Summary {#11-summary}

This directory contains summary files and results of our project. It includes:

- **Depondfi_summary**: Summary file providing an approach and algorithm for our project.

## 11) requirements.txt {#12-requirements.txt}

A file listing the required dependencies and packages for our project.

## 12) README.md {#13-readme}

A README file providing general information about our project and its directory structure.

## 13) Usage {#14-usage}

In order to implement and perform inference on **yolov8n based pipeline for fish detection** follow steps listed below:-
 
### 14.1)Cloning Github Repository
- First clone the github repository using the following command:
```
git clone https://github.com/Sparshj8287/Detectron-Depondfi-Detection-of-pond-fish-2023-Winners.git
```


### 14.2) Install

- Install dependencies (only once):

  `pip install -r requirements.txt`

### 14.3) Test on your video

- Execute the script or run the code cells in the Jupyter Notebook. **Make sure to provide the correct path to the input video.** The annotated video and the bounding box coordinates will be saved in the directory which is displayed in the output.<br><br>

- Command to run the `Demo_script/demo_script.py`:

```python
python path/to/demo_script.py --in_video_path path/to/video
```

## 15) System Specifications  {#14-system-specifications}

Inference:
- Intel Xeon CPU @2.20 GHz ( _2 cores_ )
- Google Colab Tesla T4 GPU 15.36 GB
- System RAM - 12.7 GB

Training:
  
- Intel Xeon CPU @2.20 GHz ( _6 cores_ )
- Google Colab A100 GPU 40.51 GB
- System RAM - 83.5 GB


```

```

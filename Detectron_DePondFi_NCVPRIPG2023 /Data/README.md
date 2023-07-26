
Dataset:
The dataset "DePondFi'23" comprises data collected from various aqua farms located in and around Tamil Nadu, India. The dataset includes water samples of varying turbidity levels, including high and medium turbidity. The lighting conditions under which the samples were collected include low light, direct sunlight, and no artificial lighting. The data has been collected and curated in a systematic manner to ensure accuracy and reliability. DePondFi'23 Dataset consists of six folders with subfolders. Subfolders contain both the image files and their corresponding text labels.


To perform YOLOv5 calculation, we need to convert the normalized coordinates to actual pixel values. We can do this by multiplying the normalized values with the corresponding dimension of the image.

Images in this dataset has dimensions 640x640, we can perform the YOLOv5 calculation for the given normalized coordinates as follows:
class - 'Fish'
x-coordinate of top-left corner (xmin): xmin_normalized = 120.6 / 640 = 0.18828125
y-coordinate of top-left corner (ymin): ymin_normalized = 433.6 / 640 = 0.67734375
x-coordinate of bottom-right corner (xmax): xmax_normalized = 288.6 / 640 = 0.4515625
y-coordinate of bottom-right corner (ymax): ymax_normalized = 603.2 / 640 = 0.94125

(class xmin_normalized, ymin_normalized, xmax_normalized, ymax_normalized) = (0 0.18828125, 0.67734375, 0.4515625, 0.94125)

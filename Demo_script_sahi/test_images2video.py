import re
import cv2
import os
video_name = 'test_images_video_frame3.mp4'
image_folder = 'Demo_script/Images'
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


images = sorted(os.listdir(image_folder), key=numericalSort)

frame_width, frame_height = cv2.imread(
    os.path.join(image_folder, images[0])).shape[:2]
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 3, (frame_width, frame_height))

# Iterate through each image file
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)

    # Write the frame to the video
    video.write(frame)

# Release the video writer and close the video file
video.release()

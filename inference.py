#from ultralytics import YOLO
#model = YOLO('yolov8n.pt')
#model.predict(source="0", show=True,)

import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

# Load the YOLOv8 model
model = YOLO('runs/oneshot/yolov8n_RPC/weights/best.pt')
support = Image.open("oasis.jpg")
support = np.asarray(support)
support = cv2.cvtColor(support, cv2.COLOR_BGR2RGB)
# Open the video file
cap = cv2.VideoCapture('video2.mp4')
#capture = cv2.VideoCapture('rtsp://192.168.1.64/1')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #frame = cv2.resize(frame, (640,640))
        frame = resizeAndPad(frame,(640,640))
        support = cv2.resize(support, (640,640))
        #support = resizeAndPad(support,(640,640))
        #support = cv2.cvtColor(support, cv2.COLOR_BGR2RGB)
        # Run YOLOv8 inference on the frame
        results = model(frame, support, conf=0.6, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.hconcat([annotated_frame,support])
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Arguments and their parsinga
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loading pretrained model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Video Stream initialisation
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Convert frame dimensions to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))

    # Compute detections and Predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop statements until the Video Stream is closed
    for i in range(0, detections.shape[2]):
        # Compute confidence of detection.
        confidence = detections[0, 0, i, 2]

        # Compare confidence of detection with threshold and ignore false detects.
        if confidence < args["confidence"]:
            continue

        # Compute dimensions of theti detect box.
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw detect box around face along with the confidence of detection.
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Display VideoStream Frame as output
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Instruction to exit video stream
    # enter "Esc" to exit the Live Video Stream
    if key == 27:
        break

# CLose all the associated files and perform clean up of the system.
cv2.destroyAllWindows()
vs.stop()

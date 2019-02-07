import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="patho to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffee 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load the model from disk
print("Loading model")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# save the height and width of the object
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# generate blob for the Image to pass into the network
# resize the image to (300,300)
# 1.0 is the scaling factor
# last values are mean substraction values so that our CNN model can understand it 
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

# set the input for layers as blob
net.setInput(blob)

# pass the blob into the network (forward pass)
detections = net.forward()


for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the threshold
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        print("hi")
cv2.imshow("ho",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

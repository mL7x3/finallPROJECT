import time
import cv2
from cvzone.ClassificationModule import Classifier
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

offset = 20

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

imgSize = 320

folder = "data/A"

counter = 0


labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:

        hand = hands[0]

        x, y, w, h = hand["bbox"]

        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h

            wCal = math.ceil(k*w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize-wCal)/2)

            imgWhite[:, wGap:wCal+wGap] = imgResize


        else:
            k = imgSize / w

            hCal = math.ceil(k*h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize-hCal)/2)

            imgWhite[hGap:hCal+hGap, :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
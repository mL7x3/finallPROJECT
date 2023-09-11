# استيراد المكتبات / Importing Libraries
import time
import cv2
from cvzone.ClassificationModule import Classifier
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

# إعداد النموذج للتصنيف / Initializing the Classification Model
classifire = Classifier("Model/keras_model.h5", "Model/labels.txt")

# تعيين قيم ثوابت ومتغيرات / Setting Constants and Variables
offset = 20

# تهيئة التقاط الفيديو من الكاميرا / Configuring Video Capture from the Camera
cap = cv2.VideoCapture(0)

# تهيئة متعقب اليد / Initializing the Hand Detector
detector = HandDetector(maxHands=1)

# حجم الصورة المرادة / Desired Image Size
imgSize = 320

# مجلد لحفظ الصور الملتقطة
folder = "data/I"

# متغير لعد الصور / Variable for Counting Images
counter = 0

# قائمة بتصنيفات الإيماءات / List of Gesture Classifications
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

# الحلقة الرئيسية / Main Loop
while True:
    # قراءة إطار الفيديو / Reading Video Frame
    success, img = cap.read()
    imgresult = img.copy()

    # البحث عن الأيدي في الإطار / Detecting Hands in the Frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # تجهيز الصورة للتصنيف / Preparing the Image for Classificatio
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifire.getPrediction(imgWhite)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifire.getPrediction(imgWhite)

        # عرض النتائج على الصورة الأصلية / Displaying Results on the Original Image
        cv2.putText(imgresult, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        #  Displaying Intermediate Images for Debugging / عرض الصور المتوسطة لأغراض التصحيح
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # عرض الصورة النهائية / Displaying the Final Image
    cv2.imshow("Image", imgresult)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
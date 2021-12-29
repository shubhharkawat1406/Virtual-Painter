import cv2
import os
import time
import numpy as np
import HandTrackingModule as htm


folderPath = "Images"
myList = os.listdir(folderPath)
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)
# print(len(overLayList))
header = overLayList[0]
drawColor = (0,0,255)
brushThickness = 15
erazerThickness = 80
wCam, hCam = 1280, 720
xp,yp=0,0
cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.75)
imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        # print(lmList)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp,yp=0,0
            if y1<155:
                if 250<x1<450:
                    header = overLayList[0]
                    drawColor = (0,0,255)
                elif 550<x1<750:
                    header = overLayList[1]
                    drawColor = (200,0,0)
                elif 800<x1<950:
                    header = overLayList[2]
                    drawColor = (0,230,230)
                elif 1050<x1<1250:
                    header = overLayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-30),(x2,y2+30),drawColor,cv2.FILLED)

        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1),15,drawColor, cv2.FILLED)
            if xp ==0 and yp ==0:
                xp,yp=x1,y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1), drawColor,erazerThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor,erazerThickness)
            else:
                cv2.line(img, (xp,yp),(x1,y1), drawColor,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor,brushThickness)

            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    h,w,c = header.shape
    img[0:h, 0:w] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("canvas", imgCanvas)
    cv2.waitKey(1)

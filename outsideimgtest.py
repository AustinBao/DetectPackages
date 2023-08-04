import cv2 
import matplotlib.pyplot as plt
from ultralytics import YOLO


img = cv2.imread("C:/OpenCV/DetectAmazonPackage/Images/outside_test2.jpg", cv2.IMREAD_UNCHANGED)

model = YOLO("yolov8n.pt")

while True:

    #resize the img to make it smaller
    width = int(img.shape[1] * 0.2)
    height = int(img.shape[0] * 0.2)
    smaller_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    # gray_img = cv2.cvtColor(smaller_img,cv2.COLOR_BGR2GRAY)
    # blur_img = cv2.blur(gray_img,(5,5))
    # th1 = percent_thresh(blur_img)

    results = model(smaller_img)

    cv2.imshow("original", smaller_img)
    # cv2.imshow("thresh", th1)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
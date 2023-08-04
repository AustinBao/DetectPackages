from ultralytics import YOLO
import cv2


model = YOLO("C:/OpenCV/DetectAmazonPackage/runs/detect/train4/weights/best.pt") 

image = cv2.imread('C:\OpenCV\DetectAmazonPackage\Images\webcam-toy-photo1.jpg')

# ,save =True, save_txt= True
result = model.predict("C:\OpenCV\DetectAmazonPackage\Images\webcam-toy-photo1.jpg")


normalsize = []
boxDimensions = open("C:/Users/Austi/Downloads/webcam-toy-photo1.txt", "r")
lines = boxDimensions.readlines()[0].split()
for i in lines:
    normalsize.append(float(i)*255)

name,x,y,width, height = normalsize

print(lines, name)
print(name)

print(x)
print(y)
print(width)
print(height)



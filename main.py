import os
import cv2
import time
from ultralytics import YOLO

curr = time.localtime()
hour = curr.tm_hour
minute = curr.tm_min

model = YOLO("C:/OpenCV/DetectAmazonPackage/runs/detect/train4/weights/best.pt") 



def takePicture(frame):
    newImageFilePath = "C:/OpenCV/DetectAmazonPackage/Images/mainImageFolder/hour" + str(hour) + "minute" + str(minute) + ".jpg"
    cv2.imwrite(newImageFilePath, frame)
    return newImageFilePath

def rectangleDimensions(predictFilePath):
    fileLines = open(predictFilePath, "r")
    # keep in mind, this only works when there is one box detected since im am strictly indexing at 0
    lines = fileLines.readlines()[0].split()
    xc, yc, nw, nh = float(lines[1]), float(lines[2]), float(lines[3]), float(lines[4])
    return xc, yc, nw, nh

def countNumberOfBoxesDetected(predictFilePath):
    fileLines = open(predictFilePath, "r")
    return len(fileLines.readlines())

def delete_directory(directory_path):
    file_list = os.listdir(directory_path)

    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_directory(file_path)

    os.rmdir(directory_path)
    print(f"Directory '{directory_path}' and its contents successfully deleted.")

def deleteImage(imagePath):
    os.remove(imagePath)
    print(f"File '{imagePath}' successfully deleted.")


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 1)

imgfilepath = takePicture(frame)

result = model.predict(imgfilepath,save =True, save_txt= True)

# Need an if statement
predictionsTXTFilePath = "C:/OpenCV/DetectAmazonPackage/runs/detect/predict" + "/labels/hour" + str(hour) + "minute" + str(minute) + ".txt"
x_center, y_center, r_width, r_height = rectangleDimensions(predictionsTXTFilePath)

img = cv2.imread(imgfilepath)
img_height, img_width = img.shape[0], img.shape[1]
x_center, y_center, r_width, r_height = x_center * img_width, y_center * img_height, r_width * img_width, r_height * img_height

top_left = (int(x_center-r_width/2),int(y_center-r_height/2))
bottom_right = (int(x_center+r_width/2),int(y_center+r_height/2))

cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)


while True:
    cv2.imshow("rectangle", frame)

    if cv2.waitKey(30) == 27:
        delete_directory("C:/OpenCV/DetectAmazonPackage/runs/detect/predict")
        deleteImage(imgfilepath)
        break

cap.release()
cv2.destroyAllWindows()
    






import cv2
import numpy as np
import time

def percent_thresh(gray):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
    # Calculate percent brightness to threshold. Adapts to the changing light lvl throughout the day. 
    percent_brightness = int(maxVal * 0.8)
    ret, thresh_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh_frame
    
def capture_image_and_label(curr_frame):
    # Get the current time
    current_time = time.localtime()
    hour = current_time.tm_hour
    minutes = current_time.tm_min

    # Save the image with the current hour as the filename
    cv2.imwrite(f"C:/OpenCV/PythonComputerVision/PersonalProjects/DetectAmazonPackage/Images/hour_{hour}:{minutes}.png", curr_frame)

    print(f"Image at {hour}:{minutes} captured successfully.")



cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)


    # Set the interval in seconds (1 minute = 60 seconds, 30 minutes = 1800 seconds)
    interval = 10
    capture_image_and_label(frame)
    time.sleep(interval)


    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()




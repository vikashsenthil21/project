## Aim
To write a python program using OpenCV to do the following image manipulations. i) Extract ROI from an image. ii) Perform handwritting detection in an image. iii) Perform object detection with label in an image.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
I)Perform ROI from an image
## Step1:
Import necessary packages

## Step2:
Read the image and convert the image into RGB

## Step3:
Display the image

## Step4:
Set the pixels to display the ROI

## Step5:
Perform bit wise conjunction of the two arrays using bitwise_and

## Step6:
Display the segmented ROI from an image.

II)Perform handwritting detection in an image
## Step1:
Import necessary packages

## Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.

## Step3:
Display the results.

III)Perform object detection with label in an image
Step1:
Import necessary packages

## Step2:
Set and add the config_file,weights to ur folder.

## Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)

## Step4:
Create a classLabel and print the same

## Step5:
Display the image using imshow()

## Step6:
Set the model and Threshold to 0.5

## Step7:
Flatten the index,confidence.

## Step8:
Display the result.

## Program:
## Name: vikash s
## Register Number: 212222240115
## I)Perform ROI from an image:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'nature.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([22, 93, 0])#choose the RGB values accordingly to display specific color
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
segmented_image = cv2.bitwise_and(img, img, mask=mask)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_image_rgb)
plt.title('Segmented Image (Yellow)')
plt.axis('off')
plt.show()
```

## II)Perform handwritting detection in an image:

```
get_ipython().system('pip install opencv-python numpy matplotlib')
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
image_path = 'images.jpeg'
detect_handwriting(image_path)
```
## III)Perform object detection with label in an image
```
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))
img=cv2.imread('air.jpeg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

```

## Output:

## I)Perform ROI from an image:
![image](https://github.com/user-attachments/assets/e556b94f-ca8e-42dd-a1e5-0d6a45f517fa)

## II)Perform handwritting detection in an image:
![image](https://github.com/user-attachments/assets/b50b8054-23c7-4662-89b5-ba17185a6479)

## III)Perform object detection with label in an image:

![image](https://github.com/user-attachments/assets/3f7eecc7-eef1-4f56-8f9e-67df8aab23f0)

## Result:
Thus, The python program using OpenCV to do the image manipulations is executed successfully.



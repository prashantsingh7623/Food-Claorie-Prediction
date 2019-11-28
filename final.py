"""
We will first create the bounding box using YOLO object detection algorithm.
For loading the YOLO you have to download 'yolov3.weights and yolov3.cfg.txt' files. 
"""

"""
making bounding box on images
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg.txt")   
classes = []
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading image
""" Here we will first upload the side view of the image. """
img = cv2.imread("side.png") #put the address of your image here.
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_PLAIN
label = ''
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(img, label, (x, y + 30), font, 3, color, 3) uncomment this line if you want the object name.

cv2.imwrite("apple_box_side.png",img) #saving the file with name 'apple_box_side.png'
plt.imshow(img) #showing the image on screen.


"""
Now we will apply grab cut algorithm to get the interested portion of apple.
"""

"""_____applying grabcut algorithm_____"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('apple_box_side.png')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (x, y, w, h) #these are the values which you will get from above code of YOLO object detection.
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
cv.imwrite("apple_grab_side.png", img) #saving the image with name 'apple_box_grab_side.png'.



"""
After applying the grab cut algorithm we will find the contour of image.
And then we will find the extreme points of the image.
"""

"""_____finding contour and then extreme points_____"""

#declaring points array to store the extreme points of both top and side images of apple.
points = []

# import the necessary packages
import imutils
import cv2
import matplotlib.pyplot as plt
 
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("apple_grab_side.png") #image address.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)
 
# show the output image
plt.imshow(image)
cv2.imwrite("extreme_points_side.png",image) #saving the image.

#append minor axis points in the point list.
#Here we are finding the minor_axis, hence we are only interested in top and bottom points.
points.append(extTop)
points.append(extBot)


#calculating minor_axis
#Here the results will come in pixels.
#We have to convert these results into cm.
minor_axis = (points[0][0], points[0][1], points[1][0], points[1][1])



"""
applying for top image
"""


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg.txt")
classes = []
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("top.png")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape




# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_PLAIN
label = ''
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

cv2.imwrite("apple_box_top.png",img)
plt.imshow(img)


"""_____applying grabcut algorithm_____"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('apple_box_top.png')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (x, y, w, h) #swap first two values and than put
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
cv.imwrite("apple_grab_top.png", img)



"""_____finding contour and then extreme points_____"""

#declaring points array
points = []

# import the necessary packages
import imutils
import cv2
import matplotlib.pyplot as plt
 
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("apple_grab_top.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)
 
# show the output image
plt.imshow(image)
cv2.imwrite("extreme_points_top.png",image)

#append major axis points in points list.
#Here we are only interested in major_axis points , so we take left and right points into account.
points.append(extLeft)
points.append(extRight)


#Calculating major axis
#Here the results will come in pixels.
major_axis = (points[0][0], points[0][1], points[1][0], points[1][1])


""" Finding distance between the points using distance formula """
#here we will get the distance in pixels.
import math
a = math.sqrt(pow((major_axis[0] - major_axis[2]),2) + pow((major_axis[1] - major_axis[3]),2))
b = math.sqrt(pow((minor_axis[0] - minor_axis[2]),2) + pow((minor_axis[1] - minor_axis[3]),2))


""" converting the pixels into cm by assuming ppi of 400 """
length_in_cm = (2.54 / 400) * math.ceil(a)
breadth_in_cm = (2.54 / 400) * math.ceil(b)



""" calculating volume """

"""
ratio = b / a
if(ratio >= 0.5 and ratio <= 1.5):
    v = (4 * 3.14 * length_in_cm * length_in_cm * breadth_in_cm) / 3
elif(ratio > 1.5):
    v = (4 * 3.14 * length_in_cm * breadth_in_cm * breadth_in_cm) / 3)
"""

#here we know the shape of apple is elliptical.
volume = (4 * 3.14 * length_in_cm * length_in_cm * breadth_in_cm) / 3
mass = 0.5 * volume #here we assume the density of apple be 0.5.
print("mass of {} is {} grams".format(label, mass))



"""
estimating calories
We have created a file 'energy.txt' where we stored corresponding calories of fruits in kcal/g.
"""
cal = 0.0
with open('energy.txt') as file:
    for line in file:
        key, val = line.split()
        if key == label:
            cal = float(val) * mass
print("estimated calories of {} is {} kcal".format(label, math.ceil(cal)))

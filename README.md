# Food-Claorie-Prediction
This appication basically calculate the volume of food(apple in our case) and than the calories. We use YOLO V3 model to identify the object (i.e. object detection). The result of object detection will give us the bounding box around image of the apple. Also the model will give us the cordinates of bounding box created on the image.
Then we apply grab cut algorithm by using the coordinates of bounding box on the image. We'll get the area of interest of the image. After applying grab cut algo, we will find the contour of the image and than the extreme points.
The idea is to find the major and minor axis of the food (apple in our case) by finding the distance between the extreme points using the distance formula. After finding the distance between the points we'll get the desired lenght and width of the food.
Here we assumed that the ppi of smartphone approx 400.
After finding the volume of food(apple in our case) we'll find the mass using mass volume relationship.
Here we assumed that the density of food(apple in our case) will be 0.5. 
Now we can find the desired calories of the food(apple in our case).

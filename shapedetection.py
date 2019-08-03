import cv2 as cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import random as rng


def find_Corners(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]==0:
                img[i,j]=255
            else:
                img[i,j]=0
    
    retval, labels = cv2.connectedComponents(img)
    return (retval -1)


rng.seed(12345)
shape_img = cv2.imread('C:\\Users\\akbul\\Desktop\\ISSD STAJ\\image_process_tutorial\\r\\4.jpeg')
real_img=cv2.resize(shape_img,(480,360))
shape_img = cv2.resize(shape_img,(480 , 360))
gray_img = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)

bil_f = cv2.bilateralFilter(gray_img,3,75,75)

ret, gray_img= cv2.threshold(bil_f,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
new_img=255-gray_img
kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

img_dilate = cv2.dilate(new_img, kernel, iterations=1)
kernel1=np.array([[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1]],np.uint8)
img_erosion1 = cv2.erode(img_dilate, kernel1, iterations=1)

kernel2=np.array([[1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]],np.uint8)
img_erosion2 = cv2.erode(img_dilate, kernel2, iterations=1)
bw_1= cv2.bitwise_and(img_erosion1 , img_erosion2)

kernel3=np.array([[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]],np.uint8)
img_erosion3 = cv2.erode(img_dilate, kernel3, iterations=1)

kernel4=np.array([[0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]],np.uint8)
img_erosion4 = cv2.erode(img_dilate, kernel4, iterations=1)

bw_2= cv2.bitwise_and(img_erosion3 , img_erosion4)
bw_3=cv2.bitwise_and(bw_1,bw_2)

kernel5=np.array([[0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1]],np.uint8)
img_erosion5 = cv2.erode(img_dilate, kernel5, iterations=1)

kernel6=np.array([[1, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0]],np.uint8)

img_erosion6 = cv2.erode(img_dilate, kernel6, iterations=1)

bw_4= cv2.bitwise_and(img_erosion5 , img_erosion6)
img_and_total=cv2.bitwise_and(bw_4,bw_3)
im2, contours, hierarchy = cv2.findContours(img_and_total,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
img_contour = cv2.drawContours(shape_img, contours,-1, (255,0,255), 1)

hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)
# Draw contours + hull results
drawing = np.zeros((shape_img.shape[0], shape_img.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (255,255,255)
    cv2.drawContours(drawing, hull_list, i, color)

dest = cv2.cornerHarris(img_and_total, 2, 5, 0.02) 
dest = cv2.dilate(dest, None)
shape_img[dest > 0.01 * dest.max()]=[0, 0, 0] 


img_or=cv2.bitwise_or(shape_img,img_contour)
gr=cv2.cvtColor(shape_img,cv2.COLOR_RGB2GRAY)
for i in range(len(contours)):
    x_1,y_1,w_1,h_1 = cv2.boundingRect(contours[i])
    cv2.rectangle(drawing,(x_1-5,y_1-5),(x_1+w_1+5,y_1+h_1+5),(255,255,255),1)
    crop_img_contours = drawing[y_1-5:y_1+5+h_1, x_1-5:x_1+w_1+5]
    area_convex = cv2.contourArea(hull_list[i])
    area_contour = cv2.contourArea(contours[i])
    
    x,y,w,h = cv2.boundingRect(contours[i])
    cv2.rectangle(gr,(x,y),(x+w,y+h),(255,255,0),1)
    crop_img = gr[y:y+h, x:x+w]

    number_of_corner=find_Corners(crop_img)
    if area_convex<1000:
        cv2.putText(real_img,"", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if number_of_corner<=2 or number_of_corner==0:
        cv2.putText(real_img," Circle ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    elif (area_convex-area_contour) >1000:
        cv2.putText(real_img,"Star : "+str(int(number_of_corner/2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    elif number_of_corner == 3:
         cv2.putText(real_img,"Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    elif number_of_corner == 4:
         cv2.putText(real_img,"Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    elif number_of_corner == 5:
         cv2.putText(real_img,"Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 255, 255), 1, cv2.LINE_AA)
    elif number_of_corner == 6:
         cv2.putText(real_img,"Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        cv2.putText(real_img,"Corner: "+str(number_of_corner), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

   
plt.imshow(real_img),plt.show() 
cv2.waitKey(0)
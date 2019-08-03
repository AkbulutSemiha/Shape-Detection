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
#img_fill=cv2.resize(img_fill,(800,600))
#shape_img = cv2.resize(shape_img,(480 , 360))
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
"""dest = cv2.cornerHarris(img_erosion1, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] """
# the window showing output image with corners 
"""cv2.namedWindow('kernel1',cv2.WINDOW_NORMAL)
cv2.imshow('kernel1', shape_img)"""
kernel2=np.array([[1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]],np.uint8)
img_erosion2 = cv2.erode(img_dilate, kernel2, iterations=1)
"""dest = cv2.cornerHarris(img_erosion2, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners """
"""cv2.namedWindow('kernel2',cv2.WINDOW_NORMAL)
cv2.imshow('kernel2', shape_img)
"""

bw_1= cv2.bitwise_and(img_erosion1 , img_erosion2)
"""dest = cv2.cornerHarris(bw_1, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners
cv2.namedWindow('kernel1-2',cv2.WINDOW_NORMAL)
cv2.imshow('kernel1-2', shape_img)
"""

kernel3=np.array([[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]],np.uint8)
img_erosion3 = cv2.erode(img_dilate, kernel3, iterations=1)
"""#cv2.namedWindow('Erosion3',cv2.WINDOW_NORMAL)
#cv2.imshow('Erosion3', img_erosion3)
dest = cv2.cornerHarris(img_erosion3, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
cv2.namedWindow('kernel3',cv2.WINDOW_NORMAL)
cv2.imshow('kernel3', shape_img)
"""
kernel4=np.array([[0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]],np.uint8)
img_erosion4 = cv2.erode(img_dilate, kernel4, iterations=1)
"""dest = cv2.cornerHarris(img_erosion4, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
cv2.namedWindow('kernel4',cv2.WINDOW_NORMAL)
cv2.imshow('kernel4', shape_img)
cv2.namedWindow('Erosion4',cv2.WINDOW_NORMAL)
cv2.imshow('Erosion4', img_erosion4)
"""
bw_2= cv2.bitwise_and(img_erosion3 , img_erosion4)
bw_3=cv2.bitwise_and(bw_1,bw_2)
"""cv2.namedWindow('AND',cv2.WINDOW_NORMAL)
cv2.imshow('AND', bw_3)

dest = cv2.cornerHarris(bw_3, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
plt.imshow(shape_img),plt.show()
"""

kernel5=np.array([[0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1]],np.uint8)

img_erosion5 = cv2.erode(img_dilate, kernel5, iterations=1)
"""dest = cv2.cornerHarris(img_erosion5, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
#cv2.namedWindow('Erosion5',cv2.WINDOW_NORMAL)
#cv2.imshow('Erosion5', shape_img)
#cv2.namedWindow('Erosion5',cv2.WINDOW_NORMAL)
#cv2.imshow('Erosion5', img_erosion5)
"""
kernel6=np.array([[1, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0]],np.uint8)

img_erosion6 = cv2.erode(img_dilate, kernel6, iterations=1)
"""dest = cv2.cornerHarris(img_erosion6, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
cv2.namedWindow('Erosion6',cv2.WINDOW_NORMAL)
cv2.imshow('Erosion6', shape_img)
"""

bw_4= cv2.bitwise_and(img_erosion5 , img_erosion6)
"""dest = cv2.cornerHarris(bw_2, 2, 5, 0.05) 
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
# Reverting back to the original image, 
# with optimal threshold value
shape_img[dest > 0.01 * dest.max()]=[255, 0, 255] 
# the window showing output image with corners 
cv2.namedWindow('Erosion5-6',cv2.WINDOW_NORMAL)
cv2.imshow('Erosion5-6', shape_img)
"""
img_and_total=cv2.bitwise_and(bw_4,bw_3)
im2, contours, hierarchy = cv2.findContours(img_and_total,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
img_contour = cv2.drawContours(shape_img, contours,-1, (255,0,255), 1)
#plt.imshow(img),plt.show()
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
    #plt.imshow(crop_img_contours),plt.show()
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

    #print("Corner "+str(number_of_corner))
#cv2.namedWindow('AND_Total_Harris',cv2.WINDOW_NORMAL)
#cv2.imshow('AND_Total_Harris',real_img )
plt.imshow(real_img),plt.show() 
cv2.waitKey(0)
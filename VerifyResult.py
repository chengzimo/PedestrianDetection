# coding:utf-8
import cv2
img = r'Person.jpg'
image = cv2.imread(img)
image = cv2.rectangle(image, (40, 51), (203, 369), (0, 0, 255), thickness=3)
cv2.namedWindow("Result", 0)
cv2.imshow("Result", image)
cv2.waitKey()
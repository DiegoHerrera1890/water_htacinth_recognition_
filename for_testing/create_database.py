import os
import cv2

img = cv2.imread('/home/diego/water_htacinth_recognition_/dataset/ir1/ir1_1.png', cv2.IMREAD_GRAYSCALE)

print(img.shape)

cv2.imshow('Grayscale',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
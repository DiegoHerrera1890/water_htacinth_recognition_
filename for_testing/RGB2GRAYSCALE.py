import os
import cv2
import glob

imgpath = '/home/diego/water_htacinth_recognition_/dataset'
subfolders = ['ir1', 'ir2']

for path_ in subfolders:
    print(os.path.join(imgpath, path_)) 
    subfolder_path = os.path.join(imgpath, path_)
    images_files = glob.glob(os.path.join(subfolder_path, '*.png'))

    for image_file in images_files:
        output_file = os.path.join(subfolder_path, os.path.basename(image_file))
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Grayscale',img)
        # cv2.imwrite(output_file, img)
        print("output", output_file)
        print(img.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()
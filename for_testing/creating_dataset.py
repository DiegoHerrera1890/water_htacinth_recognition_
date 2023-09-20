import os
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

noir_path = '/home/diego/water_htacinth_recognition_/dataset/noir'
ir1_path = '/home/diego/water_htacinth_recognition_/dataset/ir1/'
ir2_path = '/home/diego/water_htacinth_recognition_/dataset/ir2/'

output_dir = '/home/diego/water_htacinth_recognition_/dataset/multisp_img'
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 19):
    rgb_image = cv2.imread(os.path.join(noir_path, f"rgbd_{i}.png"), cv2.COLOR_BGR2RGB)
    print(rgb_image.shape)
    ir1_image = cv2.imread(os.path.join(ir1_path, f"ir1_{i}.png"), cv2.IMREAD_GRAYSCALE)
    ir2_image = cv2.imread(os.path.join(ir2_path, f"ir2_{i}.png"), cv2.IMREAD_GRAYSCALE)

    multispectral_image = np.dstack((rgb_image, ir1_image, ir2_image))

    output_filename = os.path.join(output_dir, f"image_{i}.tiff")

    tiff.imsave(output_filename, multispectral_image)

    print(f"Processed and saved image: {output_filename}")
    multispectral_image = tiff.imread(output_filename)
    print("Shape of TIFF image:", multispectral_image.shape)







# Read the TIFF file
multispectral_image = tiff.imread(output_filename)

# Get the number of channels in the image
num_channels = multispectral_image.shape[-1]
print('the image {} has {} channels'.format(output_filename, num_channels))
# Plot each channel separately
for channel_index in range(num_channels):
    plt.subplot(1, num_channels, channel_index + 1)
    plt.imshow(multispectral_image[..., channel_index], cmap='gray')
    plt.title(f'Channel {channel_index + 1}')
    plt.axis('off')

plt.show()

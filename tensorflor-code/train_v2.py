"""
Semantic Segmentation using U-net architecture with tensorflow-keras
@Diego Herrera Monday 13-02-2023 
"""
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
from tensorflow import keras
import segmentation_models as sm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from simple_multi_unet_model import multi_unet_model, jacard_coef
from tensorflow.keras.utils import to_categorical

sm.set_framework('tf.keras')
sm.framework()


def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == BACKGROUND, axis=-1)] = 0
    label_seg[np.all(label == Grass, axis=-1)] = 1
    label_seg[np.all(label == Pavement, axis=-1)] = 2
    label_seg[np.all(label == Traversable, axis=-1)] = 3
    label_seg[np.all(label == Branches, axis=-1)] = 4
    label_seg[np.all(label == Person, axis=-1)] = 5
    label_seg[np.all(label == Vehicle, axis=-1)] = 6
    label_seg[np.all(label == Robot, axis=-1)] = 7
    label_seg[np.all(label == Tree, axis=-1)] = 8
    label_seg[np.all(label == Dynamic, axis=-1)] = 9

    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels

    return label_seg


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


scaler = MinMaxScaler()
root_directory = '/home/diego/kisui_nav_ai-main/u-net-tensoflow/Dataset/'
# patch_size = 512
image_dataset = []

for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    print(dirname)
    if dirname == 'images_prepped_train':
        images = os.listdir(path)
        for image_name in images:
            if image_name.endswith(".png"):
                image = cv2.imread(path + "/" + image_name, 1)
                image = Image.fromarray(image)
                image = image.resize((672, 384))  #960x540
                image = np.array(image)
                image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
                # image = image[0] # Drop the extra unnecessary dimension
                single_patch_img = image
                image_dataset.append(single_patch_img)

mask_dataset = []

for path, subdirs, files in os.walk(root_directory):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'annotations_prepped_train':  # Find all 'images' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for mask_name in masks:
            if mask_name.endswith(".png"):
                mask = cv2.imread(path + "/" + mask_name,1)  # Read each image as Grey (or color but remember to map
                # each color to an integer)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = Image.fromarray(mask)
                mask = mask.resize((672, 384))  # Try not to resize for semantic segmentation
                mask = np.array(mask)
                # mask = mask[0] # Drop the extra unnecessary dimension
                single_patch_mask = mask
                mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

print(image_dataset.shape)
print(mask_dataset.shape)

image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (384, 672, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (384, 672, 3)))
plt.show()

# Convert HEX to RGB array

BACKGROUND = '#ffffff'.lstrip('#')
BACKGROUND = np.array(tuple(int(BACKGROUND[i:i + 2], 16) for i in (0, 2, 4)))  # 60, 16, 152

Grass = '#2ca02c'.lstrip('#')
Grass = np.array(tuple(int(Grass[i:i + 2], 16) for i in (0, 2, 4)))  # 132, 41, 246

Pavement = '#1f77b4'.lstrip('#')
Pavement = np.array(tuple(int(Pavement[i:i + 2], 16) for i in (0, 2, 4)))  # 110, 193, 228

Traversable = 'ff7f0e'.lstrip('#')
Traversable = np.array(tuple(int(Traversable[i:i + 2], 16) for i in (0, 2, 4)))  # 254, 221, 58

Branches = 'd62728'.lstrip('#')
Branches = np.array(tuple(int(Branches[i:i + 2], 16) for i in (0, 2, 4)))  # 226, 169, 41

Person = '#9467bd'.lstrip('#')
Person = np.array(tuple(int(Person[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

Vehicle = '#8c564b'.lstrip('#')
Vehicle = np.array(tuple(int(Vehicle[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

Robot = '#e377c2'.lstrip('#')
Robot = np.array(tuple(int(Robot[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

Tree = '#7f7f7f'.lstrip('#')
Tree = np.array(tuple(int(Tree[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

Dynamic = '#bcbd22'.lstrip('#')
Dynamic = np.array(tuple(int(Dynamic[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

label = single_patch_mask

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("label shape", label.shape)
print("labels shape", labels.shape)
print("Unique labels in label dataset are: ", np.unique(labels))

# Sanity check, view few images
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

n_classes = len(np.unique(labels))

labels_cat = to_categorical(labels, num_classes=n_classes)

print(n_classes)

# Split the data into four folders
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)
print(labels_cat.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

weights = compute_class_weight(class_weight="balanced", classes=np.unique(np.ravel(labels, order='C')),
                               y=np.ravel(labels, order='C'))
print(weights)
# weights = [0.1857, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]

model = get_model()
# model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train,
                     batch_size=4,
                     verbose=1,
                     epochs=10,
                     validation_data=(X_test, y_test),
                     shuffle=False)

model.save('models/final_model.hdf5')

history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

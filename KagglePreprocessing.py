import numpy as np
from PIL import Image
from tqdm import tqdm
import fnmatch
import glob
import os


# Function for gathering images
def get_images(folder, class_num):
    images = []
    total = len(fnmatch.filter(os.listdir(folder), "*.jpg"))
    for file in tqdm(glob.glob(f"{folder}/*.jpg"), total=total, desc="Importing images"):
        try:
            img = Image.open(file).convert("L").resize((50, 50), Image.ANTIALIAS)
            if img is not None:
                images.append([np.array(img), class_num])
        except:
            pass
    return np.array(images)


# Getting the images, Cat is 0, Dog is 1
data = get_images("KaggleCND/Cat", 0) + get_images("KaggleCND/Dog", 1)

# Shuffling data
np.random.shuffle(data)

# Separate data into features and labels
features, labels = [], []
for feature, label in tqdm(data, total=len(data), desc="Pre-processing"):
    features.append(feature/255)  # Normalize data between 0 and 1
    labels.append(label)
features = np.array(features).reshape((-1, 50*50*1))
labels = np.array(labels)

# Debug shapes
print(labels.shape)
print(features.shape)


# Save data
np.save("features", features)
np.save("labels", labels)

import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================
# Dataset
# ==========================

DATASET_PATH = "dataset"

IMG_SIZE = 224


data = []
labels = []


categories = [
    "with_mask",
    "without_mask"
]


# ==========================
# Load Images
# ==========================

for category in categories:

    folder_path = os.path.join(
        DATASET_PATH,
        category
    )

    label = categories.index(category)


    for img_name in os.listdir(folder_path):

        img_path = os.path.join(
            folder_path,
            img_name
        )


        img = cv2.imread(img_path)


        if img is not None:

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )


            data.append(img)
            labels.append(label)



print("Total Images:", len(data))


data = np.array(data)
labels = np.array(labels)



# MobileNetV2 preprocessing

data = tf.keras.applications.mobilenet_v2.preprocess_input(data)



# ==========================
# Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    data,

    labels,

    test_size=0.2,

    random_state=42

)


print("Training:", X_train.shape)
print("Testing:", X_test.shape)



# ==========================
# Data Augmentation
# ==========================

datagen = ImageDataGenerator(

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True

)



# ==========================
# MobileNetV2 Base
# ==========================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)



# Freeze base model

base_model.trainable = False



# ==========================
# Final Model
# ==========================

model = Sequential([


    base_model,


    GlobalAveragePooling2D(),


    Dense(
        128,
        activation="relu"
    ),


    Dropout(0.5),


    Dense(
        1,
        activation="sigmoid"
    )

])



model.summary()



# ==========================
# Compile
# ==========================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



# ==========================
# Train
# ==========================

history = model.fit(

    datagen.flow(
        X_train,
        y_train,
        batch_size=32
    ),

    epochs=10,

    validation_data=(X_test,y_test)

)



# ==========================
# Save
# ==========================

if not os.path.exists("models"):

    os.makedirs("models")



model.save(
    "models/mask_detector.h5"
)


print("MobileNetV2 Model Saved Successfully!")
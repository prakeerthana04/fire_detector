import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os

# -----------------------------
# Parameters
# -----------------------------
img_height, img_width = 224, 224  # MobileNetV2 preferred size
batch_size = 16
dataset_path = "dataset"
class_names = ["fire", "no_fire"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔥 Fire Detector 🔥")
st.write("Upload an image to detect Fire / No Fire")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

# -----------------------------
# Training function
# -----------------------------
def train_model():
    st.write("Training model...")

    # Load dataset
    train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )

    # Preprocess the dataset for MobileNetV2
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds   = val_ds.map(lambda x, y: (preprocess_input(x), y))

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)
    ])

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # MobileNetV2 Base
    base_model = MobileNetV2(input_shape=(img_height,img_width,3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False  # freeze pre-trained layers

    # Build model
    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_ds, validation_data=val_ds, epochs=15)

    # Save model
    model.save("fire_detection_mobilenet.h5")
    st.success("✅ Model trained and saved as fire_detection_mobilenet.h5")
    return model

# -----------------------------
# Load or Train Model
# -----------------------------
if os.path.exists("fire_detection_mobilenet.h5"):
    model = load_model("fire_detection_mobilenet.h5")
else:
    model = train_model()

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_resized = img.resize((img_width,img_height))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.write(f"**Prediction:** {label.upper()} 🔥" if label=="fire" else f"**Prediction:** {label.upper()} ✅")
    st.write(f"**Confidence:** {confidence:.2f}%")

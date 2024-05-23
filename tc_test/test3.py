import urllib
import zipfile
import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

# This function normalizes the images.
def preprocess(image, label):
    # Rescale by 1/.255 (Normalize the images)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    # Normalizes train and validation datasets using the preprocess() function.
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20
    )

    return model

# Save the model as a .h5 file
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
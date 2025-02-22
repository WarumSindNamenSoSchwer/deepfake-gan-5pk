import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import glob
from datetime import timedelta

# Get absolute path to the "Bilder" directory, no matter where the script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the running script
image_dir = os.path.abspath(os.path.join(script_dir, "..", "Bilder"))
image_size = (128, 128)

# Print directory and file information for debugging
print("Current working directory:", os.getcwd())
print("Image directory:", image_dir)
print("Files in image directory:", os.listdir(image_dir))

# Hyperparameter
noise_dim = 100  # Dimension des Rauschens
BATCH_SIZE = 36  # Best practice batch size
EPOCHS = 9000  # Definieren wir hier, damit es später leichter zugänglich ist.

# Create a single project folder when the script starts
project_timestamp = time.strftime("%d.%m.%Y_%H.%M")
project_dir = os.path.abspath(os.path.join(script_dir, f"../Generierte_Bilder/{project_timestamp}"))
os.makedirs(project_dir, exist_ok=True)
checkpoint_dir = os.path.join(project_dir, "checkpoints")  # Checkpoint Ordner innerhalb des Projektordners
os.makedirs(checkpoint_dir, exist_ok=True)  # erstellt den Ordner

# Funktion zum Generieren und Anzeigen von Bildern
def plot_generated_images(epoch, generator, project_dir, examples=25, dim=(5, 5), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 128, 128)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(project_dir, f"Bild_bei_Epoche_{epoch:04d}.png")
    plt.savefig(save_path)
    plt.close()

# Funktion zum Löschen alter Checkpoints
def delete_old_checkpoints(checkpoint_dir, days=3):
    current_time = time.time()
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        file_modified = os.path.getmtime(file_path)
        if current_time - file_modified > days * 24 * 3600:
            os.remove(file_path)
            print(f"Gelöschter alter Checkpoint: {filename}")

# Funktion zum Speichern von Checkpoints
def save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir):
    generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch:04d}.h5"))
    discriminator.save(os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch:04d}.h5"))
    print(f"Checkpoint gespeichert für Epoche {epoch} unter {checkpoint_dir}")
    delete_old_checkpoints(checkpoint_dir)

def load_and_preprocess_image(path, image_size=(128, 128)):
    image = tf.io.read_file(path)
    # Try decoding as JPEG, if it fails, decode as PNG
    try:
        image = tf.image.decode_jpeg(image, channels=1)
    except:
        image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def create_dataset(image_dir, batch_size=100, image_size=(128, 128)):
    # Use a wildcard to match any image file extension
    image_paths = tf.data.Dataset.list_files(os.path.join(image_dir, "*.*"))
    dataset = image_paths.map(lambda x: load_and_preprocess_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Generator-Modell erstellen
def make_generator():
    model = keras.Sequential([
        keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(noise_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Reshape((8, 8, 512)),
        keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
    ])
    return model

# Diskriminator-Modell erstellen
def make_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(128, 128, 1)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])
    return model

# Verlustfunktionen
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Trainingsschritt
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(tf.reshape(images, (-1, 128, 128, 1)), training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Trainingsschleife
def train(dataset, epochs, project_dir, checkpoint_dir, initial_epoch=0):
    print("\n\nStarting training...\n\n")

    for epoch in range(initial_epoch, epochs):
        start = time.time()
        gen_loss_list = []
        disc_loss_list = []
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

        gen_loss = sum(gen_loss_list) / len(gen_loss_list)
        disc_loss = sum(disc_loss_list) / len(disc_loss_list)

        print('Epoch {}, gen_loss={}, disc_loss={}, time={}'.format(epoch + 1, gen_loss, disc_loss, time.time() - start))

        if (epoch + 1) % 25 == 0:
            plot_generated_images(epoch, generator, project_dir)

        if (epoch + 1) % 50 == 0:
            save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir)

#Modelle erstellen
generator = make_generator()
discriminator = make_discriminator()

#Optimierer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Laden der Modelle (nicht die Optimierer)
def load_models_from_checkpoint(checkpoint_dir):
    generator_files = glob.glob(os.path.join(checkpoint_dir, "generator_epoch_*.h5"))
    discriminator_files = glob.glob(os.path.join(checkpoint_dir, "discriminator_epoch_*.h5"))

    generator_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    discriminator_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if generator_files and discriminator_files:
        latest_generator = generator_files[-1]
        latest_discriminator = discriminator_files[-1]

        loaded_generator = tf.keras.models.load_model(latest_generator)
        loaded_discriminator = tf.keras.models.load_model(latest_discriminator)

        epoch = int(latest_generator.split('_')[-1].split('.')[0])

        print(f"Generator geladen von: {latest_generator}")
        print(f"Diskriminator geladen von: {latest_discriminator}")
        print(f"Training wird ab Epoche {epoch + 1} fortgesetzt")

        return loaded_generator, loaded_discriminator, epoch
    else:
        print("Keine Checkpoints gefunden. Starte Training von vorn.")
        return None, None, 0

# Trainingsschleife starten oder fortsetzen
def start_training(image_dir, EPOCHS, project_dir, checkpoint_dir):
    answer = input("Soll das Training fortgesetzt werden? (j/n): ")

    dataset = create_dataset(image_dir, batch_size=BATCH_SIZE, image_size=image_size)

    if answer.lower() == 'j':
        generator, discriminator, initial_epoch = load_models_from_checkpoint(checkpoint_dir)
        if generator and discriminator:
            train(dataset, EPOCHS, project_dir, checkpoint_dir, initial_epoch=initial_epoch)
        else:
            print("Kein Checkpoint gefunden. Starte Training von vorn.")
            train(dataset, EPOCHS, project_dir, checkpoint_dir)
    else:
        print("Starte Training von vorn.")
        train(dataset, EPOCHS, project_dir, checkpoint_dir)

#Training starten
start_training(image_dir, EPOCHS, project_dir, checkpoint_dir)

# Generiere ein finales Bild nach dem Training
noise = np.random.normal(0, 1, size=[1, noise_dim])
generated_image = generator.predict(noise)
generated_image = generated_image.reshape(128, 128)
plt.imshow(generated_image, interpolation='nearest', cmap='gray_r')
plt.axis('off')
plt.show()

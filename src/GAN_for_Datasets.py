import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

# Dataset laden
dataset = load_dataset("huggan/cats", split="train")
image_size = (64, 64)
images = []

# Bilder aus dem Datensatz verarbeiten
for item in dataset:
    img = Image.fromarray(item["image"]).convert('L')  # In Graustufen umwandeln
    img = img.resize(image_size)
    img_array = np.array(img).astype('float32')
    img_array = (img_array - 127.5) / 127.5  # Werte normalisieren (-1 bis 1)
    images.append(img_array)

input_images = np.array(images)

if len(input_images) == 0:
    raise ValueError("Keine Bilder im Ordner gefunden oder alle Bilder konnten nicht geladen werden.")

print("Form der Eingabebilder:", input_images.shape)

# Hyperparameter
noise_dim = 100 #Dimension des Rauschens
BATCH_SIZE = len(images)  # Die Batch-Größe sollte der Anzahl der Bilder entsprechen

# Generator-Modell erstellen
def make_generator():
    model = keras.Sequential([
        keras.layers.Dense(4*4*512, use_bias=False, input_shape=(noise_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Reshape((4, 4, 512)),

        keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])

    return model

# Diskriminator-Modell erstellen
def make_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=(64, 64, 1)),
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

# Modelle erstellen
generator = make_generator()
discriminator = make_discriminator()

# Optimierer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

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

        # Dimensionen anpassen für den Diskriminator
        real_output = discriminator(tf.expand_dims(images, -1), training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss




# Create a single project folder when the script starts
project_timestamp = time.strftime("%d.%m.%Y_%H.%M")
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the running script
project_dir = os.path.abspath(os.path.join(script_dir, f"../Generierte_Bilder_Cats/{project_timestamp}"))
os.makedirs(project_dir, exist_ok=True)

# Funktion zum Generieren und Anzeigen von Bildern
def plot_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 64, 64)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()

    # Always save images in the global `project_dir`
    save_path = os.path.join(project_dir, f"Bild_bei_Epoche_{epoch:04d}.png")
    plt.savefig(save_path)
    plt.close()



# Trainingsschleife
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_loss_list = []
        disc_loss_list = []

        # Jedes Bild im Datensatz als separaten Batch behandeln
        for image in dataset:
            # Dimensionen für das Training anpassen (füge Batch-Dimension hinzu)
            image = np.expand_dims(image, 0)
            gen_loss, disc_loss = train_step(image)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

        gen_loss = sum(gen_loss_list) / len(gen_loss_list)
        disc_loss = sum(disc_loss_list) / len(disc_loss_list)

        print('Epoch {}, gen_loss={}, disc_loss={}, time={}'.format(epoch+1, gen_loss, disc_loss, time.time()-start))
        if (epoch + 1) % 25 == 0:
            plot_generated_images(epoch, generator)



EPOCHS = 1000

# Training starten
train(input_images, EPOCHS)

# Generiere ein finales Bild nach dem Training
noise = np.random.normal(0, 1, size=[1, noise_dim])
generated_image = generator.predict(noise)
generated_image = generated_image.reshape(64, 64)
plt.imshow(generated_image, interpolation='nearest', cmap='gray_r')
plt.axis('off')
plt.show()

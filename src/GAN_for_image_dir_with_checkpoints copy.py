import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import glob
from datetime import timedelta

# Gewünschte Auflösung (Quadratisch)
gewollte_res = 1984  # Angepasst an die tatsächliche Bildgröße, muss durch 16 teilbar sein

# Daten laden und vorbereiten
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.abspath(os.path.join(script_dir, "../Bilder"))
image_size = (gewollte_res, gewollte_res)
images = []

for filename in os.listdir(image_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(image_size, Image.LANCZOS)  # Verwenden Sie LANCZOS für bessere Qualität
            img_array = np.array(img).astype('float32')
            img_array = (img_array - 127.5) / 127.5
            images.append(img_array)
        except Exception as e:
            print(f"Fehler beim Laden von {filename}: {e}")

input_images = np.array(images)

if len(input_images) == 0:
    raise ValueError("Keine Bilder im Ordner gefunden oder alle Bilder konnten nicht geladen werden.")

print("Tatsächliche Form der Eingabebilder:", input_images.shape)

# Hyperparameter
noise_dim = 100
BATCH_SIZE = 36  # Reduzierte Batch-Größe wegen des hohen Speicherbedarfs
EPOCHS = 1000

# Generator-Modell erstellen
def make_generator():
    model = keras.Sequential([
        keras.layers.Dense((gewollte_res // 16) * (gewollte_res // 16) * 256, use_bias=False, input_shape=(noise_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Reshape(((gewollte_res // 16), (gewollte_res // 16), 256)),

        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Diskriminator-Modell erstellen
def make_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=(gewollte_res, gewollte_res, 1)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),

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

        real_output = discriminator(images, training=True)
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
project_dir = os.path.abspath(os.path.join(script_dir, f"../Generierte_Bilder/{project_timestamp}"))
os.makedirs(project_dir, exist_ok=True)
checkpoint_dir = os.path.join(project_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Funktion zum Generieren und Anzeigen von Bildern
def plot_generated_images(epoch, generator, project_dir, examples=4, dim=(2, 2), figsize=(20, 20)):
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, gewollte_res, gewollte_res)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
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
        if (current_time - os.path.getmtime(file_path)) > (days * 24 * 60 * 60):
            os.remove(file_path)
            print(f"Gelöschter alter Checkpoint: {filename}")

# Funktion zum Speichern von Checkpoints
def save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir):
    generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch:04d}.h5"))
    discriminator.save(os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch:04d}.h5"))
    print(f"Checkpoint gespeichert für Epoche {epoch} unter {checkpoint_dir}")
    delete_old_checkpoints(checkpoint_dir)

# Trainingsschleife
def train(dataset, epochs, project_dir, checkpoint_dir, initial_epoch=0):
    print("\n\nStarting training...\n\n")

    for epoch in range(initial_epoch, epochs):
        start = time.time()
        gen_loss_list = []
        disc_loss_list = []

        for image in dataset:
            image = np.expand_dims(image, 0)
            gen_loss, disc_loss = train_step(image)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)

        gen_loss = sum(gen_loss_list) / len(gen_loss_list)
        disc_loss = sum(disc_loss_list) / len(disc_loss_list)

        print('Epoch {}, gen_loss={}, disc_loss={}, time={}'.format(epoch+1, gen_loss, disc_loss, time.time()-start))

        if (epoch + 1) % 25 == 0:
            plot_generated_images(epoch, generator, project_dir)

        if (epoch + 1) % 50 == 0:
            save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir)

# Modelle erstellen
generator = make_generator()
discriminator = make_discriminator()

# Optimierer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Laden der Modelle
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
def start_training(input_images, EPOCHS, project_dir, checkpoint_dir):
    answer = input("Soll das Training fortgesetzt werden? (j/n): ")

    if answer.lower() == 'j':
        generator, discriminator, initial_epoch = load_models_from_checkpoint(checkpoint_dir)
        if generator and discriminator:
            train(input_images, EPOCHS, project_dir, checkpoint_dir, initial_epoch=initial_epoch)
        else:
            print("Kein Checkpoint gefunden. Starte Training von vorn.")
            train(input_images, EPOCHS, project_dir, checkpoint_dir)
    else:
        print("Starte Training von vorn.")
        train(input_images, EPOCHS, project_dir, checkpoint_dir)

# Training starten
start_training(input_images, EPOCHS, project_dir, checkpoint_dir)

# Generiere ein finales Bild nach dem Training
noise = np.random.normal(0, 1, size=[1, noise_dim])
generated_image = generator.predict(noise)
generated_image = generated_image.reshape(gewollte_res, gewollte_res)
plt.figure(figsize=(20, 20))
plt.imshow(generated_image, interpolation='nearest', cmap='gray_r')
plt.axis('off')
plt.show()

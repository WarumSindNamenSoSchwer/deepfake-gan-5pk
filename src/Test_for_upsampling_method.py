import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import glob

# Gewünschte Auflösung (Quadratisch, muss durch 16 teilbar sein)
gewollte_res = 176

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
            img = img.resize(image_size, Image.LANCZOS)
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
EPOCHS = 26

# Generator-Modell erstellen (mit Option für Upsampling-Block statt Conv2DTranspose)
def make_generator(gewollte_res, noise_dim=100, use_upsampling=True):
    start_dim = 4  # Startgröße im latenten Raum
    num_layers = int(np.log2(gewollte_res // start_dim))
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(start_dim * start_dim * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((start_dim, start_dim, 256)))
    
    filters = 128
    if use_upsampling:
        # Verwende UpSampling2D + Conv2D-Blöcke (reduzieren Checkerboard-Artefakte)
        for _ in range(num_layers):
            model.add(keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'))
            model.add(keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU())
            filters = max(filters // 2, 16)
    else:
        # Klassische Variante mit Conv2DTranspose
        for _ in range(num_layers):
            model.add(keras.layers.Conv2DTranspose(filters, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU())
            filters = max(filters // 2, 16)
    
    # Finale Schicht zur Ausgabe des Bildes
    # Hier kann man ebenfalls eine normale Faltung nutzen, um Details zu glätten
    model.add(keras.layers.Conv2D(1, (3, 3), padding='same', use_bias=False, activation='tanh'))
    return model

# Diskriminator-Modell erstellen
def make_discriminator(gewollte_res):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(gewollte_res, gewollte_res, 1)))
    
    filters = 64
    num_layers = int(np.log2(gewollte_res // 4))
    for _ in range(num_layers):
        model.add(keras.layers.Conv2D(filters, (5, 5), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.3))
        filters = min(filters * 2, 512)
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    return model

# Verlustfunktionen
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Trainingsschritt (als tf.function für bessere Performance)
@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_images = tf.expand_dims(images, -1)  # (gewollte_res, gewollte_res) -> (gewollte_res, gewollte_res, 1)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Erstelle Projekt- und Checkpoint-Ordner
project_timestamp = time.strftime("%d.%m.%Y_%H.%M")
project_dir = os.path.abspath(os.path.join(script_dir, f"../Generierte_Bilder/{project_timestamp}"))
os.makedirs(project_dir, exist_ok=True)
checkpoint_dir = os.path.join(project_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Funktion zum Generieren und Anzeigen von Bildern
def plot_generated_images(epoch, generator, project_dir, examples=25, dim=(5, 5), figsize=(50, 50)):
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, gewollte_res, gewollte_res)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(project_dir, f"Bild_bei_Epoche_{epoch:04d}.png")
    plt.savefig(save_path)
    plt.close()

# Funktion zum Löschen alter Checkpoints (älter als 3 Tage)
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

# Funktion zum Laden der Modelle aus Checkpoints
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

# Trainingsschleife, die den trainierten Generator und Diskriminator zurückgibt
def train(dataset, epochs, project_dir, checkpoint_dir, initial_epoch=0):
    print("\n\nStarting training...\n\n")
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # Erstelle Modelle von Grund auf mit Upsampling-Variante (setze use_upsampling=True)\n    generator = make_generator(gewollte_res, noise_dim, use_upsampling=True)\n    discriminator = make_discriminator(gewollte_res)\n    # Falls initial_epoch > 0, Modelle aus Checkpoints laden\n    if initial_epoch > 0:\n        loaded_generator, loaded_discriminator, loaded_epoch = load_models_from_checkpoint(checkpoint_dir)\n        if loaded_generator and loaded_discriminator:\n            generator = loaded_generator\n            discriminator = loaded_discriminator\n            initial_epoch = loaded_epoch + 1\n            print(f\"Training wird ab Epoche {initial_epoch} fortgesetzt\")\n    for epoch in range(initial_epoch, epochs):\n        start = time.time()\n        gen_loss_list = []\n        disc_loss_list = []\n        for image in dataset:\n            image = np.expand_dims(image, 0)\n            gen_loss, disc_loss = train_step(image, generator, discriminator, generator_optimizer, discriminator_optimizer)\n            gen_loss_list.append(gen_loss)\n            disc_loss_list.append(disc_loss)\n        gen_loss_avg = sum(gen_loss_list) / len(gen_loss_list)\n        disc_loss_avg = sum(disc_loss_list) / len(disc_loss_list)\n        print(f\"Epoch {epoch+1}, gen_loss={gen_loss_avg}, disc_loss={disc_loss_avg}, time={time.time()-start:.2f} Sekunden\")\n        if (epoch + 1) % 25 == 0:\n            plot_generated_images(epoch, generator, project_dir)\n        if (epoch + 1) % 50 == 0:\n            save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir)\n    return generator, discriminator

# Funktion zum Starten oder Fortsetzen des Trainings
def start_training(input_images, EPOCHS, project_dir, checkpoint_dir):
    answer = input("Soll das Training fortgesetzt werden? (j/n): ")
    dataset = tf.data.Dataset.from_tensor_slices(input_images).batch(BATCH_SIZE)
    if answer.lower() == 'j':
        loaded_generator, loaded_discriminator, initial_epoch = load_models_from_checkpoint(checkpoint_dir)
        if loaded_generator and loaded_discriminator:
            trained_generator, trained_discriminator = train(input_images, EPOCHS, project_dir, checkpoint_dir, initial_epoch=initial_epoch)
        else:
            print("Kein Checkpoint gefunden. Starte Training von vorn.")
            trained_generator, trained_discriminator = train(input_images, EPOCHS, project_dir, checkpoint_dir)
    else:
        print("Starte Training von vorn.")
        trained_generator, trained_discriminator = train(input_images, EPOCHS, project_dir, checkpoint_dir)
    return trained_generator, trained_discriminator


# Globale Variable für das zuletzt gezeigte Fenster
last_fig = None

def show_generated_image(generator, noise_dim, gewollte_res, figsize=(20,20)):
    global last_fig
    # Falls bereits ein Fenster offen ist, schließe es
    if last_fig is not None:
        plt.close(last_fig)
    noise = np.random.normal(0, 1, size=[1, noise_dim])
    gen_img = generator.predict(noise)
    gen_img = gen_img.reshape(gewollte_res, gewollte_res)
    fig = plt.figure(figsize=figsize)
    plt.imshow(gen_img, interpolation='nearest', cmap='gray_r')
    plt.axis('off')
    plt.show(block=False)
    # Speichere das aktuelle Figure-Handle für den nächsten Aufruf
    last_fig = fig

# Training starten und den trainierten Generator zurückgeben
generator, discriminator = start_training(input_images, EPOCHS, project_dir, checkpoint_dir)

# Generiere ein finales Bild nach dem Training mit dem trainierten Generator
noise = np.random.normal(0, 1, size=[1, noise_dim])
generated_image = generator.predict(noise)
generated_image = generated_image.reshape(gewollte_res, gewollte_res)
plt.figure(figsize=(20, 20))
plt.imshow(generated_image, interpolation='nearest', cmap='gray_r')
plt.axis('off')
plt.show()

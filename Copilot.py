import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

image_dir = "Bilder"
image_size = (28, 28)
images = []

for filename in os.listdir(image_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).convert('L')  # Konvertiere zu Graustufen
        img = img.resize(image_size)
        img_array = np.array(img).reshape(28, 28, 1).astype('float32')
        img_array = (img_array - 127.5) / 127.5  # Normalisierung
        images.append(img_array)

input_images = np.array(images)
print(input_images.shape)

# Reshape und Normalisierung
input_images = input_images.reshape(-1, 28, 28, 1).astype('float32')
input_images = (input_images - 127.5) / 127.5

print(input_images.shape)

def make_generator():
    model = keras.Sequential([
        keras.layers.Dense(7*7*256, input_shape=(100,)),
        keras.layers.Reshape((7, 7, 256)),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model


def make_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])
    return model



cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

generator = make_generator()
discriminator = make_discriminator()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
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

EPOCHS = 1000
BATCH_SIZE = len(images)

for epoch in range(EPOCHS):
    gen_loss, disc_loss = train_step(input_images)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")



noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

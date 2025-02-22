import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Laden Sie das Bild
img = Image.open("Bilder/OlafScholz.png")

# Konvertieren Sie es in Graustufen
img_gray = img.convert('L').resize((28, 28))

# Konvertieren Sie es in ein Numpy-Array
input_images = np.array(img_gray)

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



EPOCHS = 50
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    for batch in input_images:
        train_step(batch)
    
    # Generieren Sie hier ein Beispielbild und speichern Sie es



noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

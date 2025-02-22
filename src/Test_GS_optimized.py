import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm

# Hyperparameters
NOISE_DIM = 128
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 2e-4
IMAGE_SIZE = 128

# Load and prepare data
def load_and_prepare_data(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(img_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img).astype('float32')
                img_array = (img_array - 127.5) / 127.5
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images)

# Spectral Normalization
def spectral_normalization(w):
    w = tf.reshape(w, [-1, w.shape[-1]])
    u = tf.random.normal([1, w.shape[-1]])
    v = tf.random.normal([w.shape[0], 1])
    for _ in range(10):
        v = tf.matmul(w, u, transpose_a=True)
        v = v / tf.norm(v)
        u = tf.matmul(w, v)
        u = u / tf.norm(u)
    sigma = tf.matmul(tf.matmul(v, w, transpose_a=True), u, transpose_b=True)
    return w / sigma

# Generator
def make_generator():
    noise = Input(shape=(NOISE_DIM,))
    x = Dense(8*8*256, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(1, 4, strides=2, padding='same', use_bias=False, activation='tanh')(x)

    return Model(noise, x)

# Discriminator
def make_discriminator():
    image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = Conv2D(32, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(image)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1)(x)

    return Model(image, x)

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Training step
@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input, output_dir):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# Main training loop
def train(dataset, epochs):
    generator = make_generator()
    discriminator = make_discriminator()

    generator_optimizer = Adam(LEARNING_RATE_GEN, beta_1=0.5)
    discriminator_optimizer = Adam(LEARNING_RATE_DISC, beta_1=0.5)

    output_dir = 'generated_images'
    os.makedirs(output_dir, exist_ok=True)

    seed = tf.random.normal([16, NOISE_DIM])

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

        # Generate and save images
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed, output_dir)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, Time: {time.time()-start:.2f} sec')

    # Generate a final set of images
    generate_and_save_images(generator, epochs, seed, output_dir)

# Load and prepare data
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the running script
image_dir = os.path.abspath(os.path.join(script_dir, "../Bilder"))
images = load_and_prepare_data(image_dir)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(BATCH_SIZE)

# Start training
train(dataset, EPOCHS)

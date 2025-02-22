import os
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Keras imports
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import MaxNorm


# Hyperparameters
NOISE_DIM = 100
BATCH_SIZE = 36
EPOCHS = 50000
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 2e-4
IMAGE_SIZE = 512

def load_and_prepare_data(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img).astype('float32')
                img_array = (img_array - 127.5) / 127.5
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images)

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

def make_generator():
    noise = Input(shape=(NOISE_DIM,))
    x = Dense(4*4*1024, use_bias=False)(noise)  # Adjusted for new resolution
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((4, 4, 1024))(x)  # Adjusted for new resolution

    x = Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #8

    x = Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #16

    x = Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #32

    x = Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #64
    
    x = Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #128

    x = Conv2DTranspose(16, 4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) #256

    x = Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False, activation='tanh')(x) #512

    return Model(noise, x)

def make_discriminator():
    image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # Updated input shape

    x = Conv2D(16, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(image)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #256

    x = Conv2D(32, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #128

    x = Conv2D(64, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #64

    x = Conv2D(128, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #32
    
    x = Conv2D(256, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #16

    x = Conv2D(512, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #8

    x = Conv2D(1024, 4, strides=2, padding='same', 
               kernel_initializer=RandomNormal(0, 0.02),
               kernel_constraint=MaxNorm(3))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x) #4

    x = Flatten()(x)
    x = Dense(1, kernel_constraint=MaxNorm(3))(x) #1

    return Model(image, x)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

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

def generate_and_save_images(model, epoch, test_input, output_dir):
    predictions = model(test_input, training=False)
    
    # Create a figure with a 10x5 grid of subplots
    num_images = test_input.shape[0] #Use test_input instead
    rows = int(np.floor(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))  # Adjust figure size
    fig.subplots_adjust(hspace=0.4, wspace=0.4) #Increase spacing

    axs = axs.ravel()
    
    for i in range(num_images):
        img = (predictions[i] * 0.5) + 0.5
        axs[i].imshow(img)
        axs[i].axis('off')
    
    #Hide any unused subplots
    for i in range(num_images, rows*cols):
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'bild_bei_epoche_{epoch:04d}.png'), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def train(dataset, epochs):
    generator = make_generator()
    discriminator = make_discriminator()

    generator_optimizer = Adam(LEARNING_RATE_GEN, beta_1=0.5)
    discriminator_optimizer = Adam(LEARNING_RATE_DISC, beta_1=0.5)

    # Create a unique directory name
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    output_dir = os.path.join('Generierte_Bilder_Farbig', 
                              f'{timestamp}_Epochen_{EPOCHS}_ImageSize_{IMAGE_SIZE}_NoiseDim_{NOISE_DIM}')
    os.makedirs(output_dir, exist_ok=True)

    num_images = 50
    seed = tf.random.normal([num_images, NOISE_DIM]) #Here is where we adjust the seed size

    for epoch in range(epochs):
        start = time.time()

        gen_loss_avg = []
        disc_loss_avg = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
            gen_loss_avg.append(gen_loss)
            disc_loss_avg.append(disc_loss)

        avg_gen_loss = np.mean(gen_loss_avg)
        avg_disc_loss = np.mean(disc_loss_avg)

        if (epoch + 1) % 100 == 0:
            generate_and_save_images(generator, epoch + 1, seed, output_dir)

        print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time()-start:.2f} sec')

    generate_and_save_images(generator, epochs, seed, output_dir)

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the running script
image_dir = os.path.abspath(os.path.join(script_dir, "../Bilder"))
images = load_and_prepare_data(image_dir)
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(BATCH_SIZE)
train(dataset, EPOCHS)

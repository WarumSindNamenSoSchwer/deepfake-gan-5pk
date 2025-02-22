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
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import for data augmentation

# Hyperparameters
NOISE_DIM = 100
BATCH_SIZE = 1  # Crucially, set this to 1 (or a small number like 2-4).  Important for memory
EPOCHS = 5000
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
    images = np.array(images) #Convert to numpy array *before* augmentation
    num_original_images = len(images) #Store the number of original images

    # Data Augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,       # Rotate images up to 20 degrees
        width_shift_range=0.1,    # Shift images horizontally by up to 10% of the width
        height_shift_range=0.1,   # Shift images vertically by up to 10% of the height
        shear_range=0.1,        # Apply shear transformations
        zoom_range=0.1,         # Zoom in/out by up to 10%
        horizontal_flip=True,     # Flip images horizontally
        fill_mode='nearest'      # Fill in newly created pixels after rotation/shifting
    )

    datagen.fit(images)  # Important: Fit the datagen to your training data
    return datagen.flow(images, batch_size=BATCH_SIZE, shuffle=True), num_original_images  # Return a data generator AND the number of original images

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
    real_output = tf.clip_by_value(real_output, -1e7, 1e7)
    fake_output = tf.clip_by_value(fake_output, -1e7, 1e7)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    fake_output = tf.clip_by_value(fake_output, -1e7, 1e7)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, noise):
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
    
    # Create a figure with a 1x1 grid of subplots
    num_images = test_input.shape[0] #Use test_input instead
    rows = 1
    cols = 1

    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))  # Adjust figure size
    fig.subplots_adjust(hspace=0.4, wspace=0.4) #Increase spacing

    #axs = axs.ravel()
    
    for i in range(num_images):
        img = (predictions[i] * 0.5) + 0.5
        axs.imshow(img)
        axs.axis('off')
    
    #Hide any unused subplots
    #for i in range(num_images, rows*cols):
    #    axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'bild_bei_epoche_{epoch:04d}.png'), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

def train(dataset, epochs, num_original_images):
    generator = make_generator()
    discriminator = make_discriminator()

    generator_optimizer = Adam(LEARNING_RATE_GEN, beta_1=0.5)
    discriminator_optimizer = Adam(LEARNING_RATE_DISC, beta_1=0.5)

    # Create a unique directory name
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    output_dir = os.path.join('Generierte_Bilder_Farbig', 
                              f'Fixed_Noise_Vector_RNG_IMG_GEN_{timestamp}_Epochen_{EPOCHS}_ImageSize_{IMAGE_SIZE}_NoiseDim_{NOISE_DIM}')
    os.makedirs(output_dir, exist_ok=True)

    fixed_noise = tf.random.normal([1, NOISE_DIM]) # One fixed noise vector

    # Training Loop - Modified for ImageDataGenerator
    for epoch in range(epochs):
        start = time.time()

        gen_loss_avg = []
        disc_loss_avg = []

        # Iterate through the data generator (datagen.flow())
        img_count = 0 #Counter to make sure you are not training infinitely if batch size is small
        for image_batch in dataset:  # 'dataset' is now the ImageDataGenerator.flow object
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, fixed_noise)
            gen_loss_avg.append(gen_loss)
            disc_loss_avg.append(disc_loss)

            img_count += BATCH_SIZE

            # Break the inner loop to avoid infinite looping with ImageDataGenerator
            if img_count >= num_original_images:
                break

        avg_gen_loss = np.mean(gen_loss_avg)
        avg_disc_loss = np.mean(disc_loss_avg)

        if (epoch + 1) % 100 == 0:
            generate_and_save_images(generator, epoch + 1, fixed_noise, output_dir)

        print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time()-start:.2f} sec')

    generate_and_save_images(generator, epochs, fixed_noise, output_dir)

# Main execution
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the running script
image_dir = os.path.abspath(os.path.join(script_dir, "../Bilder"))
dataset, num_original_images = load_and_prepare_data(image_dir)  # dataset is now the data generator, not tf.data.Dataset
train(dataset, EPOCHS, num_original_images)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay
import numpy as np
import matplotlib.pyplot as plt

# Define a Residual Block
def residual_block(x, filters, stride=1, training=True):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x, training=training)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x, training=training)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut, training=training)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

# Define the ResVGG model with MaxPooling
def build_resvgg(input_shape=(32, 32, 3), num_classes=100):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stage 1: 64 filters, 3 blocks
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Stage 2: 128 filters, 3 blocks
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=1)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Stage 3: 256 filters, 3 blocks
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=1)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Stage 4: 512 filters, 3 blocks
    x = residual_block(x, 512, stride=1)
    x = residual_block(x, 512, stride=1)
    x = residual_block(x, 512, stride=1)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Global average pooling, dropout, and classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Load and preprocess CIFAR-100 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize pixel values
mean = np.array([0.5071, 0.4867, 0.4408])
std = np.array([0.2675, 0.2565, 0.2761])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Enhanced data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Cosine decay schedule
lr_schedule = CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    alpha=1e-6
)

# Build model
model = build_resvgg()
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max'
)

# Training parameters
batch_size = 64  # Suitable for a modest GPU
epochs = 100

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the final model
val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Final validation accuracy: {val_accuracy:.2f}')

if val_accuracy >= 0.70:
    print(f'Reached target validation accuracy of {val_accuracy:.2f}')
else:
    print(f'Validation accuracy {val_accuracy:.2f} did not reach target of 0.70')

# Plot validation loss and accuracy
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot validation loss on the first y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss', color='tab:blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Create a second y-axis for validation accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy', color='tab:orange')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Title and legend
plt.title('Validation Loss and Accuracy Over Epochs')
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()
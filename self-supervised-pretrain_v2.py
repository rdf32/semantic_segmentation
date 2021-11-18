import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, concatenate, multiply, Lambda, LeakyReLU
from tensorflow.keras import backend as kb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import segmentation_models as sm
from segmentation_models.losses import categorical_focal_jaccard_loss
from segmentation_models.metrics import iou_score
from models import *
#from keras_segmentation.models.unet import vgg_unet
sm.set_framework('tf.keras')
sm.framework()

def unet_encoder(input_size=(256, 256, 3), n_filters=64, batch_norm=True, width=128):

    inputs = Input(input_size)
    
    # encoder
    # Double the number of filters at each time step
    
    enc_block1 = encoder_block(inputs, n_filters, batch_norm)
    
    enc_block2 = encoder_block(enc_block1[0], n_filters*2)
    
    enc_block3 = encoder_block(enc_block2[0], n_filters*4)
    
    enc_block4 = encoder_block(enc_block3[0], n_filters*8, batch_norm, dropout_prob=0.3)
    
    # conv block instead of encoder block (Bridge)
    bridge = conv_block(enc_block4[0], n_filters*16, batch_norm, dropout_prob=0.3)

    flat = Flatten()(bridge)
    
    dense = Dense(width, activation="relu")(flat)

    # create model object
    model = tf.keras.Model(inputs=inputs, outputs=dense, name='unet_encoder')
    
    return model


def read_image(path):
    """
    Reads the image path and converts it into a scaled numpy array
    :param: path: path to image
    :return: image as numpy array
    """
    scaler = MinMaxScaler()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256,256))
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img.astype(np.float32)
    
    return img

def tf_pretrain_dataset(x, batch, epochs, shape):
    
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.shuffle(buffer_size=10 * batch)
    def preprocess_pretrain(img, shape=shape):
        
        def f_pretrain(img=img):
            img = img.decode()
            
            img = read_image(img)
            
            return img

        images, masks = tf.numpy_function(f_pretrain, [img], [tf.float32, tf.float32])
        images.set_shape(shape[0])       # tensor input shapes **
        

        return images, masks
    dataset = dataset.map(preprocess_pretrain)
    dataset = dataset.repeat(epochs).batch(batch) ## repeat num epochs **
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_pretrain_dataset(pretrain_images, data_shape, batch_size, epochs):
    
    pretrain_dataset = tf_pretrain_dataset(pretrain_images, shape=data_shape, batch=batch_size, epochs=epochs)    

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    pretrain_data = pretrain_dataset.with_options(options)

      #Define the model metrcis and load model. 
    num_train_imgs = len(pretrain_images) # data paths **

    steps_per_epoch = num_train_imgs // batch_size

    print("Number of training images:", num_train_imgs)
    print("Steps per epoch:", steps_per_epoch)
    

    return pretrain_data, steps_per_epoch



# Distorts the color distibutions of images
class RandomColorAffine(tf.keras.layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - tf.sqrt(min_area)
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(256,256,3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
        ]
    )



# Define the encoder architecture
def get_encoder():
    return unet_encoder(input_size=(256,256,3), n_filters=64, batch_norm=True, width=128)


# Define the contrastive model with model-subclassing
class ContrastiveModel(tf.keras.Model):
    def __init__(self, temperature, width, **contrastive_augmentation):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.encoder = get_encoder()
        # Non-linear MLP as projection head
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(width,)),   # ***WIDTH
                tf.keras.layers.Dense(width, activation="relu"), # ***WIDTH
                tf.keras.layers.Dense(width), # ***WIDTH
            ],
            name="projection_head",
        )
       
        self.encoder.summary()
        self.projection_head.summary()

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, images):
        
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}





pretrain_dataset, steps_per_epoch = get_pretrain_dataset([256,256,3], 32, 50)

# Contrastive pretraining
pretraining_model = ContrastiveModel(0.1, 128, )
pretraining_model.compile(
    contrastive_optimizer=tf.keras.optimizers.Adam(),
    probe_optimizer=tf.keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    pretrain_dataset, epochs=50, steps_per_epoch=steps_per_epoch)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(pretraining_history.history["val_p_acc"]) * 100
    )
)

# load pretraining weights into supervised model for training
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import pandas as pd
from glob import glob
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
from segmentation_models import Unet, Linknet
from segmentation_models.losses import categorical_focal_jaccard_loss
from segmentation_models.metrics import iou_score
sm.set_framework('tf.keras')
sm.framework(

def get_compiled_model(model_name, config):

    """
    contains a models dictionary where you can define all the models you wish to use and then
    returns that model after compiling it using the parameters stubbed out in the config dictionary

    :param: model_name: string of model name as stored in the internal 'models' dictionary
    :param: config: dictionary of model compile parameters
    :return: compiled model 
    """
    models = {
    'unet': Unet(backbone_name='resnet34', classes=5, input_shape=(256,256,3), encoder_weights='imagenet'),
    #'hrnet': HRNet(input_size=(256,256,3), n_filters=64, n_classes=2),
    #'linknet': Linknet(backbone_name='resnet34', classes=2, input_shape=(256,256,3), encoder_weights='imagenet'),
    # 'swin_unet' : models.swin_unet_2d((256, 256, 3), filter_num_begin=64, n_labels=5, depth=5, stack_num_down=2, stack_num_up=2,
                            #patch_size=(2, 2), num_heads=[4, 8, 8, 8, 8], window_size=[4, 2, 2, 2, 2], num_mlp=512,
                            #output_activation='Softmax', shift_window=True, name='swin_unet')
    }
    print("Compiling and returning model")
    model = models[model_name]
    
    model.compile(
        optimizer=config['optimizer'], 
        loss=config['loss'], 
        metrics=config['metrics']
    )
    return model


def load_data(img_path, mask_path):
    """
    Returns a list all image and mask files in the specified directories

    :param: img_path: path to image directory
    :param: mask_path: path to mask directory
    :return: list of image and mask paths
    """
    images = glob(img_path)
    masks = glob(mask_path)

    return images, masks

def read_image(path):
    """
    Reads the image path and converts it into a scaled numpy array

    :param: path: path to image
    :return: image as numpy array
    """
    scaler = MinMaxScaler()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img.astype(np.float32)
    
    return img

def read_mask(path, num_classes):
    """
    Reads the mask path and converts it into a categorized numpy array

    :param: path: path to mask
    :param: num_classes: number of target classes
    :return: mask as numpy array
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = to_categorical(mask, num_classes)                # num_classes **
    return mask


class Augment(tf.keras.layers.Layer):
    """
    Applies data augmentation to training data, additional augmentations can be added but must
    be conscious of which ones are being applied in segmentation tasks (only specific ones work)
    for more methods go here https://www.tensorflow.org/tutorials/images/data_augmentation
    """
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def tf_dataset(x, y, batch, epochs, shape, num_classes, train=True):
    """
    Creates a tf.data.Dataset from a list of image and mask paths
    the lists of paths are sent to the from_tensor_slices() method and then each pair image/mask pair
    is sent through the preprocess() function which reads them in convertes them to their respective np.arrays
    and then asserts their user specified shapes, data augmentation is then applied to the training dataset
    before the data is returned in batches this is for the **validation data so no augmentations are applied**

    :param: x: list of image paths - output of load_data()
    :param: y: list of mask paths - output of load_data()
    :param: batch: int specifying the batch size
    :param: shape: list of tuples specifying the shape of the image and mask i.e. [(256,256,3), (256,256,5)]
    :param: num_classes: int specifying number of target classes tells the function how many categories to assert on the mask'
    :param: train: boolean declaring if it is the training data set or the validation dataset, applies augmentation to training not validation
    
    :returns: tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    def preprocess(img, mask, shape=shape, num_classes=num_classes):
        
        def f(img=img, mask=mask, num_classes=num_classes):
            img = img.decode()
            mask = mask.decode()

            img = read_image(img)
            mask = read_mask(mask, num_classes)

            return img, mask

        images, masks = tf.numpy_function(f, [img, mask], [tf.float32, tf.float32])
        images.set_shape(shape[0])       # tensor input shapes **
        masks.set_shape(shape[1])

        return images, masks
    dataset = dataset.map(preprocess)
    if train:
        dataset = dataset.map(Augment())
    dataset = dataset.repeat(epochs).batch(batch) ## repeat num epochs **
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_dataset(paths, data_shape, batch_size, epochs, num_classes):

    """
    Creates the training and validation datasets 

    :param: paths: dictionary of data paths {'train_img_path', 'train_mask_path', 'val_img_path', 'val_mask_path'}
    :param: data_shape: list of tuples for image shape and mask shape [(256,256,3), (256,256,5)]
    :param: batch_size: int number of data points per batch
    :param: epoch: int number of epochs to train the model tells the data pipeline how many times to repeat the dataset
    :param: num_classes: int number of target classes

    :return: train_data, val_data, steps_per_epoch, val_steps_per_epoch
    """

    train_img_path = paths['train_img_path']
    train_mask_path = paths['train_mask_path']
    
    val_img_path = paths['val_img_path']
    val_mask_path = paths['val_mask_path']
 
    train_images, train_masks = load_data(train_img_path, train_mask_path)
    print(f"Images: {len(train_images)} - Masks: {len(train_masks)}")
    
    train_dataset = tf_dataset(train_images, train_masks, shape=data_shape, batch=batch_size, epochs=epochs,  num_classes=num_classes, train=True)
    
    val_images, val_masks = load_data(val_img_path, val_mask_path)
    print(f"Images: {len(val_images)} - Masks: {len(val_masks)}")
    
    val_dataset = tf_dataset(val_images, val_masks, shape=data_shape, batch=batch_size, epochs=epochs,  num_classes=num_classes, train=False)

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_dataset.with_options(options)
    val_data = val_dataset.with_options(options)

      #Define the model metrcis and load model. 
    num_train_imgs = len(train_images) # data paths **
    num_val_images = len(val_images) # data paths **

    steps_per_epoch = num_train_imgs // batch_size
    val_steps_per_epoch = num_val_images // batch_size

    print("Number of training images:", num_train_imgs)
    print("Number of val images:", num_val_images)
    print("Steps per epoch:", steps_per_epoch)
    print("Val steps per epoch:", val_steps_per_epoch)

    return train_data, val_data, steps_per_epoch, val_steps_per_epoch



def run_training(model_name, config, paths = {}, data_shape=[], batch_size=32, epochs=30, num_classes=5):
    """
    Calls all training functions and builds the specified model under a distributed mirrored strategy
    will automatically train your model in parallel across all available GPU's in your environment
    automatically saves the best version of the trained model into the specified output folder as well
    as a record of the training metrics across all epochs as a csv file

    :param: model_name: string of model name as stored in the internal 'models' dictionary passed to get_compiled_model()
    :param: config: dictionary of model compile parameters passed to get_compiled_model()
    :param: paths: dictionary of paths pointing to the data and output directories passed to get_dataset()
    :param: data_shape: list of tuples representing data shapes [0] - image shape [1] - mask shape passed to get_dataset()
    :param: epochs: number of epochs to train the model
    :param: num_classes: number of target classes

    :returns: trained model with best val_loss score (min) and training metric history csv

    """

    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
          device_type)
    
    devices_names = [d.name.split("e:")[1] for d in devices]
    print(devices_names)
    n_gpus = len(devices)
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = get_compiled_model(model_name=model_name, config=config)
    
    model_output = paths['output'] + "/{model_name}.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_output, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)]


    print("Loading data")
    train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_dataset(paths=paths, data_shape=data_shape, batch_size=batch_size,
                                                                                                epochs=epochs, num_classes=num_classes)
    
    print("Training model")
    history = model.fit(
        train_dataset,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=val_dataset,
          validation_steps=val_steps_per_epoch,
        callbacks=callbacks
    )

    
    data = {"loss": history.history['loss'], "iou_score": history.history['iou_score'],
    "val_loss": history.history['val_loss'], "val_iou_score": history.history['val_iou_score']}

    df = pd.DataFrame(data)
    print("saving model metrics")
    df.to_csv(paths['output'] + "/metric_history.csv") # metric save location **






if __name__ == '__main__':
    # configuration area create config for each model

    unet_config = {
    'optimizer': Adam(learning_rate=1e-4),
    'loss': categorical_focal_jaccard_loss,
    'metrics': [iou_score]
    }

    # train_img_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data', 'train_images', 'train/*') # data paths **
    # train_mask_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/train_masks', 'train/*') # data paths **

    # val_img_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_images', 'val/*') # data paths **
    # val_mask_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_masks', 'val/*') # data paths **
    #'/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/train_images/train'
    #'/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_images/val'
    #checkpoint_dir='/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt'

    paths = {
    'train_img_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'train_images','train/*'),
    'train_mask_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'train_masks', 'train/*'),
    'val_img_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'val_images', 'val/*'),
    'val_mask_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'val_masks', 'val/*'),
    'output' : '/home/jovyan/opt/semantic_segmentation/ckpt'}

    # shape[0] is the shape of the image, shape[1] is the shape of the mask

    run_training('unet', unet_config, paths=paths, data_shape=[(256,256,3),(256,256,5)], batch_size=16, epochs=30,
    num_classes=5)

    


    
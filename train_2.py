
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
#from keras_segmentation.models.unet import vgg_unet
sm.set_framework('tf.keras')
sm.framework()


def get_compiled_model(model_name, config):
    models = {
    'unet': Unet(backbone_name='resnet34', classes=5, input_shape=(256,256,3), encoder_weights='imagenet'),
    #'hrnet': HRNet(input_size=(256,256,3), n_filters=64, n_classes=2),
    #'linknet': Linknet(backbone_name='resnet34', classes=2, input_shape=(256,256,3), encoder_weights='imagenet')
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
    images = glob(img_path)
    masks = glob(mask_path)

    return images, masks

def read_image(path):
    scaler = MinMaxScaler()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img.astype(np.float32)
    
    return img

def read_mask(path, num_classes):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = to_categorical(mask, num_classes)                # num_classes **
    return mask


def tf_dataset(x, y, batch, epochs, shape, num_classes):
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
    dataset = dataset.repeat(epochs).batch(batch) ## repeat num epochs **
    dataset = dataset.prefetch(2)
    return dataset

def get_dataset(data_paths, data_shape, batch_size, epochs, num_classes):

    train_img_path = data_paths['train_img_path']
    train_mask_path = data_paths['train_mask_path']
    
    val_img_path = data_paths['val_img_path']
    val_mask_path = data_paths['val_mask_path']
 
    train_images, train_masks = load_data(train_img_path, train_mask_path)
    print(f"Images: {len(train_images)} - Masks: {len(train_masks)}")
    
    train_dataset = tf_dataset(train_images, train_masks, shape=data_shape, batch=batch_size, epochs=epochs,  num_classes=num_classes)
    
    val_images, val_masks = load_data(val_img_path, val_mask_path)
    print(f"Images: {len(val_images)} - Masks: {len(val_masks)}")
    
    val_dataset = tf_dataset(val_images, val_masks, shape=data_shape, batch=batch_size, epochs=epochs,  num_classes=num_classes)

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_dataset.with_options(options)
    val_data = val_dataset.with_options(options)

    #'/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/train_images/train'
    #'/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_images/val'

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



def run_training(model_name, config, models={}, data_paths = [], data_shape=[], batch_size=32, epochs=30, num_classes=5, ckpt_dir=None):
    

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
    if ckpt_dir is not None:
        model_output = ckpt_dir + "/{model_name}.h5"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(model_output, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)]
    else:
        callbacks = None

    print("Loading data")
    train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_dataset(data_paths=data_paths, data_shape=data_shape, batch_size=batch_size,
                                                                                                epochs=epochs, num_classes=num_classes)
    
    print("Training model")
    model.fit(
        train_dataset,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=val_dataset,
          validation_steps=val_steps_per_epoch,
        callbacks=callbacks
    )

    
    data = {"loss": model.history['loss'], "iou_score": model.history['iou_score'],
    "val_loss": model.history['val_loss'], "val_iou_score": model.history['val_iou_score']}

    df = pd.DataFrame(data)
    print("saving model metrics")
    df.to_csv(ckpt_dir + "/metric_history.csv") # metric save location **






if __name__ == '__main__':
    # configuration area create config for each model

    unet_config = {
    'optimizer': Adam(learning_rate=1e-4),
    'loss': categorical_focal_jaccard_loss,
    'metrics': [iou_score]
    }


    data_paths = {
    'train_img_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'train_images','train/*'),
    'train_mask_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'train_masks', 'train/*'),
    'val_img_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'val_images', 'val/*'),
    'val_mask_path' : os.path.join(os.getcwd(), 'data', 'train_val_data', 'val_masks', 'val/*')}


    run_training('unet', unet_config, data_paths = data_paths, data_shape=[(256,256,3),(256,256,5)], batch_size=16, epochs=30,
    num_classes=5, ckpt_dir='/home/jovyan/opt/semantic_segmentation/ckpt')

    


    
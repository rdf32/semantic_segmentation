import pandas as pd
from models2 import *


## make all the directories (including processing dirs)
## make sure to site the machine (tallgrass)


def load_data(img_path, mask_path):
    images = glob(img_path)
    masks = glob(mask_path)

    return images, masks

def read_image(path):
    scaler = MinMaxScaler()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = img.astype(np.float32)
    
    return img

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    #Convert mask to one-hot
    mask = to_categorical(mask, 5)                # num_classes **
    return mask

def preprocess(img, mask):
    def f(img, mask):
        #print(img, mask)
        img = img.decode()
        mask = mask.decode()
        
        img = read_image(img)
        mask = read_mask(mask)

        return img, mask

    images, masks = tf.numpy_function(f, [img, mask], [tf.float32, tf.float32])
    images.set_shape([512, 512, 3])       # tensor input shapes **
    masks.set_shape([512, 512, 5])

    return images, masks

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat(30).batch(batch) ## repeat num epochs **
    dataset = dataset.prefetch(2)
    return dataset


def get_compiled_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    model = unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), n_filters=64,
                  n_classes=5, batch_norm=True) # num_classes **
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss=categorical_focal_jaccard_loss, 
        metrics=[iou_score]
    )
    return model



def get_dataset():
    batch_size = 32 # global batch size **

    train_img_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data', 'train_images', 'train/*') # data paths **
    train_mask_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/train_masks', 'train/*') # data paths **

    val_img_path = os.path.join('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_images', 'val/*') # data paths **
    val_mask_path = ('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_masks', 'val/*') # data paths **
 
    train_images, train_masks = load_data(train_img_path, train_mask_path)
    print(f"Images: {len(train_images)} - Masks: {len(train_masks)}")
    
    train_dataset = tf_dataset(train_images, train_masks, batch=batch_size)
    
    val_images, val_masks = load_data(val_img_path, val_mask_path)
    print(f"Images: {len(val_images)} - Masks: {len(val_masks)}")
    
    val_dataset = tf_dataset(val_images, val_masks, batch=batch_size)

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)
  
 
    IMG_HEIGHT = 512 # tensor input size **
    IMG_WIDTH  = 512
    IMG_CHANNELS = 3

      #Define the model metrcis and load model. 
    num_train_imgs = len(os.listdir('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/train_images/train')) # data paths **
    num_val_images = len(os.listdir('/caldera/projects/usgs/eros/users/rfleckenstein/data/train_val_data/val_images/val')) # data paths **

    steps_per_epoch = num_train_imgs // batch_size
    val_steps_per_epoch = num_val_images // batch_size
    print("Number of training images:", num_train_imgs)
    print("Number of val images:", num_val_images)
    print("Steps per epoch:", steps_per_epoch)
    print("Val steps per epoch:", val_steps_per_epoch)
    return train_data, val_data, steps_per_epoch, val_steps_per_epoch

# class MetricHistory(tf.keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.iou_scores = []
#         self.val_losses = []
#         self.val_iou_scores = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.iou_scores.append(logs.get('iou_score'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.val_iou_scores.append(logs.get('val_iou_score'))





def make_or_restore_model():
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = '/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt'  # checkpoint dir path **

    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    #checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    #if checkpoints:
      #  latest_checkpoint = max(checkpoints, key=os.path.getctime)
     #   print("Restoring from", latest_checkpoint)
    #    return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(512,512,3)  # input tensor shape **


def run_training(epochs=1):
    checkpoint_dir = '/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt' # checkpoint dir path **

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
        model = make_or_restore_model()

    #metric_history = MetricHistory()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        ) #,logging
    print("Loading data")
    train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_dataset()
    
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
    df.to_csv("/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt/metric_history.csv") # metric save location **


if __name__ == '__main__':

    checkpoint_dir = '/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt' # checkpoint dir path **

    run_training(epochs=30)

    #metric_history = MetricHistory()
    # Running the first time creates the model
    

    # data = {"loss": metric_history.losses, "iou_score": metric_history.iou_scores,
    # "val_loss": metric_history.val_losses, "val_iou_score": metric_history.val_iou_scores}

    # df = pd.DataFrame(data)
    # print("saving model metrics")
    # df.to_csv("/caldera/projects/usgs/eros/users/rfleckenstein/unet_ckpt/metric_history.csv") # metric save location **


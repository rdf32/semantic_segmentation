
# https://keras.io/examples/vision/semisupervised_simclr/
# https://www.tensorflow.org/tutorials/images/data_augmentation
# https://github.com/google-research/simclr/blob/master/run.py
# https://github.com/google-research/simclr
# https://arxiv.org/abs/2006.10029
# https://www.youtube.com/watch?v=2lkUNDZld-4


from models import *


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

def load_pretrain_data(img_path):
    """
    Returns a list all image and mask files in the specified directories

    :param: img_path: path to image directory
    :param: mask_path: path to mask directory
    :return: list of image and mask paths
    """
    images = glob(img_path)
    

    return images

def tf_pretrain_dataset(x, batch, epochs, shape):
    
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.shuffle(buffer_size=10 * batch)
    def preprocess_pretrain(img, shape=shape):
        
        def f_pretrain(img=img):
            img = img.decode()
            
            img = read_image(img)
            
            return img

        images = tf.numpy_function(f_pretrain, [img], tf.float32)
        images.set_shape(shape)       # tensor input shapes **
        

        return images
    dataset = dataset.map(preprocess_pretrain)
    dataset = dataset.repeat(epochs).batch(batch) ## repeat num epochs **
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_pretrain_dataset(pretrain_path, shape, batch_size, epochs):
    
    pretrain_images = load_pretrain_data(pretrain_path)

    pretrain_dataset = tf_pretrain_dataset(pretrain_images, shape=shape, batch=batch_size, epochs=epochs)    

    
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
def get_encoder(encoder_name, width):
    encoders = {
        'unet': unet_encoder(input_size=(256,256,3), n_filters=64, batch_norm=True, width=width),
        'hrnet' : HREncoder(input_size=(256,256,3), width=width)
    }
    return encoders[encoder_name]


# Define the contrastive model with model-subclassing
class ContrastiveModel(tf.keras.Model):
    def __init__(self, encoder_name, temperature, width, **contrastive_augmentation):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.encoder = get_encoder(encoder_name, width)
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





# Contrastive pretraining
def run_pretraining(encoder_name, temperature, width, shape, batch_size, epochs, paths, **contrastive_augmentation):

    pretrain_dataset, steps_per_epoch = get_pretrain_dataset(paths['pretrain_images'], shape[0], batch_size, epochs)

    pretraining_model = ContrastiveModel(encoder_name, temperature, width, **contrastive_augmentation)

    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
        device_type)
    
    devices_names = [d.name.split("e:")[1] for d in devices]
    print(devices_names)
    n_gpus = len(devices)
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

    with strategy.scope():

        pretraining_model.compile(
            contrastive_optimizer=tf.keras.optimizers.Adam()
        
        )


    pretraining_history = pretraining_model.fit(
        pretrain_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
        )

    pretraining_model.encoder.save_weights(paths['output'] + f"{encoder_name}_encoder.h5")

    data = {"c_loss": pretraining_history.history['c_loss'], "c_accuracy": pretraining_history.history['c_acc']}

    df = pd.DataFrame(data)
    print("saving model metrics")
    df.to_csv(paths['output'] + f"{encoder_name}_encoder_metric_history.csv")

# load pretraining weights into supervised model for training


if __name__ == '__main__':
    
    contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
   
    paths = {
        "pretrain_images" : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'train_images','train/*'),
    
        "output" : "some path"
    }

    run_pretraining('unet', temperature=0.1, width=256, shape=[(256,256,3)], batch_size=32, epochs=200,
     paths=paths, **contrastive_augmentation)

    #run_pretraining('hrnet', temperature=0.1, width=256, shape=[(256,256,3)], batch_size=32, epochs=200,
     #paths=paths, **contrastive_augmentation)

     # encoder = unet_encoder()
     # encoder.load_weights("encoder pretrained weights")
from models import *
from self_supervised_pretrain import *
from train import *
from self_distillation_training import *


if __name__ == '__main__':

    contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
   
    paths = {
        
        "pretrain_images" : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'train_images','train/*'),

        'train_img_path' : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'train_images','train/*'),

        'train_mask_path' : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'train_masks','train/*'),

        'val_img_path' : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'val_images','val/*'),

        'val_mask_path' : os.path.join('C:/Users/rfleckenstein/OneDrive - DOI/Desktop/Projects/data/train_val_data_landconverAI',
        'val_masks','val/*'),

        'output' : '/home/jovyan/opt/semantic_segmentation/ckpt/'
        }

    # self-supervised contrastive pretraining (using image classification technique) utilizing entire unlabled dataset
    run_pretraining('unet', temperature=0.1, width=256, shape=[(256,256,3)], batch_size=32, epochs=200,
     paths=paths, **contrastive_augmentation)

    # supervised semantic segmentation (pixel wise classficiation) utilizing labeled dataset
    Unet_encoder = unet_encoder(input_size=(256, 256, 3), n_filters=64, batch_norm=True, width=256)
    Unet_encoder.load_weights(paths['output'] + 'unet_encoder.h5')
    run_training('unet', paths=paths, data_shape=[(256,256,3),(256,256,5)], batch_size=16, epochs=30, num_classes=5, fine_tune=True, encoder=Unet_encoder)

    # self-knowledge distillation (teacher and student models) utilizing entire unlabeled dataset
    unet_teacher =  tf.keras.models.load_model(paths['output'] + "unet_pt_and_ft.h5")
    # completely new model of same architecture as teacher
    unet_student = unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=5, batch_norm=True)

    # self knowledge distillation
    run_distillation('unet', unet_teacher, unet_student, data_shape=[(256,256,3),(256,256,5)], batch_size=16, epochs=30, num_classes=5, paths=paths)


# https://www.youtube.com/watch?v=2lkUNDZld-4
# https://keras.io/examples/vision/knowledge_distillation/


# run on entire unlabeled data set after fine tuning of teacher model on labeled dataset
# Freeze the teacher model only used for logit generation
# create student model of same size as teacher model and run each image through
# each model and compare the logit outputs(distribution)
# the difference is the loss of the model and we only update the student
# figure out how to do this with batches (sum of losses and such)

# Train step on unlabeled data and test step on labeled data maybe???????

from models import *
from train import *
from self_supervised_pretrain import *

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(teacher_predictions, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results






if __name__ == '__main__':


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

        'output' : '/home/jovyan/opt/semantic_segmentation/ckpt'
        }

    # trained teacher model
    # if pretrained
    #teacher_unet = tf.keras.models.load_model("""paths['output'] + f"/{model_name}_pt_and_ft.h5"/""")
    # if not pretrained
    # teacher_unet = tf.keras.models.load_model('paths['output'] + f"/{model_name}.h5"/')


    def run_distillation(model_name, teacher, student, data_shape, batch_size, epochs, num_classes, paths):

        print("Loading data")
            # Distill teacher to student
        pretrain_dataset, pre_steps_per_epoch = get_pretrain_dataset(paths['pretrain_images'], data_shape[0], batch_size, epochs)
        
        # only want this to repeat once for each evaluation
        train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_dataset(paths=paths, data_shape=data_shape, batch_size=batch_size,
            epochs=1, num_classes=num_classes)
        
        # Initialize and compile distiller
        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[iou_score],
            student_loss_fn=categorical_focal_jaccard_loss,

            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )

        

    # wanting to save model with best validation accuracy so periodically checking how it performs on labeled data

        for _ in range(5):
            epoch = 0
            distiller.fit(
                        pretrain_dataset,
                        steps_per_epoch=pre_steps_per_epoch,
                        epochs=epochs/5
                    )
            curr_epoch = epoch + (epochs/5)
            print(f"metrics for epoch: {curr_epoch}")
            distiller.evaluate(train_dataset)

            model_output = paths['output'] + f"{model_name}_distilled_{curr_epoch}.h5"
            distiller.save(model_output)

            epoch += (epochs/5)




if __name__ == '__main__':
     #run_distillation('unet', teacher, student, data_shape, batch_size, epochs, num_classes, paths)
     pass
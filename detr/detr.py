# class DETR(tf.keras.Model):
#     def __init__(self, num_classes=92, num_queries=100,
        #          backbone=None,
        #          pos_encoder=None,
        #          transformer=None,
        #          num_encoder_layers=6,
        #          num_decoder_layers=6,
        #          return_intermediate_dec=True,
        #          **kwargs):
        # super().__init__(**kwargs)
        # self.num_queries = num_queries

        # self.backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        # self.transformer = Transformer(
        #         num_encoder_layers=num_encoder_layers,
        #         num_decoder_layers=num_decoder_layers,
        #         return_intermediate_dec=return_intermediate_dec,
        #         name='transformer'
        # )
        
        # self.model_dim = self.transformer.model_dim

        # self.pos_encoder = PositionEmbeddingSine(
        #     num_pos_features=self.model_dim // 2, normalize=True, name="position_embedding_sine")

        # self.input_proj = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        # self.query_embed = FixedEmbedding((num_queries, self.model_dim),
        #                                   name='query_embed')

        # self.class_embed = Linear(num_classes, name='class_embed')

        # self.bbox_embed_linear1 = Linear(self.model_dim, name='bbox_embed_0')
        # self.bbox_embed_linear2 = Linear(self.model_dim, name='bbox_embed_1')
        # self.bbox_embed_linear3 = Linear(4, name='bbox_embed_2')
        # self.activation = tf.keras.layers.ReLU(name='re_lu')


    # def downsample_masks(self, masks, x):
    #     masks = tf.cast(masks, tf.int32)
    #     masks = tf.expand_dims(masks, -1)
    #     masks = tf.compat.v1.image.resize_nearest_neighbor(masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
    #     masks = tf.squeeze(masks, -1)
    #     masks = tf.cast(masks, tf.bool)
    #     return masks

    # def call(self, inp, training=False, post_process=False):
    #     x, masks = inp
    #     x = self.backbone(x, training=training)
    #     masks = self.downsample_masks(masks, x)

    #     pos_encoding = self.pos_encoder(masks)

    #     hs = self.transformer(self.input_proj(x), masks, self.query_embed(None),
    #                           pos_encoding, training=training)[0]

    #     outputs_class = self.class_embed(hs)

    #     box_ftmps = self.activation(self.bbox_embed_linear1(hs))
    #     box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
    #     outputs_coord = tf.sigmoid(self.bbox_embed_linear3(box_ftmps))

    #     output = {'pred_logits': outputs_class[-1],
    #               'pred_boxes': outputs_coord[-1]}

    #     if post_process:
    #         output = self.post_process(output)
    #     return output


    # def build(self, input_shape=None, **kwargs):
    #     if input_shape is None:
    #         input_shape = [(None, None, None, 3), (None, None, None)]
    #     super().build(input_shape, **kwargs)

# def add_heads_nlayers(config, detr, nb_class):
#     image_input = tf.keras.Input((None, None, 3))
#     # Setup the new layers
#     cls_layer = tf.keras.layers.Dense(nb_class, name="cls_layer")
#     pos_layer = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(256, activation="relu"),
#         tf.keras.layers.Dense(256, activation="relu"),
#         tf.keras.layers.Dense(4, activation="sigmoid"),
#     ], name="pos_layer")
#     config.add_nlayers([cls_layer, pos_layer])

#     transformer_output = detr(image_input)
#     cls_preds = cls_layer(transformer_output)
#     pos_preds = pos_layer(transformer_output)

#     # Define the main outputs along with the auxialiary loss
#     outputs = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
#     outputs["aux"] = [ {"pred_logits": cls_preds[i], "pred_boxes": pos_preds[i]} for i in range(0, 5)]

#     n_detr = tf.keras.Model(image_input, outputs, name="detr_finetuning")
#     return n_detr

# def get_detr_model(config, include_top=False, nb_class=None, weights=None, tf_backbone=False, num_decoder_layers=6, num_encoder_layers=6):
#     """ Get the DETR model
#     Parameters
#     ----------
#     include_top: bool
#         If false, the last layers of the transformers used to predict the bbox position
#         and cls will not be include. And therefore could be replace for finetuning if the `weight` parameter
#         is set.
#     nb_class: int
#         If include_top is False and nb_class is set, then, this method will automaticly add two new heads to predict
#         the bbox pos and the bbox class on the decoder.
#     weights: str
#         Name of the weights to load. Only "detr" is avaiable to get started for now.
#         More weight as detr-r101 will be added soon.
#     tf_backbone:
#         Using the pretrained weight from pytorch, the resnet backbone does not used
#         tf.keras.application to load the weight. If you do want to load the tf backbone, and not
#         laod the weights from pytorch, set this variable to True.
#     """
#     detr = DETR(num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers)

#     if weights is not None:
#         load_weights(detr, weights)

#     image_input = tf.keras.Input((None, None, 3))

#     # Backbone
#     if not tf_backbone:
#         backbone = detr.get_layer("backbone")
#     else:
#         config.normalized_method = "tf_resnet"
#         backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(None, None, 3))

#     # Transformer
#     transformer = detr.get_layer("transformer")
#     # Positional embedding of the feature map
#     position_embedding_sine = detr.get_layer("position_embedding_sine")
#     # Used to project the feature map before to fit in into the encoder
#     input_proj = detr.get_layer('input_proj')
#     # Decoder objects query embedding
#     query_embed = detr.get_layer('query_embed')


#     # Used to project the  output of the decoder into a class prediction
#     # This layer will be replace for finetuning
#     class_embed = detr.get_layer('class_embed')

#     # Predict the bbox pos
#     bbox_embed_linear1 = detr.get_layer('bbox_embed_0')
#     bbox_embed_linear2 = detr.get_layer('bbox_embed_1')
#     bbox_embed_linear3 = detr.get_layer('bbox_embed_2')
#     activation = detr.get_layer("re_lu")

#     x = backbone(image_input)

#     masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)
#     pos_encoding = position_embedding_sine(masks)

#     hs = transformer(input_proj(x), masks, query_embed(None), pos_encoding)[0]

#     detr = tf.keras.Model(image_input, hs, name="detr")
#     if include_top is False and nb_class is None:
#         return detr
#     elif include_top is False and nb_class is not None:
#         return add_heads_nlayers(config, detr, nb_class) 

#     transformer_output = detr(image_input)

#     outputs_class = class_embed(transformer_output)
#     box_ftmps = activation(bbox_embed_linear1(transformer_output))
#     box_ftmps = activation(bbox_embed_linear2(box_ftmps))
#     outputs_coord = tf.sigmoid(bbox_embed_linear3(box_ftmps))

#     outputs = {}

#     output = {'pred_logits': outputs_class[-1],
#                 'pred_boxes': outputs_coord[-1]}

#     output["aux"] = []
#     for i in range(0, num_decoder_layers - 1):
#         out_class = outputs_class[i]
#         pred_boxes = outputs_coord[i]
#         output["aux"].append({
#             "pred_logits": out_class,
#             "pred_boxes": pred_boxes
#         })

#     return tf.keras.Model(image_input, output, name="detr_finetuning")


#     def train_step(self, data):
#         (images, t_bbox, t_class) = data

#         x = backbone(input)

#         masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)
#         pos_encoding = position_embedding_sine(masks)

#         hs = transformer(input_proj(x), masks, query_embed(None), pos_encoding)[0]

#         detr = tf.keras.Model(image_input, hs, name="detr")
#         if include_top is False and nb_class is None:
#             return detr
#         elif include_top is False and nb_class is not None:
#             return add_heads_nlayers(config, detr, nb_class) 

#         # transformer_output = detr(image_input)

#         outputs_class = class_embed(transformer_output)
#         # box_ftmps = activation(bbox_embed_linear1(transformer_output))
#         # box_ftmps = activation(bbox_embed_linear2(box_ftmps))
#         # outputs_coord = tf.sigmoid(bbox_embed_linear3(box_ftmps))



def detr_model(num_classes=2, num_queries=100, num_encoder_layers=6, num_decoder_layers=6):

    image_input = tf.keras.Input((None, None, 3))

    x = backbone(image_input)

    masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)

    pos_encoding = position_embedding_sine(masks)

    transformer_output = transformer(input_proj(x), masks, query_embed((num_queries, transformer.model_dim)), pos_encoding)[0]

    cls_layer = tf.keras.layers.Dense(num_classes, name="cls_layer")

    cls_preds = cls_layer(transformer_output)
    pos_preds = pos_layer(transformer_output)


    output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}

    output["aux"] = []
    for i in range(0, num_decoder_layers - 1):
        out_class = outputs_class[i]
        pred_boxes = outputs_coord[i]
        output["aux"].append({
            "pred_logits": out_class,
            "pred_boxes": pred_boxes
        })

    return tf.keras.Model(image_input, output, name="detr_finetuning")


def disable_batchnorm_training(model):
    for l in model.layers:
        if hasattr(l, "layers"):
            disable_batchnorm_training(l)
        elif isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False

class DETR(tf.keras.Model):
    def __init__(self, num_classes=2, num_queries=100,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 return_intermediate_dec=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers

        self.backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        self.transformer = Transformer(
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                return_intermediate_dec=return_intermediate_dec,
                name='transformer'
        )
        
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True, name="position_embedding_sine")

        self.input_proj = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = FixedEmbedding((self.num_queries, self.model_dim),
                                          name='query_embed')

        self.cls_layer = tf.keras.layers.Dense(num_classes, name="cls_layer")

        self.pos_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ], name="pos_layer")

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        # self.contrastive_optimizer = contrastive_optimizer
        # self.probe_optimizer = probe_optimizer

        # # self.contrastive_loss will be defined as a method
        # self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        # self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
        #     name="c_acc"
        # )
        # self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        # self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        # return [
        #     self.contrastive_loss_tracker,
        #     self.contrastive_accuracy,
        #     self.probe_loss_tracker,
        #     self.probe_accuracy,
        # ]



    def train_step(self, data):
        
        images, t_bbox, t_class = data

        disable_batchnorm_training(self.backbone)

        with tf.GradientTape() as tape:

            x, masks = images
            x = self.backbone(x, training=training)
            masks = self.downsample_masks(masks, x)

            pos_encoding = self.pos_encoder(masks)

            transformer_output = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                                pos_encoding, training=training)[0]

            cls_preds = self.cls_layer(transformer_output)

            pos_preds = self.pos_layer(transformer_output)

            output = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}

            output["aux"] = []
            for i in range(0, self.num_decoder_layers - 1):
                out_class = cls_preds[i]
                pred_boxes = pos_preds[i]
                output["aux"].append({
                    "pred_logits": out_class,
                    "pred_boxes": pred_boxes
                })

            if post_process:
                output = self.post_process(output)
            

            loss = get_losses(output, t_bbox, t_class, config)  ###config

        gradients = tape.gradient(
            loss,

            # here we need to set batchnorm layers to not trainable in backbone
            self.backbone.trainable_weights + self.transformer.trainable_weights +
             self.cls_layer.trainable_weights + self.pos_encoder.trainable_weights,
        )
        self.detr_optimizer.apply_gradients(
            zip(
                gradients,
                self.backbone.trainable_weights + self.transformer.trainable_weights +
                 self.cls_layer.trainable_weights + self.pos_encoder.trainable_weights
            )
        )
        self.detr_loss_tracker.update_state(loss)

            # Compute gradient for each part of the network
        
            # need trainable layers of backbone, transformer, cls_layer, pos_layer
            # for computing the losses so total losses
            # need to also update auxilary losses (I believe its part of how the transformer works (should read the paper))

            









            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
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

        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
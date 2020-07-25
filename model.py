"""
build base model with keras (which will be convert into tensorflow estimator)
"""
import tensorflow as tf
from config import ModelConfig, TrainingConfig
from data_utils import preprocessing


def atrous_spatial_pyramid_pooling_keras(inputs, output_stride, depth=256):
    """
    atrous spatial pyramid pooling implementation with keras
    :param inputs:
    :param output_stride:
    :param batch_norm_decay:
    :param is_training:
    :param depth:
    :return:
    """
    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2*item for item in atrous_rates]
    with tf.name_scope('atrous_pyramid_pooling'):
        conv_1x1 = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(inputs)
        conv_3x3_list = []
        for item in atrous_rates:
            conv_3x3 = tf.keras.layers.Conv2D(depth, (3, 3), strides=1, dilation_rate=item, padding='same')(inputs)
            conv_3x3_list.append(conv_3x3)
        with tf.variable_scope("image_level_features"):
            # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
            # 1×1 convolution with 256 filters( and batch normalization)
            image_level_features = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(image_level_features)
            # bilinearly upsample features
            inputs_size = tf.shape(inputs)[1:3]
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            net = tf.concat([conv_1x1]+conv_3x3_list+[image_level_features], axis=3, name='concat')
            net = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(net)
            return net


def classification_branch(x, class_num):
    """
    classification branch
    :param x: input tensor
    :return: a tensor
    """
    with tf.name_scope('classification_branch'):
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, 'relu')(x)
        x = tf.keras.layers.Dense(class_num, activation='softmax')(x)
    return x


def segmentation_branch(x):
    """
    segmentation branch
    :param x:
    :return: a tensor
    """
    with tf.name_scope('segmentation_branch'):
        x = atrous_spatial_pyramid_pooling_keras(x, 16, 256)
    return x


def model_generator(class_num):
    # TODO: add mode param to disable BN at testing phase
    def build_model(x):
        """
        build unit model for classification and segmentation
        :param x: input tensor
        :return: tensor lost of [classification_result, segmentation_result]
        """
        # feature extraction backbone
        backbone = tf.keras.applications.VGG16(input_tensor=x, include_top=False, pooling=True, weights='imagenet', input_shape=(513, 513, 3))
        # extract block-4 of vgg with downsample of 3 times(8)
        feature_for_segmentation = backbone.get_layer('block4_conv3').output
        feature_for_classification = backbone.output
        # branch 0 for classification
        classification_result = classification_branch(feature_for_classification, class_num)
        # branch 1 for segmentation
        segmentation_feature = segmentation_branch(feature_for_segmentation)
        inputs_size = tf.shape(x)[1:3]
        # extract output tensor of block4
        with tf.variable_scope("upsampling_logits"):
            net = tf.keras.layers.Conv2D(ModelConfig.num_classes, (1, 1), strides=1, padding='same', activation='linear')(segmentation_feature)
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')
            segmentation_result = tf.nn.softmax(logits, name='softmax_tensor')
        return [classification_result, segmentation_result]
    return build_model


def model_fn(features, labels, mode, params=None):
    """
    Model function for tensorflow estimator
    """
    # input image preprocessing
    if isinstance(features, dict):
        features = features['feature']
    images = tf.cast(
        tf.map_fn(preprocessing.mean_image_addition, features),
        tf.uint8)
    # extract and process output/predictions
    network = model_generator(ModelConfig.num_classes)
    classification_p, segmentation_p = network(features)
    classification_pred_classes = tf.argmax(classification_p, axis=1)
    segmentation_pred_classes = tf.expand_dims(tf.argmax(segmentation_p, axis=3, output_type=tf.int32), axis=3)
    predictions = {
        'classification_classes': classification_pred_classes,
        'classification_probabilities': tf.reduce_max(classification_p, axis=1),
        'segmentation_classes': segmentation_pred_classes,
        'segmentation_probabilities': tf.reduce_max(segmentation_p, axis=3)
    }
    # different process for different mode-train/test
    # predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_without_decoded_labels = predictions.copy()
        del predictions_without_decoded_labels['decoded_labels']

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'preds': tf.estimator.export.PredictOutput(
                    predictions_without_decoded_labels)
            })
    # optimization opt
    # TODO: deal with classification and segmentation labels void input
    classification_labels = None
    segmentation_labels = labels
    # loss for classification branch
    if classification_labels is not None:
        classification_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(classification_labels, classification_p))
    else:
        classification_loss = tf.constant(0.0, dtype=tf.float32)
    # loss for segmentation
    if segmentation_labels is not None:
        segmentation_p = tf.reshape(segmentation_p, [-1, ModelConfig.num_classes])
        segmentation_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.keras.backend.flatten(segmentation_labels), segmentation_p))
        # TODO: runinto NaN loss error without this
        segmentation_loss = tf.keras.backend.clip(segmentation_loss, 1e-4, 1e3)
    else:
        segmentation_loss = tf.constant(0.0, dtype=tf.float32)
    overall_loss = classification_loss + segmentation_loss
    # Create a tensor -losses for logging purposes.
    tf.identity(classification_loss, name='classification_loss')
    tf.summary.scalar('classification_loss', classification_loss)
    tf.identity(segmentation_loss, name='segmentation_loss')
    tf.summary.scalar('segmentation_loss', segmentation_loss)
    tf.identity(overall_loss, name='overall_loss')
    tf.summary.scalar('overall_loss', overall_loss)
    # get trainable weight except bn params
    if not TrainingConfig.freeze_batch_norm:
        train_var_list = [v for v in tf.trainable_variables()]
    else:
        train_var_list = [v for v in tf.trainable_variables()
                          if 'beta' not in v.name and 'gamma' not in v.name]
    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        loss = overall_loss + TrainingConfig.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in train_var_list])
    if classification_labels is not None:
        classification_acc = tf.metrics.accuracy(classification_labels, classification_pred_classes)
    else:
        # set to zero
        classification_acc = tf.metrics.accuracy(tf.constant([1]), tf.constant([0]))
    if segmentation_labels is not None:
        segmentation_acc = tf.metrics.accuracy(segmentation_labels, segmentation_pred_classes)
    else:
        # set to zero
        segmentation_acc = tf.metrics.accuracy(tf.constant([1]), tf.constant([0]))
    # mean_iou = tf.metrics.mean_iou(segmentation_labels, segmentation_pred_classes, ModelConfig.num_classes)
    # metrics = {'classification_acc': classification_acc, 'segmentation_px_acc': segmentation_acc, 'mean_iou': mean_iou}
    metrics = {'segmentation_px_acc': segmentation_acc}
    # Create a tensor named train_accuracy for logging purposes
    tf.identity(classification_acc[1], name='classification_acc')
    tf.summary.scalar('classification_acc', classification_acc[1])
    tf.identity(segmentation_acc[1], name='segmentation_px_acc')
    tf.summary.scalar('segmentation_px_acc', segmentation_acc[1])
    # tf.identity(mean_iou[1], name='mean_iou')
    # tf.summary.scalar('mean_iou', mean_iou[1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        # tf.summary.image('images',
        #                  tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
        #                  max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

        global_step = tf.train.get_or_create_global_step()

        if TrainingConfig.learning_rate_policy == 'piecewise':
            # Scale the learning rate linearly with the batch size. When the batch size
            # is 128, the learning rate should be 0.1.
            initial_learning_rate = 0.1 * TrainingConfig.batch_size / 128
            batches_per_epoch = ModelConfig.num_image['train'] / TrainingConfig.batch_size
            # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
            boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
            values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
            learning_rate = tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32), boundaries, values)
        elif TrainingConfig.learning_rate_policy == 'poly':
            learning_rate = tf.train.polynomial_decay(
                TrainingConfig.initial_learning_rate,
                tf.cast(global_step, tf.int32) - TrainingConfig.initial_global_step,
                TrainingConfig.max_iter, TrainingConfig.end_learning_rate, TrainingConfig.power)
        else:
            raise ValueError('Learning rate policy must be "piecewise" or "poly"')

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        # optimizer = tf.train.MomentumOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=TrainingConfig.momentum)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
    else:
        train_op = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

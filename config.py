class ModelConfig(object):
    data_dir = None
    debug = 'store_true'
    # debug = False
    # interest_label = [6, 7, 8, 9, 10]
    # num_classes = len(interest_label) + 1 # extra for background
    num_classes = 30
    height = 513
    width = 513
    depth = 3
    min_scale = 0.5
    max_scale = 2.0
    ignore_label = 255
    bn_decay = 0.9997
    num_image = {
        'train': 2685,
        'validation': 500,
    }


class TrainingConfig(object):
    clean_model_dir = True
    train_epochs = 10
    epochs_per_eval = 1
    tensorboard_images_max_outputs = 6
    batch_size = 4
    learning_rate_policy = 'poly' # choices=['poly', 'piecewise']
    max_iter = 30000
    base_architecture = 'resnet_v2_50' # choices=['resnet_v2_50', 'resnet_v2_101']
    initial_learning_rate = 7e-1
    end_learning_rate = 1e-6
    initial_global_step = 0
    power = 0.9
    momentum = 0.9
    weight_decay = 5e-4 # regulization
    freeze_batch_norm = True
    pre_trained_model = './resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
    model_dir = './new'





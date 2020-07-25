# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for creating TFRecord data sets.
source: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os

import tensorflow as tf
from config import ModelConfig
from data_utils import preprocessing


def get_file_list(folder_dir):
    """
    iteratively get file list under folder_dir
    :param folder_dir: folder
    :return: a list of files
    """
    file_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in files:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
        for name in dirs:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
    return file_list


def read_examples_list(path):
    """Read list of training or validation examples from labels folder.
    Args:
    path: absolute path to train/valid/ labels examples list file.
    Returns:
    list of example identifiers (strings).
    """
    # get files
    file_list = get_file_list(path)
    # filter out undesired file
    file_list = list(filter(lambda x: 'label' in x, file_list))
    return file_list


def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def read_dataset(
    file_read_func, decode_func, input_files, config, num_workers=1,
    worker_index=0):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.data.Dataset.interleave, to read
      every individual file into a tf.data.Dataset.
    decode_func: Function to apply to all records.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.
    num_workers: Number of workers / shards.
    worker_index: Id for the current worker.

  Returns:
    A tf.data.Dataset based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],
                        0)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.shard(num_workers, worker_index)
  dataset = dataset.repeat(config.num_epochs or None)
  if config.shuffle:
    dataset = dataset.shuffle(config.filenames_shuffle_buffer_size,
                              reshuffle_each_iteration=True)

  # Read file records and shuffle them.
  # If cycle_length is larger than the number of files, more than one reader
  # will be assigned to the same file, leading to repetition.
  cycle_length = tf.cast(
      tf.minimum(config.num_readers, tf.size(filenames)), tf.int64)
  # TODO: find the optimal block_length.
  dataset = dataset.interleave(
      file_read_func, cycle_length=cycle_length, block_length=1)

  if config.shuffle:
    dataset = dataset.shuffle(config.shuffle_buffer_size,
                              reshuffle_each_iteration=True)

  dataset = dataset.map(decode_func, num_parallel_calls=config.num_readers)
  return dataset.prefetch(config.prefetch_buffer_size)


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return ['D:\herschel\\navigation\\tf_records\\fine_train.record']
  else:
    return ['D:\herschel\\navigation\\tf_records\\fine_val.record']


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # height = tf.cast(parsed['image/height'], tf.int32)
  # width = tf.cast(parsed['image/width'], tf.int32)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), 3)
  image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  image.set_shape([None, None, 3])

  label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
  label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  label.set_shape([None, None, 1])

  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, ModelConfig.min_scale, ModelConfig.max_scale)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, ModelConfig.height, ModelConfig.width, ModelConfig.ignore_label)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)
    # select interested labels
    # label = preprocessing.categories_selection(label, ModelConfig.interest_label)
    image.set_shape([ModelConfig.height, ModelConfig.width, 3])
    label.set_shape([ModelConfig.height, ModelConfig.width, 1])

  # image = preprocessing.mean_image_subtraction(image)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  # TODO: add classification labels
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=ModelConfig.num_image['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


if __name__ == '__main__':
    train_labels = read_examples_list('D:\herschel\\navigation\data\gtFine_trainvaltest\gtFine\\train')
    train_img = get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\train')
    pass
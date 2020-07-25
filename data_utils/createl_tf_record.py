"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys

import PIL.Image
import tensorflow as tf
from data_utils import dataset_util


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(image_path,
                       label_path):
  """Convert a single image and label to tf.Example proto.

  Args:
    image_path: Path to a single PASCAL image.
    label_path: Path to its corresponding label.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  with tf.gfile.GFile(label_path, 'rb') as fid:
    encoded_label = fid.read()
  encoded_label_io = io.BytesIO(encoded_label)
  label = PIL.Image.open(encoded_label_io)

  if image.size != label.size:
    raise ValueError('The size of image does not match with that of label.')

  width, height = image.size

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/encoded': bytes_feature(encoded_jpg),
    'image/format': bytes_feature('png'.encode('utf8')),
    'label/encoded': bytes_feature(encoded_label),
    'label/format': bytes_feature('png'.encode('utf8')),
  }))
  return example


def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     ):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir:  a list of images dir.
    label_dir: a list of files dir.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, (image_path, label_path) in enumerate(zip(image_dir, label_dir)):
    if idx % 200 == 0:
      print ('On image {} of {}'.format(idx, len(image_dir)))

    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    elif not os.path.exists(label_path):
      tf.logging.warning('Could not find %s, ignoring example.', label_path)
      continue
    if image_path.split('\\')[-1].split('_left')[0] != label_path.split('\\')[-1].split('_gt')[0]:
        tf.logging.warning('img and label not match', image_path.split('\\')[-1].split('_left')[0], label_path.split('\\')[-1].split('_gt')[0])
        continue
    try:
      tf_example = dict_to_tf_example(image_path, label_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.')
  print ('final sample:{}'.format(len(image_dir)))
  writer.close()


def main(arg):
  tf.logging.info("Reading from dataset")
  train_img = dataset_util.get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\train')
  train_labels = dataset_util.read_examples_list('D:\herschel\\navigation\data\gtFine_trainvaltest\gtFine\\train')
  val_labels = dataset_util.read_examples_list('D:\herschel\\navigation\data\gtFine_trainvaltest\gtFine\\val')
  val_img = dataset_util.get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\val')
  prefix = 'D:\herschel\\navigation\\tf_records'
  train_output_path = os.path.join(prefix, 'fine_train.record')
  val_output_path = os.path.join(prefix, 'fine_val.record')

  create_tf_record(train_output_path, train_img, train_labels)
  create_tf_record(val_output_path, val_img, val_labels)


if __name__ == '__main__':
  tf.app.run()



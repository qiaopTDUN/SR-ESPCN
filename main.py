import os
import scipy.misc
import numpy as np

from srresnet import srresnet
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 150, "Epoch to train [25]")
flags.DEFINE_string("optimizer", "Adam", "optimization methods [Adam, SGD]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_float("learning_rate", 0.0001, "the initial learning  rate [0.0001]")
flags.DEFINE_float("momentum", 0.9, "Momentum term of SGD [0.9]")
flags.DEFINE_float("lr_decay_step", 800*30, "Learning rate decay step of SGD [704]")
flags.DEFINE_float("lr_decay_rate", 0.1, "Learning rate decay rate of SGD [0.1]")
flags.DEFINE_integer("train_size", 999999, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 384, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "DIV2K", "The name of dataset [celebA, mnist, lsun, DIV2K]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sr = srresnet(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            sr.train(FLAGS)
        else:
            sr.load(FLAGS.checkpoint_dir)

if __name__ == '__main__':
    tf.app.run()

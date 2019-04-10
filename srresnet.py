import os
import time
from glob import glob
import tensorflow as tf
from scipy.misc import imresize
from generator import generator_sr
from utils import *


def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y


class srresnet(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 96 
        self.sample_size = batch_size
        self.image_shape = [image_size, image_size, 3] #image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                    name='real_images')
        try:
            self.up_inputs = tf.image.resize_images(self.inputs, self.image_shape[0], self.image_shape[1], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:
            # newer versions of tensorflow
            self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')

        # self.G = self.generator(self.inputs)
        self.G = generator_sr(self.inputs)

        self.G_sum = tf.summary.image("G", self.G)

        self.g_loss = tf.reduce_mean(tf.square(self.images-self.G))

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        #self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_vars = [var for var in t_vars]

        self.saver = tf.train.Saver()

    def train(self, config):
        # first setup validation data
        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.png")))

        global_step = tf.train.create_global_step()
        lr = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps=config.lr_decay_step,
                                        decay_rate=config.lr_decay_rate)
        if config.optimizer == 'SGD':
            g_optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum).minimize(self.g_loss, var_list=self.g_vars)
        elif config.optimizer == 'Adam':
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(max_to_keep=60)

        self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image_samp(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [int(self.batch_size/8), 8], './samples/inputs_small.png')
        save_images(sample_images, [int(self.batch_size/8), 8], './samples/reference.png')

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in range(config.epoch):
            print('epoch : {}'.format(epoch))
            data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.png")))
            # batch_idxs = min(len(data), config.train_size) // config.batch_size
            batch_idxs = min(len(data), config.train_size)

            for idx in range(0, batch_idxs):
                #batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                #batch = [get_image(batch_file, self.image_size, config.batch_size, is_crop=self.is_crop) for batch_file in batch_files]
                batch_file = data[idx]
                batch = get_image(batch_file, self.image_size, config.batch_size, is_crop=self.is_crop)
                input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                    feed_dict={ self.inputs: batch_inputs, self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG))

                if np.mod(counter, 500) == 1:
                    samples, g_loss, up_inputs = self.sess.run(
                        [self.G, self.g_loss, self.up_inputs],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )
                    if not have_saved_inputs:
                        save_images(up_inputs, [int(self.batch_size/8), 8], './samples/inputs.png')
                        have_saved_inputs = True

                    diff = (samples - sample_images)/2.
                    diff = np.reshape(diff, (self.batch_size, -1))
                    rmse = np.sqrt(np.mean(diff ** 2, 1))
                    psnr = 20 * np.log10(1 / rmse)

                    save_images(samples, [int(self.batch_size/8), 8],
                                './samples/valid_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f, PSNR: %.8f" % (g_loss, np.mean(psnr)))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "srresnet.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

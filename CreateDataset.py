import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue # pylint: disable=import-error
import traceback
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import dnnlib.tflib as tflib
import h5py

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image_complex(self, img):
        print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 2, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            
            tfr_opt = tf.compat.v1.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file =self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.compat.v1.io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=img.flatten()))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1
        
    def add_image_magnitude(self, img):
        print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            tfr_opt = tf.compat.v1.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            tfr_file = os.path.join(self.tfrecord_dir, 'train.tfrecords')
            self.tfr_writer = tf.compat.v1.io.TFRecordWriter(tfr_file, tfr_opt)
        assert img.shape == self.shape
        img = img.astype(np.float32)
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
            'data': tf.train.Feature(float_list=tf.train.FloatList(value=img.flatten()))}))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    # add coil_maps(5,256,320) to (2,5,256,320)
    def add_coil_maps(self, coil_maps):
        print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = coil_maps.shape
            assert self.shape[0] in [1,2, 3]
            tfr_opt = tf.compat.v1.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            tfr_file = os.path.join(self.tfrecord_dir, 'train.tfrecords')
            self.tfr_writer = tf.compat.v1.io.TFRecordWriter(tfr_file, tfr_opt)
        assert coil_maps.shape == self.shape
        coil_maps = coil_maps.astype(np.float32)
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=coil_maps.shape)),
            'data': tf.train.Feature(float_list=tf.train.FloatList(value=coil_maps.flatten()))}))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfrecord_dir + '/-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


#----------------------------------------------------------------------------

def create_from_hdf5_complex(tfrecord_dir, hdf5_filename, h5_key, label_index = None, shuffle = 0):
    num = 256
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_data = hdf5_file[h5_key]
    
        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[0]) as tfr:

            order = np.arange(hdf5_data.shape[0])
            if shuffle:
                np.random.shuffle(order)
            for idx in range(order.size):
                temp_slice = hdf5_data[order[idx]]
                #temp_slice = np.transpose(temp_slice, (0,2,1))
                real = np.real(temp_slice)
                imag = np.imag(temp_slice)

                #real = (temp_slice['real'])
                #imag = (temp_slice['imag'])

                #real = (temp_slice[0])
                #imag = (temp_slice[1])
                
                ones = np.ones([num,num])

                w = int((num - real.shape[0]) /2)
                h = int((num - real.shape[1]) /2)

                abs_image_max = np.max(np.abs(real + 1j * imag))

                real = real / abs_image_max
                imag = imag / abs_image_max

                real = np.pad(real, ((w,w),(h,h)), mode='constant', constant_values=0)
                imag = np.pad(imag, ((w,w),(h,h)), mode='constant', constant_values=0)
                #print(f"Shape of stacked image: {np.stack([real, imag], axis=0).shape}")
                tfr.add_image_complex(np.stack([real,imag],axis=0))
            
            if label_index != None:
                onehot = np.zeros((hdf5_data.shape[0], 5), dtype=np.float32)
                onehot[:,label_index] = 1
                tfr.add_labels(onehot)

def create_from_hdf5_magnitude(tfrecord_dir, hdf5_filename, h5_key, label_index = None, shuffle = 0):
    
    with h5py.File(hdf5_filename, 'r') as hdf5_file:

        hdf5_data = hdf5_file[h5_key]

        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[0]) as tfr:
            order = np.arange(hdf5_data.shape[0])
            if shuffle:
                np.random.shuffle(order)
            
            for idx in range(order.size):
                temp_slice = hdf5_data[order[idx],:,:]
                #w = int((256 - temp_slice.shape[0]) /2)
                #h = int((256 - temp_slice.shape[1]) /2)
                #temp_slice = np.pad(temp_slice, ((w,w),(h,h)), mode='constant', constant_values=0)
                #temp = np.expand_dims(np.transpose(temp_slice), axis=0)
                temp = np.expand_dims((temp_slice), axis=0)
                #temp = np.transpose(temp, (0,2,1))
                tfr.add_image_magnitude(temp)

            if label_index != None:
                onehot = np.zeros((hdf5_data.shape[0], 5), dtype=np.float32)
                onehot[:,label_index] = 1
                tfr.add_labels(onehot)

def create_from_hdf5_coil_maps(tfrecord_dir, hdf5_filename, h5_key, label_index = None, shuffle = 0):
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_data = hdf5_file[h5_key]
        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[1]) as tfr:
            order = np.arange(hdf5_data.shape[1])
            if shuffle:
                np.random.shuffle(order)
            for idx in range(order.size):
                temp_slice = hdf5_data[:,order[idx],:,:]
                #real = np.transpose(temp_slice)['real']
                #imag = np.transpose(temp_slice)['imag']
                real = (temp_slice)['real']
                imag = (temp_slice)['imag']
                temp = (np.stack([real, imag], axis=0))
                # transpose to [0,3,1,2]
                #temp = np.transpose(temp, (0,3,1,2))
                tfr.add_coil_maps(temp)
            if label_index != None:
                onehot = np.zeros((hdf5_data.shape[0], 5), dtype=np.float32)
                onehot[:,label_index] = 1
                tfr.add_labels(onehot)
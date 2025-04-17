# Adapted from https://github.com/hojonathanho/diffusion to work on multiple inputs
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class IXI_dataset:
  def __init__(self,
    tfr_file_us_image,            # Path to tfrecord file.
    tfr_file_mask,
    label_file,
    resolution=256,      # Dataset resolution.
    max_images=7500,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
    batch_size=1,
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file_us_image  = tfr_file_us_image
    self.tfr_file_mask      = tfr_file_mask
    self.label_file         = label_file
    self.dtype              = 'float32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]
    self.batch_size = batch_size

  def train_input_fn(self):
    # Build TF expressions.
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_us = dset_us.take(self.max_images)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_mask = dset_mask.take(self.max_images)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)    
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})
    _tf_labels_dataset = _tf_labels_dataset.take(self.max_images)

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask))
    dset = dset.map(lambda x,y: dict(us_im= x["us_im"], mask=y["mask"]))

    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    #if not one_pass:
    dset = dset.repeat()
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset



  def eval_input_fn(self):
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask))
    dset = dset.map(lambda x,y: dict(us_im=x["us_im"], mask=y["mask"]))
    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))

    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset 

  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf(self, record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        return {self.name:data}


class fastMRI_dataset:
  def __init__(self,
    tfr_file_us_image,            # Path to tfrecord file.
    tfr_file_mask,
    tfr_file_coil_map,
    label_file,
    resolution=512,      # Dataset resolution.
    max_images=7500,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
    batch_size=1,
  ):
    
    self.tfr_file_us_image  = tfr_file_us_image
    self.tfr_file_mask      = tfr_file_mask
    self.tfr_file_coil_map  = tfr_file_coil_map
    self.label_file         = label_file
    self.dtype              = 'float32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 6         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]
    self.batch_size = batch_size

  def train_input_fn(self):
    # Build TF expressions.
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_us = dset_us.take(self.max_images)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_mask = dset_mask.take(self.max_images)

    dset_coil_map = tf.data.TFRecordDataset(self.tfr_file_coil_map,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'coil_map'
    dset_coil_map = dset_coil_map.map(self._parse_tfrecord_tf_4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_coil_map = dset_coil_map.take(self.max_images)



    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)    
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})
    _tf_labels_dataset = _tf_labels_dataset.take(self.max_images)

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask, dset_coil_map))
    dset = dset.map(lambda x,y,z : dict(us_im=x["us_im"], mask=y["mask"],  coil_map=z["coil_map"]))


    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    dset = dset.repeat()
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset

  def eval_input_fn(self):
    dset_us = tf.data.TFRecordDataset(self.tfr_file_us_image,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'us_im'
    dset_us = dset_us.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_mask = tf.data.TFRecordDataset(self.tfr_file_mask,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'mask'
    dset_mask = dset_mask.map(self._parse_tfrecord_tf_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dset_coil_map = tf.data.TFRecordDataset(self.tfr_file_coil_map,compression_type='', buffer_size=self.buffer_mb<<20)
    self.name = 'coil_map'
    dset_coil_map = dset_coil_map.map(self._parse_tfrecord_tf_4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset_coil_map = dset_coil_map.take(self.max_images)

    _np_labels = np.load(self.label_file)
    _tf_labels_var = tf.convert_to_tensor(_np_labels, name = "label")
    _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(_tf_labels_var)
    _tf_labels_dataset = _tf_labels_dataset.map(lambda x: {'label':x})

    dset = tf.data.TFRecordDataset.zip((dset_us, dset_mask, dset_coil_map))
    dset = dset.map(lambda x,y,z : dict(us_im=x["us_im"], mask=y["mask"], coil_map = z['coil_map']))
    dset = tf.data.TFRecordDataset.zip((dset,_tf_labels_dataset))
    # Shuffle and prefetch
   # dset = dset.shuffle(50000)
    dset = dset.batch(self.batch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset 


  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf_4(self, record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([4], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        return {self.name:data}

  def _parse_tfrecord_tf_3(self, record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True,default_value=0.0)})
        data = tf.cast(features['data'], tf.float32)
        data = tf.reshape(data, features['shape'])
        return {self.name:data}


def get_dataset(name, *, batch_size, phase='train'):

  if name == "ixi":
    if phase == 'train':
        return IXI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/train_us_sv/-r08.tfrecords",
                           
                           "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/train_map_sv/-r08.tfrecords",
                           '/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/train_fs/-r08.tfrecords',
                           "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/train_us/-rxx.labels", batch_size=batch_size)
    elif phase == 'test':
      return IXI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/IXI/test/test_us_8x/test_us_8x-r08.tfrecords",
                         "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/IXI/test/test_map_8x/train.tfrecords",
                         "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/IXI/test/test_us_8x/-rxx.labels", batch_size=batch_size)
      """ return IXI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/IXI/TFRecords/IXI_T1/data_4x/-r08.tfrecords",
                         "/home/yuxuan/Dual_SDiff/data/datasets/IXI/TFRecords/IXI_T1/map_4x/-r08.tfrecords",
                         "/home/yuxuan/Dual_SDiff/data/datasets/IXI/TFRecords/IXI_T1/data_4x/-rxx.labels", batch_size=batch_size) """
    elif phase == 'val':
      return IXI_dataset("/home/yuxuan/SSDiffRecon/data/datasets/tfrecords-datasets/ixi_mixed_val_us_images/-r08.tfrecords",
                         "/home/yuxuan/SSDiffRecon/data/datasets/tfrecords-datasets/ixi_mixed_val_mask_images/-r08.tfrecords",
                         "/home/yuxuan/SSDiffRecon/data/datasets/tfrecords-datasets/ixi_mixed_val_us_images/-rxx.labels", batch_size=batch_size)
    else:
      print("none of the phases is selected")
  elif name == 'fastMRI':
    if phase == 'train':
      return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/ssdiff_train_us/-r09.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/ssdiff_train_mask/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/ssdiff_train_coil_maps/train.tfrecords", 
                              #"/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/ssdiff_train_fs/-r09.tfrecords",
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/ssdiff_train_us/-rxx.labels", batch_size=batch_size)
    elif phase == 'val':
      return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/test/test_us/-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/test/test_mask/test.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/test/test_coil_maps/test.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/ssdiff/test/test_us/-rxx.labels", batch_size=batch_size)
    elif phase == 'test':
      if 1:
        return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_us_4x/test_us_4x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_map_4x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_us_4x/-rxx.labels", batch_size=batch_size)  
        """ return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_us_4x/test_us_4x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_map_4x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_us_4x/-rxx.labels", batch_size=batch_size) """  
        """ return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_us_4x/test_us_4x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_map_4x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_us_4x/-rxx.labels", batch_size=batch_size) """ 
      else :
        """ return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_us_8x/test_us_8x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_map_8x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_us_8x/-rxx.labels", batch_size=batch_size) """ 
        """ return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_us_8x/test_us_8x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_map_8x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T1/test_us_8x/-rxx.labels", batch_size=batch_size) """
        return fastMRI_dataset("/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_us_8x/test_us_8x-r09.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_map_8x/train.tfrecords", 
                             "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_coil_maps/train.tfrecords", 
                              "/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/T2/test_us_8x/-rxx.labels", batch_size=batch_size)                      
      
  else:
     print("Dataset is not defined")
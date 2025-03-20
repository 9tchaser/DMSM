#COMPLEX AND MAGNITUDE TFRECORDS DATA PREPARATION TOOL, CONFIGURE ACCORDINGLY
import warnings
import os
warnings.filterwarnings("ignore")

from Dual_SDiff.CreateDataset import create_from_hdf5_magnitude
from Dual_SDiff.CreateDataset import create_from_hdf5_complex
from Dual_SDiff.CreateDataset import create_from_hdf5_coil_maps


""" h5_path = f"/home/yuxuan/Dual_SDiff/data/datasets/h5_dataset/fastMRI/test/FLAIR/FLAIR_downsampled_test.mat"
tf_path = f"/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_map_8x"
#tf_path = f"/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/fastMRI/test/FLAIR/test_us_8x"
#create_from_hdf5_complex(tf_path, h5_path, h5_key = "image_8x", label_index = 0, shuffle = 0)  
create_from_hdf5_magnitude(tf_path, h5_path, h5_key = "map_8x", label_index = 0, shuffle = 0) """

h5_path = f'/home/yuxuan/Dual_SDiff/data/datasets/IXI/HDF5/data_US.mat'
tf_path = f"/home/yuxuan/Dual_SDiff/data/datasets/tfrecords_dataset/IXI/test/test_map_8x"
#create_from_hdf5_complex(tf_path, h5_path, h5_key = "image_4x", label_index = 0, shuffle = 0) 
create_from_hdf5_magnitude(tf_path, h5_path, h5_key = "map_8x", label_index = 0, shuffle = 0)
#create_from_hdf5_coil_maps(tf_path, h5_path, h5_key = "coil_maps", label_index = 0, shuffle = 0)   
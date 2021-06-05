'''
Script to convert the 2017 3D kitti training data into a 
linemod/occlusion format used by EfficientPose training.

Expected folder structure for kitti:

$PWD/training/
  ├── calib/
  |    └── .txt files  (camera intrinsics)
  ├── image_2/         
  |    └── .png images (left RGB camera)
  └── label_2/
       └── .txt files

Output folder structure for Linemod/Occlusion data:

$PWD/kitti_linemod/
  └── data/
       ├── models_info.yml
       └── 02/
           ├── gt.yml
           ├── info.yml (camera intrinsics)
           ├── test.txt
           ├── train.txt
           └── rgb/
                └── .png images

The kitti label format:

 Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

'''

import os
import sys
import random
import pandas as pd
from pathlib import Path
from shutil import copyfile

# ensure $PWD is in python's path
sys.path.append(os.getcwd())

TEST_SPLIT = 0.1
SHUFFLE_SEED = 666

KITTI_TRAIN_DIR = os.path.join("training/")
# the "_2" suffix specifies the images from the left RGB camera
KITTI_CALIB_DIR = os.path.join(KITTI_TRAIN_DIR, "calib/")
KITTI_IMAGE_DIR = os.path.join(KITTI_TRAIN_DIR, "image_2/")
KITTI_LABEL_DIR = os.path.join(KITTI_TRAIN_DIR, "label_2/")

DATA_DIR = os.path.join("data/02")
MODELS_INFO_FILE = os.path.join(DATA_DIR, "models_info.yml")
GT_FILE = os.path.join(DATA_DIR, "gt.yml")
INFO_FILE = os.path.join(DATA_DIR, "info.yml")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
RGB_DIR = os.path.join(DATA_DIR, "rgb/")

for f in [KITTI_CALIB_DIR, KITTI_IMAGE_DIR, KITTI_LABEL_DIR]:
    assert os.path.isdir(f), "kitti data not found: " + str(f)

# make directories and files
os.makedirs(os.path.dirname(RGB_DIR), exist_ok=True)

# remove the .png from the path string
all_examples = [fn[:-4] for fn in os.listdir(KITTI_IMAGE_DIR)]
# make copy and sort
all_examples_sorted = sorted(all_examples.copy())
# remove leading zeros
all_examples_stripped = [s.lstrip("0") for s in all_examples_sorted]
all_examples_stripped[0] = "0"

random.Random(SHUFFLE_SEED).shuffle(all_examples)
split_index = int(len(all_examples) * TEST_SPLIT)
train_examples = all_examples[split_index:]
test_examples = all_examples[:split_index]
assert len(all_examples) == len(train_examples) + len(test_examples)

with open(TRAIN_FILE, "w") as f:
    for ex in train_examples:
        f.write(ex + '\n')

with open(TEST_FILE, "w") as f:
    for ex in test_examples:
        f.write(ex + '\n')

# diameter is in mm and used to calculate ADD(-S) metrics default correct
# threshold being avg point-wise error being less than 1/10th the diameter.
# Current code takes the average bcube from this file as the bcube prior. 
# ^ Hence why the 3 lines are essentially repeated.
# Human/pedestrian class should have the smallest bcube.
with open(MODELS_INFO_FILE, 'w') as yml:
    yml.write("0: {diameter: 3000, min_x: -1000, min_y: -1000, min_z: -1000, size_x: 2000, size_y: 2000, size_z: 2000}\n")
    yml.write("1: {diameter: 3000, min_x: -1000, min_y: -1000, min_z: -1000, size_x: 2000, size_y: 2000, size_z: 2000}\n")
    yml.write("2: {diameter: 3000, min_x: -1000, min_y: -1000, min_z: -1000, size_x: 2000, size_y: 2000, size_z: 2000}\n")

# copy over the images
for fn in os.listdir(KITTI_IMAGE_DIR):
    src = os.path.join(KITTI_IMAGE_DIR, fn)
    dst = os.path.join(RGB_DIR, fn)
    copyfile(src, dst)

# make info file with camera intrinsics/matrix
with open(INFO_FILE, "w") as yml:
    for i in range(len(all_examples)):
        yml.write(str(i) + ":\n")
        # The calibration files all seem very similar. Seems to be same cam + lense.
        # Download from: http://www.cvlibs.net/datasets/kitti/raw_data.php
        # Format: https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
        yml.write("  cam_K: [9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]\n")
        yml.write("  depth_scale: 1.0\n")

'''
Example of original linemod gt.yml with 2 objects detected in image 0.png.
We add a bcube property & set it to a list of bounding cube vertices (shape: 8x3)

0:
- cam_R_m2c: [1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00] # row major identity matrix for testing
  cam_t_m2c: [183.63633301, -131.49685045, 1147.30061109]
  obj_bb: [397, 155, 32, 50]
  obj_id: 1
- cam_R_m2c: [1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00]
  cam_t_m2c: [16.07958566, -104.70886953, 1030.66843587]
  obj_bb: [299, 143, 68, 119]
  obj_id: 2

'''

with open(GT_FILE, "w") as yml:
    for j, ex_path in enumerate(all_examples_sorted):
        label_path = os.path.join(KITTI_LABEL_DIR, ex_path + ".txt")
        names = ['label', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 
                 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'dim_height', 'dim_width']
        names.extend(['dim_length', 'loc_x', 'loc_y', 'loc_z', 'rotation_y'])
        labels_df = pd.read_csv(label_path, sep=" ", names=names)
        num_detections = len(labels_df.index)
        print(ex_path)
        yml.write(str(j) + ":\n")
        for i in range(num_detections):
            label = labels_df.loc[i, "label"]
            if label == "Car":
                label = 0
            elif label == "Cyclist":
                label = 1
            elif label == "Pedestrian" or label == "Person_sitting":
                label = 2
            else:
                print("ignored label: ", label)
                continue

            bb_xmin = labels_df.loc[i, "bbox_xmin"]
            bb_ymin = labels_df.loc[i, "bbox_ymin"]
            bb_xmax = labels_df.loc[i, "bbox_xmax"]
            bb_ymax = labels_df.loc[i, "bbox_ymax"]
            bbox = [bb_xmin, bb_ymin, (bb_xmax - bb_xmin), (bb_ymax - bb_ymin)]
            
            # convert from mm to m
            bcw = float(labels_df.loc[i, "dim_width"])  * 1000
            bch = float(labels_df.loc[i, "dim_height"]) * 1000
            bcl = float(labels_df.loc[i, "dim_length"]) * 1000
            
            bcx = float(labels_df.loc[i, "loc_x"]) * 1000 
            bcy = float(labels_df.loc[i, "loc_y"]) * 1000 
            bcz = float(labels_df.loc[i, "loc_z"]) * 1000 

            tvec = [bcx, bcy, bcz] 

            x = bcw / 2
            y = bch / 2
            z = bcl / 2
            
            # tlf: top, left, front bcube vertice
            tlf = [x, y, z]
            tlb = [x, -y, z]
            trb = [-x, -y, z]
            trf = [-x, y, z]
            blf = [x, y, -z]
            blb = [x, -y, -z]
            brb = [-x, -y, -z]
            brf = [-x, y, -z]
            # centered at origin: (0,0,0)
            bcube = [tlf, tlb, trb, trf, blf, blb, brb, brf]

            yml.write("- cam_R_m2c: [1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00]\n")
            yml.write("  cam_t_m2c: " + str(tvec) + "\n")
            yml.write("  obj_bb: " + str(bbox) + "\n")
            yml.write("  obj_id: " + str(label) + "\n")
            yml.write("  bcube: " + str(bcube) + "\n")

# make sure everything was created
for f in [MODELS_INFO_FILE, GT_FILE, INFO_FILE, TEST_FILE, TRAIN_FILE]:
    assert os.path.isfile(f), "Error! File not created: " + f

import os
import random
from collections import defaultdict

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from generators.common import Generator


class NuScenesGenerator(Generator):
    """
    Generator for the NuScenes Full dataset (v1.0)
    downloaded from https://www.nuscenes.org/download
    """
    SENSORS = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT'
    ]

    def __init__(self,
                 dataset_base_path,
                 image_extension=".png",
                 shuffle_dataset=True,
                 symmetric_objects={"glue", 11, "eggbox", 10}, #set with names and indices of symmetric objects
                 **kwargs):
        """
        Initializes a NuScenes generator
        Args:
            dataset_base_path: path to the dataset
            object_id: Integer object id of the object on which to generate data
            image_extension: String containing the image filename extension
            shuffle_dataset: Boolean whether to shuffle the dataset or not
            symmetric_objects: set with names and indices of symmetric objects
        """
        self._data = NuScenes(version='v1.0-trainval',
                              dataroot=dataset_base_path,
                              verbose=True)

        self.init_num_rotation_parameters(**kwargs)
        self.name_to_mask_value = defaultdict(int)
        # TODO - Just using index order as `object_id` and `class` and `class_label`
        self.class_to_name = {i: n for i, n in enumerate(self.class_names)}
        self.name_to_class = {n: i for i, n in enumerate(self.class_names)}
        self.object_ids_to_class_labels = {i: i for i, _ in enumerate(self.class_names)}
        self.class_labels_to_object_ids = self.object_ids_to_class_labels

        self.dataset_base_path = dataset_base_path
        self.dataset_path = os.path.join(self.dataset_base_path, "data")
        self.model_path = os.path.join(self.dataset_base_path, "models")
        self.image_extension = image_extension
        self.shuffle_dataset = shuffle_dataset
        self.translation_parameter = 3
        self.symmetric_objects = symmetric_objects

        self._all_image_tokens = [
            sample['data'][sensor]
            for sample in self._data.sample
            for sensor in NuScenesGenerator.SENSORS
        ]

        super().__init__(**kwargs)

    def shuffle_sequences(self, *seqs):
        """
        Shuffles lists so that the corresponding entries still match
        """
        concatenated = list(zip(*seqs))
        random.shuffle(concatenated)
        return zip(*concatenated)

    def get_bbox_3d_dict(self, class_idx_as_key=True):
        """
       Returns a dictionary which either maps the class indices or the class names to the 3D model cuboids
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model cuboids as values

        """
        # TODO
        box = np.array([[1,  1,  1,  1, -1, -1, -1, -1],
                        [1, -1, -1,  1,  1, -1, -1,  1],
                        [1,  1, -1, -1,  1,  1, -1, -1]]).T
        if class_idx_as_key:
            return {i: box for i in range(len(self.class_names))}
        else:
            return {name: box for name in self.class_names}

    def get_num_rotation_parameters(self):
        """
       Returns the number of rotation parameters. For axis angle representation there are 3 parameters used

        """
        return self.rotation_parameter

    def get_models_3d_points_dict(self, class_idx_as_key=True):
        """
       Returns either the 3d model points dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model points as values

        """
        # TODO
        return self.get_bbox_3d_dict(class_idx_as_key)


    def get_objects_diameter_dict(self, class_idx_as_key=True):
        """
       Returns either the diameter dict with class idx as key or the model name as key
        Args:
            class_idx_as_key: Boolean indicating wheter to return the class indices or the class names as keys
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values

        """
        # TODO
        return defaultdict(lambda: 1.0)


    @property
    def class_names(self):
        return [c['name'] for c in self._data.category]

    def size(self):
        return len(self._data.sample)

    def num_classes(self):
        return len(self._data.category)

    def has_label(self, label):
        """
        Returns true if label is a known label

        :param label: int
        :return: boolean
        """
        return label < self.num_classes()

    def has_name(self, name):
        return name in self.class_names

    def name_to_label(self, name):
        return self.class_names.index(name)

    def label_to_name(self, label):
        return self.class_names[label]

    def image_aspect_ratio(self, image_index):
        return 1600. / 900.

    def load_image(self, image_index):
        token = self._all_image_tokens[image_index]
        path = self._data.get_sample_data_path(token)
        return cv2.imread(path)

    def load_mask(self, image_index):
        """ Load mask at the image_index.
        """
        # TODO - if this causes trouble, probably need to change common loader or training code
        return np.zeros((1600, 900), dtype=np.float32)

    def load_annotations(self, image_index):
        num_all_rotation_parameters = self.rotation_parameter + 2 #+1 for class id and +1 for is_symmetric flag

        sample_index = image_index // len(NuScenesGenerator.SENSORS)
        sample = self._data.sample[sample_index]
        sensor = NuScenesGenerator.SENSORS[image_index % len(NuScenesGenerator.SENSORS)]
        sample_data_token = sample['data'][sensor]
        path, boxes, cam_data = self._data.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ALL)

        n = len(boxes)
        # first dimension of each is the number of annotations for this image
        annotations = {'labels': np.zeros((n,)),
                       'bboxes': np.zeros((n, 4)),
                       'rotations': np.zeros((n, num_all_rotation_parameters)),
                       'translations': np.zeros((n, self.translation_parameter)),
                       'translations_x_y_2D': np.zeros((n, 2))}

        for i, box in enumerate(boxes):
            label = self.name_to_label(box.name)
            annotations["labels"][i] = label
            annotations["bboxes"][i, :] = self.get_2d_bbox(box, cam_data)
            #transform rotation into the needed representation
            annotations["rotations"][i, :-2] = self.transform_rotation(box.rotation_matrix, self.rotation_representation)
            annotations["rotations"][i, -2] = float(self.is_symmetric_object(box.name))
            annotations["rotations"][i, -1] = label

            annotations["translations"][i, :] = box.center
            trans_2d = view_points(box.center[:, np.newaxis], cam_data, normalize=True)
            annotations["translations_x_y_2D"][i, :] = trans_2d.squeeze()[:2]

        return annotations

    def load_camera_matrix(self, image_index):
        """ Load intrinsic camera parameter for an image_index.
        """
        sample_index = image_index // len(NuScenesGenerator.SENSORS)
        sample = self._data.sample[sample_index]
        sensor = NuScenesGenerator.SENSORS[image_index % len(NuScenesGenerator.SENSORS)]
        sample_data_token = sample['data'][sensor]
        _, _, cam_data = self._data.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ALL)
        return cam_data

    def is_symmetric_object(self, name_or_object_id):
        """
       Check if the given object is considered to be symmetric or not
        Args:
            name_or_object_id: The name of the object or the id of the object
        Returns:
            Boolean indicating whether the object is symmetric or not
        """
        # TODO
        """
        animal                      n=  787, width= 0.37±0.13, len= 0.86±0.36, height= 0.60±0.20, lw_aspect= 2.35±0.69
        human.pedestrian.adult      n=208240, width= 0.67±0.13, len= 0.73±0.19, height= 1.77±0.18, lw_aspect= 1.11±0.26
        human.pedestrian.child      n= 2066, width= 0.51±0.14, len= 0.53±0.15, height= 1.38±0.25, lw_aspect= 1.05±0.23
        human.pedestrian.constructi n= 9161, width= 0.72±0.20, len= 0.71±0.20, height= 1.74±0.30, lw_aspect= 1.02±0.29
        human.pedestrian.personal_m n=  395, width= 0.62±0.12, len= 1.18±0.31, height= 1.71±0.27, lw_aspect= 1.98±0.64
        human.pedestrian.police_off n=  727, width= 0.73±0.14, len= 0.69±0.13, height= 1.83±0.14, lw_aspect= 0.97±0.18
        human.pedestrian.stroller   n= 1072, width= 0.63±0.13, len= 0.95±0.27, height= 1.17±0.15, lw_aspect= 1.58±0.68
        human.pedestrian.wheelchair n=  503, width= 0.77±0.10, len= 1.09±0.23, height= 1.37±0.09, lw_aspect= 1.42±0.23
        movable_object.barrier      n=152087, width= 2.53±0.64, len= 0.50±0.17, height= 0.98±0.15, lw_aspect= 0.21±0.14
        movable_object.debris       n= 3016, width= 1.01±0.67, len= 1.08±1.17, height= 1.26±0.46, lw_aspect= 1.09±0.72
        movable_object.pushable_pul n=24605, width= 0.60±0.27, len= 0.67±0.44, height= 1.06±0.27, lw_aspect= 1.11±0.30
        movable_object.trafficcone  n=97959, width= 0.41±0.13, len= 0.41±0.14, height= 1.07±0.27, lw_aspect= 1.03±0.24
        static_object.bicycle_rack  n= 2713, width= 6.79±5.55, len= 4.69±4.81, height= 1.32±0.26, lw_aspect= 2.03±2.74
        vehicle.bicycle             n=11859, width= 0.60±0.16, len= 1.70±0.26, height= 1.28±0.34, lw_aspect= 3.03±0.83
        vehicle.bus.bendy           n= 1820, width= 2.96±0.23, len= 9.83±1.70, height= 3.45±0.21, lw_aspect= 3.33±0.59
        vehicle.bus.rigid           n=14501, width= 2.93±0.33, len=11.23±2.07, height= 3.47±0.52, lw_aspect= 3.84±0.63
        vehicle.car                 n=493322, width= 1.95±0.19, len= 4.62±0.46, height= 1.73±0.24, lw_aspect= 2.37±0.20
        vehicle.construction        n=14671, width= 2.85±1.06, len= 6.37±3.13, height= 3.19±1.02, lw_aspect= 2.27±0.89
        vehicle.emergency.ambulance n=   49, width= 2.18±0.10, len= 5.40±0.24, height= 2.45±0.04, lw_aspect= 2.49±0.21
        vehicle.emergency.police    n=  638, width= 2.03±0.14, len= 5.04±0.33, height= 1.85±0.19, lw_aspect= 2.49±0.15
        vehicle.motorcycle          n=12617, width= 0.77±0.17, len= 2.11±0.32, height= 1.47±0.23, lw_aspect= 2.81±0.55
        vehicle.trailer             n=24860, width= 2.90±0.53, len=12.29±4.52, height= 3.87±0.75, lw_aspect= 4.17±1.39
        vehicle.truck               n=88519, width= 2.51±0.45, len= 6.93±2.17, height= 2.84±0.84, lw_aspect= 2.75±0.56
        """

        return True

    def get_2d_bbox(self, box, camera):
        """
        Computes the 2D bounding box containing the given 3D bounding box as projected on the given camera.

        :param box: nuscenes.utils.data_classes.Box
        :param camera: numpy array with shape (3, 3)
        :return: numpy array with shape (4,) containing the 2D bounding box
        """
        corners = box.corners()
        projected_corners = view_points(corners, camera, normalize=True)
        min_x = np.min(projected_corners[0, :])
        max_x = np.max(projected_corners[0, :])
        min_y = np.min(projected_corners[1, :])
        max_y = np.max(projected_corners[1, :])
        return np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

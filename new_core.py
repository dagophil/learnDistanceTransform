import os
import vigra
import numpy
import skneuro.learning


class LPData(object):

    @staticmethod
    def e_power(data, lam):
        """Apply exp(-lam * data) to the given data.

        :param data: Some data.
        :param lam: Exponent.
        :return: exp(-lam * data)
        """
        return numpy.exp(1)**(-lam*data)

    @staticmethod
    def e_power_inv(data, lam):
        """Apply -log(data)/lam to the given data. This is inverse to e_power(data, lam).

        :param data: Some data
        :param lam: Exponent
        :return: -log(data)/lam
        """
        return - numpy.log(data) / lam

    @staticmethod
    def normalize(feat):
        """Normalizes the data between 0 and 1.

        :param feat: some data
        :return: the normalized data
        """
        feat = feat - numpy.min(feat)
        return feat / numpy.max(feat)

    def __init__(self, cache_folder):
        self.cache_folder = cache_folder
        if not os.path.isdir(cache_folder):
            os.mkdir(cache_folder)

        self.feat_h5_key = "feat"
        self.dists_h5_key = "dists"

        self.raw_train_path = None
        self.raw_train_key = None
        self.gt_train_path = None
        self.gt_train_key = None
        self.dists_train_path = None

        self.raw_test_path = None
        self.raw_test_key = None
        self.gt_test_path = None
        self.gt_test_key = None
        self.dists_test_path = None

        self.feature_file_names_train = None
        self.feature_file_names_test = None

    def clean_cache_folder(self):
        """Delete all files in the cache folder.
        """
        if os.path.isdir(self.cache_folder):
            for f in os.listdir(self.cache_folder):
                file_path = os.path.join(self.cache_folder, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def get_raw_train(self):
        """Returns the raw training data.

        :return: raw training data
        """
        return vigra.readHDF5(self.raw_train_path, self.raw_train_key).astype(numpy.float32)

    def get_raw_test(self):
        """Returns the raw test data.

        :return: raw test data
        """
        return vigra.readHDF5(self.raw_test_path, self.raw_test_key).astype(numpy.float32)

    def get_gt_train(self):
        """Returns the ground truth training data.

        :return: ground truth training data
        """
        return vigra.readHDF5(self.gt_train_path, self.gt_train_key).astype(numpy.uint32)

    def get_gt_test(self):
        """Returns the ground truth test data.

        :return: ground truth test data
        """
        return vigra.readHDF5(self.gt_test_path, self.gt_test_key).astype(numpy.uint32)

    def get_feature_train(self, feat_id):
        """Returns the training feature with the given id.

        :param feat_id: id of the feature
        :return: the requested feature
        """
        return vigra.readHDF5(self.feature_file_names_train[feat_id], self.feat_h5_key).astype(numpy.float32)

    def get_feature_test(self, feat_id):
        """Returns the test feature with the given id.

        :param feat_id: id of the feature
        :return: the requested feature
        """
        return vigra.readHDF5(self.feature_file_names_test[feat_id], self.feat_h5_key).astype(numpy.float32)

    def set_train(self, raw_path, raw_key, gt_path, gt_key):
        """Set path and h5 key for raw and ground truth for training data.

        :param raw_path: Path to h5 file for raw data.
        :param raw_key: Key in h5 file for raw data.
        :param gt_path: Path to h5 file for ground truth data.
        :param gt_key: Key in h5 file for ground truth data.
        """
        assert os.path.isfile(raw_path)
        assert os.path.isfile(gt_path)
        self.raw_train_path = raw_path
        self.raw_train_key = raw_key
        self.gt_train_path = gt_path
        self.gt_train_key = gt_key

    def set_test(self, raw_path, raw_key, gt_path, gt_key):
        """Set path and h5 key for raw and ground truth for test data.

        :param raw_path: Path to h5 file for raw data.
        :param raw_key: Key in h5 file for raw data.
        :param gt_path: Path to h5 file for ground truth data.
        :param gt_key: Key in h5 file for ground truth data.
        """
        assert os.path.isfile(raw_path)
        assert os.path.isfile(gt_path)
        self.raw_test_path = raw_path
        self.raw_test_key = raw_key
        self.gt_test_path = gt_path
        self.gt_test_key = gt_key

    def compute_and_save_features(self, feature_list, target="train", normalize=True):
        """Compute the features from the list on training or test data and saves them to the cache folder.

        :param feature_list: List with function calls of features. Each list item must be
                             of the form [number_of_features, function_name, function_args].
        :param target: Either "train" or "test".
        :return: List with file names of the computed features.
        """
        # Read the data.
        if target == "train":
            data = self.get_raw_train()
            prefix = "train_"
        elif target == "test":
            data = self.get_raw_test()
            prefix = "test_"
        else:
            raise Exception('Parameter "target" must be "train" or "test".')

        # Find the zero fill in for the file names.
        num_features = sum([p[0] for p in feature_list])
        zfill = int(numpy.ceil(numpy.log10(num_features)))

        # Compute and save the features.
        file_names = []
        for feat_item in feature_list:
            # Compute the feature.
            feat = feat_item[1](*([data]+feat_item[2:]))

            # Save the feature.
            if feat_item[0] == 1:
                file_names.append(os.path.join(self.cache_folder, prefix + str(len(file_names)).zfill(zfill) + ".h5"))
                if normalize:
                    feat = LPData.normalize(feat)
                vigra.writeHDF5(feat, file_names[-1], self.feat_h5_key, compression="lzf")
            else:
                for k in range(feat_item[0]):
                    file_names.append(os.path.join(self.cache_folder, prefix + str(len(file_names)).zfill(zfill) + ".h5"))
                    if normalize:
                        feat[..., -1] = LPData.normalize(feat[..., -1])
                    vigra.writeHDF5(feat[..., -1], file_names[-1], self.feat_h5_key, compression="lzf")

        if target == "train":
            self.feature_file_names_train = file_names
        elif target == "test":
            self.feature_file_names_test = file_names

    def load_features(self, feature_list, target="train"):
        """Generate the file name list for the training or test features.

        :param feature_list: List with function calls of features. Each list item must be
                             of the form [number_of_features, function_name, function_args].
        :param target: Either "train" or "test".
        """
        # Get the file name prefix.
        if target == "train":
            prefix = "train_"
        elif target == "test":
            prefix = "test_"
        else:
            raise Exception('Parameter "target" must be "train" or "test".')

        # Find the zero fill in for the file names.
        num_features = sum([p[0] for p in feature_list])
        zfill = int(numpy.ceil(numpy.log10(num_features)))

        # Create the file name list.
        file_names = [os.path.join(self.cache_folder, prefix + str(i).zfill(zfill) + ".h5")
                      for i in range(num_features)]

        if target == "train":
            self.feature_file_names_train = file_names
        elif target == "test":
            self.feature_file_names_test = file_names

    def compute_distance_transform_on_gt(self, target="train"):
        """
        Compute the distance transform of the edge image of ground truth of training or test data and save it to the
        cache folder.

        :param target: Either "train" or "test".
        """
        # Read the data.
        if target == "train":
            data = self.get_gt_train()
            file_name = os.path.join(self.cache_folder, "dists_train.h5")
            self.dists_train_path = file_name
        elif target == "test":
            data = self.get_gt_test()
            file_name = os.path.join(self.cache_folder, "dists_test.h5")
            self.dists_test_path = file_name
        else:
            raise Exception('Parameter "target" must be "train" or "test".')

        # Compute the edge image and the distance transform.
        edges = skneuro.learning.regionToEdgeGt(data)
        edges[edges == 1] = 0
        edges[edges == 2] = 1
        dists = vigra.filters.distanceTransform3D(edges.astype(numpy.float32))

        # Save the result.
        vigra.writeHDF5(dists, file_name, self.dists_h5_key, compression="lzf")

    def load_dists(self, target="train"):
        """Load distance transform.

        :param target: Either "train" or "test".
        """
        if target == "train":
            self.dists_train_path = os.path.join(self.cache_folder, "dists_train.h5")
        elif target == "test":
            self.dists_test_path = os.path.join(self.cache_folder, "dists_test.h5")
        else:
            raise Exception('Parameter "target" must be "train" or "test".')
import os
import vigra
import numpy
import skneuro.learning
from sklearn.ensemble import RandomForestRegressor
import logging as log
import opengm


def round_to_nearest(arr, l):
    """Round all values in arr to the nearest value in l and return the result.

    :param arr: numpy array
    :param l: list
    :return: rounded array
    """
    l = sorted(l)
    splits = [(l[i] + l[i+1]) / 2.0 for i in xrange(len(l)-1)]
    arr_rounded = numpy.zeros(arr.shape)
    arr_rounded[arr < splits[0]] = l[0]
    arr_rounded[arr >= splits[-1]] = l[-1]
    for i in xrange(0, len(splits) - 1):
        arr_rounded[numpy.logical_and(splits[i] <= arr, arr < splits[i+1])] = l[i+1]
    return arr_rounded


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
        self.pred_h5_key = "pred"

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

        self.pred_path = None
        self.pred_cap = None

        self.rf_regressor = None

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
            raise Exception('LPData.compute_and_save_features(): Parameter "target" must be either "train" or "test".')

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
            raise Exception('LPData.load_features(): Parameter "target" must be either "train" or "test".')

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

    def get_data_x(self, data_name="train"):
        """Returns the desired data as a ready-to-use n x d sample (n instances with d features).

        :param data_name: name of the data, either "train" or "test"
        :return: data
        """
        if not data_name in ["train", "test"]:
            raise Exception('LPData.get_data_x(): Parameter data_name must be either "train" or "test".')

        if data_name == "train":
            file_names = self.feature_file_names_train
        elif data_name == "test":
            file_names = self.feature_file_names_test

        if len(file_names) == 0:
            raise Exception("LPData.get_data_x(): There is no data that can be returned.")

        # Load the first feature to get the number of instances.
        d = vigra.readHDF5(file_names[0], self.feat_h5_key).flatten()
        data = numpy.zeros((d.shape[0], len(file_names)))
        data[:, 0] = d

        # Load the other features.
        for i, file_name in enumerate(file_names[1:]):
            data[:, i+1] = vigra.readHDF5(file_name, self.feat_h5_key).flatten()

        return data

    def get_data_y(self, data_name="train", data_type="gt"):
        """Returns the desired ground truth labels.

        :param data_name: name of the data, either "train" or "test"
        :param data_type: type of the ground truth, either "gt" or "dists"
        :return:
        """
        if not data_name in ["train", "test"]:
            raise Exception('LPData.get_data_y(): Parameter data_name must be either "train" or "test".')
        if not data_type in ["gt", "dists"]:
            raise Exception('LPData.get_data_y(): Parameter data_type must be either "gt" or "dists".')

        # Load the desired data.
        if data_name == "train" and data_type == "gt":
            return vigra.readHDF5(self.gt_train_path, self.gt_train_key).flatten()
        if data_name == "train" and data_type == "dists":
            return vigra.readHDF5(self.dists_train_path, self.dists_h5_key).flatten()
        if data_name == "test" and data_type == "gt":
            return vigra.readHDF5(self.gt_test_path, self.gt_test_key).flatten()
        if data_name == "test" and data_type == "dists":
            return vigra.readHDF5(self.dists_test_path, self.dists_h5_key).flatten()
        raise Exception("LPData.get_data_y(): Congratulations, you have reached unreachable code.")

    def get_train_x(self):
        """Return the training data as a ready-to-use n x d sample (n instances with d features).

        :return: training data
        """
        return self.get_data_x("train")

    def get_train_y(self, data="gt"):
        """Return the training labels.

        :param data: which ground truth shall be returned, either "gt" or "dists"
        :return: training labels
        """
        return self.get_data_y("train", data)

    def get_test_x(self):
        """Return the test data as a ready-to-use n x d sample (n instances with d features).

        :return: test data
        """
        return self.get_data_x("test")

    def get_test_y(self, data="gt"):
        """Return the test labels.

        :param data: which ground truth shall be returned, either "gt" or "dists"
        :return: test labels
        """
        return self.get_data_y("test", data)

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
            raise Exception('LPData.compute_distance_transform_on_gt(): Parameter "target" must be "train" or "test".')

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
            raise Exception('LPData.load_dists(): Parameter "target" must be "train" or "test".')

    def learn(self, gt_name="gt", n_estimators=10, n_jobs=1, invert_gt=False, lam=0.1, cap=0):
        """Learn the training data.

        :param gt_name: which ground truth shall be used, either "gt" or "dists"
        :param n_estimators: number of estimators for the random forest regressor
        :param n_jobs: number of jobs that will be used in the random forest regressor
        :param invert_gt: whether the ground truth values shall be modified by exp(-lam * gt)
        :param lam: the value of lam used for the inversion
        :param cap: maximum value of the ground truth data (ignored if 0), all larger values will be set to cap
        """
        if not gt_name in ["gt", "dists"]:
            raise Exception('LPData.learn_dists(): Parameter gt_name must be either "gt" or "dists".')

        # Get ground truth and cap and invert it.
        gt = self.get_train_y(gt_name)
        if cap != 0:
            gt[gt > cap] = cap
        if invert_gt:
            gt = LPData.e_power(gt, lam)

        self.rf_regressor = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
        log.info("Fitting the random forest regressor with %d estimators and %d cores." % (n_estimators, n_jobs))
        self.rf_regressor.fit(self.get_train_x(), gt)
        log.info("... done with fitting.")

    def predict(self, file_name=None, invert_gt=False, lam=0.1):
        """Predict the test data and [optional] save the predicted labels.

        :param file_name: output file name, if file_name is None, no output file will be produced
        :param invert_gt: whether the ground truth values where modified by exp(-lam * gt)
        :param lam: the value of lam used for the inversion
        :return: predicted labels of the test data
        """
        log.info("Predicting.")
        pred = self.rf_regressor.predict(self.get_test_x())
        log.info("... done with predicting.")

        # Revert the values.
        if invert_gt:
            pred = LPData.e_power_inv(pred, lam)

        # Save the output.
        if file_name is not None:
            vigra.writeHDF5(pred, file_name, self.pred_h5_key)
            self.pred_path = file_name

        return pred

    def build_gm_dists(self):
        """Build a graphical model from the predicted data.

        :return: graphical model
        """
        # Get the original shape.
        raw = vigra.readHDF5(self.raw_test_path, self.raw_test_key)
        sh = raw.shape
        del raw

        # Get the allowed values of the distance transform.
        test_y = self.get_test_y("dists")
        cap = self.pred_cap
        if cap is None:
            cap = numpy.max(test_y)
        test_y[test_y > cap] = cap
        allowed_vals = sorted(numpy.unique(test_y))

        # Read the predicted data.
        data = vigra.readHDF5(self.pred_path, self.pred_h5_key).reshape(sh)
        data = round_to_nearest(data, allowed_vals)

        # Build the graphical model.
        num_vars = data.size
        num_labels = len(allowed_vals)
        variable_space = numpy.ones(data.size) * num_labels
        gm = opengm.gm(variable_space)

        # Add the unaries.
        # TODO: Add unaries.
        # unaries = numpy.ones()

        # print gm.numberOfVariables
        # print gm.numberOfLabels(0)

        import sys
        raise sys.exit(0)

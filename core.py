import vigra
import numpy
import skneuro.blockwise_filters as filters
import skneuro.learning
import h5py


def compute_special_features(h5_key="feat"):
    """Compute some features on the 900 block and save them.
    """
    h5file = h5py.File("/home/philip/data/900_block/semantic_prob_r0.h5")
    thresh = 0.5

    for i in range(5):
        print "Step %d of %d" % (i+1, 5)
        print "Loading training data."
        train = h5file['data'][:900, :901, 702:902, i]
        print "Applying threshold."
        train[numpy.where(train < thresh)] = 0
        train[numpy.where(train > 0)] = 1
        print "Computing distance transform."
        train = vigra.filters.distanceTransform3D(train.astype(numpy.float32))
        train = train.reshape((train.size,))
        vigra.writeHDF5(train, "train_feature_%s.h5" % str(i).zfill(2), h5_key, compression="lzf")
        del train

        print "Loading test data."
        test = h5file['data'][25:725, :700, :700, i]
        print "Applying threshold."
        test[numpy.where(test < thresh)] = 0
        test[numpy.where(test > 0)] = 1
        print "Computing distance transform."
        test = vigra.filters.distanceTransform3D(test.astype(numpy.float32))
        test = test.reshape((test.size,))
        vigra.writeHDF5(test, "test_feature_%s.h5" % str(i).zfill(2), h5_key, compression="lzf")
        del test

    h5file.close()


def load_data():
    """Load 4 datasets: raw (training), ground truth (training), raw (test), ground truth (test)

    :return: raw_train, gt_train, raw_test, gt_test
    """
    # raw_path = "/home/philip/data/dataset03/training/raw.h5"
    # raw_key = "raw"
    # gt_path = "/home/philip/data/dataset03/training/gt_reg.h5"
    # gt_key = "gt_reg"
    #
    # raw_test_path = "/home/philip/data/dataset03/test/raw.h5"
    # raw_test_key = "raw"
    # gt_test_path = "/home/philip/data/dataset03/test/gt_reg.h5"
    # gt_test_key = "gt_reg"


    raw_path = "/home/philip/data/dataset02_100/training/raw.h5"
    raw_key = "raw"
    gt_path = "/home/philip/data/dataset02_100/training/gt_reg.h5"
    gt_key = "gt_reg"

    raw_test_path = "/home/philip/data/dataset02_100/test/raw.h5"
    raw_test_key = "raw"
    gt_test_path = "/home/philip/data/dataset02_100/test/gt_reg.h5"
    gt_test_key = "gt_reg"


    # raw_path = "/home/philip/data/900_block/train_raw.h5"
    # raw_key = "raw"
    # gt_path = "/home/philip/data/900_block/train_gt.h5"
    # gt_key = "gt"
    #
    # raw_test_path = "/home/philip/data/900_block/test_raw.h5"
    # raw_test_key = "raw"
    # gt_test_path = "/home/philip/data/900_block/test_gt.h5"
    # gt_test_key = "gt"


    # raw_path = "/home/philip/data/dataset01/training/data.h5"
    # raw_key = "data"
    # gt_path = "/home/philip/data/dataset01/training/groundtruth_t.h5"
    # gt_key = "stack"
    #
    # raw_test_path = "/home/philip/data/dataset01/test/data.h5"
    # raw_test_key = "data"
    # gt_test_path = "/home/philip/data/dataset01/test/groundtruth_t.h5"
    # gt_test_key = "stack"


    raw = vigra.readHDF5(raw_path, raw_key).astype(numpy.float32)
    gt = vigra.readHDF5(gt_path, gt_key)
    raw_test = vigra.readHDF5(raw_test_path, raw_test_key).astype(numpy.float32)
    gt_test = vigra.readHDF5(gt_test_path, gt_test_key)


    # raw = vigra.readHDF5(raw_path, raw_key)[:200, :200, :200]
    # gt = vigra.readHDF5(gt_path, gt_key)[:200, :200, :200]
    # raw_test = vigra.readHDF5(raw_test_path, raw_test_key)[:200, :200, :200]
    # gt_test = vigra.readHDF5(gt_test_path, gt_test_key)[:200, :200, :200]
    return raw, gt, raw_test, gt_test


def pseudo_dist_on_hog_eigenvalue(data, threshold=0.5):
    """Center the data between 0 and 1, apply the threshold and compute the distance transform on the result.

    :param data: some 3d numpy array
    :param threshold: threshold
    :return: distance transform after centering the data and applying the threshold
    """
    assert 0 <= threshold <= 1
    c = data - numpy.min(data)
    c /= numpy.max(c)
    c[numpy.where(c < threshold)] = 0
    c[numpy.where(c > 0)] = 1
    c = c.astype(numpy.uint32)
    return vigra.filters.distanceTransform3D(c.astype(numpy.float32))


def load_sub_samples(filenames, h5_key, sample_indices):
    """Loads the given filenames as h5 file and extracts a random sub sample of the loaded data.

    :param filenames: list with filenames
    :param h5_key: h5 key
    :param sample_indices: indices of the samples that is taken
    :return: n x d numpy array with n samples and d features
    """
    features = numpy.zeros((len(sample_indices), len(filenames)))
    for i in range(len(filenames)):
        print "Loading feature %d of %d." % (i+1, len(filenames))
        f = vigra.readHDF5(filenames[i], h5_key)
        features[:, i] = f.reshape((f.size,))[sample_indices]
    return features


def generate_feature_list(data):
    """Generates a list with features.

    Each list item is of the form [number_of_features, function_name, function_args].
    :param data: the data on which the features are computed
    :return: list with features
    """
    return [[1, filters.blockwiseGaussianSmoothing, data, 1.0],
            [1, filters.blockwiseGaussianSmoothing, data, 2.0],
            [1, filters.blockwiseGaussianSmoothing, data, 4.0],
            [1, filters.blockwiseGaussianGradientMagnitude, data, 1.0],
            [1, filters.blockwiseGaussianGradientMagnitude, data, 2.0],
            [1, filters.blockwiseGaussianGradientMagnitude, data, 4.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, data, 1.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, data, 2.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, data, 4.0],
            [1, filters.blockwiseLaplacianOfGaussian, data, 1.0],
            [1, filters.blockwiseLaplacianOfGaussian, data, 2.0],
            [1, filters.blockwiseLaplacianOfGaussian, data, 4.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, data, 0.5, 1.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, data, 1.0, 2.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, data, 2.0, 4.0]]


def generate_feature_filenames(feature_list, filename_prefix, special=None):
    """Generates a list with filenames for all features in feature_list.

    :param feature_list: list with features (see generate_feature_list())
    :param filename_prefix: prefix of the filenames
    :param special: append the special 900 block features for test (special == "test") or training (special == "train")
    :return: filenames of the features
    """
    count = 0
    for feat in feature_list:
        count += feat[0]
    filenames = [filename_prefix + str(c).zfill(2) + ".h5" for c in range(count)]

    # Append the special features for the 900 block.
    if special == "train":
        for i in range(5):
            filenames.append("train_feature_%s.h5" % str(i).zfill(2))
    if special == "test":
        for i in range(5):
            filenames.append("test_feature_%s.h5" % str(i).zfill(2))

    return filenames


def compute_features(data, filename_prefix="feature_", h5_key="feat", special=None):
    """Compute some features on the given dataset and save them as h5 file.

    Features will be saved as h5 file with the filename filename_prefix01.h5, filename_prefix02.h5, etc.
    :param data: numpy array with raw data
    :param filename_prefix: prefix of the used filename
    :param h5_key: the h5 key for the features
    :aram special: append the special 900 block features for test (special == "test") or training (special == "train")
    :return: list of the filenames
    """
    # Get the desired features.
    features = generate_feature_list(data)
    filenames = generate_feature_filenames(features, filename_prefix, special=special)

    # Call all functions in the features list and save the result as h5 file.
    count = data.size
    filenames_TMP = filenames[:]
    filenames_TMP.reverse()
    for feat in features:
        f = feat[1](*feat[2:])
        if feat[0] == 1:
            filename = filenames_TMP.pop()
            vigra.writeHDF5(f.reshape((count,)), filename, h5_key, compression="lzf")
        else:
            f = f.reshape((count, feat[0]))
            for k in range(feat[0]):
                filename = filenames_TMP.pop()
                vigra.writeHDF5(f[:, k], filename, h5_key, compression="lzf")

    # feat[:, 27] = pseudo_dist_on_hog_eigenvalue(feat[:, 8].reshape(data.shape)).reshape((count,))
    # feat[:, 28] = pseudo_dist_on_hog_eigenvalue(feat[:, 11].reshape(data.shape)).reshape((count,))
    # feat[:, 29] = pseudo_dist_on_hog_eigenvalue(feat[:, 14].reshape(data.shape)).reshape((count,))

    return filenames


def compute_edge_image(img):
    """Take a label image and return an edge image (1: edge, 0: no edge).

    :param img: labeled image
    :return: edge image of img
    """
    tmp_edge_image = skneuro.learning.regionToEdgeGt(img)
    edge_image = numpy.zeros(tmp_edge_image.shape, dtype=tmp_edge_image.dtype)
    edge_image[tmp_edge_image == 1] = 0
    edge_image[tmp_edge_image == 2] = 1
    return edge_image

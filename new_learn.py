import sys
import new_core as core
import skneuro.blockwise_filters as filters
import argparse
import vigra
import numpy
import matplotlib.pyplot as plt


def create_dummy_feature_list():
    """Generate a list with function calls of features that shall be computed.

    Each list item is of the form [number_of_features, function_name, function_args].
    :return: List with features.
    """
    return [[1, filters.blockwiseGaussianSmoothing, 1.0],
            [1, filters.blockwiseGaussianSmoothing, 2.0],
            [1, filters.blockwiseGaussianSmoothing, 4.0],
            [1, filters.blockwiseGaussianGradientMagnitude, 1.0],
            [1, filters.blockwiseGaussianGradientMagnitude, 2.0],
            [1, filters.blockwiseGaussianGradientMagnitude, 4.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, 1.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, 2.0],
            [3, filters.blockwiseHessianOfGaussianSortedEigenvalues, 4.0],
            [1, filters.blockwiseLaplacianOfGaussian, 1.0],
            [1, filters.blockwiseLaplacianOfGaussian, 2.0],
            [1, filters.blockwiseLaplacianOfGaussian, 4.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, 0.5, 1.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, 1.0, 2.0],
            [3, filters.blockwiseStructureTensorSortedEigenvalues, 2.0, 4.0]]


def data_names_dataset02_training():
    """Return raw_path, raw_key, gt_path, gt_key of training dataset02.
    """
    return "/home/philip/data/dataset02_100/training/raw.h5", "raw", \
           "/home/philip/data/dataset02_100/training/gt_reg.h5", "gt_reg"


def data_names_dataset02_test():
    """Return raw_path, raw_key, gt_path, gt_key of test dataset02.
    """
    return "/home/philip/data/dataset02_100/test/raw.h5", "raw", \
           "/home/philip/data/dataset02_100/test/gt_reg.h5", "gt_reg"


def show_plots(shape, plots):
    """Create a plot of the given shape and show the plots.

    :param shape: 2-tuple of integers
    :param plots: iterable of plots
    """
    assert 2 == len(shape)
    assert shape[0]*shape[1] == len(plots)
    fig, rows = plt.subplots(*shape)
    for i, p in enumerate(plots):
        ind0, ind1 = numpy.unravel_index(i, shape)
        if shape[0] == 1:
            rows[ind1].imshow(p)
        elif shape[1] == 1:
            rows[ind0].imshow(p)
        else:
            rows[ind0][ind1].imshow(p)
    plt.show()


def TESTHYSTERESIS(lp_data):
    """

    :param lp_data:
    :return:
    """
    raw = lp_data.get_raw_train()
    hog1 = lp_data.get_feature_train(8)
    hys1 = vigra.filters.hysteresisThreshold(hog1, 0.4, 0.01).astype(numpy.float32)

    sl = numpy.index_exp[:, :, 10]
    show_plots((1, 3), (raw[sl], hog1[sl], hys1[sl]))


def process_command_line():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="There is no description.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("workflow", type=str, nargs="+", help="names of the workflows that will be executed")
    parser.add_argument("-c", "--cache", type=str, default="cache", help="name of the cache folder")
    return parser.parse_args()


def main():
    """
    """
    # Read command line arguments.
    args = process_command_line()

    # ==========================
    # =====   Parameters   =====
    # ==========================
    raw_train_path, raw_train_key, gt_train_path, gt_train_key = data_names_dataset02_training()
    raw_test_path, raw_test_key, gt_test_path, gt_test_key = data_names_dataset02_test()
    feature_list = create_dummy_feature_list()
    # ==========================
    # ==========================
    # ==========================

    # Create the LPData object.
    lp_data = core.LPData(args.cache)
    lp_data.set_train(raw_train_path, raw_train_key, gt_train_path, gt_train_key)
    lp_data.set_test(raw_test_path, raw_test_key, gt_test_path, gt_test_key)

    # Parse the command line arguments and do the according stuff.
    for w in args.workflow:
        if w == "clean":
            lp_data.clean_cache_folder()
        elif w == "compute_train":
            lp_data.compute_and_save_features(feature_list, "train")
        elif w == "compute_test":
            lp_data.compute_and_save_features(feature_list, "test")
        elif w == "compute_dists_train":
            lp_data.compute_distance_transform_on_gt("train")
        elif w == "compute_dists_test":
            lp_data.compute_distance_transform_on_gt("test")
        elif w == "load_train":
            lp_data.load_features(feature_list, "train")
        elif w == "load_test":
            lp_data.load_features(feature_list, "test")
        elif w == "load_dists_train":
            lp_data.load_dists("train")
        elif w == "load_dists_test":
            lp_data.load_dists("test")
        elif w == "load_all":
            lp_data.load_features(feature_list, "train")
            lp_data.load_features(feature_list, "test")
            lp_data.load_dists("train")
            lp_data.load_dists("test")
        elif w == "hysteresis":
            TESTHYSTERESIS(lp_data)
        else:
            raise Exception("Unknown workflow: %s" % w)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)

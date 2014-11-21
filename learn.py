import sys
import core
import skneuro.blockwise_filters as filters
import argparse
import vigra
import numpy
import matplotlib.pyplot as plt
import logging as log
import data


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


def show_plots(shape, plots, interpolation="bilinear"):
    """Create a plot of the given shape and show the plots.

    :param shape: 2-tuple of integers
    :param plots: iterable of plots
    :param interpolation: interpolation argument to imshow
    """
    assert 2 == len(shape)
    assert shape[0]*shape[1] == len(plots)
    fig, rows = plt.subplots(*shape)
    for i, p in enumerate(plots):
        ind0, ind1 = numpy.unravel_index(i, shape)
        if shape[0] == 1:
            rows[ind1].imshow(p, interpolation=interpolation)
        elif shape[1] == 1:
            rows[ind0].imshow(p, interpolation=interpolation)
        else:
            rows[ind0][ind1].imshow(p, interpolation=interpolation)
    plt.show()


def TESTHYSTERESIS(lp_data):
    """

    :param lp_data:
    :return:
    """
    raw = lp_data.get_raw_train()
    hog1 = lp_data.get_feature_train(8)
    hys1 = vigra.filters.hysteresisThreshold(hog1, 0.4, 0.0125).astype(numpy.float32)

    sl = numpy.index_exp[:, :, 10]
    show_plots((1, 3), (raw[sl], hog1[sl], hys1[sl]))


def round_to_nearest(arr, l):
    """Round all values in arr to the nearest value in l and return the result.

    :param arr: numpy array
    :param l: list
    :return: rounded array
    """
    l = sorted(l)
    arr_rounded = numpy.zeros(arr.shape)
    arr_rounded[arr < (l[0] + l[1]) / 2.0] = l[0]
    arr_rounded[arr >= (l[-1] + l[-2]) / 2.0] = l[-1]
    for i in range(1, len(l) - 1):
        arr_rounded[numpy.logical_and((l[i-1] + l[i]) / 2.0 <= arr, arr < (l[i] + l[i+1]) / 2.0)] = l[i]
    return arr_rounded


def TESTCOMPARE(lp_data):
    """

    :param lp_data:
    :return:
    """
    sh = (100, 100, 100)
    dists_test = lp_data.get_data_y("test", "dists").reshape(sh)

    cap = 5
    dists_test[dists_test > cap] = cap
    allowed_vals = sorted(numpy.unique(dists_test))

    pred_inv = vigra.readHDF5("cache/pred_cap_lam_01.h5", "pred").reshape(sh)
    pred_inv_nearest = round_to_nearest(pred_inv, allowed_vals)

    sl = numpy.index_exp[:, :, 50]
    show_plots((1, 3), (dists_test[sl], pred_inv[sl], pred_inv_nearest[sl]), interpolation="nearest")


def process_command_line():
    """Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="There is no description.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("workflow", type=str, nargs="+", help="names of the workflows that will be executed")
    parser.add_argument("-c", "--cache", type=str, default="cache", help="name of the cache folder")
    parser.add_argument("--jobs", type=int, default=1, help="number of cores that can be used")
    parser.add_argument("--estimators", type=int, default=10, help="number of estimators for random forest regressor")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose output")
    parser.add_argument("--cap", type=float, default=0,
                        help="maximum value of distance transform (ignored if 0), all larger values will be set to this")
    return parser.parse_args()


def main():
    """
    """
    # Read command line arguments.
    args = process_command_line()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # ==========================
    # =====   Parameters   =====
    # ==========================
    raw_train_path, raw_train_key, gt_train_path, gt_train_key = data.data_names_dataset02_training()
    raw_test_path, raw_test_key, gt_test_path, gt_test_key = data.data_names_dataset02_test()
    feature_list = create_dummy_feature_list()
    # ==========================
    # ==========================
    # ==========================

    # Create the LPData object.
    lp_data = core.LPData(args.cache)
    lp_data.set_train(raw_train_path, raw_train_key, gt_train_path, gt_train_key)
    lp_data.set_test(raw_test_path, raw_test_key, gt_test_path, gt_test_key)

    # Check beforehand if the workflow arguments are usable.
    allowed_workflows = ["clean", "compute_train", "compute_test", "compute_dists_train", "compute_dists_test",
                         "load_train", "load_test", "load_dists_train", "load_dists_test", "load_all",
                         "learn_dists", "predict", "TESThysteresis", "TESTcompare"]
    for w in args.workflow:
        if not w in allowed_workflows:
            raise Exception("Unknown workflow: %s" % w)

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
        elif w == "learn_dists":
            lp_data.learn(gt_name="dists", n_estimators=args.estimators, n_jobs=args.jobs, invert_gt=True, cap=5.0)
            # lp_data.learn(gt_name="dists", n_estimators=args.estimators, n_jobs=args.jobs, cap=5.0)
        elif w == "predict":
            # lp_data.predict(file_name="cache/pred_lam_01.h5", invert_gt=True)
            lp_data.predict(file_name="cache/pred_cap_lam_01.h5", invert_gt=True)
            # lp_data.predict(file_name="cache/pred_cap.h5")
        elif w == "TESThysteresis":
            # TODO: Is this workflow still needed?
            TESTHYSTERESIS(lp_data)
        elif w == "TESTcompare":
            # TODO: Is this workflow still needed?
            TESTCOMPARE(lp_data)
        else:
            raise Exception("Unknown workflow: %s" % w)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)

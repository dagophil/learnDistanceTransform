import vigra
import numpy
from sklearn.ensemble import RandomForestRegressor
import core
import random
import sys
import os
import matplotlib.pyplot as plt


def e_power(data, lam):
    """Apply exp(-lam * data) to the given data.

    :param data: numpy array
    :param lam: exponent
    :return: exp(-lam * data)
    """
    return numpy.exp(1)**(-lam*data)


def e_power_inv(data, lam):
    """Apply log(data) / (-lam) to the given data. This is the inverse function to e_power(data, lam).

    :param data: numpy array
    :param lam: exponent
    :return: log(data) / (-lam)
    """
    return numpy.log(data) / (-lam)


def learn_and_predict(train_x, train_y, test_x, n_estimators=10, n_jobs=6):
    """Train a RandomForestRegressor and predict some data.

    :param train_x: training features
    :param train_y: training classes
    :param test_x: test features
    :param n_estimators: number of estimators
    :param n_jobs: number of threads
    :return: predicted test classes
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
    rf.fit(train_x, train_y)
    return rf.predict(test_x)


def clean_features(prefixes, prefixes_test):
    """Deletes the previously created feature files.

    :param prefixes: file prefix of training features
    :param prefixes_test: file prefix of test features
    """
    for prefix in [prefixes, prefixes_test]:
        folder, pref = os.path.split(prefix)
        for dir_name, subdir_list, file_list in os.walk(folder):
            for f in file_list:
                if f.startswith(pref):
                    os.remove(os.path.join(dir_name, f))


def random_train_predict_loop(filename_prefix, filename_prefix_test, steps, train_rate, test_rate, h5_key):
    """

    :return:
    """
    # Load the data and the features.
    print "Loading data."
    raw, gt, raw_test, gt_test = core.load_data()
    del gt
    del gt_test
    print "Loading the features."
    feature_list = core.generate_feature_list(raw)
    raw_size = raw.size
    raw_shape = raw.shape
    del raw
    features = core.generate_feature_filenames(feature_list, filename_prefix, special="train")
    del feature_list
    feature_list_test = core.generate_feature_list(raw_test)
    raw_test_size = raw_test.size
    raw_test_shape = raw_test.shape
    del raw_test
    features_test = core.generate_feature_filenames(feature_list_test, filename_prefix_test, special="test")
    del feature_list_test

    # Do the train-predict loop with a random sample of the data.
    for i in range(steps):
        print "Taking random sample of training data (step %d of %d)" % (i+1, steps)

        # Training:
        print "Computing sample indices (training)."
        sample_count = int(numpy.ceil(raw_size * train_rate))
        sample_indices = random.sample(range(raw_size), sample_count)
        print "Loading sub samples for training features."
        features_sub = core.load_sub_samples(features, h5_key, sample_indices)
        print "Getting distances for training sub samples."
        dists_sub = vigra.readHDF5(filename_prefix + "dists.h5", "dists").flatten()[sample_indices]
        # dists_sub = dists.reshape((dists.size,))[sample_indices]
        del sample_indices

        # Test:
        print "Computing sample indices (test)."
        sample_count_test = int(numpy.ceil(raw_test_size * test_rate))
        sample_indices_test = random.sample(range(raw_test_size), sample_count_test)
        print "Loading sub samples for test features."
        features_sub_test = core.load_sub_samples(features_test, h5_key, sample_indices_test)
        print "Getting distances for test sub samples."
        # dists_sub_test = dists_test.reshape((dists_test.size,))[sample_indices_test]
        # dists_sub_test = vigra.readHDF5(filename_prefix_test + "dists.h5", "dists").flatten()[sample_indices_test]
        del sample_indices_test

        # Learn and predict the distances.
        print "Learning and predicting distances."
        dists_01 = e_power(dists_sub, lam=lam)
        pred_dists_01 = learn_and_predict(features_sub, dists_01.flatten(), features_sub_test).reshape(raw_test_shape)
        pred_dists_01 = e_power_inv(pred_dists_01, lam=lam)
        vigra.writeHDF5(pred_dists_01, "pred_dists_%s.h5" % str(i).zfill(2), "pred", compression="lzf")


if __name__ == "__main__":
    # ==========================
    # =====   Parameters   =====
    # ==========================
    steps = 3
    # filename_prefix = "/media/philip/AZURA/TEMP/features_train/feature_"
    # filename_prefix_test = "/media/philip/AZURA/TEMP/features_test/feature_"
    filename_prefix = "features_train/feature_"
    filename_prefix_test = "features_test/feature_"
    h5_key = "feat"
    train_rate = 0.1
    test_rate = 0.04
    lam = 0.1
    # ==========================
    # ==========================
    # ==========================

    if len(sys.argv) > 1:
        # Clean the feature files.
        if sys.argv[1] == "clean":
            print "Cleaning feature files."
            clean_features(filename_prefix, filename_prefix_test)
            raise sys.exit(0)

        # Compute the special 900 block features.
        if sys.argv[1] == "special_features":
            print "Computing 900 block features."
            core.compute_special_features(h5_key)
            raise sys.exit(0)

        # Compute the features.
        if sys.argv[1] == "features":
            # Load the data.
            print "Loading data."
            raw, gt, raw_test, gt_test = core.load_data()

            # Compute some features on the raw data.
            print "Computing training features."
            features = core.compute_features(raw, filename_prefix=filename_prefix, h5_key=h5_key, special="train")
            print "Computing test features."
            features_test = core.compute_features(raw_test, filename_prefix=filename_prefix_test, h5_key=h5_key, special="test")
            raise sys.exit(0)

        # Compute the distances.
        if sys.argv[1] == "distances":
            # Load the data.
            print "Loading data."
            raw, gt, raw_test, gt_test = core.load_data()
            del raw
            del raw_test

            # Compute the distance transform on the edge image.
            print "Computing distance transform on edge image."
            edges = core.compute_edge_image(gt)
            dists = vigra.filters.distanceTransform3D(edges.astype(numpy.float32))
            vigra.writeHDF5(dists, filename_prefix + "dists.h5", "dists", compression="lzf")
            del dists
            del edges
            edges_test = core.compute_edge_image(gt_test)
            dists_test = vigra.filters.distanceTransform3D(edges_test.astype(numpy.float32))
            vigra.writeHDF5(dists_test, filename_prefix_test + "dists.h5", "dists", compression="lzf")
            del dists_test
            del edges_test
            raise sys.exit(0)

        # Train-predict loop with random features.
        if sys.argv[1] == "random":
            random_train_predict_loop(filename_prefix, filename_prefix_test, steps, train_rate, test_rate, h5_key)
            raise sys.exit(0)

        # Do some test with the hysteresis.
        if sys.argv[1] == "hysteresis":
            # Load the data and the features.
            print "Loading data."
            raw, gt, raw_test, gt_test = core.load_data()
            print "Loading the features."
            feature_list = core.generate_feature_list(raw)
            features = core.generate_feature_filenames(feature_list, filename_prefix, special="train")
            del feature_list
            feature_list_test = core.generate_feature_list(raw_test)
            raw_test_size = raw_test.size
            raw_test_shape = raw_test.shape
            features_test = core.generate_feature_filenames(feature_list_test, filename_prefix_test, special="test")
            del feature_list_test

            hog1 = vigra.readHDF5(features[8], "feat").reshape(raw.shape)
            hog1 = hog1 - numpy.min(hog1)
            hog1 = hog1 / numpy.max(hog1)
            hog2 = vigra.readHDF5(features[11], "feat").reshape(raw.shape)
            hog2 = hog2 - numpy.min(hog2)
            hog2 = hog2 / numpy.max(hog2)
            hog4 = vigra.readHDF5(features[14], "feat").reshape(raw.shape)
            hog4 = hog4 - numpy.min(hog4)
            hog4 = hog4 / numpy.max(hog4)
            thr1 = numpy.zeros(hog1.shape)
            thr1[numpy.where(hog1 > 0.4)] = 1
            thr2 = numpy.zeros(hog2.shape)
            thr2[numpy.where(hog2 > 0.5)] = 1
            thr4 = numpy.zeros(hog4.shape)
            thr4[numpy.where(hog4 > 0.5)] = 1
            hys1 = vigra.filters.hysteresisThreshold(hog1, 0.4, 0.0100).astype(numpy.float32)
            hys2 = vigra.filters.hysteresisThreshold(hog2, 0.5, 0.0095).astype(numpy.float32)
            hys4 = vigra.filters.hysteresisThreshold(hog4, 0.5, 0.004).astype(numpy.float32)

            sl = numpy.index_exp[:, :, 10]
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4)
            ax1.imshow(raw[sl])
            ax2.imshow(hog1[sl])
            ax3.imshow(hog2[sl])
            ax4.imshow(hog4[sl])
            fig.delaxes(ax5)
            ax6.imshow(thr1[sl])
            ax7.imshow(thr2[sl])
            ax8.imshow(thr4[sl])
            fig.delaxes(ax9)
            ax10.imshow(hys1[sl])
            ax11.imshow(hys2[sl])
            ax12.imshow(hys4[sl])
            plt.show()


            raise sys.exit(0)



    else:  # No command line arguments are given.
        print "Please give the desired workflow as commandline argument."
        raise sys.exit(0)

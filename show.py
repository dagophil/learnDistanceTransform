import vigra
import numpy
import core
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the data
    print "Loading data."
    raw, gt, raw_test, gt_test = core.load_data()
    raw = raw.astype(numpy.float32)
    raw_test = raw_test.astype(numpy.float32)

    # Compute edge image.
    print "Computing edge images."
    edges = core.compute_edge_image(gt)
    edges_test = core.compute_edge_image(gt_test)

    # Compute the distance transform.
    print "Computing distance transform."
    dists = vigra.filters.distanceTransform3D(edges.astype(numpy.float32))
    dists_test = vigra.filters.distanceTransform3D(edges_test.astype(numpy.float32))

    # Show the results.
    z = 10
    fig, ((ax1, ax2, ax7, ax10), (ax3, ax4, ax8, ax11), (ax5, ax6, ax9, ax12)) = plt.subplots(3, 4)
    # fig, ((ax1, ax2, ax7), (ax3, ax4, ax8), (ax5, ax6, ax9)) = plt.subplots(3, 3)

    ax1.imshow(raw[:, :, z], interpolation="nearest")
    ax1.set_title("raw train")
    ax3.imshow(edges[:, :, z], interpolation="nearest")
    ax3.set_title("edges train")
    ax5.imshow(dists[:, :, z], interpolation="nearest")
    ax5.set_title("dists train")

    ax2.imshow(raw_test[:, :, z], interpolation="nearest")
    ax2.set_title("raw test")
    # pred_edges = vigra.readHDF5("pred_edges.h5", "pred")
    # ax4.imshow(pred_edges[:, :, z], interpolation="nearest")
    # del pred_edges
    ax4.set_title("edges test PRED")
    # pred_dists = vigra.readHDF5("pred_dists.h5", "pred")
    # ax6.imshow(pred_dists[:, :, z], interpolation="nearest")
    # del pred_dists
    ax6.set_title("dists test PRED")

    fig.delaxes(ax7)

    ax8.imshow(edges_test[:, :, z], interpolation="nearest")
    ax8.set_title("edges test")
    ax9.imshow(dists_test[:, :, z], interpolation="nearest")
    ax9.set_title("dists test")

    pred_dists_01 = vigra.readHDF5("pred_dists_01.h5", "pred")
    ax10.imshow(pred_dists_01[:, :, z], interpolation="nearest")
    del pred_dists_01
    ax10.set_title("dists test PRED 0.1")
    # pred_dists_005 = vigra.readHDF5("pred_dists_005.h5", "pred")
    # ax11.imshow(pred_dists_005[:, :, z], interpolation="nearest")
    # del pred_dists_005
    ax11.set_title("dists test PRED 0.05")
    # pred_dists_002 = vigra.readHDF5("pred_dists_002.h5", "pred")
    # ax12.imshow(pred_dists_002[:, :, z], interpolation="nearest")
    # del pred_dists_002
    ax12.set_title("dists test PRED 0.02")

    plt.show()

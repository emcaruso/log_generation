from volume_generator import VolumeGenerator
import cv2
import pyvista as pv
import numpy as np
from visualizer import Visualizer
import time


def main():

    resolution = 500
    n_slices = 500
    show = True

    # volume generator
    vol_gen = VolumeGenerator(
        resolution=resolution,
        n_slices=n_slices,
        show=show,
        random=True,
    )

    print("Generating volume...")

    # generate random volume
    t1 = time.time()
    volume = vol_gen.generate_volume()

    print("Volume generated in {:.2f} seconds".format(time.time() - t1))

    # visualize the log
    try:
        vis = Visualizer(volume)
        vis.show()
    except Exception as e:
        print("Visualization failed: ", e)

    # # show slices
    # for i in range(n_slices):
    #     cv2.imshow("slice", volume[i])
    #     key = cv2.waitKey(0)
    #     if key == ord("l"):
    #         break
    #     if key == ord("q"):
    #         exit(1)
    #
    #


if __name__ == "__main__":
    main()

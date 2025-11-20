from volume_generator import VolumeGenerator
import cv2
import pyvista as pv
import numpy as np
from visualizer import Visualizer


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
    volume = vol_gen.generate_volume()

    # visualize the log
    if False
        vol = volume[:1000, ...]
        vis = Visualizer(vol)
        vis.show()

    # show slices
    if False:
        cv2.imshow("slice", volume[i])
        key = cv2.waitKey(0)
        if key == ord("l"):
            break
        if key == ord("q"):
            exit(1)


if __name__ == "__main__":
    main()

from volume_generator import VolumeGenerator
import cv2
import pyvista as pv
import numpy as np
from visualizer import Visualizer
import time
from omegaconf import OmegaConf
from pathlib import Path


def main():

    # load yaml with omegaconf
    path = Path(__file__).parent.parent / "configs" / "log_generation.yaml"
    cfg = OmegaConf.load(path)

    # volume generator
    vol_gen = VolumeGenerator(cfg=cfg)

    print("Generating volume...")

    # generate random volume
    t1 = time.time()
    volume = vol_gen.generate_volume()

    print("Volume generated in {:.2f} seconds".format(time.time() - t1))

    # # visualize the log
    # try:
    #     vis = Visualizer(volume)
    #     vis.show()
    # except Exception as e:
    #     print("Visualization failed: ", e)

    # show slices
    for i in range(volume.shape[0]):
        cv2.imshow("slice", volume[i])
        key = cv2.waitKey(0)
        if key == ord("l"):
            break
        if key == ord("q"):
            exit(1)


if __name__ == "__main__":
    main()

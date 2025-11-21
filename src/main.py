from volume_generator import VolumeGenerator
import cv2
import time
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from tiff import volume_to_tiff


def main():

    # load yaml with omegaconf
    path = Path(__file__).parent.parent / "configs" / "log_generation.yaml"
    cfg = OmegaConf.load(path)

    # volume generator
    vol_gen = VolumeGenerator(cfg=cfg)

    # set seed of numpy
    np.random.seed(cfg.seed)

    print("Generating volume...")

    # generate random volume
    t1 = time.time()
    volume = vol_gen.generate_volume()

    print("Volume generated in {:.2f} seconds".format(time.time() - t1))

    volume_to_tiff(
        volume,
        str(Path(__file__).parent.parent / "data" / "tiffs" / "generated_volume.tiff"),
    )

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

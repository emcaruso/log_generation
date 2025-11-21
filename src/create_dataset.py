from volume_generator import VolumeGenerator
import cv2
import time
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from tiff import volume_to_tiff
from tqdm import tqdm
import os


def main():

    n_volumes = 500
    dir = "/mnt/NAS20/quality/microtec/static_CT/synthetic/"
    os.makedirs(dir, exist_ok=True)

    # load yaml with omegaconf
    path = Path(__file__).parent.parent / "configs" / "log_generation.yaml"
    cfg = OmegaConf.load(path)

    # volume generator
    vol_gen = VolumeGenerator(cfg=cfg)

    # set seed of numpy
    np.random.seed(cfg.seed)

    progress_bar = tqdm(range(n_volumes), desc="Generating volumes")

    # generate random volume
    for i in tqdm(progress_bar):
        volume = vol_gen.generate_volume()
        volume_to_tiff(volume, dir + f"{i:04d}.tiff")


if __name__ == "__main__":
    main()

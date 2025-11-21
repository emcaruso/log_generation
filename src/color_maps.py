import numpy as np
from PIL import Image
import cv2
from typing import Tuple
from omegaconf import DictConfig


class ColorMaps:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def generate_color_map(self):

        background = np.random.uniform(
            self.cfg.background_range[0], self.cfg.background_range[1]
        )
        sigmoid_stepness = np.random.uniform(
            self.cfg.sigmoid_steepness_range[0], self.cfg.sigmoid_steepness_range[1]
        )
        humidity_pos = np.random.uniform(
            self.cfg.humidity_pos_range[0], self.cfg.humidity_pos_range[1]
        )

        sigmoid_minimum = np.random.uniform(
            self.cfg.sigmoid_minimum_range[0], self.cfg.sigmoid_minimum_range[1]
        )

        image_log = np.zeros((self.cfg.first_res, 1), dtype="float32")
        end = np.random.uniform(self.cfg.end_range[0], self.cfg.end_range[1])
        ring_step = np.random.uniform(
            self.cfg.ring_step_range[0], self.cfg.ring_step_range[1]
        )
        # image_log *= background

        # inject rings
        x = 0
        while True:

            # sample from gaussian
            step = np.random.normal(ring_step, self.cfg.ring_step_variability)
            step = (
                np.random.uniform(
                    self.cfg.ring_step_range[0], self.cfg.ring_step_range[1]
                )
                / end
            )

            x += step

            if x >= self.cfg.first_res:
                break

            # gaussian centered on x
            std = (
                np.random.uniform(
                    self.cfg.ring_std_range[0], self.cfg.ring_std_range[1]
                )
                / end
            )
            pick = np.random.uniform(
                self.cfg.ring_pick_range[0], self.cfg.ring_pick_range[1]
            )
            gauss = pick * np.exp(
                -0.5 * ((np.arange(self.cfg.first_res) - x) / std) ** 2
            )
            image_log[:, 0] += gauss

        # add background
        image_log += background

        # apply gamma to image_log according to sigmoid
        xs = (
            np.arange(self.cfg.first_res).reshape(self.cfg.first_res, 1)
            / self.cfg.first_res
        )
        sigm = 1 - 1 / (1 + np.exp(-sigmoid_stepness * (xs - humidity_pos)))
        sigm = sigm * (1 - sigmoid_minimum) + sigmoid_minimum
        image_log = np.power(image_log, sigm)

        # resize image log according to end
        # show
        image_log = cv2.resize(
            image_log,
            (1, int(self.cfg.first_res * end)),
            dst=image_log,
            interpolation=cv2.INTER_LINEAR,
        )

        image = np.zeros((self.cfg.first_res, 1), dtype="float32")
        image[: image_log.shape[0], 0] = image_log[:, 0]

        # resize image to self.cfg.height
        image = cv2.resize(
            image,
            (1, self.cfg.width),
            dst=image,
            interpolation=cv2.INTER_LINEAR,
        ).T
        image = np.repeat(image, self.cfg.height, 0)
        # from L to RGB
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image, mode="L").convert("RGB")

        return image, end

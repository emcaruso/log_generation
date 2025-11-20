import numpy as np
from PIL import Image
import cv2
from typing import Tuple


class ColorMaps:

    def __init__(
        self,
        # background_range: Tuple[float, float] = (0.15, 0.25),
        background_range: Tuple[float, float] = (0.25, 0.35),
        ring_pick_range: Tuple[float, float] = (0.02, 0.09),
        ring_std_range: Tuple[float, float] = (1.0, 2.0),
        sigmoid_steepness_range: Tuple[float, float] = (100.0, 200.0),
        sigmoid_minimum_range: Tuple[float, float] = (0.3, 0.4),
        # sigmoid_minimum_range: Tuple[float, float] = (0.7, 0.71),
        # humidity_pos_range: Tuple[float, float] = (0.6, 0.8),
        humidity_pos_range: Tuple[float, float] = (1.6, 1.8),
        end_range: Tuple[float, float] = [0.4, 0.6],
        # end_range: Tuple[float, float] = [0.0199, 0.02],
        ring_step_range: Tuple[float, float] = (10, 20),
        first_res: int = 500,
        width: int = 1504,
        height: int = 56,
    ):

        self.background_range = background_range
        self.ring_pick_range = ring_pick_range
        self.ring_std_range = ring_std_range
        self.sigmoid_steepness_range = sigmoid_steepness_range
        self.sigmoid_minimum_range = sigmoid_minimum_range
        self.humidity_pos_range = humidity_pos_range
        self.end_range = end_range
        self.ring_step_range = ring_step_range
        self.first_res = first_res
        self.width = width
        self.height = height

    def generate_color_map(self):

        background = np.random.uniform(
            self.background_range[0], self.background_range[1]
        )
        sigmoid_stepness = np.random.uniform(
            self.sigmoid_steepness_range[0], self.sigmoid_steepness_range[1]
        )
        humidity_pos = np.random.uniform(
            self.humidity_pos_range[0], self.humidity_pos_range[1]
        )

        sigmoid_minimum = np.random.uniform(
            self.sigmoid_minimum_range[0], self.sigmoid_minimum_range[1]
        )

        image_log = np.zeros((self.first_res, 1), dtype="float32")
        end = np.random.uniform(self.end_range[0], self.end_range[1])
        # image_log *= background

        # inject rings
        x = 0
        while True:

            step = np.random.uniform(self.ring_step_range[0], self.ring_step_range[1])

            x += step

            if x >= self.first_res:
                break

            # gaussian centered on x
            std = np.random.uniform(self.ring_std_range[0], self.ring_std_range[1])
            pick = np.random.uniform(self.ring_pick_range[0], self.ring_pick_range[1])
            gauss = pick * np.exp(-0.5 * ((np.arange(self.first_res) - x) / std) ** 2)
            image_log[:, 0] += gauss

        # add background
        image_log += background

        # apply gamma to image_log according to sigmoid
        xs = np.arange(self.first_res).reshape(self.first_res, 1) / self.first_res
        sigm = 1 - 1 / (1 + np.exp(-sigmoid_stepness * (xs - humidity_pos)))
        sigm = sigm * (1 - sigmoid_minimum) + sigmoid_minimum
        image_log = np.power(image_log, sigm)

        # resize image log according to end
        # show
        image_log = cv2.resize(
            image_log,
            (1, int(self.first_res * end)),
            dst=image_log,
            interpolation=cv2.INTER_LINEAR,
        )

        image = np.zeros((self.first_res, 1), dtype="float32")
        image[: image_log.shape[0], 0] = image_log[:, 0]

        # resize image to self.height
        image = cv2.resize(
            image,
            (1, self.width),
            dst=image,
            interpolation=cv2.INTER_LINEAR,
        ).T
        image = np.repeat(image, self.height, 0)
        # from L to RGB
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image, mode="L").convert("RGB")

        return image, end

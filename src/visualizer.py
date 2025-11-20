import numpy as np
from vedo import Volume, Plotter
from vedo import show as show_volume
import vedo
from vedo.applications import RayCastPlotter, IsosurfaceBrowser


class Visualizer:

    def __init__(self, volume):
        if volume.dtype == np.uint8:
            vol = volume.astype(np.float32) / 255.0
        else:
            vol = volume.mean(axis=-1)

        if len(vol.shape) == 4:
            vol = vol.mean(axis=-1)

        self.plt = RayCastPlotter(Volume(vol), bg="white", bg2="blackboard", axes=7)
        # self.plt = IsosurfaceBrowser(Volume(vol), use_gpu=True, c="gold")

    def save(self, path):
        self.plt.screenshot(path)

    def show(self):
        self.plt.show()
        # self.plotter.show()


if __name__ == "__main__":
    import tifffile

    path = "/home/emcarus/Desktop/log_generation/gibboni/tronky/W44476880.tiff"
    with tifffile.TiffFile(path) as tif:
        # extract metadata for tiff file (tags)
        page = tif.pages[0]
        tags = list(page.tags)
        tags_dict = {tag.name: tag.value for tag in tags}

        # read info for number of images to apply correct reshaping of the data
        page_number = tags_dict["PageNumber"]

        # convert and reshape data
        image = page.asarray()
        image = image.reshape(page_number[1], -1, image.shape[1])

        Visualizer(image).show()

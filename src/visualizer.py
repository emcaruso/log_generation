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
            vol = volume

        self.plt = RayCastPlotter(Volume(vol), bg="white", bg2="blackboard", axes=7)
        # self.plt = IsosurfaceBrowser(Volume(vol), use_gpu=True, c="gold")
        # self.vol = Volume(vol)
        # self.plotter = Plotter()
        # self.plotter.background("white")
        # self.plotter += self.vol
        #
        # self.plotter.add_slider(
        #     self.update_opacity,
        #     xmin=0.0,
        #     xmax=1.0,
        #     value=0.5,
        #     pos=(0.05, 0.9),
        #     title="Opacity",
        # )

        # # Create a slider for the cut plane position
        # cut_plane_slider = vedo.widgets.Slider2D(
        #     pos=(0.05, 0.05),  # Position of the cut plane slider
        #     length=0.8,  # Length of the slider
        #     value=0.5,  # Initial position (normalized between 0 and 1)
        #     min_value=0.0,  # Min value (cut on one side of the volume)
        #     max_value=1.0,  # Max value (cut on the other side)
        #     action=self.update_cut_plane,  # Function to update the cut plane position
        #     title="Cut Plane",
        #     font_size=15,
        # )
        # self.plotter += cut_plane_slider
        #

    def save(self, path):
        self.plt.screenshot(path)

    def show(self):
        self.plt.show()
        # self.plotter.show()

    # # Interactive slider for opacity
    # def update_opacity(self, val):
    #     """Callback to adjust the opacity based on slider value."""
    #     self.vol.alpha(val)  # Update alpha value of the volume (opacity)
    #     self.plotter.render()
    #
    # def update_cut_plane(self, x_val):
    #     """Callback to adjust the position of the cut plane."""
    #     # Move the cut plane along the x-axis
    #     self.vol.cut_plane(x_pos=x_val)
    #     self.plotter.render()


if __name__ == "__main__":
    import tifffile

    path = "/home/emcarus/Desktop/procedural_knots/gibboni/tronky/W44476880.tiff"
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

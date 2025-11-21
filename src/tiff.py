import tifffile
import numpy as np


def volume_to_tiff(volume: np.ndarray, path: str):
    volume = volume.mean(axis=-1)

    length = len(volume)
    volume = np.vstack(volume)
    log_tags = [
        (
            tifffile.TIFF.TAGS["DocumentName"],
            tifffile.DATATYPE.ASCII,
            15,
            "MiCROTEC3DFAST",
            True,
        ),
        (
            tifffile.TIFF.TAGS["ImageWidth"],
            tifffile.DATATYPE.SHORT,
            1,
            volume.shape[1],
            True,
        ),
        (
            tifffile.TIFF.TAGS["ImageLength"],
            tifffile.DATATYPE.LONG,
            1,
            volume.shape[0],
            True,
        ),
        (tifffile.TIFF.TAGS["SamplesPerPixel"], tifffile.DATATYPE.SHORT, 1, 1, True),
        (
            tifffile.TIFF.TAGS["PageNumber"],
            tifffile.DATATYPE.SHORT,
            2,
            (volume.shape[1], length),
            True,
        ),
        (
            tifffile.TIFF.TAGS["PhotometricInterpretation"],
            tifffile.DATATYPE.SHORT,
            15,
            tifffile.PHOTOMETRIC.MINISBLACK,
            True,
        ),
    ]
    with tifffile.TiffWriter(path) as tiff_writer:
        tiff_writer.write(volume, compression="lzw", extratags=log_tags)

#!/usr/bin/env python3

"""

"""


import os
import sys
import h5py as h5
import numpy 
import numpy.typing
import argparse
from typing import Any, List, Dict, Union, Tuple, TextIO, cast
from om.utils import crystfel_geometry
from numpy.typing import NDArray
import fabio


os.nice(0)


class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)
    parser.add_argument('-i', '--i', type=str, help="Input input_file")
    parser.add_argument('-h5p', '--h5p', type=str, help="hdf5 path for the cxi input_file data")
    parser.add_argument('-o', '--o',type=str, help="Output path")
    parser.add_argument('-g', '--g',type=str, help="Geometry input_file")
    parser.add_argument('-n', '--n', type=int, help="Number of patterns that you are going to sum every time")
    return parser.parse_args()



def load_geometry(geometry_lines: List[str]) -> None:
    # Loads CrystFEL goemetry using om.utils module.
    parameter_geometry: crystfel_geometry.TypeDetector
    beam: crystfel_geometry.TypeBeam
    parameter_geometry, beam, __ = crystfel_geometry.read_crystfel_geometry(
        text_lines=geometry_lines
    )
    parameter_pixelmaps: crystfel_geometry.TypePixelMaps = (
        crystfel_geometry.compute_pix_maps(geometry=parameter_geometry)
    )

    first_panel: str = list(parameter_geometry["panels"].keys())[0]
    parameter_pixel_size: float = parameter_geometry["panels"][first_panel]["res"]
    parameter_clen_from: str = parameter_geometry["panels"][first_panel]["clen_from"]
    if parameter_clen_from == "":
        parameter_clen: float = parameter_geometry["panels"][first_panel]["clen"]
    parameter_coffset: float = parameter_geometry["panels"][first_panel]["coffset"]
    parameter_photon_energy_from: str = beam["photon_energy_from"]
    if parameter_photon_energy_from == "":
        parameter_photon_energy: float = beam["photon_energy"]
    parameter_mask_filename = parameter_geometry["panels"][first_panel]["mask_file"]
    parameter_mask_hdf5_path = parameter_geometry["panels"][first_panel]["mask"]

    y_minimum: int = (
        2
        * int(max(abs(parameter_pixelmaps["y"].max()), abs(parameter_pixelmaps["y"].min())))
        + 2
    )
    x_minimum: int = (
        2
        * int(max(abs(parameter_pixelmaps["x"].max()), abs(parameter_pixelmaps["x"].min())))
        + 2
    )
    parameter_data_shape: Tuple[int, ...] = parameter_pixelmaps["x"].shape
    parameter_visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    parameter_img_center_x: int = int(parameter_visual_img_shape[1] / 2)
    parameter_img_center_y: int = int(parameter_visual_img_shape[0] / 2)
    parameter_visual_pixelmap_x: NDArray[numpy.int32] = cast(
        NDArray[numpy.int32],
        numpy.array(parameter_pixelmaps["x"], dtype=numpy.int32)
        + parameter_visual_img_shape[1] // 2
        - 1,
    )
    parameter_visual_pixelmap_y: NDArray[numpy.int32] = cast(
        NDArray[numpy.int32],
        numpy.array(parameter_pixelmaps["y"], dtype=numpy.int32)
        + parameter_visual_img_shape[0] // 2
        - 1,
    )
    
    
    return parameter_visual_pixelmap_x, parameter_visual_pixelmap_y, parameter_visual_img_shape


def get_header():
    pass

if __name__ == "__main__":

    args = parse_cmdline_args()
    input_file = args.i
    h5path = args.h5p
    output_folder = args.o
    geometry_input_file = args.g
    chunk_size = args.n
    
    images = h5.File(input_file, 'r')[h5path]
    
    
    geometry_lines = open(geometry_input_file, 'r').readlines()
    parameter_visual_pixelmap_x, parameter_visual_pixelmap_y, parameter_visual_img_shape = load_geometry(geometry_lines)
    frame_data_img: NDArray[Any] = numpy.zeros(parameter_visual_img_shape)
    


    f_idx = 0
    for idx in range(0, images.shape[0], chunk_size):
        frame_data_img[parameter_visual_pixelmap_y.ravel(), parameter_visual_pixelmap_x.ravel()] = numpy.sum(images[idx:idx+chunk_size,], axis=0).ravel()
        prefix = os.path.basename(input_file).split('.')[0]
        tif = fabio.tifimage.tifimage(data=frame_data_img[()])
        fname = os.path.join(output_folder, f"{prefix}_%06i.tif"%f_idx)
        f_idx += 1
        tif.write(fname)
        print(fname)


import os, argparse
import subprocess, logging, ray
from hichip_object.load_loop_data import process_peak_to_anchor_bins
from hichip_object.glm_loop_model import Loop_ZIP

# command line args
parser = argparse.ArgumentParser(
    description="Call significant interactions from HiChIP data"
)
parser.add_argument("peak_file", type=str, help="path to the peaks file")
parser.add_argument("validpairs", type=str, help="path to validpairs file")
parser.add_argument(
    "chro_size", type=str, help="chromosome size file of the genome"
)
parser.add_argument(
    "--resolution",
    type=int,
    help="resolution of anchors, default 2500 bp",
    default=2500,
)

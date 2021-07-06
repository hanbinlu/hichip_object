import os, subprocess, logging
import pandas as pd
from mvp import dump_PETs_to_bed
from load_loop_data import (
    genomic_bins,
    genomic_anchor_bins,
    merge_anchors_bins,
    merge_anchors,
    extend_anchors,
)


def call_anchors_from_hichip(
    bam,
    digestion_sites,
    prefix,
    exclude_interchro=True,
    local_range=2000000,
    macs2_path="/home/coco/miniconda3/envs/hichip-loop/bin/macs2",
    macs2_qval=0.01,
    macs2_genome="mm",
    mapq=10,
    nproc=30,
    parallel=15,
):
    """
    Call peaks from HiChIP mapped data in BAM format.
    """
    macs2_result_dir = f"{prefix}_MACS2_results/"
    try:
        os.mkdir(macs2_result_dir)
    except FileExistsError:
        subprocess.run(["rm", "-r", macs2_result_dir])
        os.mkdir(macs2_result_dir)
    # implemented control for macs2_qval, local_range, exclude inter chromosomal PETs
    bed_reads = os.path.join(macs2_result_dir, f"{prefix}.PETs.bed")
    if type(bam) == list:
        temp_bed = []
        for i, f in enumerate(bam):
            temp_bed.append(f"{bed_reads}.{i}")
            dump_PETs_to_bed(
                f,
                mapq,
                digestion_sites,
                temp_bed[-1],
                nproc,
                parallel,
                exclude_interchro,
                local_range,
            )

        # merge
        with open(bed_reads, "w") as o:
            subprocess.run(["cat", *temp_bed], stdout=o)

        subprocess.run(["rm", *temp_bed])
        tot = subprocess.run(["wc", "-l", bed_reads], stdout=subprocess.PIPE)
        logging.info(
            f"{tot.stdout.decode().split()[0]} reads input for peak calling"
        )

    else:
        dump_PETs_to_bed(
            bam,
            mapq,
            digestion_sites,
            bed_reads,
            nproc,
            parallel,
            exclude_interchro,
            local_range,
        )

    # call peaks
    logging.info("Calling peaks by MACS2")
    call_peaks = subprocess.run(
        [
            macs2_path,
            "callpeak",
            "-t",
            bed_reads,
            "--keep-dup",
            "all",
            "-q",
            str(macs2_qval),
            "--extsize",
            "147",
            "--nomodel",
            "-g",
            macs2_genome,
            "-B",
            "-f",
            "BED",
            "--verbose",
            "1",
            "-n",
            os.path.join(macs2_result_dir, prefix),
        ],
    )

    peak_file = os.path.join(macs2_result_dir, f"{prefix}_peaks.narrowPeak")
    count_peaks = subprocess.run(
        [
            "wc",
            "-l",
            peak_file,
        ],
        stdout=subprocess.PIPE,
    )
    logging.info(
        f"MACS2 called {count_peaks.stdout.decode().split()[0]} peaks at q < {macs2_qval}"
    )
    return peak_file


def process_peak_to_anchor_bins(
    peak_file,
    chro_size,
    format="macs2_narrow",
    resolution=2500,
    # merge_adjacent=True,
):
    """
    Extend peak and merge into non-overlapping anchors
    """
    gbs = genomic_bins(chro_size, resolution)
    if format == "macs2_narrow":
        peaks = pd.read_csv(
            peak_file,
            sep="\t",
            skiprows=1,
            header=None,
        ).iloc[:, 0:3]
        peaks.columns = ["chro", "start", "end"]
        # peak_anchor_bins, _ = genomic_anchor_bins(gbs, peaks, merge_adjacent)
        gbs["Num_Anchors"] = genomic_anchor_bins(
            gbs, peaks, merge_adjacent=False
        )[1]
        # add up anchor number too
        gbs_merged = merge_anchors_bins(gbs)
    elif format == "homer_peaks":
        peaks = pd.read_csv(
            peak_file,
            sep="\t",
            skiprows=39,
        ).iloc[:, 1:4]
        peaks.columns = ["chro", "start", "end"]
        # peak_anchor_bins, _ = genomic_anchor_bins(gbs, peaks, merge_adjacent)
        gbs["Num_Anchors"] = genomic_anchor_bins(
            gbs, peaks, merge_adjacent=False
        )[1]
        # add up anchor number too
        gbs_merged = merge_anchors_bins(gbs)
    else:
        raise ValueError("Peak file format unknown")

    return gbs_merged


def process_peak_to_anchors_centered(
    peak_file,
    format="macs2_narrow",
    extend_peak_size=2500,
    merge_peak_gap=1000,
):
    """
    Extend peak and merge into non-overlapping anchors
    """
    if format == "macs2_narrow":
        extensive_peaks = pd.read_csv(
            peak_file,
            sep="\t",
            skiprows=1,
            header=None,
        ).iloc[:, 0:3]
        extensive_peaks.columns = ["chro", "start", "end"]
        extensive_anchors = merge_anchors(
            extend_anchors(extensive_peaks, extend_peak_size), merge_peak_gap
        )
    elif format == "homer_peaks":
        extensive_peaks = pd.read_csv(
            peak_file,
            sep="\t",
            skiprows=39,
            header=None,
        ).iloc[:, 1:4]
        extensive_peaks.columns = ["chro", "start", "end"]
        extensive_anchors = merge_anchors(
            extend_anchors(extensive_peaks, extend_peak_size), merge_peak_gap
        )
    else:
        raise ValueError("Peak file format unknown")

    return extensive_anchors

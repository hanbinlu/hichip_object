import os, ray
import subprocess, logging
from multipass_process.mvp import dump_PETs_to_bed, genome_digestion


def call_anchors_from_hichip(
    bam,
    digestion_site,
    genome_fa,
    prefix,
    result_path=None,
    exclude_interchro=True,
    local_range=2000000,
    macs2_path="/home/coco/miniconda3/envs/hichip-loop/bin/macs2",
    macs2_qval=0.01,
    macs2_genome="mm",
    mapq=10,
    nproc=30,
    procs_per_pysam=2,
):
    """
    Call peaks from HiChIP mapped data in BAM format.
    """
    if result_path:
        macs2_result_dir = os.path.join(result_path, f"{prefix}_MACS2_results/")
    else:
        # current folder
        macs2_result_dir = f"{prefix}_MACS2_results/"

    try:
        os.mkdir(macs2_result_dir)
    except FileExistsError:
        subprocess.run(["rm", "-r", macs2_result_dir])
        os.mkdir(macs2_result_dir)

    digestion_sites = genome_digestion(genome_fa, digestion_site)
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
                procs_per_pysam,
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
            procs_per_pysam,
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

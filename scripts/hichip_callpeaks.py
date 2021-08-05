import os, argparse, ray
import subprocess, logging
from multipass_process.mvp import dump_PETs_to_bed, genome_digestion

# logging to the console
logging.basicConfig(level=logging.INFO, formatter="%(asctime)s: %(message)s")

# command line args
parser = argparse.ArgumentParser(description="Call peaks from HiChIP data")
parser.add_argument("bam", type=str, nargs="+", help="Paired bam files")
parser.add_argument("prefix", type=str, help="prefix for the result")
parser.add_argument("genome_fa", type=str, help="genome sequence file")
parser.add_argument(
    "--result_path",
    type=str,
    help="directory for call peak result, default cwd",
    default="./",
)
parser.add_argument(
    "--digestion_site",
    type=str,
    help="digestion site used for the Hi-C experiment, default MseI + DdeI",
    default="(CT[ATCG]AG)|(TTAA)",
)
parser.add_argument(
    "--num_cpus",
    type=int,
    help="number of cpu cores to use, default 8",
    default=8,
)

# control for PETs to keep
parser.add_argument(
    "--exclude_interchro",
    type=bool,
    help="whether to discard interchro PETs for peak calling, default True",
    default=True,
)
parser.add_argument(
    "--local_range",
    type=int,
    help="longest separated PETs to keep, default 1000",
    default=1000,
)
parser.add_argument(
    "--mapq", type=int, help="MAPQ filter, default 10", default=10
)

# macs2 parameters
parser.add_argument(
    "--macs2_path",
    type=str,
    help="path to macs2 program, default to look from env path",
    default="macs2",
)
parser.add_argument(
    "--macs2_genome",
    type=str,
    help="genome id for macs2, default mm",
    default="mm",
)
parser.add_argument(
    "--macs2_qval",
    type=float,
    help="q value filter for macs2 calling peaks, default 0.01",
    default=0.01,
)
args = parser.parse_args()


def call_anchors_from_hichip(
    bam,
    digestion_sites,
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


digested_frags = genome_digestion(args.genome_fa, args.digestion_site)
ray.init(num_cpus=args.num_cpus)
call_anchors_from_hichip(
    args.bam,
    digested_frags,
    args.prefix,
    args.result_path,
    args.exclude_interchro,
    args.local_range,
    args.macs2_path,
    args.macs2_qval,
    args.macs2_genome,
    args.mapq,
    args.num_cpus,
)

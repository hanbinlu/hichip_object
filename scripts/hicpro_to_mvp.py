import argparse, subprocess, logging
from multipass_process.mpmap import multipass_mapping_from_hicpro
from multipass_process.mvp import genome_digestion, construct_mpp_validpair

# logging to the console
logging.getLogger().setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s: %(message)s")
logger = logging.getLogger("hicpro_to_mvp")
hdl = logging.StreamHandler()
hdl.setLevel(logging.INFO)
hdl.setFormatter(formatter)
logger.addHandler(hdl)

# command line args
parser = argparse.ArgumentParser(
    description="Multipass mapping from HiC-Pro results and process to multipass-processed validpairs (MVP)"
)
parser.add_argument("hicpro_results", type=str, help="HiC-Pro output directory")
parser.add_argument(
    "project_name", type=str, help="project/sample name of HiC-Pro output"
)
parser.add_argument("genome_index", type=str, help="bowtie2 genome index")
parser.add_argument("genome_fa", type=str, help="genome sequence file")
parser.add_argument(
    "--bowtie2_path",
    type=str,
    help="bowtie2 program path, default to find in env PATH",
    default="bowtie2",
)
parser.add_argument(
    "--samtools_path",
    type=str,
    help="samtools program path, default to find in env PATH",
    default="samtools",
)
parser.add_argument(
    "--num_cpus",
    type=int,
    help="number of cpu cores to use, default 8",
    default=8,
)
parser.add_argument(
    "--ligation_site",
    type=bytes,
    help="ligation partern used for the Hi-C experiment, default MseI + DdeI",
    default=b"(CT[ATCG]AT[ATCG]AG)|(CT[ATCG]ATAA)|(TTAT[ATCG]AG)|(TTATAA)",
)
parser.add_argument(
    "--digestion_site",
    type=str,
    help="digestion site used for the Hi-C experiment, default MseI + DdeI",
    default="(CT[ATCG]AG)|(TTAA)",
)
parser.add_argument(
    "--mapq", type=int, help="MAPQ filter, default 10", default=10
)
args = parser.parse_args()

# Processing data
logger.info("@multipass mapping: start")
multipass_mapped_bam = multipass_mapping_from_hicpro(
    args.hicpro_results,
    args.project_name,
    args.ligation_site,
    args.genome_index,
    args.num_cpus,
    args.bowtie2_path,
    args.samtools_path,
)
logger.info("@multipass mapping: finished")
logger.info("@parse to MVP: start")
# digestion fragments
digested_frags = genome_digestion(args.genome_fa, args.digestion_site)
# process to MVP file
subprocess.run(f"mkdir {args.hicpro_results}/mvp_results/", shell=True)
construct_mpp_validpair(
    multipass_mapped_bam,
    args.mapq,
    digested_frags,
    f"{args.hicpro_results}/mvp_results/{args.project_name}.mppValidPairs",
    args.num_cpus,
)
logger.info("@parse to MVP: finished")

import os, gzip, mmap
import pysam, subprocess, logging
import re2 as re

# %%
def multipass_mapping_from_hicpro(
    hicpro_results,
    project_name,
    ligation_site,
    genome_index,
    nthd,
    bowtie2_path,
    samtools_path,
):
    """
    Map all the segments of a read splitted by `ligation_site`.

    Description
    -----------
    The first pass mapping (global + local) has done by HiC-Pro. During the local mapping, unmapped reads from global mapping
    were splited by `ligation_site`. 5' parts were subject to "local" mapping. New pass of global and local mapping is done
    for the remanant 3' splited parts. Iterate the process until no 3' splitted parts are left for another pass of mapping.

    Parameters
    ----------
    hicpro_results: The hicpro output dir. The function picks up from the first pass mapping and add result to the same dir for better output arrangement.
    project_name: This is also from the step of hicpro. hicpro input structure: raw_fastq_dir/project_name/fqs
    ligation_site: raw regexp string of ligation pattern to split the reads
    genome_index: bowtie index
    nthd: number of total cores to use
    """

    result_dir = f"{hicpro_results}bowtie_results/bwt2_multipass/{project_name}"
    try:
        os.makedirs(result_dir)
    except OSError as error:
        print(error)

    logger = logging.getLogger("mpmap")
    fh = logging.FileHandler(f"{result_dir}/mpmap.log")
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    bwt_opts = "--very-sensitive -L 30 --score-min L,-0.6,-0.2"
    prefix, _ = solve_pair_end_samples(hicpro_results, project_name)
    intermediates = {
        pfx: {"bams": [], "5primeFq": [], "3primeFq": [], "ToMap": -1}
        for pfx in prefix
    }

    # Pass 1 result from hicpro. However, add "P1RxG/L" group tag to the bam files
    logger.info("@Pass 1: processing HiC-Pro results")
    ipass = 0
    global_dir = f"{hicpro_results}bowtie_results/bwt2_global/{project_name}/"
    local_dir = f"{hicpro_results}bowtie_results/bwt2_local/{project_name}/"
    for pfx in prefix:
        # currently only recognize of R1 and R2 naming pattern
        ri = paired_side(pfx)
        # global
        tag = f"P0{ri}G"
        inp = os.path.join(global_dir, f"{pfx}.bwt2glob.bam")
        out = os.path.join(result_dir, f"{pfx}.{tag}.bam")
        unmap_pfx = os.path.join(result_dir, f"{pfx}.{tag}")
        # add tag to hicpro mapped bam file
        pysam.addreplacerg("-r", f"ID:{tag}", "-@", f"{nthd}", "-o", out, inp)
        # split by ligation site. 5' part for local mapping; 3' part for new pass
        leftover = split_fastq_by_motif(
            os.path.join(global_dir, f"{pfx}.bwt2glob.unmap.fastq"),
            ligation_site,
            unmap_pfx,
        )
        intermediates[pfx]["bams"].append(out)
        intermediates[pfx]["5primeFq"].append(unmap_pfx + ".5prime.fastq")
        intermediates[pfx]["3primeFq"].append(unmap_pfx + ".3prime.fastq")
        intermediates[pfx]["ToMap"] = leftover

        # local
        tag = f"P0{ri}L"
        inp = os.path.join(local_dir, f"{pfx}.bwt2glob.unmap_bwt2loc.bam")
        out = os.path.join(result_dir, f"{pfx}.{tag}.bam")
        pysam.addreplacerg("-r", f"ID:{tag}", "-@", f"{nthd}", "-o", out, inp)
        intermediates[pfx]["bams"].append(out)

        logger.info(f"@Pass 1, {pfx}: {leftover} reads for next pass to map")
        logger.info(f"@Pass 1, {pfx}: Finished")

    logger.info(f"@Pass 1: Done")

    # multi pass mapping
    while any([intermediates[pfx]["ToMap"] != 0 for pfx in prefix]):
        ipass += 1
        logger.info(f"@Pass {ipass+1} mapping")

        for pfx in prefix:
            if intermediates[pfx]["ToMap"] == 0:
                continue

            ri = paired_side(pfx)

            # global
            logger.info(f"@Pass {ipass+1}, {pfx}, global mapping")
            tag = f"P{ipass}{ri}G"
            # 3' part for new pass of mapping
            inp = intermediates[pfx]["3primeFq"][-1]
            out = os.path.join(result_dir, f"{pfx}.{tag}.bam")
            with open(out, "w") as o:
                map_proc = subprocess.Popen(
                    [
                        bowtie2_path,
                        *bwt_opts.split(" "),
                        "--end-to-end",
                        "--reorder",
                        "--un",
                        "temp.fq",
                        "--rg-id",
                        tag,
                        "--rg",
                        f"SM:{pfx}",
                        "-p",
                        str(nthd),
                        "-x",
                        genome_index,
                        "-U",
                        inp,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                sam_filt = subprocess.Popen(
                    [samtools_path, "view", "-F", "4", "-bS", "-"],
                    stdin=map_proc.stdout,
                    stdout=o,
                )

                # mapping stats
                for line in iter(map_proc.stderr.readline, b""):
                    line = line.decode().rstrip()
                    if (
                        line.startswith("#")
                        or line.startswith("perl")
                        or line.startswith("\s")
                        or line.startswith("Warn")
                    ):
                        continue
                    else:
                        logger.info(f"@Pass {ipass+1}, {pfx}, {line}")

                sam_filt.wait()

            # logger.info(f"@Pass {ipass+1}, {pfx}, global mapping stats")
            # for line in map_proc.stderr.decode().split("\n"):
            #    logger.info(f"@Pass {ipass+1}, {pfx}, {line}")

            # local: split unmapped: 5' for local mapping of this pass; 3' for next pass
            if os.stat("temp.fq").st_size != 0:
                # split reads
                logger.info(f"@Pass {ipass+1}, {pfx}, local mapping")
                unmap_pfx = os.path.join(result_dir, f"{pfx}.{tag}")
                leftover = split_fastq_by_motif(
                    "temp.fq", ligation_site, unmap_pfx,
                )
                os.remove("temp.fq")
                intermediates[pfx]["bams"].append(out)
                intermediates[pfx]["5primeFq"].append(
                    unmap_pfx + ".5prime.fastq"
                )
                intermediates[pfx]["3primeFq"].append(
                    unmap_pfx + ".3prime.fastq"
                )
                intermediates[pfx]["ToMap"] = leftover

                # local mapping
                tag = f"P{ipass}{ri}L"
                inp = intermediates[pfx]["5primeFq"][-1]
                out = os.path.join(result_dir, f"{pfx}.{tag}.bam")
                with open(out, "w") as o:
                    logger.info(f"@Pass {ipass+1}, {pfx}, local mapping stats")

                    map_proc = subprocess.Popen(
                        [
                            bowtie2_path,
                            *bwt_opts.split(" "),
                            "--end-to-end",
                            "--reorder",
                            "--un",
                            "temp.fq",
                            "--rg-id",
                            tag,
                            "--rg",
                            f"SM:{pfx}",
                            "-p",
                            str(nthd),
                            "-x",
                            genome_index,
                            "-U",
                            inp,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    sam_filt = subprocess.Popen(
                        [samtools_path, "view", "-F", "4", "-bS", "-"],
                        stdin=map_proc.stdout,
                        stdout=o,
                    )

                    # mapping stats
                    for line in iter(map_proc.stderr.readline, b""):
                        line = line.decode().rstrip()
                        if (
                            line.startswith("#")
                            or line.startswith("perl")
                            or line.startswith("\s")
                            or line.startswith("Warn")
                        ):
                            continue
                        else:
                            logger.info(f"@Pass {ipass+1}, {pfx}, {line}")

                    sam_filt.wait()

                # log process
                os.remove("temp.fq")
                intermediates[pfx]["bams"].append(out)
                if leftover != 0:
                    logger.info(
                        f"@Pass {ipass+1}, {pfx}: {leftover} reads for next pass to map"
                    )
                    logger.info(f"@Pass {ipass+1}, {pfx}: Finished")
                else:
                    logger.info(f"@Pass {ipass+1}, {pfx}: Finished")
                    logger.info(f"{pfx} multipass mapping done")
            else:
                intermediates[pfx]["ToMap"] = 0
                logger.info(f"@Pass {ipass+1}, {pfx}: Finished")
                logger.info(f"{pfx} multipass mapping done")

        logger.info(f"@Pass {ipass+1}: Done")
    logger.info(f"Multipass mapping done")

    # merge all bams
    logger.info(f"Merging bam files")
    pysam.merge(
        "-@",
        str(nthd),
        "-n",
        "-f",
        f"{result_dir}/merged.bam",
        *[b for pfx in prefix for b in intermediates[pfx]["bams"]],
    )
    # sort by name. multiple mapped segment of a read and the mate pair will be grouped together
    logger.info(f"Sorting merged bam file by name")
    pysam.sort(
        "-m",
        "500M",
        "-@",
        str(nthd),
        "-n",
        "-o",
        f"{result_dir}/{project_name}.merged.bam",
        f"{result_dir}/merged.bam",
    )
    os.remove(f"{result_dir}/merged.bam")

    # clean intermidiate files
    for pfx, files in intermediates.items():
        [os.remove(f) for f in files["bams"]]
        [os.remove(f) for f in files["5primeFq"]]
        [os.remove(f) for f in files["3primeFq"]]

    logger.info("Done")

    return f"{result_dir}/{project_name}.merged.bam"


# multipass_mapping_from_hicpro(
#    hicpro_results,
#    project_name,
#    MseI_DdeI_Ligated_Site,
#    "/home/software/bowtie2-2.2.9/genome/mm9/mm9",
#    35,
# )

# %%
def split_fastq_by_motif(fastq, motif, prefix):
    search_pattern = re.compile(motif).search
    cnt = 0
    with open(f"{prefix}.5prime.fastq", "wb", 1024 * 1024) as out5, open(
        f"{prefix}.3prime.fastq", "wb", 1024 * 1024
    ) as out3:
        out5_write, out3_write = out5.write, out3.write
        for name1, seq, name2, qual_seq in read_fastq(fastq):
            m = search_pattern(seq)
            if m:
                cnt += 1
                start, end = m.span()
                out5_write(
                    name1
                    + seq[:start]
                    + b"\n"
                    + name2
                    + qual_seq[:start]
                    + b"\n"
                )
                out3_write(name1 + seq[end:] + name2 + qual_seq[end:])

    return cnt


def read_fastq(fastq):
    if fastq.endswith("gz"):
        inp_ = gzip.open(fastq, "rb")
    else:
        inp_ = open(fastq, "rb")
    inp = mmap.mmap(inp_.fileno(), 0, access=mmap.ACCESS_READ)
    readline = inp.readline

    while 1:
        name1 = readline()
        if not name1:  # EOF
            inp_.close()
            inp.close()
            break
        seq = readline()
        name2 = readline()
        qual_seq = readline()

        yield name1, seq, name2, qual_seq


def paired_side(str):
    if "R1" in str:  # or "_1" in str:
        return "R1"
    elif "R2" in str:  # or "_2" in str:
        return "R2"
    else:
        raise ValueError("Undetermined PE data")


def solve_pair_end_samples(hicpro_results, project_name):
    bams = [
        f
        for f in os.listdir(
            f"{hicpro_results}bowtie_results/bwt2_global/{project_name}/"
        )
        if f.endswith("bam")
    ]
    if len(bams) % 2 != 0:
        raise Exception("Files are not paired")

    prefix = [f.split(".")[0] for f in bams]
    return prefix, bams


# split_fastq_by_motif(
#    "/Extension_HDD2/Hanbin/ES_Cell/E14/HiC3_HL/HL12_HiChIP_Test/E14_201206_HiC3_out/bowtie_results/bwt2_global/data/HL12_1515_S16_L002_R1_001_mm9.bwt2glob.unmap.fastq",
#    MseI_DdeI_Ligated_Site,
#    "test",
# )

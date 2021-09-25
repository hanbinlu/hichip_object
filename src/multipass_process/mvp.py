import os, subprocess
import logging
import operator, itertools
import pysam
import re2 as re  # pip install google-re2
import numpy as np
from typing import NamedTuple
from random import choice
import ray

# MseI_DdeI_Digestion_Site = r"(CT[ATCG]AG)|(TTAA)"
# MseI_DdeI_Ligated_Site = (
#     b"(CT[ATCG]AT[ATCG]AG)|(CT[ATCG]ATAA)|(TTAT[ATCG]AG)|(TTATAA)"
# )
# MboI_Digestion_Site = r"GATC"

########## Process Multiple Pass Mapped BAM File to ValidPair  ############
def construct_mpp_validpair(
    bam_file,
    mapq,
    digestion_sites,
    out,
    nprocs,
    procs_per_pysam=2,
    mvp_selection_rule="longest_cis",
):
    """
    Process multiple-pass mapped and paired BAM to validpair records. Output is sorted and duplication removed.

    Parameters
    ----------
    bam_file: path to the multiple-pass mapped and paired BAM.
    mapq: keep high mapq reads.
    digestion_sites: dictionary, chr->[array of digestion sitepositions]
    out: path for the output mvp file
    nprocs: number of cpu to be used
    procs_per_pysam: the function split `nprocs // procs_per_pysam` pysam jobs to parallel process the `bam_file` to speedup.
        Each pysam job is using `pysam_parallel` cores. (Note: one pysam jobs using `nprocs` cores is not efficient)
    """
    logger = logging.getLogger("create_mvp")
    fh = logging.FileHandler(f"{out}.stats")
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    parallel, workers, workers_out = nprocs // procs_per_pysam, [], []
    kernel = count_high_order_pet.options(num_cpus=procs_per_pysam)
    # worker i takes care of `itertools.islice(i, None, parallel)` PET records in `bam_file`
    logger.info(f"Start parsing {bam_file}")
    for i in range(parallel):
        temp_out = f"{out}.{i}"
        # temp mvp files
        workers_out.append(temp_out)
        workers.append(
            kernel.remote(
                bam_file,
                mapq,
                digestion_sites,
                procs_per_pysam,
                i,
                parallel,
                temp_out,
                mvp_selection_rule,
            )
        )

    # merge temp outputs
    results = [ray.get(j) for j in workers]
    cnt = sum([x[0] for x in results])
    vp = sum([x[1] for x in results])
    re_de_dump = sum([x[2] for x in results])
    inter_chro = sum([x[3] for x in results])
    cnt_2frags = sum([x[4] for x in results])

    logger.info(
        f"{cnt} multiple mapped fragment PETs, {cnt_2frags} ({cnt_2frags/cnt}) are two-frag PETs"
    )
    logger.info(
        f"{vp} validpairs, {inter_chro/vp*100}% are interchromosomal validpairs"
    )
    logger.info(f"{re_de_dump} religation, dangling end, and dump pairs")

    # mvp tag: chr1,x1,chr2,x2,chr3,x3... record all mapped segment of a PET
    logger.info("Removing duplications")
    with open(f"{out}.sorted", "w") as o:
        cat_proc = subprocess.Popen(
            ["cat", *workers_out], stdout=subprocess.PIPE
        )
        sort_mvp = subprocess.Popen(
            ["sort", "-k13,13", "--parallel", str(nprocs), "-S", "20G"],
            stdin=cat_proc.stdout,
            stdout=o,
        )
        sort_mvp.wait()

    num_rmdup_vps = mvp_rmdup(f"{out}.sorted", out)

    [os.remove(f) for f in workers_out]
    os.remove(f"{out}.sorted")

    logger.info(
        f"{num_rmdup_vps} ({num_rmdup_vps / vp}) validpairs are kept from duplication removal"
    )


@ray.remote(num_returns=5)
def count_high_order_pet(
    bam_file,
    mapq,
    digestion_sites,
    nthd,
    worker_id,
    n_workers,
    temp_file,
    mvp_selection_rule,
):
    """
    Parse (worker_id:n_workers:End) th PET record in paired BAM file to Hi-C validpair data and add mvp tag for all mapped segs of the PET

    For records has more than 2 mapped fragment, keep the "longest" pair.
    """
    cnt, vp, re_de_dump, inter_chro, cnt_2frag = 0, 0, 0, 0, 0
    pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_file, threads=nthd) as bfh, open(
        temp_file, "w", 1024 * 100
    ) as o:
        # slicing records in the BAM file to parse
        worker_items_iter = itertools.islice(
            mp_bam_rec_reader(bfh, mapq), worker_id, None, n_workers
        )
        # each sequence PET is parsed to a list of MappedFrags
        for qname, raw_data in worker_items_iter:
            data = list(raw_data)
            if len(data) >= 2:
                cnt += 1
                cnt_2frag += len(data) == 2
                frags = list(map(bam_rec_to_mapped_seg, data))
                frags.sort(key=operator.attrgetter("chromosome", "middle"))
                # for PET that has more than 2 mapped fragment, keep the longest separated pairs
                frag_i, frag_j = _select_pair(frags, how=mvp_selection_rule)
                if frag_i.chromosome != frag_j.chromosome:
                    is_vp = True
                    inter_chro += 1
                    res_i = np.searchsorted(
                        digestion_sites[frag_i.chromosome],
                        frag_i.middle,
                    )
                    res_j = np.searchsorted(
                        digestion_sites[frag_j.chromosome],
                        frag_j.middle,
                    )
                else:
                    res_i, res_j = np.searchsorted(
                        digestion_sites[frag_i.chromosome],
                        [frag_i.middle, frag_j.middle],
                    )
                    if res_j - res_i > 1:
                        is_vp = True
                    else:
                        # religation, dangling end, dump pair
                        is_vp = False
                        re_de_dump += 1

                # write result in validpair format suffix with tag describing all the mapped segments
                if is_vp:
                    vp += 1
                    mvp_tag = _mvp_tag(frags)
                    mvp_rec = (
                        _write_mvp_rec(
                            qname, frag_i, frag_j, res_i, res_j, mvp_tag
                        )
                        + "\n"
                    )
                    o.write(mvp_rec)

    return cnt, vp, re_de_dump, inter_chro, cnt_2frag


def _select_pair(frags, low=5000, high=2000000, how="rand"):
    """
    Selection rules:
    ---------------
    rand pair (rand)
    rand cis (rand_cis)
    longest cis (longest_cis)
    rand midrange (rand_mrng)
    longest midrange (longest_mrng)
    """

    pairs = list(itertools.combinations(frags, 2))
    if how == "rand":
        return choice(pairs)

    pairs = list(itertools.combinations(frags, 2))
    if how == "rand":
        return choice(pairs)

    # cis priority
    cis_pairs = list(
        filter(lambda x: x[0].chromosome == x[1].chromosome, pairs)
    )
    if len(cis_pairs) == 0:
        return choice(pairs)
    elif how == "rand_cis":
        return choice(cis_pairs)
    elif how == "longest_cis":
        return max(cis_pairs, key=lambda x: x[1].middle - x[0].middle)

    # midrange priority
    mid_rng_pairs = list(
        filter(
            lambda x: (x[1].middle - x[0].middle >= low)
            and (x[1].middle - x[0].middle <= high),
            cis_pairs,
        )
    )
    if len(cis_pairs) == 0:
        return choice(pairs)
    elif len(mid_rng_pairs) == 0:
        return choice(cis_pairs)
    elif how == "rand_mrng":
        return choice(mid_rng_pairs)
    elif how == "longest_mrng":
        return max(mid_rng_pairs, key=lambda x: x[1].middle - x[0].middle)


########## Select PETs from Multiple Pass Mapped BAM to Dump as Reads ############
def dump_PETs_to_bed(
    bam_file,
    mapq,
    digestion_sites,
    out,
    nprocs,
    procs_per_pysam,
    exclude_interchro=False,
    local_range=1e20,
    read_length=100,
    mvp_selection_rule="longest_cis",
    flatten_as_one_frag=True,
):
    """
    Pipline to process Hi-C BAM file to filtered reads in bed format.

    Steps
    -----
    1. Parsing BAM, write filtered PETs to intermediate files -- chr1,x1,chr2,x2,chr3,x3...
    2. call linux `sort` and `uniq` to remove duplication
        (1 and 2 is done by `construct_stringent_local_pair`)

        Considering following situation of a PET with `n` mapped segments:
        1. n == 2: Always includes re-ligation, self-ligation, dangling-end pairs;
                For interchrom and validpairs determine by `exclude_interchro` and `local_range` respectively
        2. n > 2: if any pair of fragment were determined to `False` base on situation 1, discard the PET.

    3. if `flatten_as_one_frag` is `False`, each chr_i,x_i flatten to one bed record and dump to output file. Otherwise, treat as a frag: chr,x_min,x_max

    Parameters
    ----------
    bam_file: path to the multiple-pass mapped and paired BAM.
    mapq: keep high mapq reads.
    digestion_sites: raw regexp string, digestion sites use for the Hi-C experiment.
    out: path for the output bed file
    nprocs: number of cpu to be used
    procs_per_pysam: the function split `nprocs // procs_per_pysam` pysam jobs to parallel process the `bam_file` to speedup.
        Each pysam job is using `pysam_parallel` cores. (Note: one pysam jobs using `nprocs` cores is not efficient)
    local_range: 1e20 in default to include all cis pairs
    exclude_interchro: default `False` to include interchromosomal pairs (Note that it dumps all (>=2) mapped fragments)
    """

    construct_stringent_local_pair(
        bam_file,
        local_range,
        exclude_interchro,
        mapq,
        digestion_sites,
        "temp.out",
        nprocs,
        procs_per_pysam,
        mvp_selection_rule,
    )

    logging.info(f"Flattening to PETs to {read_length} bp reads")
    if flatten_as_one_frag:
        if not exclude_interchro:
            raise ValueError(
                "If flatten to one frag, please set exclude_interchro to be True"
            )

        with open("temp.out") as inp, open(out, "w") as o:
            for line in inp:
                fields = line.split(",")
                chro = fields[0]
                pos = [int(fields[i]) for i in range(1, len(fields), 2)]
                start, end = min(pos), max(pos)
                o.write(chro + "\t" + str(start) + "\t" + str(end) + "\n")
    else:
        half_rl = read_length // 2
        with open("temp.out") as inp, open(out, "w") as o:
            for line in inp:
                fields = line.split(",")
                for i in range(0, len(fields), 2):
                    chro, pos = fields[i], int(fields[i + 1])
                    o.write(
                        chro
                        + "\t"
                        + str(pos - half_rl)
                        + "\t"
                        + str(pos + half_rl)
                        + "\n"
                    )

    subprocess.run(["rm", "temp.out"])
    count_reads = subprocess.run(["wc", "-l", out], stdout=subprocess.PIPE)
    num_reads = count_reads.stdout.decode().split(" ")[0]
    logging.info(f"{num_reads} reads were dumped.")


def construct_stringent_local_pair(
    bam_file,
    local_range,
    exclude_interchro,
    mapq,
    digestion_sites,
    out,
    nprocs,
    proc_per_pysam,
    mvp_selection_rule,
):
    """
    Parse (worker_id:n_workers:End) PET record in paired BAM file to Hi-C validpair data and dump the filterd record and dump the filterd records

    Considering following situation of a PET of `n` mapped segments:
    1. n == 2: Always includes re-ligation, self-ligation, dangling-end pairs;
            For interchrom and validpairs determine by `exclude_interchro` and `local_range` respectively
    2. n > 2: if any pair of fragment were determined to `False` base on situation 1, discard the PET.

    Parameters
    ----------
        `?dump_PETs_to_bed` for details
    """
    parallel, workers, workers_out = nprocs // proc_per_pysam, [], []
    kernel = count_stringent_local_high_order_pet.options(
        num_cpus=proc_per_pysam
    )
    logging.info("Parsing bam data to Hi-C pairs")
    for i in range(parallel):
        temp_out = f"{out}.{i}"
        # temp mvp files
        workers_out.append(temp_out)
        workers.append(
            kernel.remote(
                bam_file,
                local_range,
                exclude_interchro,
                mapq,
                digestion_sites,
                proc_per_pysam,
                i,
                parallel,
                temp_out,
                mvp_selection_rule,
            )
        )

    results = [ray.get(j) for j in workers]
    cnt = sum(results)

    # the outfile only has mvp tags, call unique to remove duplication
    logging.info("Removing duplications")
    with open(out, "w") as o:
        cat_proc = subprocess.Popen(
            ["cat", *workers_out], stdout=subprocess.PIPE
        )
        sort_mvp = subprocess.Popen(
            ["sort", "-T", ".", "--parallel", str(nprocs), "-S", "20G"],
            stdin=cat_proc.stdout,
            stdout=subprocess.PIPE,
        )
        rmdup_mvp = subprocess.Popen(
            ["uniq"],
            stdin=sort_mvp.stdout,
            stdout=o,
        )
        rmdup_mvp.wait()

    [os.remove(f) for f in workers_out]
    count_rmdup = subprocess.run(["wc", "-l", out], stdout=subprocess.PIPE)
    num_rmdup = count_rmdup.stdout.decode().split(" ")[0]
    logging.info(
        f"Processed {cnt} paired PETs. {num_rmdup} paired PETs were kept after duplication removal"
    )


@ray.remote(num_returns=1)
def count_stringent_local_high_order_pet(
    bam_file,
    local_range,
    exclude_interchro,
    mapq,
    digestion_sites,
    nthd,
    worker_id,
    n_workers,
    temp_file,
    mvp_selection_rule,
):
    """
    Parse (worker_id:n_workers:End) th PET records in BAM file to Hi-C type pairs.

    Considering following situation of a PET of `n` mapped segments:
    1. n == 2: Always includes re-ligation, self-ligation, dangling-end pairs;
            For interchrom and validpairs determine by `exclude_interchro` and `local_range` respectively
    2. n > 2: if any pair of fragment were determined to `False` base on situation 1, discard the PET.

    Parameters
    ----------
        `?dump_PETs_to_bed` for details
    """
    cnt, local_pet = 0, 0
    pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_file, threads=nthd) as bfh, open(
        temp_file, "w", 1024 * 100
    ) as o:
        # chunks in the BAM file to parse
        worker_items_iter = itertools.islice(
            mp_bam_rec_reader(bfh, mapq), worker_id, None, n_workers
        )
        # each sequence PET is parsed to a list of MappedFrags
        # This function only keeps stringent local pairs of following configurations:
        # len(MappedSeg) == 2: Re-ligation, Self-ligation, dangling-end, inter chromasomal (by `exclude_interchro`), and dist <= `short_cis`
        # len(MappedSeg) > 2: if all cis _longest_cis <= `short_cist`; all inter chro, keep all (by `exclude_interchro`); inter chro + cis <= `short_dist`
        for _, raw_data in worker_items_iter:
            data = list(raw_data)
            is_local = None
            if len(data) == 2:
                cnt += 1
                frags = list(map(bam_rec_to_mapped_seg, data))
                frags.sort(key=operator.attrgetter("chromosome", "middle"))
                frag_i, frag_j = _select_pair(frags, how=mvp_selection_rule)
                if frag_i.chromosome != frag_j.chromosome:
                    # inter chro
                    if exclude_interchro:
                        is_local = False
                    else:
                        is_local = True
                else:
                    res_i, res_j = np.searchsorted(
                        digestion_sites[frag_i.chromosome],
                        [frag_i.middle, frag_j.middle],
                    )
                    if res_j - res_i > 1:
                        # validpair
                        if frag_j.middle - frag_i.middle <= local_range:
                            # short cis
                            is_local = True
                        else:
                            is_local = False
                    else:
                        # religation, self-ligation, dangling end, dump pair
                        # dump pair are very few
                        is_local = True
            elif len(data) > 2:  # high order situation
                cnt += 1
                frags = list(map(bam_rec_to_mapped_seg, data))
                frags.sort(key=operator.attrgetter("chromosome", "middle"))
                # group by chro
                res = {}
                for frg in frags:
                    res.setdefault(frg.chromosome, []).append(frg)

                if len(res.keys()) == 1:
                    # all frags are from same chromosome
                    res_i, res_j = np.searchsorted(
                        digestion_sites[frags[0].chromosome],
                        [frags[0].middle, frags[-1].middle],
                    )
                    if res_j - res_i > 1:
                        if frags[-1].middle - frags[0].middle <= local_range:
                            # short cis
                            is_local = True
                        else:
                            is_local = False
                    else:
                        # religation, self-ligation, dangling end, dump pair
                        # dump pair are very few
                        is_local = True
                else:  # intra + inter
                    if exclude_interchro:
                        is_local = False
                    else:
                        # single fragment default 0 <= local_range
                        is_local = all(
                            [
                                v[-1].middle - v[0].middle <= local_range
                                for _, v in res.items()
                            ]
                        )

            # write result in validpair format suffix with tag describing all the mapped segments
            if is_local:
                local_pet += 1
                mvp_tag = _mvp_tag(frags)
                o.write(mvp_tag + "\n")

    return cnt


# def _group_frags_by_chro(frags):
#    res = {}
#    for frg in frags:
#        res.setdefault(frg.chromosome, []).append(frg.middle)
#
#    for k, v in res.items():
#        v.sort()
#
#    return res

########### Functions that operates MVP data ############
def _mvp_tag(frags):
    # chr1,x1,chr2,x2,chr3,x3 ...
    tag_components = []
    for frg in frags:
        tag_components.append(frg.chromosome)
        tag_components.append(str(frg.middle))

    return ",".join(tag_components)


def _write_mvp_rec(qname, frag_i, frag_j, res_i, res_j, mvp_tag):
    # A00953:185:HN7TMDSXY:1:1419:26494:17033 chr1    3000368 +       chr1    3000709 -       213    HIC_chr1_9       HIC_chr1_12     40      42 weight VPFF/VPFR/VPRR/VPRF pet_coor(added field)
    fields = [
        qname,
        frag_i.chromosome,
        str(frag_i.middle),
        "+",
        frag_j.chromosome,
        str(frag_j.middle),
        "+",
        "NA",
        str(res_i),
        str(res_j),
        str(frag_i.mapq),
        str(frag_j.mapq),
        mvp_tag,
    ]
    return "\t".join(fields)


def mvp_rmdup(inp, out):
    count = 0
    with open(inp) as f, open(out, "w", 1024 * 1024) as o:
        mvp_tag_gen = (line.rsplit("\t", 1) for line in f)
        for _, rec in itertools.groupby(
            mvp_tag_gen, key=operator.itemgetter(1)
        ):
            count += 1
            o.write(next(rec)[0] + "\n")

    return count


########### Functions that parsed MP BAM file into MappedSegment ############
class MappedSegment(NamedTuple):
    """Not differentiating R1 and R2"""

    chromosome: str
    start: int
    middle: int
    mapq: int


def bam_rec_to_mapped_seg(rec):
    p1, p2 = rec.reference_start, rec.reference_end - 1
    return MappedSegment(
        rec.reference_name,
        p2 if rec.is_reverse else p1,
        (p1 + p2) // 2,
        rec.mapping_quality,
    )


def mp_bam_rec_reader(bfh, mapq):
    def is_discard_rec(rec):
        return rec.is_unmapped or rec.mapping_quality < mapq

    mp_bam_rec_iter = itertools.groupby(
        itertools.filterfalse(is_discard_rec, bfh),
        key=operator.attrgetter("query_name"),
    )
    return mp_bam_rec_iter


def genome_digestion(genome_fa, motif):
    pattern = re.compile(motif)
    dig_pos = {}
    with pysam.FastxFile(genome_fa) as f:
        for entry in f:
            dig_pos[entry.name] = np.array(
                [m.end() - 1 for m in pattern.finditer(entry.sequence.upper())]
            )

    return dig_pos

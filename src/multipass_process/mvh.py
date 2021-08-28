import logging, os
import subprocess, pysam
import itertools, operator
import numpy as np
from .mvp import MappedSegment, bam_rec_to_mapped_seg, mp_bam_rec_reader

import ray

########## Process Multiple Pass Mapped BAM File to Multi-way validhubs (mvh) ############
def construct_mpp_validhub(
    bam_file, mapq, digestion_sites, out, nprocs, procs_per_pysam=2
):
    """
    Process multiple-pass mapped and paired BAM to validhub records. Output is sorted and duplication removed.

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
    logger = logging.getLogger("create_mvh")
    fh = logging.FileHandler(f"{out}.mvh.stats")
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
            )
        )

    # merge temp outputs
    # mppValidHubs concise format:
    # chros positions   mapqs
    # chr1,chr2,chr3    x1,x2,x3    mapq1,mapq2,mapq3
    results = [ray.get(j) for j in workers]
    cnt = sum([x[0] for x in results])
    vh = sum([x[1] for x in results])
    inter_chro = sum([x[2] for x in results])
    cis_vh = sum([x[3] for x in results])
    cis_vh_3 = sum([x[4] for x in results])
    cis_vh_4 = sum([x[5] for x in results])
    cis_vh_5 = sum([x[6] for x in results])
    cis_vh_6_plus = sum([x[7] for x in results])

    logger.info(f"{cnt} >3 mapped fragment PETs, {vh} ({vh/cnt}) are validhubs")
    logger.info(
        f"{cis_vh} ({cis_vh/vh}) cis validhubs, {inter_chro} ({inter_chro/cnt}) are interchromosomal validhubs"
    )
    logger.info(f"n = 3 cis validhubs: {cis_vh_3}")
    logger.info(f"n = 4 cis validhubs: {cis_vh_4}")
    logger.info(f"n = 5 cis validhubs: {cis_vh_5}")
    logger.info(f"n >= 6 cis validhubs: {cis_vh_6_plus}")

    logger.info("Removing duplications")
    with open(f"{out}.unique", "w") as o:
        cat_proc = subprocess.Popen(
            ["cat", *workers_out], stdout=subprocess.PIPE
        )
        sort_mvh = subprocess.Popen(
            ["sort", "-k12,12", "--parallel", str(nprocs), "-S", "20G"],
            stdin=cat_proc.stdout,
            stdout=subprocess.PIPE,
        )
        uniq_mvh = subprocess.Popen(["uniq"], stdin=sort_mvh.stdout, stdout=o)
        uniq_mvh.wait()

    # mppValidHubs format:
    # record: chr1,chr2,chr3    x1,x2,x3    mapq1,mapq2,mapq3
    # flattern to sam hub id bed records
    # hub_x chr1 x1
    # hub_x chr2 x2
    # hub_x chr3 x3
    hub_cnt = 0
    with open(f"{out}.unique") as inp, open(out, "w", 1024 * 1024) as o:
        for line in inp:
            hub_cnt += 1
            chros, positions = line.rstrip().split("\t")
            hub_id = f"hub_{hub_cnt}"
            for chro, pos in zip(chros.split(","), positions.split(",")):
                o.write("\t".join([hub_id, chro, str(pos)]) + "\n")

    [os.remove(f) for f in workers_out]
    os.remove(f"{out}.unique")

    logger.info(
        f"{hub_cnt} ({hub_cnt / vh}) validpairs are kept from duplication removal"
    )


@ray.remote(num_returns=8)
def count_high_order_pet(
    bam_file, mapq, digestion_sites, nthd, worker_id, n_workers, temp_file
):
    """
    Parse (worker_id:n_workers:End) th PET record in paired BAM file to Hi-C validpair data and add mvp tag for all mapped segs of the PET
    """
    cnt = vh = inter_chro = 0
    cis_vh = cis_vh_3 = cis_vh_4 = cis_vh_5 = cis_vh_6_plus = 0
    pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_file, threads=nthd) as bfh, open(
        temp_file, "w", 1024 * 100
    ) as o:
        # slicing records in the BAM file to parse
        worker_items_iter = itertools.islice(
            mp_bam_rec_reader(bfh, mapq), worker_id, None, n_workers
        )
        # each sequence PET is parsed to a list of MappedFrags
        for _, raw_data in worker_items_iter:
            data = list(raw_data)
            if len(data) > 2:
                # only high order PETs
                cnt += 1
                frags = list(map(bam_rec_to_mapped_seg, data))
                nfrags = len(frags)
                frags.sort(key=operator.attrgetter("chromosome", "middle"))
                # find all resfrag id
                resfrags = [
                    np.searchsorted(digestion_sites[frg.chromosome], frg.middle)
                    for frg in frags
                ]
                # vp_type code: 0->re_de_dump, 1->vp, 2->interchro
                vp_path = []
                for j in range(1, nfrags):
                    frag_i, frag_j = frags[j - 1], frags[j]
                    res_i, res_j = resfrags[j - 1], resfrags[j]
                    if frag_i.chromosome != frag_j.chromosome:
                        vp_path.append(2)
                    else:
                        if res_j - res_i > 1:
                            vp_path.append(1)
                        else:
                            # religation, dangling end, self-circle, dump pair
                            vp_path.append(0)

                # validhub require at least 2 different validpair (1|2)
                # flag frags that is vp and merge continues re_de_dump
                merged_frags, vp_frag_flag = [frags[0]], np.ones(nfrags)
                # if a frag involed in a re_de_dump pairs, flag as 0
                for (i, vp_type) in enumerate(vp_path):
                    # vp_type of frags[i-1] and frags[i]
                    vp_frag_flag[i - 1] *= vp_type
                    vp_frag_flag[i] *= vp_type

                for (flag, _frg_group) in itertools.groupby(
                    zip(frags, vp_frag_flag), key=operator.itemgetter(1)
                ):
                    frg_group = [x[0] for x in _frg_group]
                    if flag:
                        merged_frags.extend(frg_group)
                    else:  # merge
                        merged_frags.append(
                            MappedSegment(
                                chromosome=frg_group[0].chromosome,
                                start=int(
                                    np.mean([frg.start for frg in frg_group])
                                ),
                                middle=int(
                                    np.mean([frg.middle for frg in frg_group])
                                ),
                                mapq=np.min([frg.mapq for frg in frg_group]),
                            )
                        )

                # summary stats
                n = len(merged_frags)
                if n >= 3:
                    is_vh = True
                    vh += 1
                    ref_chr = merged_frags[0].chromosome
                    if all([frg.chromosome == ref_chr for frg in merged_frags]):
                        cis_vh += 1
                        if n == 3:
                            cis_vh_3 += 1
                        elif n == 4:
                            cis_vh_4 += 1
                        elif n == 5:
                            cis_vh_5 += 1
                        else:
                            cis_vh_6_plus += 1
                    else:
                        inter_chro += 1
                else:
                    is_vh = False

                # write result in multiway chr/position/mapq format
                if is_vh:
                    mbed_data = [
                        ",".join([frg.chromosome for frg in merged_frags]),
                        ",".join([str(frg.middle) for frg in merged_frags]),
                    ]
                    o.write("\t".join(mbed_data) + "\n")

    return (
        cnt,
        vh,
        inter_chro,
        cis_vh,
        cis_vh_3,
        cis_vh_4,
        cis_vh_5,
        cis_vh_6_plus,
    )


# def mvh_rmdup(inp, out):
#    count = 0
#    with open(inp) as f, open(out, "w", 1024 * 1024) as o:
#        mvh_tag_gen = (line.rsplit("\t", 1) for line in f)
#        for _, rec in itertools.groupby(
#            mvh_tag_gen, key=operator.itemgetter(0)
#        ):
#            count += 1
#            o.write("\t".join(next(rec)) + "\n")
#
#    return count

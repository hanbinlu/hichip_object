import numpy as np
import pandas as pd
from collections import Counter
import logging

# import itertools, operator, ray


def gb_hubs_to_gff(gb_hubs, gb_anchors, out):
    # GFF format:
    # chro  source  feature start   end score   strand  frame group
    # "chr22	TeleGene	enhancer	10000000	10001000	500	+	.	touch1"
    # feature -> anchor or normal genomic bin
    # score -> PET count of the hub
    # frame ->
    source = "MVH"
    gb_chrs, gb_starts, gb_ends, gb_nanchors = (
        gb_anchors.chro.values,
        gb_anchors.start.values,
        gb_anchors.end.values,
        gb_anchors.Num_Anchors.values,
    )
    with open(out, "w") as o:
        for i, (bins, count) in enumerate(gb_hubs.items()):
            hub_name = f"hub_{i}"
            for bid in bins:
                is_anchor = "anchor" if gb_nanchors[bid] > 0 else "normal"
                o.write(
                    "\t".join(
                        [
                            gb_chrs[bid],
                            source,
                            is_anchor,
                            str(gb_starts[bid]),
                            str(gb_ends[bid]),
                            str(count),
                            ".",
                            ".",
                            hub_name,
                        ]
                    )
                    + "\n"
                )


def assign_genomic_bins(mvh_file, gb_anchors, parallel=20, chunk_size=500000):
    """ """
    if any(gb_anchors.index.values != np.arange(len(gb_anchors))):
        raise ValueError("Anchors index must be RangeIndex")
    gb_hubs = Counter()

    mvh_data_chunks = pd.read_csv(
        mvh_file,
        sep="\t",
        header=None,
        chunksize=chunk_size,
        names=["hid", "chro", "pos"],
    )
    orphan_hub = pd.DataFrame()
    logging.info("Start")
    for i, chunk in enumerate(mvh_data_chunks):
        # adopt last orphan
        chunk = pd.concat((orphan_hub, chunk))
        # drop the last record since it might be incomplete hub record
        orphan_hub_id = chunk.hid.iloc[-1]
        orphan_hub = chunk.loc[chunk.hid == orphan_hub_id]
        complete_chunk = chunk.loc[chunk.hid != orphan_hub_id].reset_index(
            drop=True
        )
        binned_chunk = _locate_genomic_bin(complete_chunk, gb_anchors)
        complete_chunk["bid"] = binned_chunk

        # aggregate interacting bins
        for _, group in complete_chunk.groupby("hid"):
            bin_coor = tuple(np.sort(group.bid.values))
            if -1 not in bin_coor:
                gb_hubs[bin_coor] += 1
        logging.info(f"@Chunk {i}")

    # finish the real orphan
    orphan_hub = orphan_hub.reset_index(drop=True)
    # only one hub
    orphan_hub_bid = tuple(np.sort(_locate_genomic_bin(orphan_hub, gb_anchors)))
    if -1 not in orphan_hub_bid:
        gb_hubs[orphan_hub_bid] += 1

    return gb_hubs


def _locate_genomic_bin(positions, gb_anchors):
    chro_set = gb_anchors.chro.unique()
    chros = positions.chro.unique()
    which_gb = -np.ones(len(positions), dtype=int)
    for chro in chros:
        if chro not in chro_set:
            continue
        chro_gb_anchors = gb_anchors.loc[gb_anchors.chro == chro]
        bins = chro_gb_anchors.start.values
        # binned positions
        chro_pos_idx = positions.chro == chro
        chro_pos = positions.loc[chro_pos_idx]
        bin_idx = np.searchsorted(bins, chro_pos.pos.values, side="right") - 1
        which_gb[chro_pos_idx.values] = chro_gb_anchors.index.values[bin_idx]

    return which_gb


# def mvh_reader(mvh_fh):
#    mvh_line_gen = (line.rstrip().split("\t") for line in mvh_fh)
#    mvh_iter = itertools.groupby(mvh_line_gen, key=operator.getitem(0))
#    return mvh_iter

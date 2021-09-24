import numpy as np
import pandas as pd
import itertools, ray, logging
from numba.typed import List
from numba import njit
from scipy.sparse import csc_matrix


def loop_PET_count(
    vp,
    anchors,
    parallel=20,
    chunk_size=500000,
):
    """
    Anchor-Anchor entries count from PET data
    """
    if any(anchors.index.values != np.arange(len(anchors))):
        raise ValueError("Anchors index must be RangeIndex")

    n = len(anchors)
    loop_pet = csc_matrix((n, n), dtype=int)

    process_jobs = []
    for job_id in range(parallel):
        process_jobs.append(
            _loop_PET_count.remote(vp, anchors, chunk_size, job_id, parallel)
        )

    for v in ray.get(process_jobs):
        loop_pet += v

    return loop_pet


@ray.remote
def _loop_PET_count(vp, anchors, chunk_size, job_id, parallel):
    """
    Loop is defined as anchor-anchor connection.
    """
    n = len(anchors)
    loop_pet = csc_matrix((n, n), dtype=int)
    vp_chunk = itertools.islice(
        pd.read_csv(
            vp,
            sep="\t",
            header=None,
            usecols=range(12),
            names=[
                "PET_name",
                "chr1",
                "x1",
                "strand1",
                "chr2",
                "x2",
                "strand2",
                "Frag_Size",
                "ResFrag1",
                "ResFrag2",
                "MAPQ1",
                "MAPQ2",
            ],
            chunksize=chunk_size,
        ),
        job_id,
        None,
        parallel,
    )

    for vp_df in vp_chunk:
        is_in_anchor_1, is_in_anchor_2 = _check_PET_from_anchors(vp_df, anchors)
        intra_chro_index = (vp_df.chr1 == vp_df.chr2).values
        loop_index = (is_in_anchor_1 != -1) & (is_in_anchor_2 != -1)
        selected_index = intra_chro_index & loop_index
        I, J = [], []
        for x, y in zip(
            is_in_anchor_1[selected_index], is_in_anchor_2[selected_index]
        ):
            # make sure upper triangle entries
            if x > y:
                x, y = y, x

            I.append(x)
            J.append(y)

        loop_pet += csc_matrix(
            (np.ones(sum(selected_index)), (np.array(I), np.array(J))),
            shape=(n, n),
        )

    return loop_pet


def putative_loops_from_anchors(anchors, inter_range=(5000, 2000000)):
    """
    Label qualified anchor-anchor entries by value of interaction distance
    """
    if any(anchors.index.values != np.arange(len(anchors))):
        raise ValueError("Anchors index must be RangeIndex")

    n = len(anchors)
    qualified_anchor_pairs = csc_matrix((n, n), dtype=int)
    for _, chro_anchors in anchors.groupby(by="chro"):
        centers = (chro_anchors.start + chro_anchors.end).values // 2
        I, J, V = _midrange_anchor_pairs(
            centers, chro_anchors.index.values, inter_range[0], inter_range[1]
        )
        qualified_anchor_pairs += csc_matrix(
            (V, (I, J)),
            shape=(n, n),
        )

    # if mask_zero_anchors is not None:
    #     mask_opt = diags(mask_zero_anchors, format="csc", dtype=int)
    #     qualified_anchor_pairs = mask_opt * qualified_anchor_pairs * mask_opt
    #     qualified_anchor_pairs.eliminate_zeros()

    return qualified_anchor_pairs


# def midrange_anchor_pairs_depth_product(qualified_anchor_pairs, anchor_depth):
#    """
#    Qualified anchor-anchor entries = product of anchor depth
#    """
#    n = qualified_anchor_pairs.shape[0]
#    product_depth = qualified_anchor_pairs.copy()
#    for j in range(n):
#        for i_ in range(
#            qualified_anchor_pairs.indptr[j],
#            qualified_anchor_pairs.indptr[j + 1],
#        ):
#            i = qualified_anchor_pairs.indices[i_]
#            v = anchor_depth[i] * anchor_depth[j]
#            product_depth.data[i_] = v
#
#    return product_depth


@njit
def _midrange_anchor_pairs(centers, index, low, high):
    I, J, V = List(), List(), List()
    n = len(centers)

    for i in range(n):
        for j in range(i, n):
            d = centers[j] - centers[i]
            if d >= low and d <= high:
                I.append(index[i])
                J.append(index[j])
                V.append(d)

    return I, J, V


def _check_PET_from_anchors(vp_df, anchors):
    is_in_anchor_1, is_in_anchor_2 = np.empty(len(vp_df), dtype=int), np.empty(
        len(vp_df), dtype=int
    )
    chros = np.union1d(vp_df.chr1.unique(), vp_df.chr2.unique())

    for chro in chros:
        chro_anchors = anchors.loc[anchors.chro == chro, :]
        # check read 1 with anchors
        chro_idx = vp_df.chr1 == chro
        chro_vp = vp_df.loc[chro_idx, :]
        is_in_anchor = _anchor_PET(
            chro_vp.x1.values,
            chro_anchors.index.values,
            chro_anchors.start.values,
            chro_anchors.end.values,
        )
        is_in_anchor_1[chro_idx.values] = is_in_anchor
        # check read 2 with anchors
        chro_idx = vp_df.chr2 == chro
        chro_vp = vp_df.loc[chro_idx, :]
        is_in_anchor = _anchor_PET(
            chro_vp.x2.values,
            chro_anchors.index.values,
            chro_anchors.start.values,
            chro_anchors.end.values,
        )
        is_in_anchor_2[chro_idx.values] = is_in_anchor

    return is_in_anchor_1, is_in_anchor_2


def anchor_depth_by_A2N_PETs(
    vp, anchors, parallel=5, chunk_size=500000, vp_filter=None
):
    """
    All PETs of Anchor-Non summation, not limited to mid range PETs.
    """
    mode = 1
    return _anchor_depth(
        vp,
        anchors,
        mode,
        parallel,
        chunk_size,
        vp_filter,
    )


def anchor_depth_by_anchor_PETs(
    vp, anchors, parallel=5, chunk_size=500000, vp_filter=None
):
    """
    All PETs of summation at anchors, not limited to mid range PETs and not exclude anchor-anchor
    """
    mode = 0
    return _anchor_depth(
        vp,
        anchors,
        mode,
        parallel,
        chunk_size,
        vp_filter,
    )


def _anchor_depth(
    vp,
    anchors,
    mode,
    parallel=5,
    chunk_size=500000,
    vp_filter=None,
):
    """
    Count depth of anchors either from: all PETs (`mode=0`), anchor-to-non PETs (`mode=1`), loop PETs (`mode=2`).
    """

    if any(anchors.index.values != np.arange(len(anchors))):
        raise ValueError("Anchors index must be RangeIndex")

    anchor_depth = pd.Series(np.zeros(len(anchors)), index=anchors.index)

    process_jobs = []
    for job_id in range(parallel):
        process_jobs.append(
            _anchor_PET_sum.remote(
                vp, anchors, mode, chunk_size, job_id, parallel, vp_filter
            )
        )

    for v in ray.get(process_jobs):
        anchor_depth += v

    return anchor_depth


def _mid_range_pets(vp_df, inter_range=(5000, 2000000)):
    intra_chro_index = vp_df.chr1 == vp_df.chr2
    # meet dist range
    d = vp_df.x2 - vp_df.x1
    dist_qualified_index = (d >= inter_range[0]) & (d <= inter_range[1])
    return vp_df.loc[intra_chro_index & dist_qualified_index, :]


# def anchor_depth_by_loop_PETs(vp, anchors, parallel=5, chunk_size=500000):
#    """
#    Loop PETs summation, not limited to mid range PETs.
#    """
#    mode = 2
#    return _anchor_depth(
#        vp, anchors, mode, parallel, chunk_size, vp_filter=None
#    )
#
#


@ray.remote
def _anchor_PET_sum(
    vp,
    anchors,
    mode,
    chunk_size,
    job_id,
    parallel,
    vp_filter,
):
    vp_chunk = itertools.islice(
        pd.read_csv(
            vp,
            sep="\t",
            header=None,
            usecols=range(12),
            names=[
                "PET_name",
                "chr1",
                "x1",
                "strand1",
                "chr2",
                "x2",
                "strand2",
                "Frag_Size",
                "ResFrag1",
                "ResFrag2",
                "MAPQ1",
                "MAPQ2",
            ],
            chunksize=chunk_size,
        ),
        job_id,
        None,
        parallel,
    )

    anchor_depth = pd.Series(np.zeros(len(anchors)), index=anchors.index)

    for vp_df in vp_chunk:
        if vp_filter:
            is_in_anchor_1, is_in_anchor_2 = _check_PET_from_anchors(
                _mid_range_pets(vp_df, inter_range=vp_filter), anchors
            )
        else:
            is_in_anchor_1, is_in_anchor_2 = _check_PET_from_anchors(
                vp_df, anchors
            )

        # anchor to non anchor
        anchor_of_selected_pet = []
        for x, y in zip(is_in_anchor_1, is_in_anchor_2):
            if mode == 1:
                # anchor to non: 0+1 or 1+0
                if (x != -1) & (y == -1):
                    anchor_of_selected_pet.append(x)
                elif (x == -1) & (y != -1):
                    anchor_of_selected_pet.append(y)
            elif mode == 0:
                # any anchor derived PET
                if x != -1:
                    anchor_of_selected_pet.append(x)
                elif y != -1:
                    anchor_of_selected_pet.append(y)
            elif mode == 2:
                # anchor to anchor: 1+1
                if (x != -1) & (y != -1):
                    anchor_of_selected_pet.append(x)
                    anchor_of_selected_pet.append(y)

        anchor_index, anchor_sum = np.unique(
            np.array(anchor_of_selected_pet), return_counts=True
        )

        # update anchor depth
        for i, v in zip(anchor_index, anchor_sum):
            anchor_depth[i] += v

    return anchor_depth


@njit
def _anchor_PET(pet_x, anchor_index, anchor_start, anchor_end):
    is_in_anchor = np.zeros(len(pet_x)) - 1

    for i, p in enumerate(pet_x):
        for idx, x, y in zip(anchor_index, anchor_start, anchor_end):
            if p >= x and p <= y:
                is_in_anchor[i] = idx

    return is_in_anchor


############## Given an interaction list, count number of PETs from the validpair dataset ############
def count_interaction_strength(vp, interactions):
    """
    interactions: bedpe format: chr1, x1, y1, chr2, x2, y2, ...
    """
    anchors = pd.DataFrame(
        {
            "chro": np.append(
                interactions.chr1.values.copy(), interactions.chr2.values.copy()
            ),
            "start": np.append(
                interactions.x1.values.copy(), interactions.x2.values.copy()
            ),
            "end": np.append(
                interactions.y1.values.copy(), interactions.y2.values.copy()
            ),
        }
    ).drop_duplicates(ignore_index=True)
    anchors["indexer"] = range(len(anchors))
    loop_counts = loop_PET_count(vp, anchors)

    # anchor 1D signal
    anchors["Depth"] = (
        anchor_depth_by_A2N_PETs(
            vp,
            anchors,
        ).values
        / (anchors.end.values - anchors.start.values)
        * 1000
    )

    I = interactions.merge(
        anchors, left_on=["chr1", "x1", "y1"], right_on=["chro", "start", "end"]
    ).indexer.values
    J = interactions.merge(
        anchors, left_on=["chr2", "x2", "y2"], right_on=["chro", "start", "end"]
    ).indexer.values

    counts = [
        loop_counts[i, j] if i <= j else loop_counts[j, i] for i, j in zip(I, J)
    ]
    return np.array(counts)


def count_interaction_anchor_depth(vp, interactions):
    """
    interactions: bedpe format: chr1, x1, y1, chr2, x2, y2, ...
    """
    anchors = pd.DataFrame(
        {
            "chro": np.append(
                interactions.chr1.values.copy(), interactions.chr2.values.copy()
            ),
            "start": np.append(
                interactions.x1.values.copy(), interactions.x2.values.copy()
            ),
            "end": np.append(
                interactions.y1.values.copy(), interactions.y2.values.copy()
            ),
        }
    ).drop_duplicates(ignore_index=True)
    anchors["indexer"] = range(len(anchors))

    # anchor 1D signal
    anchor_depth = (
        anchor_depth_by_A2N_PETs(
            vp,
            anchors,
        ).values
        / (anchors.end.values - anchors.start.values)
        * 1000
    )

    I = interactions.merge(
        anchors, left_on=["chr1", "x1", "y1"], right_on=["chro", "start", "end"]
    ).indexer.values
    J = interactions.merge(
        anchors, left_on=["chr2", "x2", "y2"], right_on=["chro", "start", "end"]
    ).indexer.values

    depth_i, depth_j = anchor_depth[I], anchor_depth[J]
    return depth_i, depth_j


############## Function to process anchors ##############
def genomic_bins(chro_size, s):
    gb = {"chro": [], "start": [], "end": []}
    with open(chro_size) as inp:
        for line in inp:
            chro, l = line.split()
            if "_" in chro:
                continue
            else:
                pos = list(range(0, int(l), s))
                if pos[-1] != int(l):
                    pos.append(int(l))

                gb["chro"].extend([chro] * (len(pos) - 1))
                gb["start"].extend(pos[:-1].copy())
                gb["end"].extend(pos[1:].copy())

    return pd.DataFrame(gb).sort_values(by=["chro", "start"], ignore_index=True)


def genomic_anchor_bins(genomic_bins, anchors, merge_adjacent=True):
    """
    Mask genomic bins overlapping with anchors as anchor_bins. `genomic_bins` and `anchors` are (chro, start) sorted bed format pd.DataFrame.
    """
    anchor_count = np.zeros(len(genomic_bins), dtype=int)
    for chro, chro_anchors in anchors.groupby(by="chro"):
        chro_bins_idx = np.arange(len(genomic_bins))[
            genomic_bins.chro.values == chro
        ]
        if sum(chro_bins_idx) == 0:
            continue
        #  gb[i-1] <= anchor_i < gb[i]
        assigned_bin = (
            np.searchsorted(
                genomic_bins.start.values[chro_bins_idx],
                chro_anchors.start.values,
                side="right",
            )
            - 1
        )
        anchor_count[chro_bins_idx[assigned_bin]] += 1

        #  gb[i-1] <= anchor_i < gb[i]
        assigned_bin = (
            np.searchsorted(
                genomic_bins.start.values[chro_bins_idx],
                chro_anchors.end.values,
                side="right",
            )
            - 1
        )
        anchor_count[chro_bins_idx[assigned_bin]] += 1

    if merge_adjacent:
        return (
            merge_anchors(genomic_bins.loc[anchor_count != 0], 0),
            anchor_count,
        )
    else:
        return (
            genomic_bins.loc[anchor_count != 0].copy().reset_index(drop=True),
            anchor_count,
        )


def merge_anchors_bins(genomic_bins):
    """
    merge adjacent anchor bins
    """
    chro, start, end, num_anchors = [], [], [], []
    for ch, chro_gbs in genomic_bins.groupby("chro"):
        anchor_count = chro_gbs.Num_Anchors.values
        (
            merged_anchor_count,
            merged_start,
            merged_end,
        ) = _merge_adjacent_anchor_bins(
            anchor_count, chro_gbs.start.values, chro_gbs.end.values
        )
        chro.extend([ch] * len(merged_start))
        start.extend(merged_start)
        end.extend(merged_end)
        num_anchors.extend(merged_anchor_count)

    return pd.DataFrame(
        {"chro": chro, "start": start, "end": end, "Num_Anchors": num_anchors}
    ).sort_values(by=["chro", "start"], ignore_index=True)


@njit
def _merge_adjacent_anchor_bins(anchor_count, start, end):
    n = len(anchor_count)
    merged_start, merged_end, merged_anchor_count = List(), List(), List()
    merged_start.append(start[0])
    merged_end.append(end[0])
    merged_anchor_count.append(anchor_count[0])
    for i in range(1, n):
        if anchor_count[i]:
            # current bin is an anchor
            if anchor_count[i - 1]:
                # last bin is anchor, merge
                merged_end[-1] = end[i]
                merged_anchor_count[-1] += anchor_count[i]
            else:
                merged_start.append(start[i])
                merged_end.append(end[i])
                merged_anchor_count.append(anchor_count[i])
        else:
            merged_start.append(start[i])
            merged_end.append(end[i])
            merged_anchor_count.append(anchor_count[i])
    return merged_anchor_count, merged_start, merged_end


def extend_anchors(anchors, size):
    """
    Extend anchors to size. Input anchors is a pandas dataframe encoding a bed format with exactly 3 columns: chro, start, end
    """

    center = (anchors.start + anchors.end) // 2
    half_size = size // 2
    return pd.DataFrame(
        {
            "chro": anchors.chro,
            "start": center - half_size,
            "end": center + half_size,
        }
    )


def merge_anchors(anchors, gap):
    """
    merge anchors within distance of `gap`
    """

    sorted_anchors = anchors.sort_values(
        by=["chro", "start"], ignore_index=True
    )
    chro, start, end = [], [], []
    curr_chro, curr_start, curr_end = "", 0, 0
    for _, rec in sorted_anchors.iterrows():
        if rec.chro != curr_chro:
            if curr_chro:
                chro.append(curr_chro)
                start.append(curr_start)
                end.append(curr_end)
            curr_chro, curr_start, curr_end = rec.chro, rec.start, rec.end
        else:
            if rec.start - curr_end > gap:
                # new rec and store last merged rec
                chro.append(curr_chro)
                start.append(curr_start)
                end.append(curr_end)
                # update
                curr_chro, curr_start, curr_end = rec.chro, rec.start, rec.end
            else:
                # merge the record
                curr_end = rec.end

    if end[-1] != curr_end:
        chro.append(curr_chro)
        start.append(curr_start)
        end.append(curr_end)

    logging.info(f"{len(anchors)} are merged to {len(chro)} with {gap} bp gap")
    return pd.DataFrame(
        {
            "chro": chro,
            "start": start,
            "end": end,
        }
    )


def process_peak_to_anchor_bins(
    peak_file,
    chro_size,
    format="macs2_narrow",
    resolution=2500,
    filter_qval=0.01,
    merge_adjacent=True,
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
        )
        peaks = peaks[peaks.iloc[:, 8] >= -np.log10(filter_qval)].iloc[:, 0:3]
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
    elif format == "bed":
        peaks = pd.read_csv(
            peak_file,
            sep="\t",
            comment="#",
            header=None,
        ).iloc[:, 0:3]
        peaks.columns = ["chro", "start", "end"]
        # peak_anchor_bins, _ = genomic_anchor_bins(gbs, peaks, merge_adjacent)
        gbs["Num_Anchors"] = genomic_anchor_bins(
            gbs, peaks, merge_adjacent=False
        )[1]
        # add up anchor number too
        gbs_merged = merge_anchors_bins(gbs)
    else:
        raise ValueError("Peak file format unknown")

    if merge_adjacent:
        return gbs_merged
    else:
        return gbs


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

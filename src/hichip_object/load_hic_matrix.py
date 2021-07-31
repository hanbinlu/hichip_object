import numpy as np
import pandas as pd
import ray, itertools
from numba import njit
from scipy.sparse import csc_matrix

# from numba.typed import List
# from ray.util.multiprocessing import Pool
# from scipy.sparse.csr import csr_matrix
# from model_ILD import _bin_x_along_y_eq_count


def build_hic_matrix(vp, gbins, parallel, chunk_size=500000):
    if any(gbins.index.values != np.arange(len(gbins))):
        raise ValueError("Anchors index must be RangeIndex")

    n = len(gbins)
    mtx = csc_matrix((n, n), dtype=int)
    process_jobs = []
    for job_id in range(parallel):
        process_jobs.append(
            _build_hic_matrix_kernel.remote(
                vp, gbins, chunk_size, job_id, parallel
            )
        )

    for v in ray.get(process_jobs):
        mtx += v

    return mtx


@ray.remote
def _build_hic_matrix_kernel(vp, gbins, chunk_size, job_id, parallel):
    n = len(gbins)
    mtx = csc_matrix((n, n), dtype=int)
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
        vp_df["I"] = np.empty(len(vp_df), dtype=int)
        vp_df["J"] = np.empty(len(vp_df), dtype=int)
        # coor to gbin id
        for chro, chro_vp in vp_df.groupby(by="chr1"):
            chro_gbins = gbins[gbins.chro == chro]
            if len(chro_gbins) == 0:
                vp_df.loc[chro_vp.index, "I"] = -1
                continue
            I_ = _assign_pos_to_bins(chro_vp.x1.values, chro_gbins.start.values)
            vp_df.loc[chro_vp.index, "I"] = chro_gbins.index.values[I_]

        for chro, chro_vp in vp_df.groupby(by="chr2"):
            chro_gbins = gbins[gbins.chro == chro]
            if len(chro_gbins) == 0:
                vp_df.loc[chro_vp.index, "J"] = -1
                continue
            J_ = _assign_pos_to_bins(chro_vp.x2.values, chro_gbins.start.values)
            vp_df.loc[chro_vp.index, "J"] = chro_gbins.index.values[J_]

        I, J = [], []
        for i, j in zip(vp_df.I.values, vp_df.J.values):
            if i == -1 or j == -1:
                continue
            else:
                if i > j:
                    i, j = j, i
                I.append(i)
                J.append(j)

        mtx += csc_matrix(
            (np.ones(len(I), dtype=int), (I, J)),
            shape=(n, n),
        )

    return mtx


@njit
def _assign_pos_to_bins(x, bin_pos):
    return np.searchsorted(bin_pos, x, side="right") - 1


# def putative_anchor_to_non_loops(gbs, inter_range=(5000, 2000000), parallel=20):
#     """
#     Label qualified anchor-non entries by value of interaction distance
#     """
#     if any(gbs.index.values != np.arange(len(gbs))):
#         raise ValueError("Anchors index must be RangeIndex")
#
#     n = len(gbs)
#     qualified_a2n_pairs = csc_matrix((n, n), dtype=int)
#
#     def _chro_a2n_dist_pairs(chro_bins, low, high):
#         centers = (chro_bins.start + chro_bins.end).values // 2
#         I, J, V = _anchor_to_non_dist_pairs(
#             centers,
#             chro_bins.index.values,
#             low,
#             high,
#             chro_bins.Num_Anchors.values != 0,
#         )
#         return csc_matrix(
#             (V, (I, J)),
#             shape=(n, n),
#         )
#
#     with Pool(processes=parallel) as pool:
#         for chro_result in pool.imap_unordered(
#             lambda x: _chro_a2n_dist_pairs(
#                 x[1], inter_range[0], inter_range[1]
#             ),
#             gbs.groupby("chro"),
#         ):
#             qualified_a2n_pairs += chro_result
#
#     return qualified_a2n_pairs
#
#
# @njit
# def _anchor_to_non_dist_pairs(centers, index, low, high, is_anchor):
#     I, J, V = List(), List(), List()
#     n = len(centers)
#
#     for i in range(n):
#         for j in range(i, n):
#             # is it anchor to non
#             if is_anchor[i] + is_anchor[j] != 1:
#                 continue
#             d = centers[j] - centers[i]
#             if d >= low and d <= high:
#                 I.append(index[i])
#                 J.append(index[j])
#                 V.append(d)
#
#             if d > high:
#                 break
#
#     return I, J, V
#
#
#
#
# def hic_background_matrix(
#     hic_mtx,
#     gbins,
#     inter_range=(5000, 2000000),
# ):
#     """
#     Only keep non-anchor contacts
#
#     `gbins`: continuous and sorted genomic bins in bed format and added anchor column
#     """
#     # bin raw vp data into triplet df
#     mtx = hic_mtx.tocoo()
#     df = pd.DataFrame(
#         {
#             "I": mtx.row,
#             "J": mtx.col,
#             "C": mtx.data,
#         }
#     )
#     # logging.info("Filtering contacts and add anchor info")
#     # keep intrachro and contacts meet inter range
#     intra_chro_df = df[gbins.chro.values[mtx.row] == gbins.chro.values[mtx.col]]
#     dist = (
#         gbins.start.values[intra_chro_df.J.values]
#         - gbins.start.values[intra_chro_df.I.values]
#     )
#     dist_qual_index = (dist >= inter_range[0]) & (dist <= inter_range[1])
#     dist_qual_df = intra_chro_df[dist_qual_index]
#     # non anchor contacts
#     non_anchor_contact_idx = (
#         gbins.Num_Anchors.values[dist_qual_df.I.values] == 0
#     ) & (gbins.Num_Anchors.values[dist_qual_df.J.values] == 0)
#
#     filtered_triplet = dist_qual_df[non_anchor_contact_idx]
#     n = hic_mtx.shape[0]
#
#     return csr_matrix(
#         (
#             filtered_triplet.C.values.copy(),
#             (
#                 filtered_triplet.I.values.copy(),
#                 filtered_triplet.J.values.copy(),
#             ),
#         ),
#         (n, n),
#     )
#
#
#
#
# @njit
# def _num_bin_pairs_for_dist_bins(bin_pos, dist_low, dist_high):
#     count, ysum = np.zeros_like(dist_low), np.zeros_like(dist_low)
#     n, max_dist = len(bin_pos), dist_high.max()
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             d = bin_pos[j] - bin_pos[i]
#             if d > max_dist:
#                 break
#             for k, (x, y) in enumerate(zip(dist_low, dist_high)):
#                 if d >= x and d < y:
#                     count[k] += 1
#                     ysum[k] += d
#
#     return count, ysum
#
#
# def _eq_occupancy_bin_along_feature(df, feature, nbins):
#     # based binning
#     df_to_bin = df.loc[df.C != 0]
#
#     def _sum_counts(df):
#         return df.C.sum()
#
#     model_data = _bin_x_along_y_eq_count(df_to_bin, feature, nbins, _sum_counts)
#     # expand to cover whole data range
#     # model_data.loc[0, "Low"] = df[feature].min()
#     # model_data.loc[model_data.index[-1], "High"] = df[feature].max() + 1
#
#     model_data["Sum_PETs"] = [
#         df_to_bin.C.loc[
#             (df_to_bin[feature] >= x) & (df_to_bin[feature] < y)
#         ].sum()
#         for x, y in zip(model_data.Low.values, model_data.High.values)
#     ]
#
#     return model_data
#
#
# # Model response Cij of HiChIP as Poisson Regression by explanatory variables IP efficient and Hi-C distant expected counts.
#
# ######### Hi-C scaling background buiding from non-anchor contacts ########
# def non_anchor_dist_scaling(
#     vp,
#     gbins,
#     nbins=200,
#     inter_range=(10000, 2000000),
#     chunk_size=500000,
#     parallel=10,
# ):
#     """
#     Expected counts as function of genomic distance from non-anchor contacts
#
#     `gbins`: continuous and sorted genomic bins in bed format and added anchor column
#     """
#     # bin raw vp data into triplet df
#     logging.info("Loading validpair data into Triplet dataframe")
#     mtx = build_hic_matrix(vp, gbins, parallel, chunk_size).tocoo()
#     df = pd.DataFrame(
#         {
#             "I": mtx.row,
#             "J": mtx.col,
#             "C": mtx.data,
#         }
#     )
#     logging.info("Filtering contacts and add anchor info")
#     # keep intrachro and contacts meet inter range
#     intra_chro_df = df[gbins.chro.values[mtx.row] == gbins.chro.values[mtx.col]]
#     dist = (
#         gbins.start.values[intra_chro_df.J.values]
#         - gbins.start.values[intra_chro_df.I.values]
#     )
#     dist_qual_index = (dist >= inter_range[0]) & (dist <= inter_range[1])
#     dist_qual_df = intra_chro_df[dist_qual_index].copy().reset_index(drop=True)
#     # I, J bin has anchors?
#     dist_qual_df["L"] = dist[dist_qual_index]
#     dist_qual_df["Anchor_I"] = gbins.Num_Anchors.values[dist_qual_df.I.values]
#     dist_qual_df["Anchor_J"] = gbins.Num_Anchors.values[dist_qual_df.J.values]
#
#     # non anchor contacts to equal occupancy binsgg
#     non_anchor_contact_idx = (dist_qual_df.Anchor_I == 0) & (
#         dist_qual_df.Anchor_J == 0
#     )
#     model_data = _eq_occupancy_bin_along_feature(
#         dist_qual_df[non_anchor_contact_idx], "L", nbins
#     )
#
#     # base on eq occupancy bin borders calculate possible pairs
#     logging.info("Counting possible bin pairs at different distance ranges")
#     all_possible_pairs = np.zeros(len(model_data), dtype=int)
#     all_possible_pairs_sum_dist = np.zeros(len(model_data), dtype=int)
#     with Pool(processes=parallel) as pool:
#         result = pool.map(
#             lambda x: _num_bin_pairs_for_dist_bins(
#                 x[1].start.values, model_data.Low.values, model_data.High.values
#             ),
#             gbins[gbins.Num_Anchors == 0].groupby(by="chro"),
#         )
#     for v in result:
#         all_possible_pairs += v[0]
#         all_possible_pairs_sum_dist += v[1]
#     # for _, chro_gb in gbins[gbins.Num_Anchors == 0].groupby(by="chro"):
#     #     all_possible_pairs += _num_bin_pairs_for_dist_bins(
#     #         chro_gb.start.values, model_data.Low.values, model_data.High.values
#     #     )
#     model_data["All_Items"] = all_possible_pairs
#     model_data["x"] = all_possible_pairs_sum_dist / all_possible_pairs
#     model_data["y"] = (
#         model_data.Sum_PETs / model_data.Sum_PETs.sum() / model_data.All_Items
#     )
#
#     return model_data
#

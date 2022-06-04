import logging

import numpy as np
import pandas as pd
from scipy.sparse import diags
from statsmodels.stats.multitest import multipletests

from .load_hic_matrix import build_hic_matrix
from .load_loop_data import (
    anchor_depth_by_A2N_PETs,
    anchor_depth_by_anchor_PETs,
    loop_PET_count,
    putative_loops_from_anchors,
    putative_p2np_loops,
)
from .model_ILD import _bin_x_along_y_eq_count


class GenericLoopCallHandler(object):
    """
    A generic HiChIP loop calling object provides following infrastructures:
    1. Process HiChIP data `vp` and anchors of interest into building blocks for modeling:
        a. `L`: List the putative loops within the distance range. Currently support the anchor/anchor pairs.
            The implements for anchor/any pairs and any/any are experimental.
        b. `D`: Calculate the coverage (depth) of given anchors `genomic_bins` by the HiChIP dataset `vp`.
        c. `C`: Count interaction numbers at the putative loops.
        d. Joining L, D, C into `loop_metric`
    2. Iteratively fit model: `fit_model` then `refit_model` till `eps` return by fitting smaller than threshold.
    3. Adjust p values, `P` field in the `loop_metric` added by fitting model.
    4. Write results.

    An inheritance for a concrete model should at least implement following functions:
    1. `fit_model`: fits `loop_metric` to the model and calculate the p values for putative loops.
    2. `re_fit`: remove significant loops from previous model fitting. Then, fit the model again.
            Besides calculating p values, it should also return summary statistics for `iteratively_fit_model` to terminate.
    """

    def __init__(
        self,
        vp,
        genomic_bins,
        inter_range=(5000, 2000000),
        parallel=20,
        anchor_depth_mode=2,
        get_hic_matrix=False,
    ):
        # genomic_bins are pd.DataFrame in bed format and are sorted
        # colnames ["chro", "start", "end"] (add check for colname obligation)
        # 4th column "Num_Anchors", non zero value indicates the correpondent gb has anchors
        if any(genomic_bins.index.values != np.arange(len(genomic_bins))):
            raise ValueError("Anchors index must be RangeIndex")
        self.genomic_bins = genomic_bins.copy()
        self.is_anchor = genomic_bins.Num_Anchors.values != 0
        self.anchors = (
            self.genomic_bins[self.is_anchor].copy().reset_index(drop=True)
        )

        # interaction distance to be considered
        self.dist_low, self.dist_high, self.parallel = (
            inter_range[0],
            inter_range[1],
            parallel,
        )
        # intra, dist qualified anchor pairs
        putative_loops = putative_loops_from_anchors(self.anchors, inter_range)

        # read interaction data into anchor-anchor sparse matrix
        logging.info("Loading loop PETs")
        loop_pet = loop_PET_count(vp, self.anchors)
        # drop loop_pets of unqualified anchor pairs
        loop_pet = loop_pet.multiply(putative_loops != 0)

        # count anchor depth. mode=0: summation on any/any PETs;
        # mode=1: on anchor/non-anchor PETs; mode=2: anchor/anchor PETs
        logging.info("Counting anchor depth")
        if anchor_depth_mode == 1:
            anchor_depth = (
                anchor_depth_by_A2N_PETs(
                    vp, self.anchors, parallel=parallel, vp_filter=inter_range
                ).values
                / (self.anchors.end.values - self.anchors.start.values)
                * 1000
            )
        elif anchor_depth_mode == 0:
            anchor_depth = (
                anchor_depth_by_anchor_PETs(
                    vp, self.anchors, parallel=parallel, vp_filter=inter_range
                ).values
                / (self.anchors.end.values - self.anchors.start.values)
                * 1000
            )
        elif anchor_depth_mode == 2:
            anchor_depth = loop_pet.sum(axis=0).A1 + loop_pet.sum(axis=1).A1

        sig_mean = anchor_depth[anchor_depth != 0].mean()
        anchor_depth /= sig_mean

        # mask 0 depth anchors
        mask_opt = diags(anchor_depth != 0, format="csc", dtype=int)
        putative_loops = mask_opt * putative_loops * mask_opt
        loop_pet = mask_opt * loop_pet * mask_opt

        if get_hic_matrix:
            # loading hic matrix
            self.contact_matrix = build_hic_matrix(
                vp, self.genomic_bins, self.parallel
            )
        # logging.info("Building Hi-C background model")
        # self.hic_background_model_data = non_anchor_dist_scaling(
        #     vp, genomic_bins, nbins, inter_range, parallel=parallel
        # )
        # self.hic_background_model = _spline_fit_model(
        #     self.hic_background_model_data, "x", "y"
        # )

        # organize loop_pet and putative_loops into metric table
        # loop_metric is data for all putative loops
        logging.info("Joining data")
        self.loop_metric = _compile_loop_metric(
            loop_pet, putative_loops, anchor_depth
        )
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()
        # self.loop_metric["BG"] = self.hic_background_model(
        #     self.loop_metric.L.values
        # )
        self.anchors["Depth"] = anchor_depth
        self.count_cut = self.loop_metric.C[self.loop_metric.C != 0].mean()

        logging.info(
            f"{len(self.anchors)} of anchors forms {len(self.loop_metric)} putative mid-range loops"
        )
        logging.info(
            f"{len(self.loop_metric[self.loop_metric.C != 0])} observed loops that contains {self.loop_metric.C.sum()} PETs (avg = {self.count_cut})"
        )

    @classmethod
    def P2NP_Loops(
        self,
        vp,
        genomic_bins,
        inter_range=(5000, 2_000_000),
        parallel=20,
    ):
        """Call significant loops from P2A bins"""
        # genomic_bins are pd.DataFrame in bed format and are sorted
        # colnames ["chro", "start", "end"] (add check for colname obligation)
        # 4th column "Num_Anchors", non zero value indicates the correpondent gb has anchors
        if any(genomic_bins.index.values != np.arange(len(genomic_bins))):
            raise ValueError("Anchors index must be RangeIndex")
        self.genomic_bins = genomic_bins.copy()

        # interaction distance to be considered
        self.dist_low, self.dist_high, self.parallel = (
            inter_range[0],
            inter_range[1],
            parallel,
        )
        # intra, dist qualified anchor pairs
        putative_loops = putative_p2np_loops(
            self.genomic_bins, inter_range, parallel
        )

        # read interaction data into anchor-anchor sparse matrix
        logging.info("Loading loop PETs")
        # loading hic matrix
        self.contact_matrix = build_hic_matrix(
            vp, self.genomic_bins, self.parallel
        )
        # drop loop_pets of unqualified anchor pairs
        loop_pet = self.contact_matrix.multiply(putative_loops != 0)
        # count bin depth by all vp
        gb_anchor_depth = (
            self.contact_matrix.sum(axis=0).A1
            + self.contact_matrix.sum(axis=1).A1
        )

        # anchor and non-anchor signal mean
        # anchor_bin depth is the sum of interaction with non anchor bin
        # non anchor bin depth is the sum of anchor bin
        is_anchor = (genomic_bins.Num_Anchors != 0).values
        gb_anchor_depth[is_anchor] /= gb_anchor_depth[is_anchor].mean()
        gb_anchor_depth[~is_anchor] /= gb_anchor_depth[~is_anchor].mean()

        # mask 0 depth anchors
        mask_opt = diags(gb_anchor_depth != 0, format="csc", dtype=int)
        putative_loops = mask_opt * putative_loops * mask_opt
        loop_pet = mask_opt * loop_pet * mask_opt

        # organize loop_pet and putative_loops into metric table
        # loop_metric is data for all putative loops
        logging.info("Joining data")
        self.loop_metric = _compile_loop_metric(
            loop_pet, putative_loops, gb_anchor_depth, is_anchor
        )
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()

        self.genomic_bins["Depth"] = gb_anchor_depth
        self.count_cut = self.loop_metric.C[self.loop_metric.C != 0].mean()

        logging.info(
            f"{sum(self.genomic_bins.Num_Anchors != 0)} of anchors forms {len(self.loop_metric)} putative mid-range P2A loops"
        )
        logging.info(
            f"{len(self.loop_metric[self.loop_metric.C != 0])} observed loops that contains {self.loop_metric.C.sum()} PETs (avg = {self.count_cut})"
        )

    @classmethod
    def A2A_Loops(
        self,
        vp,
        genomic_bins,
        inter_range=(5000, 2_000_000),
        parallel=20,
    ):
        """Call significant loops from A2A bins"""
        # genomic_bins are pd.DataFrame in bed format and are sorted
        # colnames ["chro", "start", "end"] (add check for colname obligation)
        # 4th column "Num_Anchors", non zero value indicates the correpondent gb has anchors
        if any(genomic_bins.index.values != np.arange(len(genomic_bins))):
            raise ValueError("Anchors index must be RangeIndex")
        self.genomic_bins = genomic_bins.copy()

        # interaction distance to be considered
        self.dist_low, self.dist_high, self.parallel = (
            inter_range[0],
            inter_range[1],
            parallel,
        )
        # intra, dist qualified anchor pairs
        fake_anchors = genomic_bins.copy()
        fake_anchors["Num_Anchors"] += 1
        putative_loops = putative_loops_from_anchors(
            fake_anchors, inter_range, parallel
        )

        # read interaction data into anchor-anchor sparse matrix
        logging.info("Loading loop PETs")
        # loading hic matrix
        self.contact_matrix = build_hic_matrix(
            vp, self.genomic_bins, self.parallel
        )
        # drop loop_pets of unqualified anchor pairs
        loop_pet = self.contact_matrix.multiply(putative_loops != 0)
        # count bin depth by all vp
        gb_anchor_depth = (
            self.contact_matrix.sum(axis=0).A1
            + self.contact_matrix.sum(axis=1).A1
        )

        # anchor and non-anchor signal mean
        # anchor_bin depth is the sum of interaction with non anchor bin
        # non anchor bin depth is the sum of anchor bin
        gb_anchor_depth /= gb_anchor_depth[gb_anchor_depth != 0].mean()

        # mask 0 depth anchors
        mask_opt = diags(gb_anchor_depth != 0, format="csc", dtype=int)
        putative_loops = mask_opt * putative_loops * mask_opt
        loop_pet = mask_opt * loop_pet * mask_opt

        # organize loop_pet and putative_loops into metric table
        # loop_metric is data for all putative loops
        logging.info("Joining data")
        self.loop_metric = _compile_loop_metric(
            loop_pet, putative_loops, gb_anchor_depth
        )
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()

        self.genomic_bins["Depth"] = gb_anchor_depth
        self.count_cut = self.loop_metric.C[self.loop_metric.C != 0].mean()

        logging.info(
            f"{sum(self.genomic_bins.Num_Anchors != 0)} of anchors forms {len(self.loop_metric)} putative mid-range P2A loops"
        )
        logging.info(
            f"{len(self.loop_metric[self.loop_metric.C != 0])} observed loops that contains {self.loop_metric.C.sum()} PETs (avg = {self.count_cut})"
        )

    def iteratively_fit_model(
        self,
        *args,
        **kwargs,
    ):
        ic, s, eps = 1, 1e8, 1
        while s >= eps:
            logging.info(f"========Iter {ic}========")
            if ic == 1:
                self.fit_model(*args, **kwargs)
            else:
                s = self.re_fit(*args, **kwargs)
                logging.info(f"Sum square changes of E: {s}")
            ic += 1

    def fit_model(self):
        return NotImplemented

    def re_fit(self):
        return NotImplemented

    def calc_qval(self):
        self.interaction_statistics["Q"] = multipletests(
            self.interaction_statistics.P.values,
            1,
            "fdr_bh",
        )[1]

        # keep loop_mertric data untouch
        significant_interactions = (self.interaction_statistics.Q <= 0.1) & (
            self.interaction_statistics.C >= self.count_cut
        )
        logging.info(
            f"{sum(significant_interactions)} interactions were called at qval <= 0.1, count >= {self.count_cut}"
        )

    def estimate_mean_var(
        self,
        loop_metric_,
        nbins,
        compute_grid_stats=True,
        return_loop_metric=False,
    ):
        """
        Create L, D grid. For loops in each grid with similar L and D, assume they were generated by same distribution.
        """
        # equal-bin-number meta bin by "D", "L"
        loop_metric = loop_metric_.copy()
        borders_L = _bin_x_along_y_eq_count(loop_metric, "L", nbins, len)
        borders_D = _bin_x_along_y_eq_count(loop_metric, "D", nbins, len)

        logging.info("Assigning loops to meta group")
        loop_metric["L_BID"] = (
            np.searchsorted(
                borders_L.Low.values, loop_metric.L.values, side="right"
            )
            - 1
        )
        loop_metric["D_BID"] = (
            np.searchsorted(
                borders_D.Low.values, loop_metric.D.values, side="right"
            )
            - 1
        )

        if compute_grid_stats:
            logging.info("Inferring mean-var relation")
            grid_data = loop_metric.groupby(["L_BID", "D_BID"])
            grid_means = grid_data.C.mean().values
            grid_var = grid_data.C.var().values
            mask = (grid_means == 0) | (grid_var == 0)
            # inflation GLM infered probability
            self.grid_mean, self.grid_var = (
                grid_means[~mask],
                grid_var[~mask],
            )

        if return_loop_metric:
            return loop_metric

    def write_interaction_statistics(self, f, count=0, q=1):
        # build pandas table to write
        df = pd.concat(
            [
                self.anchors.loc[
                    self.interaction_statistics.I.values
                ].reset_index()[["chro", "start", "end"]],
                self.anchors.loc[
                    self.interaction_statistics.J.values
                ].reset_index()[["chro", "start", "end"]],
                self.interaction_statistics[["C", "Q"]].reset_index()[
                    ["C", "Q"]
                ],
            ],
            axis=1,
        )
        df.columns = ["chr1", "x1", "y1", "chr2", "x2", "y2", "counts", "qval"]
        df[(df.counts >= count) & (df.qval <= q)].to_csv(
            f, sep="\t", header=None, index=None
        )


def _compile_loop_metric(
    loop_pet, putative_loops, anchor_depth, is_anchor=None
):
    """
    Integrate sparse matrix data `loop_pet`, `dist_anchor_pairs`, `product_depth` into one DataFrame
    """
    putative_loops_data = putative_loops.tocoo()

    # group the data together. np.array([I, J, counts, features])
    if is_anchor is None:
        df = pd.DataFrame(
            {
                "I": putative_loops_data.row,
                "J": putative_loops_data.col,
                "L": putative_loops_data.data,
                "D_i": anchor_depth[putative_loops_data.row],
                "D_j": anchor_depth[putative_loops_data.col],
                "D": anchor_depth[putative_loops_data.row]
                * anchor_depth[putative_loops_data.col],
                # add count values; only dist qualified anchor-anchor pairs
                "C": [
                    loop_pet[i, j]
                    for i, j in zip(
                        putative_loops_data.row, putative_loops_data.col
                    )
                ],
            }
        ).sort_values(by=["L", "C"], ignore_index=True)
    else:
        # D_i as anchor, D_j non_anchor, no product depth
        # only use for p2np mode
        is_anchor_i, is_anchor_j = (
            is_anchor[putative_loops_data.row],
            is_anchor[putative_loops_data.col],
        )
        if not all(np.logical_xor(is_anchor_i, is_anchor_j)):
            raise ValueError("P2NP mode only")

        I = (
            putative_loops_data.row * is_anchor_i
            + putative_loops_data.col * is_anchor_j
        )
        J = putative_loops_data.row * np.logical_not(
            is_anchor_i
        ) + putative_loops_data.col * np.logical_not(is_anchor_j)

        df = pd.DataFrame(
            {
                "I": I,
                "J": J,
                "L": putative_loops_data.data,
                "D_i": anchor_depth[I],
                "D_j": anchor_depth[J],
                "D": anchor_depth[putative_loops_data.row]
                * anchor_depth[putative_loops_data.col],
                # add count values; only dist qualified anchor-anchor pairs
                "C": [
                    loop_pet[i, j]
                    for i, j in zip(
                        putative_loops_data.row, putative_loops_data.col
                    )
                ],
            }
        ).sort_values(by=["L", "C"], ignore_index=True)

    return df

import numpy as np
import pandas as pd
import logging
from ray.util.multiprocessing import Pool
from numba import njit
from scipy.sparse import diags
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
import matplotlib.pyplot as plt
from load_loop_data import (
    loop_PET_count,
    putative_loops_from_anchors,
    # anchor_depth_by_anchor_PETs,
    anchor_depth_by_A2N_PETs,
)
from model_ILD import build_feature_PET_models, _spline_fit_model
from load_hic_matrix import (
    build_hic_matrix,
    hic_background_matrix,
    putative_anchor_to_non_loops,
    non_anchor_dist_scaling,
)


class Loop_MANGO:
    """
    Loop is defined as a pair of anchor. Loop object is to process the given anchors to putative loops calculated their metric from interaction data.
    """

    def __init__(
        self, anchors, vp, nbins, inter_range=(5000, 2000000), parallel=20
    ):
        # anchors are pd.DataFrame in bed format and are sorted
        # colnames ["chro", "start", "end"] (add check for colname obligation)
        if any(anchors.index.values != np.arange(len(anchors))):
            raise ValueError("Anchors index must be RangeIndex")
        self.anchors = anchors.copy()

        # interaction distance to be considered
        self.dist_low, self.dist_high = inter_range
        self.nbins = nbins
        # intra, dist qualified anchor pairs
        putative_loops = putative_loops_from_anchors(anchors, inter_range)

        # read interaction data into anchor-anchor sparse matrix
        logging.info("Loading loop PETs")
        loop_pet = loop_PET_count(vp, anchors)
        # drop loop_pets of unqualified anchor pairs
        loop_pet = loop_pet.multiply(putative_loops != 0)

        # count anchor depth using anchor-non PETs
        # anchor_depth = loop_pet.sum(axis=0).A1 + loop_pet.sum(axis=1).A1
        logging.info("Counting anchor depth")
        anchor_depth = (
            anchor_depth_by_A2N_PETs(
                vp, anchors, parallel=parallel, vp_filter=True
            ).values
            / (anchors.end.values - anchors.start.values)
            * 1000
        )

        # mask 0 depth anchors; notice that loop_pet already masked
        mask_opt = diags(anchor_depth != 0, format="csc", dtype=int)
        putative_loops = mask_opt * putative_loops * mask_opt

        # organize loop_pet and putative_loops into pd.DataFrame structure
        # loop_metric is data of all putative loops
        self.loop_metric = _compile_loop_metric(
            loop_pet, putative_loops, anchor_depth
        )
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()
        self.anchors["Depth"] = anchor_depth
        self.count_cut = self.interaction_statistics.C.mean()

        logging.info(
            f"{len(anchors)} of anchors forms {len(self.loop_metric)} putative mid-range loops"
        )
        logging.info(
            f"{len(self.loop_metric[self.loop_metric.C != 0])} observed loops that contains {self.loop_metric.C.sum()} PETs (avg = {self.count_cut})"
        )

    def iteratively_fit_model(self, qval=0.2, eps=1, plot_model="END"):
        ic, s = 1, 1e8
        while s >= eps:
            logging.info(f"========Iter {ic}========")
            if ic == 1:
                self.fit_model()
            else:
                s = self.re_fit_model_removing_interactions(qval)
                logging.info(f"Sum square changes of E: {s}")
                if plot_model == "ALL":
                    plt.show()
                elif s >= eps:
                    plt.close()
            ic += 1

            sig_inter = (self.interaction_statistics.Q <= 0.1) & (
                self.interaction_statistics.C >= self.count_cut
            )
            logging.info(
                f"Detected {sum(sig_inter)} interactions at FDR < 0.1, Count >= {self.count_cut}"
            )

        logging.info("Finished")
        plt.show()

    def fit_model(self, plot_model=True):
        logging.info("Fitting model")
        self.build_dist_model(plot_model)
        self.build_depth_model(plot_model)
        self.calculate_loop_significancy()
        logging.info("Finished")

    def re_fit_model_removing_interactions(
        self, qval=0.05, count=0, plot_model=True
    ):
        if "Q" not in self.interaction_statistics.columns:
            raise KeyError("Run fit_model first")

        # keep loop_mertric data untouch
        potential_interactions = (self.interaction_statistics.Q <= qval) & (
            self.interaction_statistics.C >= count
        )
        logging.info(
            f"{sum(potential_interactions)} interactions were adjusted by averaging non-interaction neighborhood to re-build the background model"
        )

        logging.info("Adjusting and re-fitting model")
        dist_y_adjust = []
        for _, row in self.dist_model_data.iterrows():
            candidate_interactions = self.interaction_statistics.loc[
                potential_interactions
                & (self.interaction_statistics.L >= row.Low)
                & (self.interaction_statistics.L < row.High)
            ]

            backgound_sum_pet = row.Sum_PETs - candidate_interactions.C.sum()
            background_mean = backgound_sum_pet / (
                row.Num_NZ_Items - len(candidate_interactions)
            )
            dist_y_adjust.append(
                backgound_sum_pet
                + background_mean * len(candidate_interactions)
            )

        dist_y_adjust = np.array(dist_y_adjust)
        self.dist_model_data["yAdjust"] = dist_y_adjust / dist_y_adjust.sum()
        self.dist_PET_model = _spline_fit_model(
            self.dist_model_data, "x", "yAdjust"
        )

        depth_y_adjust = []
        for _, row in self.depth_model_data.iterrows():
            candidate_interactions = self.interaction_statistics.loc[
                potential_interactions
                & (self.interaction_statistics.D >= row.Low)
                & (self.interaction_statistics.D < row.High)
            ]

            backgound_sum_pet = row.Sum_PETs - candidate_interactions.C.sum()
            background_mean = backgound_sum_pet / (
                row.Num_NZ_Items - len(candidate_interactions)
            )
            depth_y_adjust.append(
                backgound_sum_pet
                + background_mean * len(candidate_interactions)
            )

        depth_y_adjust = np.array(depth_y_adjust)
        self.depth_model_data["yAdjust"] = depth_y_adjust / depth_y_adjust.sum()
        self.depth_PET_model = _spline_fit_model(
            self.depth_model_data, "x", "yAdjust"
        )

        if plot_model:
            fig, axs = plt.subplots(ncols=2, nrows=1)
            axs[0].scatter(
                np.log10(self.dist_model_data.x), self.dist_model_data.yAdjust
            )
            axs[0].scatter(
                np.log10(self.dist_model_data.x),
                self.dist_model_data.y,
                marker="x",
            )
            axs[0].plot(
                np.log10(self.dist_model_data.x),
                self.dist_PET_model(self.dist_model_data.x),
            )
            axs[0].set_title("L model")

            axs[1].scatter(
                np.log10(self.depth_model_data.x), self.depth_model_data.yAdjust
            )
            axs[1].scatter(
                np.log10(self.depth_model_data.x),
                self.depth_model_data.y,
                marker="x",
            )
            axs[1].plot(
                np.log10(self.depth_model_data.x),
                self.depth_PET_model(self.depth_model_data.x),
            )
            axs[1].set_title("D model")

        eps = self.calculate_loop_significancy()
        return eps

    def build_dist_model(self, plot_model=True):
        (
            self.dist_model_data,
            self.dist_PET_model,
            self.dist_NLOOP_model,
        ) = build_feature_PET_models(
            self.loop_metric, "L", self.nbins, plot_model
        )

    def build_depth_model(self, plot_model=True):
        (
            self.depth_model_data,
            self.depth_PET_model,
            self.depth_NLOOP_model,
        ) = build_feature_PET_models(
            self.loop_metric, "D", self.nbins, plot_model
        )

    def calculate_loop_significancy(self, parallel=20):
        if not hasattr(self, "dist_PET_model"):
            raise KeyError(
                "Please build_dist_model before calling calculate_loop_significancy"
            )
        if not hasattr(self, "depth_PET_model"):
            raise KeyError(
                "Please build_depth_model before calling calculate_loop_significancy"
            )

        N, n = self.loop_metric.C.sum(), len(self.loop_metric)
        # probability of generating PETs with specific L or D
        logging.info("Infering probability of PET number")
        prob_L, prob_D = self.dist_PET_model(
            self.interaction_statistics.L.values
        ), self.depth_PET_model(self.interaction_statistics.D.values)
        prob_L = np.clip(
            prob_L, self.dist_model_data.y.min(), self.dist_model_data.y.max()
        )
        prob_D = np.clip(
            prob_D, self.depth_model_data.y.min(), self.depth_model_data.y.max()
        )
        # probaility of generating PETs with combined L and D assuming L, D independent
        prob_LD = prob_L * prob_D

        # expected number of loops (anchor pairs) with combined L and D
        # they are considered to have same probability
        logging.info("Infering probability of loop number")
        prob_nloop_L = self.dist_NLOOP_model(
            self.interaction_statistics.L.values
        )
        prob_nloop_D = self.depth_NLOOP_model(
            self.interaction_statistics.D.values
        )
        prob_nloop_L = np.clip(
            prob_nloop_L,
            self.dist_model_data.yItems.min(),
            self.dist_model_data.yItems.max(),
        )
        prob_nloop_D = np.clip(
            prob_nloop_D,
            self.depth_model_data.yItems.min(),
            self.depth_model_data.yItems.max(),
        )
        nloop_LD = prob_nloop_L * prob_nloop_D * n

        # probability of a PET linking a loop with L and D
        p = prob_LD / nloop_LD
        self.interaction_statistics["p"] = p
        E = N * p
        eps = None
        if "E" in self.interaction_statistics.columns:
            # changes of adjusting
            eps = np.square(E - self.interaction_statistics.E.values).sum()
        self.interaction_statistics["E"] = E

        # parallel speed up
        logging.info("Calculating and correcting P values")

        with Pool(processes=parallel) as pool:
            self.interaction_statistics["P"] = pool.map(
                lambda x: binom.sf(x[0] - 1, N, x[1]),
                self.interaction_statistics[["C", "p"]].values,
            )
        # self.interaction_statistics["P"] = np.apply_along_axis(
        #    lambda x: binom.sf(x[0] - 1, N, x[1]),
        #    axis=1,
        #    arr=self.interaction_statistics[["C", "p"]].values,
        # )

        self.interaction_statistics["Q"] = multipletests(
            self.interaction_statistics.P.values,
            1,
            "fdr_bh",
        )[1]

        return eps

    def write_interaction_statistics(self, f):
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
        df.to_csv(f, sep="\t", header=None, index=None)


def _compile_loop_metric(loop_pet, putative_loops, anchor_depth):
    """
    Integrate sparse matrix data `loop_pet`, `dist_anchor_pairs`, `product_depth` into one DataFrame
    """
    putative_loops_data = putative_loops.tocoo()

    # group the data together. np.array([I, J, counts, features])
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

    return df


class Loop_MANGO_A2N:
    """
    Loop is defined as a pair of anchor. Loop object is to process the given anchors to putative loops calculated their metric from interaction data.
    """

    def __init__(
        self,
        vp,
        genomic_bins,
        nbins,
        inter_range=(5000, 2000000),
        parallel=20,
    ):
        # genomic_bins are pd.DataFrame in bed format and are sorted
        # colnames ["chro", "start", "end"] (add check for colname obligation)
        # 4th column "Num_Anchors", non zero value indicates the correpondent gb has anchors
        if any(genomic_bins.index.values != np.arange(len(genomic_bins))):
            raise ValueError("Anchors index must be RangeIndex")
        self.genomic_bins = genomic_bins.copy()
        self.is_anchor = genomic_bins.Num_Anchors.values != 0

        # interaction distance to be considered
        self.dist_low, self.dist_high = inter_range
        self.nbins = nbins

        # load hic matrix
        logging.info("Loading ValidPair data to contact matrix")
        hic_mtx = build_hic_matrix(vp, self.genomic_bins, parallel)
        hic_bg_mtx = hic_background_matrix(hic_mtx, genomic_bins, inter_range)

        # Determine putative anchor-to-non pairs within distance range
        logging.info(
            f"Determine putative A2N bin pairs within {inter_range[0]} - {inter_range[1]} bp"
        )
        putative_a2n_loops = putative_anchor_to_non_loops(
            self.genomic_bins, inter_range, parallel
        )

        # drop loop_pets of unqualified anchor pairs
        logging.info("Dropping data from zero depth genomic bins")
        hic_mtx = hic_mtx.multiply(putative_a2n_loops != 0)
        # count genoimc bins' depth using anchor-non PETs
        gbins_depth = hic_mtx.sum(axis=0).A1 + hic_mtx.sum(axis=1).A1
        # non anchor depth set to be sum of N2N contacts
        gbins_depth[~self.is_anchor] = (
            hic_bg_mtx.sum(0).A1 + hic_bg_mtx.sum(1).A1
        )[~self.is_anchor]

        # mask 0 depth anchors
        mask_opt = diags(gbins_depth != 0, format="csc", dtype=int)
        putative_a2n_loops = mask_opt * putative_a2n_loops * mask_opt
        hic_mtx = mask_opt * hic_mtx * mask_opt

        # organize loop_pet and putative_loops into pd.DataFrame structure
        # loop_metric is data of all putative loops
        logging.info("Compiling loop data: distance, depth, contact counts")
        self.loop_metric = _compile_a2n_loop_metric(
            hic_mtx, putative_a2n_loops, gbins_depth, self.is_anchor
        )
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()
        self.count_cut = self.interaction_statistics.C.mean()

        logging.info(
            f"{sum(genomic_bins.Num_Anchors != 0)} of anchors forms {len(self.loop_metric)} putative A2N mid-range loops"
        )
        logging.info(
            f"{len(self.loop_metric[self.loop_metric.C != 0])} observed loops that contains {self.loop_metric.C.sum()} PETs (avg = {self.count_cut})"
        )

    def fit_model(self, plot_model=True):
        logging.info("Fitting model")
        self.build_dist_model(plot_model)
        self.build_depth_model(plot_model)
        self.calculate_loop_significancy()
        logging.info("Finished")

    def build_dist_model(self, plot_model=True):
        (
            self.dist_model_data,
            self.dist_PET_model,
            self.dist_NLOOP_model,
        ) = build_feature_PET_models(
            self.loop_metric, "L", self.nbins, plot_model
        )

    def build_depth_model(self, plot_model=True):
        (
            self.depth_model_data,
            self.depth_PET_model,
            self.depth_NLOOP_model,
        ) = build_feature_PET_models(
            self.loop_metric, "D", self.nbins, plot_model
        )

    def calculate_loop_significancy(self, parallel=20):
        if not hasattr(self, "dist_PET_model"):
            raise KeyError(
                "Please build_dist_model before calling calculate_loop_significancy"
            )
        if not hasattr(self, "depth_PET_model"):
            raise KeyError(
                "Please build_depth_model before calling calculate_loop_significancy"
            )

        N, n = self.loop_metric.C.sum(), len(self.loop_metric)
        # probability of generating PETs with specific L or D
        logging.info("Infering probability of PET number")
        prob_L, prob_D = self.dist_PET_model(
            self.interaction_statistics.L.values
        ), self.depth_PET_model(self.interaction_statistics.D.values)
        prob_L = np.clip(
            prob_L, self.dist_model_data.y.min(), self.dist_model_data.y.max()
        )
        prob_D = np.clip(
            prob_D, self.depth_model_data.y.min(), self.depth_model_data.y.max()
        )
        # probaility of generating PETs with combined L and D assuming L, D independent
        prob_LD = prob_L * prob_D

        # expected number of loops (anchor pairs) with combined L and D
        # they are considered to have same probability
        logging.info("Infering probability of loop number")
        prob_nloop_L = self.dist_NLOOP_model(
            self.interaction_statistics.L.values
        )
        prob_nloop_D = self.depth_NLOOP_model(
            self.interaction_statistics.D.values
        )
        prob_nloop_L = np.clip(
            prob_nloop_L,
            self.dist_model_data.yItems.min(),
            self.dist_model_data.yItems.max(),
        )
        prob_nloop_D = np.clip(
            prob_nloop_D,
            self.depth_model_data.yItems.min(),
            self.depth_model_data.yItems.max(),
        )
        nloop_LD = prob_nloop_L * prob_nloop_D * n

        # probability of a PET linking a loop with L and D
        p = prob_LD / nloop_LD
        self.interaction_statistics["p"] = p
        E = N * p
        eps = None
        if "E" in self.interaction_statistics.columns:
            # changes of adjusting
            eps = np.square(E - self.interaction_statistics.E.values).sum()
        self.interaction_statistics["E"] = E

        # parallel speed up
        logging.info("Calculating and correcting P values")

        with Pool(processes=parallel) as pool:
            self.interaction_statistics["P"] = pool.map(
                lambda x: binom.sf(x[0] - 1, N, x[1]),
                self.interaction_statistics[["C", "p"]].values,
            )
        # self.interaction_statistics["P"] = np.apply_along_axis(
        #    lambda x: binom.sf(x[0] - 1, N, x[1]),
        #    axis=1,
        #    arr=self.interaction_statistics[["C", "p"]].values,
        # )

        self.interaction_statistics["Q"] = multipletests(
            self.interaction_statistics.P.values,
            1,
            "fdr_bh",
        )[1]

        return eps

    def write_interaction_statistics(self, f):
        # build pandas table to write
        df = pd.concat(
            [
                self.genomic_bins.loc[
                    self.interaction_statistics.I.values
                ].reset_index()[["chro", "start", "end"]],
                self.genomic_bins.loc[
                    self.interaction_statistics.J.values
                ].reset_index()[["chro", "start", "end"]],
                self.interaction_statistics[["C", "Q"]].reset_index()[
                    ["C", "Q"]
                ],
            ],
            axis=1,
        )
        df.columns = ["chr1", "x1", "y1", "chr2", "x2", "y2", "counts", "qval"]
        df.to_csv(f, sep="\t", header=None, index=None)


def _compile_a2n_loop_metric(
    hic_mtx, putative_a2n_loops, gbins_depth, is_anchor
):
    """
    Integrate sparse matrix data `hic_mtx`, `dist_a2n_loop`, `anchor_depth` into one DataFrame
    """
    putative_a2n_loops_data = putative_a2n_loops.tocoo()

    # make sure I is anchor
    I, J = (
        putative_a2n_loops_data.row.copy(),
        putative_a2n_loops_data.col.copy(),
    )
    _anchor_as_bin_i(I, J, is_anchor)

    # upgrade hic_mtx same non zero structure in order to speed up generating C
    count_data = (hic_mtx + putative_a2n_loops).tocoo()
    # make sure underlying data is in same order
    if all(count_data.row - putative_a2n_loops_data.row == 0) and all(
        count_data.col - putative_a2n_loops_data.col == 0
    ):
        C = count_data.data - putative_a2n_loops_data.data

    # group the data together. np.array([I, J, counts, features])
    df = pd.DataFrame(
        {
            "I": I,
            "J": J,
            "L": putative_a2n_loops_data.data,
            "D_i": gbins_depth[I],
            "D_j": gbins_depth[J],
            "D": gbins_depth[I] * gbins_depth[J],
            # add count values; only dist qualified anchor-anchor pairs
            "C": C,
        }
    ).sort_values(by=["L", "C"], ignore_index=True)

    return df


@njit
def _anchor_as_bin_i(I, J, is_anchor):
    for k in range(len(I)):
        i, j = I[k], J[k]
        if is_anchor[j]:
            I[k], J[k] = j, i

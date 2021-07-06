import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.sparse import diags
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import poisson
from statsmodels.distributions import zipoisson
from patsy import dmatrix
from load_loop_data import (
    loop_PET_count,
    putative_loops_from_anchors,
    anchor_depth_by_anchor_PETs,
    anchor_depth_by_A2N_PETs,
)
from mango_loop_model import _compile_loop_metric
from model_ILD import _bin_x_along_y_eq_count

# from load_hic_matrix import non_anchor_dist_scaling
# from model_ILD import _spline_fit_model


def _zipoisson_p_vals(model_res, counts, exog, exog_infl):
    model, params = model_res.model, model_res.params
    params_infl = params[: model.k_inflate]
    params_main = params[model.k_inflate :]

    # base on fitted model, predict sf
    if len(exog_infl.shape) < 2:
        w = np.atleast_2d(model.model_infl.predict(params_infl, exog_infl))[
            :, None
        ]
    else:
        w = model.model_infl.predict(params_infl, exog_infl)[:, None]

    w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps).flatten()
    mu = model.model_main.predict(params_main, exog)[:, None].flatten()

    return zipoisson.sf(counts, mu, w), w, (1 - w) * mu


class Loop_ZIP:
    """
    The object processes the given anchors to putative loops calculated their metric from interaction data.
    Note: loop is a term used for a pair of anchor regardless whether it is linked by PETs or significant.
        In contrast, interaction is a term used for significant loops.
    """

    def __init__(
        self,
        vp,
        genomic_bins,
        inter_range=(5000, 2000000),
        parallel=20,
        anchor_depth_mode=1,
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

        # count anchor depth. mode=0: any anchor PETs;
        # mode=1: anchor to non PETs; mode=2: anchor to anchor PETs
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

    def iteratively_fit_model(
        self,
        model_distri,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        exog_infl_formula="1",
        disp_glm_summary=True,
        eps=1,
    ):
        ic, s = 1, 1e8
        while s >= eps:
            logging.info(f"========Iter {ic}========")
            if ic == 1:
                self.fit_model(
                    model_distri,
                    exog_model_formula,
                    exog_infl_formula,
                    disp_glm_summary,
                )
            else:
                s = self.re_fit(
                    model_distri,
                    exog_model_formula,
                    exog_infl_formula,
                    disp_glm_summary,
                )
                logging.info(f"Sum square changes of E: {s}")
            ic += 1

    def fit_model(
        self,
        model_distri,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        exog_infl_formula="1",
        disp_glm_summary=True,
    ):
        self.model_distri = model_distri
        logging.info(f"Fitting data to {model_distri} model")
        # only used for zero_inflated model
        exog_infl = dmatrix(exog_infl_formula, self.loop_metric).view()

        if model_distri == "poisson":
            self.glm_res = sm.Poisson.from_formula(
                formula=exog_model_formula,
                data=self.loop_metric,
            ).fit_regularized(maxiter=500, disp=0)

            self.loop_metric["E"] = self.glm_res.predict(self.loop_metric)
            self.loop_metric["P"] = poisson.sf(
                self.loop_metric.C.values - 1,
                self.loop_metric.E.values,
            )
        elif model_distri == "zipoisson":
            # fit distri
            self.glm_res = sm.ZeroInflatedPoisson.from_formula(
                formula=exog_model_formula,
                data=self.loop_metric,
                exog_infl=exog_infl,
            ).fit_regularized(maxiter=500, disp=0)

            # predict
            (
                self.loop_metric["P"],
                self.loop_metric["P_Infl"],
                self.loop_metric["E"],
            ) = _zipoisson_p_vals(
                self.glm_res,
                self.loop_metric.C.values - 1,
                dmatrix(
                    exog_model_formula.split("~")[1], self.loop_metric
                ).view(),
                exog_infl,
            )
        # elif model_distri == "nbinom":
        #     # assume commonly used NB2
        #     self.glm_res = sm.GLM.from_formula(
        #         formula=exog_model_formula,
        #         data=self.loop_metric,
        #         family=sm.families.NegativeBinomial(alpha=1),
        #         # subset=np.random.rand(len(self.loop_metric)) < 0.3,
        #     ).fit(maxiter=500)

        #     mu = self.glm_res.predict(self.loop_metric)
        #     alpha = 1  # np.exp(self.glm_res.lnalpha)
        #     size = 1 / alpha
        #     prob = size / (size + mu)
        #     self.loop_metric["P"] = nbinom.sf(
        #         self.loop_metric.C.values - 1, size, prob
        #     )
        else:
            NotImplementedError

        logging.info(f"AIC: {self.glm_res.aic}")
        if disp_glm_summary:
            print(self.glm_res.summary())

        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()

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

    def re_fit(
        self,
        model_distri=None,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        exog_infl_formula="1",
        disp_glm_summary=True,
    ):
        confident_interaction_index = self.interaction_statistics.index[
            (self.interaction_statistics.P <= 1 / len(self.loop_metric))
        ]
        old_expected = self.interaction_statistics.E.copy()
        # confident_interaction_index = self.interaction_statistics.index[
        #     self.interaction_statistics.Q <= 1e-3
        # ]
        filter_loop_metric = self.loop_metric.loc[
            ~self.loop_metric.index.isin(confident_interaction_index)
        ]
        logging.info(
            f"{len(confident_interaction_index)} high confident interactions were removed from fitting"
        )

        if model_distri is None:
            model_distri = self.model_distri

        exog_infl = dmatrix(exog_infl_formula, filter_loop_metric).view()
        if model_distri == "poisson":
            self.glm_res_1 = sm.Poisson.from_formula(
                formula=exog_model_formula,
                data=filter_loop_metric,
            ).fit_regularized(maxiter=500, disp=0)
            self.loop_metric["E"] = self.glm_res_1.predict(self.loop_metric)
            self.loop_metric["P"] = poisson.sf(
                self.loop_metric.C.values - 1, self.loop_metric.E.values
            )
        elif model_distri == "zipoisson":
            # fit distri
            self.glm_res_1 = sm.ZeroInflatedPoisson.from_formula(
                formula=exog_model_formula,
                data=filter_loop_metric,
                exog_infl=exog_infl,
            ).fit_regularized(maxiter=500, disp=0)
            # predict
            (
                self.loop_metric["P"],
                self.loop_metric["P_Infl1"],
                self.loop_metric["E"],
            ) = _zipoisson_p_vals(
                self.glm_res_1,
                self.loop_metric.C.values - 1,
                dmatrix(
                    exog_model_formula.split("~")[1], self.loop_metric
                ).view(),
                dmatrix(exog_infl_formula, self.loop_metric).view(),
            )

        self.loop_metric_for_refit = self.loop_metric.loc[
            self.loop_metric.index.isin(confident_interaction_index)
        ]

        logging.info(f"AIC: {self.glm_res_1.aic}")
        if disp_glm_summary:
            print(self.glm_res_1.summary())

        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()
        self.interaction_statistics["Q"] = multipletests(
            self.interaction_statistics.P.values,
            1,
            "fdr_bh",
        )[1]

        eps = np.square(
            old_expected - self.interaction_statistics.E.values
        ).sum()

        # keep loop_mertric data untouch
        significant_interactions = (self.interaction_statistics.Q <= 0.1) & (
            self.interaction_statistics.C >= self.count_cut
        )
        logging.info(
            f"{sum(significant_interactions)} interactions were called at qval <= 0.1, count >= {self.count_cut}"
        )

        return eps

    def estimate_mean_var(self, loop_metric_, nbins, infl_prob=0):
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

        logging.info("Inferring mean-var relation")
        grid_data = loop_metric.groupby(["L_BID", "D_BID"])
        grid_means = grid_data.C.mean().values
        grid_var = grid_data.C.var().values
        mask = grid_means == 0
        if infl_prob == 0:
            grid_infl = grid_data.P_Infl.mean().values
            self.grid_infl = grid_infl[~mask]
        elif infl_prob == 1:
            grid_infl = grid_data.P_Infl1.mean().values
            self.grid_infl = grid_infl[~mask]
        self.grid_mean, self.grid_var = (
            grid_means[~mask],
            grid_var[~mask],
        )

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

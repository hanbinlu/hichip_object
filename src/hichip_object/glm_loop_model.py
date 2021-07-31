import numpy as np
import logging
import statsmodels.api as sm
from scipy.stats import poisson
from statsmodels.distributions import zipoisson
from patsy import dmatrix
from .model_ILD import _bin_x_along_y_eq_count
from .loop_calling_handler import GenericLoopCallHandler


class Loop_ZIP(GenericLoopCallHandler):
    """ """

    def fit_model(
        self,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        exog_infl_formula="1",
        disp_glm_summary=True,
    ):
        logging.info("Fitting data to ZeroInflatedPoisson model")
        # only used for zero_inflated model
        exog_infl = dmatrix(exog_infl_formula, self.loop_metric).view()

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
        ) = self._zipoisson_p_vals(
            self.glm_res,
            self.loop_metric.C.values - 1,
            dmatrix(exog_model_formula.split("~")[1], self.loop_metric).view(),
            exog_infl,
        )

        logging.info(f"AIC: {self.glm_res.aic}")
        if disp_glm_summary:
            print(self.glm_res.summary())

        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()

        super().calc_qval()

    def re_fit(
        self,
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

        exog_infl = dmatrix(exog_infl_formula, filter_loop_metric).view()
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
        ) = self._zipoisson_p_vals(
            self.glm_res_1,
            self.loop_metric.C.values - 1,
            dmatrix(exog_model_formula.split("~")[1], self.loop_metric).view(),
            dmatrix(exog_infl_formula, self.loop_metric).view(),
        )

        logging.info(f"AIC: {self.glm_res_1.aic}")
        if disp_glm_summary:
            print(self.glm_res_1.summary())

        self.loop_metric_for_refit = self.loop_metric.loc[
            self.loop_metric.index.isin(confident_interaction_index)
        ]
        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()
        eps = np.square(
            old_expected - self.interaction_statistics.E.values
        ).sum()
        super().calc_qval()

        return eps

    def _zipoisson_p_vals(self, model_res, counts, exog, exog_infl):
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


class Loop_Poisson(GenericLoopCallHandler):
    """ """

    def fit_model(
        self,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        disp_glm_summary=True,
    ):
        logging.info("Fitting data to Poisson model")
        self.glm_res = sm.Poisson.from_formula(
            formula=exog_model_formula,
            data=self.loop_metric,
        ).fit_regularized(maxiter=500, disp=0)

        # predict
        self.loop_metric["E"] = self.glm_res.predict(self.loop_metric)
        self.loop_metric["P"] = poisson.sf(
            self.loop_metric.C.values - 1,
            self.loop_metric.E.values,
        )

        logging.info(f"AIC: {self.glm_res.aic}")
        if disp_glm_summary:
            print(self.glm_res.summary())

        self.interaction_statistics = self.loop_metric.loc[
            self.loop_metric.C != 0
        ].copy()

        super().calc_qval()

    def re_fit(
        self,
        exog_model_formula="C ~ np.log(D) + np.log(L)",
        disp_glm_summary=True,
    ):
        confident_interaction_index = self.interaction_statistics.index[
            (self.interaction_statistics.P <= 1 / len(self.loop_metric))
        ]
        old_expected = self.interaction_statistics.E.copy()
        filter_loop_metric = self.loop_metric.loc[
            ~self.loop_metric.index.isin(confident_interaction_index)
        ]
        logging.info(
            f"{len(confident_interaction_index)} high confident interactions were removed from fitting"
        )

        self.glm_res_1 = sm.Poisson.from_formula(
            formula=exog_model_formula,
            data=filter_loop_metric,
        ).fit_regularized(maxiter=500, disp=0)
        self.loop_metric["E"] = self.glm_res_1.predict(self.loop_metric)
        self.loop_metric["P"] = poisson.sf(
            self.loop_metric.C.values - 1, self.loop_metric.E.values
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

        eps = np.square(
            old_expected - self.interaction_statistics.E.values
        ).sum()

        super().calc_qval()

        return eps

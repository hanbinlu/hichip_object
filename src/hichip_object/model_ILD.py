import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from sklearn.isotonic import IsotonicRegression
from numba import njit
from numba.typed import List
import matplotlib.pyplot as plt

# At this point of building model, it should have already prepared the following data
# `loop_pet`: sparse matrix storing interaction values of anchor_i to anchor_j (all anchor pairs that has PET linked)
# `dist_anchor_pairs`: sparse matrix storing interaction distance of anchor_i to anchor_j (only midrange anchor pairs)
# `product_depth_anchor_pairs`: sparse matrix storing calculated seq depth production of anchor_i to anchor_j (only midrange anchor pairs)
# `anchors`: data frame of anchors order and coordinate in bed format
# `anchor_cal_depth`: pandas series of anchor's calcualted seq depth


def build_feature_PET_models(loop_metric, feature, nbins, plot_model=True):
    feature_model_data = _eq_nz_loop_number_bin_along_feature(
        loop_metric, feature, nbins
    )
    feature_PET_model = _spline_fit_model(feature_model_data, "x", "y")
    feature_NLOOP_model = _spline_fit_model(
        feature_model_data, "xItems", "yItems"
    )
    if plot_model:
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        axs[0].scatter(np.log10(feature_model_data.x), feature_model_data.y)
        axs[0].plot(
            np.log10(feature_model_data.x),
            feature_PET_model(feature_model_data.x),
        )
        axs[1].scatter(
            np.log10(feature_model_data.xItems), feature_model_data.yItems
        )
        axs[1].plot(
            np.log10(feature_model_data.xItems),
            feature_NLOOP_model(feature_model_data.xItems),
        )
        fig.suptitle(f"{feature} model")
        plt.show()
    return feature_model_data, feature_PET_model, feature_NLOOP_model


def _bin_x_along_y_eq_count(
    df, y_col_name, num_bins, how_to_count, is_sorted=False
):
    """
    Split df along column `y` to `num_bins`. Each resulting bin `[y_low, y_high)` will have nearly equal sum counts, calculating by:
    `how_to_count(filtered_df[y_low <= filtered_df[y_col] < y_high])`
    """
    if not is_sorted:
        df_sort_along_y = df.sort_values(by=y_col_name)
    else:
        df_sort_along_y = df
    y_sum = how_to_count(df_sort_along_y)

    unique_y_vals = df_sort_along_y[y_col_name].unique()
    last_index = len(unique_y_vals) - 1
    y_group_sum = df_sort_along_y.groupby(y_col_name).apply(how_to_count)
    break_points = _bin_y_group_sum(
        y_group_sum.values, num_bins, y_sum, last_index
    )

    borders = np.array(df[y_col_name].min())
    borders = np.append(borders, unique_y_vals[break_points])
    borders[-1] = unique_y_vals[-1] + 1

    model_data = pd.DataFrame(
        {
            "Low": borders[:-1],
            "High": borders[1:],
            "Num_NZ_Items": np.histogram(
                df_sort_along_y[y_col_name].values, bins=borders
            )[0],
        }
    )

    return model_data


@njit
def _bin_y_group_sum(y_group_sum, remain_num_bin, remain_sum, max_index):
    break_points, sum_count_of_bin = List(), 0
    bin_size = remain_sum // remain_num_bin
    for i in range(len(y_group_sum)):
        # try to be as even as possible
        this_group_sum = y_group_sum[i]
        if sum_count_of_bin + this_group_sum >= bin_size:
            # decide break point by the margin to `bin_size`
            # right open
            excessive = sum_count_of_bin + this_group_sum - bin_size
            shortage = bin_size - sum_count_of_bin
            if excessive > shortage:
                if sum_count_of_bin == 0:
                    break_points.append(min(i + 1, max_index))
                    remain_sum -= this_group_sum
                else:
                    break_points.append(i)
                    remain_sum -= sum_count_of_bin
                    sum_count_of_bin = this_group_sum
            else:
                break_points.append(min(i + 1, max_index))
                remain_sum -= sum_count_of_bin + this_group_sum
                sum_count_of_bin = 0

            remain_num_bin -= 1
            # adjust bin_size based on binned
            if remain_num_bin:
                bin_size = remain_sum // remain_num_bin
        else:
            sum_count_of_bin += this_group_sum
    return break_points


def _eq_nz_loop_number_bin_along_feature(df, feature, nbins):
    # based binning
    df_to_bin = df.loc[df.C != 0]
    model_data = _bin_x_along_y_eq_count(df_to_bin, feature, nbins, len)
    # expand to cover whole data range
    # model_data.loc[0, "Low"] = df[feature].min()
    # model_data.loc[model_data.index[-1], "High"] = df[feature].max() + 1

    # add x, y data for fitting
    model_data["Num_ALL_Items"] = [
        sum((df[feature] >= x) & (df[feature] < y))
        for x, y in zip(model_data.Low.values, model_data.High.values)
    ]
    model_data["Sum_PETs"] = [
        df_to_bin.C.loc[
            (df_to_bin[feature] >= x) & (df_to_bin[feature] < y)
        ].sum()
        for x, y in zip(model_data.Low.values, model_data.High.values)
    ]

    model_data["x"] = [
        df_to_bin[feature]
        .loc[(df_to_bin[feature] >= low) & (df_to_bin[feature] < high)]
        .mean()
        for low, high in zip(model_data.Low.values, model_data.High.values)
    ]
    model_data["y"] = model_data.Sum_PETs / df_to_bin.C.sum()
    model_data["xItems"] = [
        df[feature].loc[(df[feature] >= x) & (df[feature] < y)].mean()
        for x, y in zip(model_data.Low.values, model_data.High.values)
    ]
    model_data["yItems"] = (
        model_data.Num_ALL_Items / model_data.Num_ALL_Items.sum()
    )

    return model_data


# def build_logarithmic_grid_model(df, x, y, x_step, y_step, x_range, y_range):
#     df_to_bin = df.loc[df.C != 0]
#
#     # create logarithmic borders of x and y
#     x_borders = [np.log10(x_range[0])]
#     while x_borders[-1] <= np.log10(x_range[1]):
#         x_borders.append(x_borders[-1] + x_step)
#     x_borders = 10 ** np.array(x_borders)
#     y_borders = [np.log10(y_range[0])]
#     while y_borders[-1] <= np.log10(y_range[1]):
#         y_borders.append(y_borders[-1] + y_step)
#     y_borders = 10 ** np.array(y_borders)
#
#     # create grid structure
#     x_low, x_high, y_low, y_high = [], [], [], []
#     for i, j in product(range(len(x_borders) - 1), range(len(y_borders) - 1)):
#         x_low.append(x_borders[i])
#         x_high.append(x_borders[i + 1])
#         y_low.append(y_borders[j])
#         y_high.append(y_borders[j + 1])
#
#     # bin data in to the gride
#     N = df_to_bin.C.sum()
#     sum_pets, nz_items = [], []
#     x_mean, y_mean = [], []
#     for x1, x2, y1, y2 in zip(x_low, x_high, y_low, y_high):
#         within_x_block = (df_to_bin[x] >= x1) & (df_to_bin[x] < x2)
#         within_y_block = (df_to_bin[y] >= y1) & (df_to_bin[y] < y2)
#         sum_pets.append(df_to_bin.loc[within_x_block & within_y_block].C.sum())
#
#         # regardless whether there is a PET linked
#         within_x_block = (df[x] >= x1) & (df[x] < x2)
#         within_y_block = (df[y] >= y1) & (df[y] < y2)
#         x_mean.append(df[x].loc[within_x_block & within_y_block].mean())
#         y_mean.append(df[y].loc[within_x_block & within_y_block].mean())
#         nz_items.append(sum(within_x_block & within_y_block))
#
#     grid_data = pd.DataFrame(
#         {
#             f"{x}_low": x_low,
#             f"{x}_high": x_high,
#             f"{y}_low": y_low,
#             f"{y}_high": y_high,
#             "NZ_Items": nz_items,
#             "x": x_mean,
#             "y": y_mean,
#             "p": np.array(sum_pets) / N / np.array(nz_items),
#         }
#     )
#
#     # spline fit 2D
#     f = SmoothBivariateSpline(
#         np.log10(grid_data.x.values),
#         np.log10(grid_data.y.values),
#         np.log10(grid_data.p.values),
#         s=0.75,
#     )
#
#     return grid_data, lambda x, y: 10 ** f.ev(np.log10(x), np.log10(y))


def _spline_fit_model(model_data, x, y):
    # f = UnivariateSpline(
    #    np.log10(model_data[x].values),
    #    np.log10(model_data[y].values),
    #    s=0.75,
    # )
    r_y = robjects.FloatVector(np.log10(model_data[y].values))
    r_x = robjects.FloatVector(np.log10(model_data[x].values))
    r_smooth_spline = robjects.r["smooth.spline"]
    spline1 = r_smooth_spline(x=r_x, y=r_y, spar=0.75)
    ySpline = np.array(robjects.r["predict"](spline1, r_x).rx2("y"))
    iso_reg = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        np.log10(model_data[x].values), ySpline
    )

    return lambda x: 10 ** iso_reg.predict(np.log10(x))

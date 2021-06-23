import itertools, logging
from typing import overload
import pandas as pd
import numpy as np
import networkx as nx
from numba import njit
from numba.typed import List
from ray.util.multiprocessing import Pool
from functools import partial

############## Functions to Combine Loops #################
def combine_two_loop_list(
    lo1, lo2, connect_dist, names=("lo1", "lo2"), return_merged_result=False
):
    """
    Merge two loop list into non-redundant loop set by building a combined graph.
    """
    logging.info(
        f'{len(lo1)} records in "{names[0]}" loop set; {len(lo2)} records in "{names[1]}" loop set'
    )
    # combine lo1, lo2 into single data frame and add origin columne
    df1 = lo1.iloc[:, 0:6].copy()
    df1["Source"] = names[0]
    df2 = lo2.iloc[:, 0:6].copy()
    df2["Source"] = names[1]
    all_loops = df1.append(df2, ignore_index=True)
    all_loops["Non_Redundant_Loops"] = np.append(
        lo1.Non_Redundant_Loops.values, lo2.Non_Redundant_Loops.values
    )

    # split df of loops by chromosomes
    gen_chro_loops = (
        chro_loops
        for chro, chro_loops in all_loops.groupby(by="chr1")
        if "_" not in chro
    )

    logging.info("Building Graph")
    with Pool(processes=30) as pool:
        # each process build graph from loops of one chromosomes
        # nodes are loops; add edge to two nodes if eucledian distance < `connect_dist`
        chro_loop_graph = pool.map(
            partial(_build_loop_graph, connect_dist=connect_dist),
            gen_chro_loops,
        )

    logging.info("Assigning Components")
    component_id = np.empty(len(all_loops), dtype=int)
    component_cnt = 0
    for G in chro_loop_graph:
        for component in nx.connected_components(G):
            component_cnt += 1
            component_id[list(component)] = component_cnt

    combined_results = pd.DataFrame(
        {
            "Source": all_loops.Source.values.copy(),
            "Component_ID": component_id,
            "Num_Loops_From_1": np.zeros(len(all_loops), dtype=int),
            "Num_Loops_From_2": np.zeros(len(all_loops), dtype=int),
        }
    )

    only_1, overlap, only_2 = 0, 0, 0
    for _, component_df in combined_results.groupby(by="Component_ID"):
        source, source_cnt = np.unique(
            component_df.Source.values, return_counts=True
        )
        source = source.tolist()
        index = component_df.index
        if len(source) == 1:
            if source[0] == names[0]:
                combined_results.loc[index, "Num_Loops_From_1"] = sum(
                    source_cnt
                )
                only_1 += 1
            else:
                combined_results.loc[index, "Num_Loops_From_2"] = sum(
                    source_cnt
                )
                only_2 += 1
        else:
            combined_results.loc[index, "Num_Loops_From_1"] = source_cnt[
                source.index(names[0])
            ]
            combined_results.loc[index, "Num_Loops_From_2"] = source_cnt[
                source.index(names[1])
            ]
            overlap += 1

    logging.info(
        f"Combined to {component_cnt} components (merged loops combined graph of both loop set)"
    )
    only_1_non_redundant = len(
        all_loops.Non_Redundant_Loops[
            (combined_results.Num_Loops_From_1 != 0)
            & (combined_results.Num_Loops_From_2 == 0)
        ].unique()
    )
    logging.info(
        f'Only "{names[0]}": {only_1} component are consist of {only_1_non_redundant} "{names[0]}" non-redundant loops'
    )
    only_2_non_redundant = len(
        all_loops.Non_Redundant_Loops[
            (combined_results.Num_Loops_From_1 == 0)
            & (combined_results.Num_Loops_From_2 != 0)
        ].unique()
    )
    logging.info(
        f'Only "{names[1]}": {only_2} component are consist of {only_2_non_redundant} "{names[1]}" non-redundant loops'
    )
    overlap_idx = (combined_results.Num_Loops_From_1 != 0) & (
        combined_results.Num_Loops_From_2 != 0
    )
    overlap_1_non_redundant = len(
        all_loops.Non_Redundant_Loops[
            overlap_idx & (all_loops.Source == names[0])
        ].unique()
    )
    overlap_2_non_redundant = len(
        all_loops.Non_Redundant_Loops[
            overlap_idx & (all_loops.Source == names[1])
        ].unique()
    )
    logging.info(
        f'{overlap} overlaping components are equal to {overlap_1_non_redundant} non-redundant loops for "{names[0]}" or {overlap_2_non_redundant} for "{names[1]}"'
    )

    if return_merged_result:
        all_loops["Component_ID"] = combined_results["Component_ID"].values
        all_loops["Num_Loops_From_1"] = combined_results[
            "Num_Loops_From_1"
        ].values
        all_loops["Num_Loops_From_2"] = combined_results[
            "Num_Loops_From_2"
        ].values
        return all_loops


def non_redundant_loops(loops, connect_dist):
    """
    Merge loops that is connecting each other in `connect_dist` radios by building Graph of loops.
    """
    if any(loops.index.values != np.arange(len(loops))):
        raise ValueError("Loops index must be RangeIndex")
    # split df of loops by chromosomes
    gen_chro_loops = (
        chro_loops
        for chro, chro_loops in loops.groupby(by="chr1")
        if "_" not in chro
    )

    with Pool(processes=30) as pool:
        # each process build graph from loops of one chromosomes
        # nodes are loops; add edge to two nodes if eucledian distance < `connect_dist`
        # find components and use the lowest qvalue loop in each component as its representative
        chro_loop_graph = pool.map(
            partial(_build_loop_graph, connect_dist=connect_dist),
            gen_chro_loops,
        )

    # merge chromosome results
    # non_redundant_loop_index = [
    #    ix for chro_ix in final_loop_index for ix in chro_ix
    # ]
    non_redundant_id = np.empty(len(loops), dtype=int)
    component_cnt = 0
    for G in chro_loop_graph:
        for component in nx.connected_components(G):
            component_cnt += 1
            non_redundant_id[list(component)] = component_cnt

    logging.info(f"{len(loops)} reduces to {len(np.unique(non_redundant_id))}")

    result = loops.copy()
    result["Non_Redundant_Loops"] = non_redundant_id

    return result.sort_values(by=["chr1", "x1", "x2"], ignore_index=True)


def _low_qval_component_representative(loops, connect_dist):
    G = _build_loop_graph(loops, connect_dist)
    return [
        sorted(list(component), key=lambda x: loops.qval[x])[0]
        for component in nx.connected_components(G)
    ]


def _build_loop_graph(loops, connect_dist, prefix=None):
    G = nx.Graph()
    center1 = ((loops.x1 + loops.y1) // 2).values
    center2 = ((loops.x2 + loops.y2) // 2).values
    index = loops.index.values

    # nodes with loop coordinates
    for i, ix in enumerate(index):
        if prefix:
            G.add_node(f"{prefix}_{ix}", x=center1[i], y=center2[i])
        else:
            G.add_node(ix, x=center1[i], y=center2[i])

    # add edges based on two nodes/loops euclidean distance
    G.add_edges_from(_compute_edges(index, center1, center2, connect_dist))

    return G


@njit
def _compute_edges(ix, x, y, d):
    n = len(ix)
    edges = List()
    for i in np.arange(n - 1):
        for j in np.arange(i + 1, n):
            if np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) <= d:
                edges.append((ix[i], ix[j]))

    return edges


############## Functions to Merge Loops with 1D peaks #################
def loops_cobound_by_peaks(peaks, loops, connect_dist):
    """
    Check loop's anchors are bound by `peaks` within `connect_dist`.
    """
    if any(loops.index.values != np.arange(len(loops))):
        raise ValueError("Loops index must be RangeIndex")
    # mark as 1 if found peak cobound
    anchor1_bound, anchor2_bound = np.zeros(len(loops), dtype=int), np.zeros(
        len(loops), dtype=int
    )
    for chro, chro_peaks in peaks.groupby(by="chro"):
        chro_loop_index = loops.chr1 == chro
        chro_loops = loops.loc[chro_loop_index, :]
        peak_centers = (chro_peaks.start.values + chro_peaks.end.values) // 2
        bound1, bound2 = np.zeros(len(chro_loops), dtype=int), np.zeros(
            len(chro_loops), dtype=int
        )

        # peak bound at anchor 1
        center1 = (chro_loops.x1.values + chro_loops.y1.values) // 2
        for p in peak_centers:
            dist = np.abs(p - center1)
            bound1[dist <= connect_dist] += 1
        # peak bound at anchor 2
        center2 = (chro_loops.x2.values + chro_loops.y2.values) // 2
        for p in peak_centers:
            dist = np.abs(p - center2)
            bound2[dist <= connect_dist] += 1

        # loops index is same as row number
        anchor1_bound[chro_loop_index.values] = bound1
        anchor2_bound[chro_loop_index.values] = bound2

    # add result to original df
    loops["peak_cobound1"] = anchor1_bound
    loops["peak_cobound2"] = anchor2_bound

    both_bound, only_bound_one, unbound = (
        _both_bound_loops(loops),
        _one_anchor_bound_loops(loops),
        _unbound_loops(loops),
    )
    logging.info(f"Of {len(loops)} loops:")
    logging.info(
        f"{len(both_bound)} ({len(both_bound)/len(loops)}) loops are bound by both anchors"
    )
    logging.info(
        f"{len(only_bound_one)} ({len(only_bound_one)/len(loops)}) loops are bound by one anchor"
    )
    logging.info(
        f"{len(unbound)} ({len(unbound)/len(loops)}) loops are not bound"
    )


def _both_bound_loops(loops):
    idx = (loops.peak_cobound1 > 0) & (loops.peak_cobound2 > 0)
    return loops.loc[idx, :]


def _unbound_loops(loops):
    idx = (loops.peak_cobound1 == 0) & (loops.peak_cobound2 == 0)
    return loops.loc[idx, :]


def _bound_loops(loops):
    idx = (loops.peak_cobound1 > 0) | (loops.peak_cobound2 > 0)
    return loops.loc[idx, :]


def _one_anchor_bound_loops(loops):
    idx_1 = (loops.peak_cobound1 > 0) & (loops.peak_cobound2 == 0)
    idx_2 = (loops.peak_cobound1 == 0) & (loops.peak_cobound2 > 0)
    return loops.loc[idx_1 | idx_2, :]


def read_hicuups_loop(f):
    """
    HiCUUPS output loops in bedpe format
    """
    loops = pd.read_csv(f, sep="\t", skiprows=[1])
    first_6_col_names = ["chr1", "x1", "y1", "chr2", "x2", "y2"]
    loops.rename(
        columns={
            ori: rename
            for ori, rename in zip(loops.columns[:6], first_6_col_names)
        },
        inplace=True,
    )
    return loops


def significant_loops(
    loops, qval, count=1, qval_col="qval", coverage_col="counts"
):
    filt_index = (loops[qval_col] <= qval) & (loops[coverage_col] >= count)
    return loops.loc[filt_index, :]


def mid_range_loops(loops, interaction_range=(5000, 2000000)):
    loop_dists = loops.x2 - loops.x1
    idx = (loop_dists >= interaction_range[0]) & (
        loop_dists <= interaction_range[1]
    )
    return loops.loc[idx, :]


def drop_unplaced_chromosome_data(loops):
    idx = ~loops.chr1.str.contains("_")
    return loops.loc[idx, :]

""" This File provides the functionality for a RAFT-knowledgebase website. """

import sort_RAFT_table as sRt
import pandas as pd
import numpy as np
import plotly.express as px
import re
from scipy.optimize import curve_fit
import itertools

# rectifying datatable(s)
sRt.df["t6h-conversion"] = np.nan
sRt.df["t10h-conversion"] = np.nan

# get all conversion headers, sort them by the hours
conversion_list = []
for column in sRt.df.columns:
    if "conversion" in column:
        conversion_list.append(column)

conversion_list.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

# get all the Mn and Mw headers
Mn_list = []
Mw_list = []
for column in sRt.df.columns:
    if "Mn" in column:
        Mn_list.append(column)
    if "Mw" in column:
        Mw_list.append(column)


def change_time_format_h(time_format):
    h_m_s = str(time_format).split(":")
    h_format = int(h_m_s[0]) + (int(h_m_s[1]) + int(h_m_s[2]) / 60) / 60
    return h_format


# creating a table as a lookup to correct all sample measurement times

# get the hours from the headers with regex
hours_list = []
two_digit_regex = r"\d+"
for column in conversion_list:
    hours_list.append(int(re.findall(two_digit_regex, column)[0]))
hours_list.sort()

exact_times = pd.read_excel(sRt.INPUT_FILE_PATH, sheet_name="exact sampling times")

time_correction_df = pd.DataFrame(data=exact_times.iloc[3:12, 3:])
time_correction_df.columns = exact_times.loc[2][3:]

time_correction_df.reset_index(inplace=True, drop=True)
ext_time_corr_df_data = []
for row in time_correction_df.iterrows():
    reactor_nr = row[1]["Reactor"]
    if type(reactor_nr) == int or reactor_nr == "closing of reactors":  # that applies to reactor 15 and the closing 
        # of reactors 
        row[1]["Reactor"] = str(reactor_nr)
        ext_time_corr_df_data.append(row[1])
    else:  # that applies to all reactors which are described per row in pairs like "3+4"
        reactor_nr_s = (row[1]["Reactor"]).split("+")
        for reactor_nr in reactor_nr_s:
            row[1]["Reactor"] = reactor_nr
            ext_time_corr_df_data.append(row[1].copy())
ext_time_corr_df = pd.DataFrame(data=ext_time_corr_df_data, columns=exact_times.loc[2][3:])
ext_time_corr_df.reset_index(drop=True, inplace=True)
ext_time_corr_df.columns = ["Reactor", 0, 1, 2, 4, 6, 8, 10, 15]

# change time format to minutes and set
time_cols = ext_time_corr_df.columns.difference(["Reactor"])
ext_time_corr_df[time_cols] = ext_time_corr_df[time_cols].apply(lambda x: [change_time_format_h(d) for d in x])
ext_time_corr_df[0] = ext_time_corr_df[0].apply(lambda x: 0)

# set index to reactor
ext_time_corr_df.set_index("Reactor", inplace=True)

# replace the "sample determiner" with columns describing for the experiment number (next cell) and the actual 
# reagents that made out the "determiner" 
reaction_descriptors_dict = {}
abbreviation_keys = pd.read_excel(sRt.INPUT_FILE_PATH, sheet_name="Legend for Abbreviations")
abbreviation_keys.dropna(inplace=True)
for row in abbreviation_keys.itertuples():
    reaction_descriptors_dict[str(row.Symbol)] = row.Name

# generate a reformatted table with all entries attributing to a sample analysis taken, described with a column of 
# the right time 
kinetic_curves = []
for index, polymerisation_kinetic in sRt.df.iterrows():
    kinetic_curve_entries = pd.DataFrame(index=range(len(polymerisation_kinetic[conversion_list])),
                                         data={"time": hours_list,
                                               "conversion": polymerisation_kinetic[conversion_list].values,
                                               "Mn": polymerisation_kinetic[Mn_list].values,
                                               "Mw": polymerisation_kinetic[Mw_list].values,
                                               # , "reactor" : polymerisation_kinetic["reactor"] reactor is not 
                                               # needed since the time is corrected 
                                               })

    kinetic_curve_entries["exp_nr"] = polymerisation_kinetic["Experiment number"]
    kinetic_curve_entries["monomer"] = reaction_descriptors_dict[polymerisation_kinetic["monomer"]]
    kinetic_curve_entries["RAFT-Agent"] = reaction_descriptors_dict[polymerisation_kinetic["RAFT-Agent"]]
    kinetic_curve_entries["solvent"] = reaction_descriptors_dict[polymerisation_kinetic["solvent"]]

    # the times are dependent on the current reactor, get current
    current_reactor_nr = str(polymerisation_kinetic["reactor"])
    current_time_list = ext_time_corr_df.loc[current_reactor_nr]
    kinetic_curve_entries["time"] = list(current_time_list)

    kinetic_curves.append(kinetic_curve_entries)


#  fitting functions
def neg_growth(x, l, k):
    y = l * (1 - np.exp(k * (-x)))
    return y


def neg_growth_derivative(x, l, k):
    y = l * k * np.exp(k * (-x))
    return y


# as the Mn values do not all start at 0
def neg_growth_abscissae(x, l, k, b):
    y = l * (1 - np.exp(k * (-x))) + b
    return y


def linear_growth(x, m):
    y = m * x
    return y


def linear_growth_derivative(m):
    return m


# settings for plot functions
colors = px.colors.qualitative.Plotly  # set up a simple color palette
extended_xdata = np.linspace(-1, 16.5, 100)  # x data array for plotting the fits


def add_fits_to_plot(figure, fit_func, fit_func_params, fit_func_derivative=None, *args, **kwargs):
    figure.add_scatter(
        x=extended_xdata, y=fit_func(extended_xdata, *fit_func_params),
        opacity=1, line=dict(dash="dot"), name=f"{fit_func.__name__} fit", *args, **kwargs)
    if fit_func_derivative:
        figure.add_scatter(
            x=extended_xdata, y=fit_func_derivative(extended_xdata, *fit_func_params),
            opacity=0.3, line=dict(dash="dash"), name=f"{fit_func_derivative.__name__}", *args, **kwargs)


def fit_and_exclude_outliers(x, y, fit_func, p0, bounds, nan_policy="omit", iteration=1, outliers=None):
    outliers = outliers if outliers is not None else []

    # exclude the nan values from the data
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]

    cf_data = curve_fit(f=fit_func, xdata=x, ydata=y, p0=p0, nan_policy=nan_policy, maxfev=800 * 10, bounds=bounds)

    # calculate the fit points
    fit_points = np.array([fit_func(x, *cf_data[0]) for x in x])
    # calculate the standard deviation of the residuals between the fit and the data points
    sigma = np.std(fit_points - y)

    # exclude the outliers
    msk = ~(np.abs(fit_points - y) > 2 * sigma)
    if not msk.all():
        return fit_and_exclude_outliers(x=x[msk], y=y[msk], fit_func=fit_func, p0=cf_data[0], bounds=bounds,
                                        nan_policy=nan_policy, iteration=iteration + 1, outliers=outliers + [x[~msk]])

    # calculate the squared error of fit and data points
    sq_err = np.mean(np.square(y - fit_points))

    result = {"x": x, "y": y, "p_opt": cf_data[0], "p_cov": cf_data[1], "sq_err": sq_err,
              "excluded_points": (x[~msk], y[~msk]), "iteration": iteration, "outliers": outliers}
    return result


kinetics_df = pd.DataFrame()  # create new dataframe with kinetics per row
for idx, kinetic_curve in enumerate(kinetic_curves):
    # first make sure the datapoints are in the right format and not sometimes int sometimes float
    xdata = np.array(kinetic_curve["time"].values, dtype=float)
    ydata_conv = np.array(kinetic_curve["conversion"].values, dtype=float)
    ydata_Mn = np.array(kinetic_curve["Mn"].values, dtype=float) / 100000  # make it more comparable to conversion
    ydata_Mw = np.array(kinetic_curve["Mw"].values, dtype=float) / 100000

    # fitting section for conversion
    p_initial = [max(ydata_conv), 0.1]  # this is a mandatory initial guess
    ng_fit = fit_and_exclude_outliers(x=xdata, y=ydata_conv, fit_func=neg_growth, p0=p_initial,
                                      bounds=([0, -np.inf], [1, np.inf]))
    # l_fit = fit_and_exclude_outliers(x=xdata, y=ydata, fit_func=linear_growth, p0=[max(ydata)/7], bounds=([0], 
    # [np.inf])) 

    popt, pcov = ng_fit["p_opt"], ng_fit["p_cov"]
    conv_time_data = np.array([ng_fit["x"], ng_fit["y"]])
    squared_error = ng_fit["sq_err"]

    # fitting section for Mn
    p_initial = [max(ydata_Mn[~np.isnan(ydata_Mn)]), 0.1, 0]
    ng_fit_Mn = fit_and_exclude_outliers(x=xdata, y=ydata_Mn, fit_func=neg_growth_abscissae, p0=p_initial,
                                         bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]))

    popt_Mn, pcov_Mn = ng_fit_Mn["p_opt"], ng_fit_Mn["p_cov"]
    Mn_time_data = np.array([ng_fit_Mn["x"], ng_fit_Mn["y"]])
    squared_error_Mn = ng_fit_Mn["sq_err"]

    # fitting section for Mw
    ng_fit_Mw = fit_and_exclude_outliers(x=xdata, y=ydata_Mw, fit_func=neg_growth_abscissae, p0=p_initial,
                                         bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]))

    popt_Mw, pcov_Mw = ng_fit_Mw["p_opt"], ng_fit_Mw["p_cov"]
    Mw_time_data = np.array([ng_fit_Mw["x"], ng_fit_Mw["y"]])
    squared_error_Mw = ng_fit_Mw["sq_err"]

    new_row = pd.DataFrame({"exp_nr": kinetic_curve["exp_nr"].iloc[1], "max_con": max(ydata_conv),
                            "theo_max_con": "yet to calc", "theo_react_end": "yet to calc",
                            "monomer": kinetic_curve["monomer"].iloc[1],
                            "RAFT-Agent": kinetic_curve["RAFT-Agent"].iloc[1],
                            "solvent": kinetic_curve["solvent"].iloc[1],
                            "fit_p1": [popt[0]], "fit_p2": [popt[1]],
                            "p1_variance": [pcov[0][0]], "p1_p2_covariance": [pcov[0][1]], "p2_variance": [pcov[1][1]],
                            "squared_error": squared_error, "conv_time_data": [conv_time_data],
                            "Mn_time_data": [Mn_time_data], "Mw_time_data": [Mw_time_data]})
    kinetics_df = pd.concat([kinetics_df, new_row])

kinetics_df.reset_index(drop=True, inplace=True)
kinetics_df.drop(axis="index", index=kinetics_df[kinetics_df["max_con"] <= 0].index, inplace=True)
kinetics_df.reset_index(drop=True, inplace=True)


# normalize the errors by dividing them by their respective standard deviation
def normalize_errors(err):
    return err / np.std(err)


for error in ["squared_error", "p1_variance", "p2_variance", "p1_p2_covariance"]:
    kinetics_df[error] = normalize_errors(kinetics_df[error])


def get_quartile_indexes(error_type):
    quartile_len = len(kinetics_df) / 4
    quartile_ranges = np.array([(a * quartile_len, (a + 1) * quartile_len) for a in range(4)], dtype=int)
    quartiles_list = list()
    for q in range(4):
        quartiles_list.append(
            kinetics_df.sort_values(by=[error_type]).iloc[quartile_ranges[q][0]:quartile_ranges[q][1]])
    return quartiles_list


# descry when a function aligns to the datapoints in a reasonable way Hence, wheneth' the blunder exaggerates,
# an 80% betweeneth' of the maximum conversion in that kinetic should be assessed to be the maximum conversion. let's
# give a point for every quartile further from the first for the single errors divided by the maximum score (that is
# 3*4=12)
error_list = ["squared_error", "p1_variance", "p2_variance", "p1_p2_covariance"]
error_dic = {}
score = np.zeros(len(kinetics_df), int)
for error in error_list:
    quartiles = get_quartile_indexes(error)
    # for every error we want to give an error per index
    for sc, quartile in enumerate(quartiles):
        if sc == 0:
            continue
        # each quartile is a dataframe, where the latter one raise a higher error ( 0, 1, 2, 3)
        for num in quartile.index:
            score[num] += sc
kinetics_df["error_score"] = score

# determining apposite reaction end values.
# The moment where 90% of the maximum conversion have been reached can be seen as a practical maximum conversion point
kinetics_df["theo_max_con"] = kinetics_df["fit_p1"].apply(lambda x: x * 0.9)


def y_converted_negative_growth(y, l, k):
    return -np.log(1 - y / l) / k


kinetics_df["theo_react_end"] = [y_converted_negative_growth(y, fit_p1, fit_p2) for y, fit_p1, fit_p2 in
                                 zip(kinetics_df["theo_max_con"], kinetics_df["fit_p1"], kinetics_df["fit_p2"])]

# the theoretical maximal conversion must be capped at reasonable time (we take two days here) that is applying 
# 139/313 entries 
kinetics_df["theo_react_end"] = [30 if x > 30 else x for x in kinetics_df["theo_react_end"].values]
# recalculate the apposite maximal conversion
kinetics_df["theo_max_con"] = [neg_growth(x, p1, p2) for x, p1, p2 in
                               zip(kinetics_df["theo_react_end"], kinetics_df["fit_p1"], kinetics_df["fit_p2"])]

# to find the optimal threshold parameters for search one has to keep in mind that with high conversion (assuming 
# around 80%, research/citation needed) increasing side reaction take place. After that the reactions should be 
# sorted after time then error score. Maybe a multiple-decreasing-threshold-sorting-algorithm would be good. so first
# priority would be sorting after nearest to 80% conversion. 

# create a score ingesting the importance of the different kinetic descriptors
#     Conversion*1 + time*(-0.8) + error_score*(0.5)
#     while spanning between the optimum and the least bearable values like in the following:
#          Conversion: |con-0.8| - 0 (0.8 is the optimum)
#             using a linear decreasing function -x*m+b
#          Time: 0 - np.inf (0 is the optimum) (more than 72 is not bearable)
#             using a negative potential function -x**2+b
#          Error: 0 - 12 (0 is the optimum) (the error is more negligible)
#             using a linear decreasing function -x*m+b
score = []
for row in kinetics_df.itertuples():
    score.append(((0.8 - np.abs(row.theo_max_con - 0.8)) / 0.8 * 1 + (-(row.theo_react_end / 72) ** 2 + 1) * 0.8 + (
            (12 - row.error_score) / 12) * 0.5))
kinetics_df["score"] = score

""" Functions for website-interaction-elements """


def color_variant(hex_color, brightness_offset=1):
    """ takes a color like #87c95f and produces a lighter or darker variant """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x + 2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]  # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"; also we zfill to pad with leading zeros if necessary, e.g. 9 -> 09
    return "#" + "".join([hex(i)[2:].zfill(2) for i in new_rgb_int])


def search_for_exp(exp_nr: str | list) -> pd.DataFrame:
    """ Search for the experimental and descriptive info of the given experiment number(s)
    Parameters
        ----------
        exp_nr
            The experiment number(s) to search for

        Returns
        -------
        DataFrame
            The DataFrame containing the experimental and descriptive info of the given experiment number(s)
    """
    if type(exp_nr) == str:
        return kinetics_df[kinetics_df['exp_nr'] == exp_nr]
    else:
        return kinetics_df[kinetics_df['exp_nr'].isin(exp_nr)]


def plot_exp(exp_nr: str | list, plot_mn: bool = False, plot_mw: bool = False, fit_curves: tuple = (True, True)):
    """ Plot the kinetic curves for the experiment number(s) exp_nr.
    Parameters
        ----------
        exp_nr
            The experiment number(s) to plot
        plot_mn
            Whether to plot the Mn data
        plot_mw
            Whether to plot the Mw data
        fit_curves
            A tuple with two boolean values, the first one determines whether to plot the fit curves, the second one
            determines whether to plot the derivative of the fit curves.

        Returns
        -------
        Figure

    """
    plot_data = search_for_exp(exp_nr)
    exp_fig = px.line(title=f"Kinetic Curve Fit for {exp_nr}")
    for kinetic_to_plot in plot_data.itertuples():
        x_data, ydata = kinetic_to_plot.conv_time_data
        marker_dict = dict(color=colors[int(kinetic_to_plot.Index) % len(colors)])
        exp_fig.add_scatter(x=x_data, y=ydata, mode="lines+markers", name=kinetic_to_plot.exp_nr, marker=marker_dict,
                            legendgroup=str(kinetic_to_plot.exp_nr))
        if fit_curves[0]:
            if fit_curves[1]:
                add_fits_to_plot(exp_fig, neg_growth, [kinetic_to_plot.fit_p1, kinetic_to_plot.fit_p2],
                                 fit_func_derivative=neg_growth_derivative, marker=marker_dict,
                                 legendgroup=str(kinetic_to_plot.exp_nr), showlegend=False)
            else:
                add_fits_to_plot(exp_fig, neg_growth, [kinetic_to_plot.fit_p1, kinetic_to_plot.fit_p2],
                                 marker=marker_dict, legendgroup=str(kinetic_to_plot.exp_nr), showlegend=False)

        if plot_mn:
            marker_dict["color"] = color_variant(marker_dict["color"], 30)
            x2_data, y2_data = kinetic_to_plot.Mn_time_data
            exp_fig.add_scatter(x=x2_data, y=y2_data, mode="lines+markers", name="Mn of " + kinetic_to_plot.exp_nr,
                                marker=marker_dict, legendgroup=str(kinetic_to_plot.exp_nr))
        if plot_mw:
            marker_dict["color"] = color_variant(marker_dict["color"], -60)
            x2_data, y2_data = kinetic_to_plot.Mw_time_data
            exp_fig.add_scatter(x=x2_data, y=y2_data, mode="lines+markers", name="Mw of " + kinetic_to_plot.exp_nr,
                                marker=marker_dict, legendgroup=str(kinetic_to_plot.exp_nr), opacity=0.5)

    exp_fig.update_layout(yaxis=dict(range=[-0.1, 1]), xaxis_title="Time [h]", yaxis_title="Conversion [%]")
    return exp_fig


def find_optimal_synthesis(monomer: str | list):
    search_q_monomer = [x in monomer for x in kinetics_df["monomer"]]
    result_df = kinetics_df[search_q_monomer].sort_values(by=["score"], ascending=False)
    return result_df


def refine_search(dataframe: pd.DataFrame, monomer: str | list = None, solvent: str | list = None,
                  raft_agent: str | list = None):
    len_df = len(dataframe)
    search_q_monomer, search_q_solvent, search_q_raft_agent = [np.array([True] * len_df) for _ in range(3)]
    if monomer:
        search_q_monomer = dataframe["monomer"].apply(lambda x: x in [*monomer])
    if solvent:
        search_q_solvent = dataframe["solvent"].apply(lambda x: x in [*solvent])
    if raft_agent:
        search_q_raft_agent = dataframe["RAFT-Agent"].apply(lambda x: x in [*raft_agent])
    return dataframe[search_q_monomer & search_q_solvent & search_q_raft_agent]




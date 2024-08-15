""" This File provides the functionality for a RAFT-knowledgebase website. """

import pandas as pd
import numpy as np
import plotly.express as px
from .reformat_database import format_database_to_kinetics_df, neg_growth, neg_growth_derivative

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


def add_fits_to_plot(figure, fit_func, fit_func_params, fit_func_derivative=None, *args, **kwargs):
    extended_xdata = np.linspace(-1, 16.5, 100)  # x data array for plotting the fits
    figure.add_scatter(
        x=extended_xdata, y=fit_func(extended_xdata, *fit_func_params),
        opacity=1, line=dict(dash="dot"), name=f"{fit_func.__name__} fit", *args, **kwargs)
    if fit_func_derivative:
        figure.add_scatter(
            x=extended_xdata, y=fit_func_derivative(extended_xdata, *fit_func_params),
            opacity=0.3, line=dict(dash="dash"), name=f"{fit_func_derivative.__name__}", *args, **kwargs)


class KnowledgeBase:
    def __init__(self):
        self.kinetics_df = format_database_to_kinetics_df()
        self.colors = px.colors.qualitative.Plotly  # set up a simple color palette
        self.dropdown_options = self.get_dropdown_options()

    def get_dropdown_options(self) -> dict:
        """ Get the dropdown options for the search form
        Returns
            -------
            dict
                A dictionary containing the dropdown options for the search form
        """
        return {
            "monomer": self.kinetics_df["monomer"].unique().tolist(),
            "solvent": self.kinetics_df["solvent"].unique().tolist(),
            "raft_agent": self.kinetics_df["RAFT-agent"].unique().tolist()
        }

    def search_for_exp(self, exp_nr: str | list) -> pd.DataFrame:
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
        if exp_nr is str:
            return self.kinetics_df[self.kinetics_df['exp_nr'] == exp_nr]
        else:
            return self.kinetics_df[self.kinetics_df['exp_nr'].isin(exp_nr)]

    def plot_exp(
            self, exp_nr: str | list, plot_mn: bool = False, plot_mw: bool = False, fit_curves=None, *args, **kwargs):
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
            args and kwargs
                Additional arguments and keyword arguments will be passed to the plotly express line function

            Returns
            -------
            Figure

        """
        if fit_curves is None:
            fit_curves = [True, True]
        plot_data = self.search_for_exp(exp_nr)
        exp_fig = px.line(title=f"Kinetic Curve Fit for {exp_nr}", *args, **kwargs)
        for kinetic_to_plot in plot_data.itertuples():
            x_data, ydata = kinetic_to_plot.conv_time_data
            marker_dict = dict(color=self.colors[int(kinetic_to_plot.Index) % len(self.colors)])
            exp_fig.add_scatter(x=x_data, y=ydata, mode="lines+markers", name=kinetic_to_plot.exp_nr,
                                marker=marker_dict, legendgroup=str(kinetic_to_plot.exp_nr))
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
        if any([plot_mn, plot_mw]):
            exp_fig.update_layout(yaxis_title=
                                  "Conversion [%] and M<sub>n</sub>/M<sub>w</sub> [g/mol] · 10<sup>-5</sup>",
                                  overwrite=True)

        return exp_fig

    def find_optimal_synthesis(self, monomer: str | list):
        search_q_monomer = [x in monomer for x in self.kinetics_df["monomer"]]
        result_df = self.kinetics_df[search_q_monomer].sort_values(by=["score"], ascending=False)
        return result_df

    def refine_search(self, dataframe: pd.DataFrame = None, monomer: list = None, solvent: list = None,
                      raft_agent: list = None):
        if dataframe is None:
            dataframe = self.kinetics_df
        len_df = len(dataframe)
        search_q_monomer, search_q_solvent, search_q_raft_agent = [np.array([True] * len_df) for _ in range(3)]
        if monomer and not monomer == [""]:
            search_q_monomer = dataframe["monomer"].apply(lambda x: x in [*monomer])
        if solvent and not solvent == [""]:
            search_q_solvent = dataframe["solvent"].apply(lambda x: x in [*solvent])
        if raft_agent and not raft_agent == [""]:
            search_q_raft_agent = dataframe["RAFT-agent"].apply(lambda x: x in [*raft_agent])

        search_q = dataframe[search_q_monomer & search_q_solvent & search_q_raft_agent].copy(deep=True)
        search_q["exp_nr"] = search_q["exp_nr"].apply(lambda x: x.zfill(3))

        return search_q[["exp_nr", "max_con", "theo_react_end", "monomer", "solvent", "RAFT-agent", "score"]].sort_values(by=["score"], ascending=False)

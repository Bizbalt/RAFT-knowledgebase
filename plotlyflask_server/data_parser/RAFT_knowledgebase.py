""" This File provides the functionality for a RAFT-knowledgebase website. """

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
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


new_headers = {"exp_nr": "Exp. Nr.",
               "max_con": "Max. Conv. [%]",
               "theo_react_end": "Theo. React. End [h]",
               "max_mn": "Max. Mn [10<sup>-5</sup> g/mol]",
               "monomer": "Monomer",
               "solvent": "Solvent",
               "RAFT-agent": "RAFT-agent",
               "score": "Score"}


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
            self, exp_nr: str | list, plot_conv: bool = True, plot_mn: bool = False, plot_mw: bool = False,
            fit_curves=None, stacked_plots=False, *args, **kwargs):
        """ Plot the kinetic curves for the experiment number(s) exp_nr.
        Parameters
            ----------
            exp_nr
                The experiment number(s) to plot
            plot_conv
                Whether to plot the conversion
            plot_mn
                Whether to plot the Mn data
            plot_mw
                Whether to plot the Mw data
            fit_curves
                A tuple with two boolean values, the first one determines whether to plot the fit curves, the second one
                determines whether to plot the derivative of the fit curves.
            stacked_plots
                Whether to stack the plots
            args and kwargs
                Additional arguments and keyword arguments will be passed to the plotly express line function

            Returns
            -------
            Figure

        """
        if fit_curves is None:
            fit_curves = [True, True]
        plot_data = self.search_for_exp(exp_nr).dropna()

        # stacked_plot_keywords = {}
        if stacked_plots:
            stack_amount = sum([plot_conv, plot_mn, plot_mw])
            exp_fig = make_subplots(rows=stack_amount, cols=1, shared_xaxes=True)
            exp_fig.update_xaxes(title_text="Time [h]", row=stack_amount, col=1)

        else:
            exp_fig = px.line(title=f"Kinetic Curve Fit for {exp_nr}", *args, **kwargs)

        for kinetic_to_plot in plot_data.itertuples():
            stack_amount = 1
            marker_dict = dict(color=self.colors[int(kinetic_to_plot.Index) % len(self.colors)])
            additional_plot_keywords = {"mode": "lines+markers",
                                        "marker": marker_dict,
                                        "name": kinetic_to_plot.exp_nr,
                                        "legendgroup": str(kinetic_to_plot.exp_nr),
                                        "col": 1}
            if plot_conv:
                x_data, ydata = kinetic_to_plot.conv_time_data
                exp_fig.add_scatter(x=x_data, y=ydata, row=stack_amount, **additional_plot_keywords)

                if fit_curves[0]:
                    if fit_curves[1]:
                        add_fits_to_plot(exp_fig, neg_growth, [kinetic_to_plot.fit_p1, kinetic_to_plot.fit_p2],
                                         fit_func_derivative=neg_growth_derivative, marker=marker_dict,
                                         legendgroup=str(kinetic_to_plot.exp_nr), showlegend=False, col=1, row=1)
                    else:
                        add_fits_to_plot(exp_fig, neg_growth, [kinetic_to_plot.fit_p1, kinetic_to_plot.fit_p2],
                                         marker=marker_dict, legendgroup=str(kinetic_to_plot.exp_nr), showlegend=False,
                                         col=1, row=1)
                if stacked_plots:
                    exp_fig.update_yaxes(title_text="Conv [%]", row=stack_amount, col=1)
                    stack_amount += 1

            if plot_mn:
                marker_dict["color"] = color_variant(marker_dict["color"], 30)
                x2_data, y2_data = kinetic_to_plot.Mn_time_data
                if not plot_conv or stacked_plots:
                    y2_data = y2_data * 10 ** 5
                exp_fig.add_scatter(x=x2_data, y=y2_data, row=stack_amount,
                                    **additional_plot_keywords | {"name": "Mn of " + kinetic_to_plot.exp_nr})
                if stacked_plots:
                    exp_fig.update_yaxes(title_text="M<sub>n</sub> [g/mol]", row=stack_amount, col=1)
                    stack_amount += 1

            if plot_mw:
                marker_dict["color"] = color_variant(marker_dict["color"], -60)
                x2_data, y2_data = kinetic_to_plot.Mw_time_data
                if not plot_conv or stacked_plots:
                    y2_data = y2_data * 10 ** 5
                exp_fig.add_scatter(x=x2_data, y=y2_data, row=stack_amount,
                                    **additional_plot_keywords | {"name": "Mw of " + kinetic_to_plot.exp_nr},
                                    opacity=0.5)

                if stacked_plots:
                    exp_fig.update_yaxes(title_text="M<sub>w</sub> [g/mol]", row=stack_amount, col=1)

        if not stacked_plots:
            axis_case = [plot_conv, plot_mn, plot_mw]
            match axis_case:
                # conversion only
                case [True, False, False]:
                    exp_fig.update_layout(
                        yaxis=dict(range=[-0.1, 1]), xaxis_title="Time [h]", yaxis_title="Conversion [%]")
                # any of Mn Mw
                case [False, True, True] | [False, False, True] | [False, True, False]:
                    exp_fig.update_layout(
                        yaxis=dict(range=[0, 50000]), xaxis_title="Time [h]",
                        yaxis_title="M<sub>n</sub>/M<sub>w</sub> [g/mol]")
                # conversion and any of Mn Mw
                case [True, True, True] | [True, True, False] | [True, False, True]:
                    exp_fig.update_layout(
                        yaxis_title="Conversion [%] and M<sub>n</sub>/M<sub>w</sub> [g/mol] · 10<sup>-5</sup>",
                        overwrite=True)
                case _:
                    pass
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

        # reformatting for the website-look
        search_q["exp_nr"] = search_q["exp_nr"].apply(lambda x: x.zfill(3))
        search_q["max_con"] = search_q["max_con"].apply(lambda x: x * 100)
        # truncate the conversion reaction end and score to 2 decimal places
        search_q[["max_con", "theo_react_end", "score"]] = search_q[["max_con", "theo_react_end", "score"]].map(
            lambda x: round(x, 2))
        reformatted_search = search_q[
            ["exp_nr", "max_con", "theo_react_end", "max_mn", "monomer", "RAFT-agent", "solvent", "score"]].sort_values(
            by=["score"], ascending=False)
        reformatted_search.columns = [new_headers[col] for col in reformatted_search.columns]

        return reformatted_search

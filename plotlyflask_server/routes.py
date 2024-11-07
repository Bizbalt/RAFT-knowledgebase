""" This file contains the routes for the web app. """
import os
from flask import render_template, jsonify, redirect, request
from flask import current_app as app
from flask import send_from_directory
from plotlyflask_server.data_parser.RAFT_knowledgebase import KnowledgeBase
import json

kb = KnowledgeBase()


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "chart_curve.ico", mimetype='image/vnd.microsoft.icon')


@app.route("/")
def home():
    return redirect("/raft_knowledge_base")


@app.route("/raft_knowledge_base", methods=["GET", "POST"])
def intro_page():
    if request.method == "POST":
        query = request.form.to_dict(flat=False)
        dataframe = kb.refine_search(**query)
        prior_search = ("\n".join("{}\t{}".format(*item) for item in query.items() if item[1] != [""]))
        return render_template("raft_knowledge_base.html", dataframe=dataframe.to_html(index=False, escape=False), prior_search=prior_search)
    else:
        return render_template("raft_knowledge_base.html")


@app.route("/get_dropdown_options")
def get_dropdown_options():
    return jsonify(kb.dropdown_options)


@app.route("/test_site")
def test_site():
    return render_template(
        "test_site.html", figure=kb.plot_exp(["241", "145", "343"], True, True).to_html(full_html=False))


@app.route("/plot_exp", methods=["POST"])
def plot_exp():
    # create dictionary from json object
    exp_nr_n_settings = request.json
    return kb.plot_exp(**exp_nr_n_settings, template=None).to_json()  # "plotly_dark"

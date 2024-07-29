import os
from flask import render_template, jsonify, redirect, request
from flask import current_app as app
from flask import send_from_directory
from plotlyflask_server.data_parser.RAFT_knowledgebase import KnowledgeBase

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

        return render_template("raft_knowledge_base.html", dataframe=dataframe.to_html(index=False), prior_search=query)
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

    print("test", exp_nr_n_settings,  sep="\n")
    return kb.plot_exp(**exp_nr_n_settings).to_json()

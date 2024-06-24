from flask import Blueprint, render_template, jsonify, redirect, request
from RAFT_knowledgebase import KnowledgeBase

views = Blueprint(__name__, "views")

kb = KnowledgeBase()


@views.route("/")
def home():
    return redirect("/raft_knowledge_base")


@views.route("/raft_knowledge_base", methods=["GET", "POST"])
def intro_page():
    if request.method == "POST":
        query = request.form.to_dict(flat=False)
        print(request.form, query, sep="\n")
        dataframe = kb.refine_search(**query)

        return render_template("raft_knowledge_base.html", dataframe=dataframe.to_html(), prior_search=query)
    else:
        return render_template("raft_knowledge_base.html")


@views.route("/get_dropdown_options")
def get_dropdown_options():
    return jsonify(kb.dropdown_options)


@views.route("/test_site")
def test_site():
    return render_template(
        "test_site.html", figure=kb.plot_exp(["241", "145", "343"], True, True).to_html(full_html=False))

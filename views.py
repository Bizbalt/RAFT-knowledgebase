from flask import Blueprint, render_template, jsonify, send_from_directory, redirect, request, session
from RAFT_knowledgebase import KnowledgeBase

views = Blueprint(__name__, "views")

kb = KnowledgeBase()


@views.route("/")
def home():
    return redirect("/raft_knowledge_base")


@views.route("/raft_knowledge_base")
def intro_page():
    return render_template("raft_knowledge_base.html")


@views.route("/get_dropdown_options")
def get_dropdown_options():
    return jsonify(kb.dropdown_options)


@views.route("/search_query", methods=["POST"])
def search_query():
    query = dict(request.form)
    search_result = kb.refine_search(monomer=query["monomer"], solvent=query["solvent"], raft_agent=query["RAFT-agent"])
    return search_result.to_html()


@views.route("/test_site")
def test_site():
    return render_template(
        "test_site.html", figure=kb.plot_exp(["241", "145", "343"], True, True).to_html(full_html=False))



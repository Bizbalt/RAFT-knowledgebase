from flask import Blueprint, render_template, jsonify, send_from_directory, redirect, request, session

views = Blueprint(__name__, "views")

user = {"username": "abc", "password": "123"}


@views.route("/")
def home():
    return redirect("/raft_knowledge_base")


@views.route("/raft_knowledge_base")
def intro_page():
    return render_template("raft_knowledge_base.html")


@views.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == user["username"] and password == user["password"]:
            session["user"] = username
            return redirect("/New_temp")

        return "<h1>Wrong username or password</h1>"

    return render_template("login.html")

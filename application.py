from flask import Flask
from views import views

# instantiate the app
# ToDo: I love this post on how to embed Dash in Flask: https://hackersandslackers.com/plotly-dash-with-flask/
app = Flask(__name__)

app.register_blueprint(views, url_prefix='/')


if __name__ == '__main__':
    app.run(debug=True, port=5000)

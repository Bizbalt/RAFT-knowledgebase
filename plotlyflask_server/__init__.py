from flask import Flask


# instantiate the app
# ToDo: I love this post on how to embed Dash in Flask: https://hackersandslackers.com/plotly-dash-with-flask/

def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    # app.config.from_object('config.Config') we do that later if needed

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .plotlydash.dashboard import create_dashboard
        app = create_dashboard(app)

        return app

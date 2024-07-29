from flask import Flask


# instantiate the app
def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    # app.config.from_object('config.Config') we do that later if needed

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .plotlydash.dashboard import init_dashboard
        app = init_dashboard(app)

        return app

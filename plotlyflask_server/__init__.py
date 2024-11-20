from flask import Flask


# instantiate the app
def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    # import routes
    with app.app_context():
        from . import routes

    return app

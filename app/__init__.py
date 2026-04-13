import os
from flask import Flask


def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "..", "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit
    app.secret_key = os.urandom(24)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from app.routes import bp
    app.register_blueprint(bp)

    return app

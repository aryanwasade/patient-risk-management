"""
app.py -- Flask Application Factory for Healthcare Patient Risk Mining.
"""

import os
from flask import Flask, send_from_directory, render_template_string
from backend.routes import api_bp
from src.data_loader import get_project_root


def create_app():
    """Create and configure the Flask application."""
    project_root = get_project_root()
    frontend_dir = os.path.join(project_root, 'frontend')
    static_dir = os.path.join(project_root, 'static')

    app = Flask(
        __name__,
        static_folder=static_dir,
        static_url_path='/static'
    )

    # Configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    # Serve Frontend UI
    @app.route('/')
    def index():
        index_file = os.path.join(frontend_dir, 'index.html')
        if os.path.exists(index_file):
            return send_from_directory(frontend_dir, 'index.html')
        return "<h1>Healthcare Patient Risk Pattern Mining</h1><p>Frontend under construction.</p>"

    @app.route('/<path:filename>')
    def serve_frontend_static(filename):
        # Serve styles, scripts, or assets directly from frontend directory
        if os.path.exists(os.path.join(frontend_dir, filename)):
            return send_from_directory(frontend_dir, filename)
        # If static asset requested
        if os.path.exists(os.path.join(static_dir, filename)):
            return send_from_directory(static_dir, filename)
        return ("File not found", 404)

    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f" Healthcare Patient Risk Pattern Mining Server")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)

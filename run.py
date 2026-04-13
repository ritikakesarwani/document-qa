"""
run.py
------
Application entry point.

Run with:
    python run.py

The server starts on http://localhost:5000 by default.
Models are downloaded on first request and cached locally.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 60)
    print("  Document Q&A System")
    print("  Running at: http://localhost:5000")
    print("  Note: Models will download on first use (~500 MB)")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)

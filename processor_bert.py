# processor_bert.py  (lazy load, safe fallback)
import os
import traceback

_embedding_model = None
_classifier = None
_MODEL_PATH = os.path.join("models", "log_classifier_model.joblib")  # adjust if yours is named differently

def _ensure_models_loaded():
    global _embedding_model, _classifier
    if _embedding_model is not None and _classifier is not None:
        return
    try:
        # Import inside function to avoid top-level TF/huge imports on module import
        from sentence_transformers import SentenceTransformer
        import joblib
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if os.path.exists(_MODEL_PATH):
            _classifier = joblib.load(_MODEL_PATH)
        else:
            print(f"Warning: classifier model not found at {_MODEL_PATH}")
            _classifier = None
    except Exception as e:
        print("Error loading BERT models:", e)
        print(traceback.format_exc())
        _embedding_model = None
        _classifier = None

def classify_with_bert(log_message: str) -> str:
    """
    Returns a string label or 'Unclassified' if model unavailable or low-confidence.
    """
    _ensure_models_loaded()
    if _embedding_model is None or _classifier is None:
        return "Unclassified"

    try:
        emb = _embedding_model.encode([log_message])
        # Some scikit-learn classifiers expect 2D, others accept embeddings directly
        if hasattr(_classifier, "predict_proba"):
            probs = _classifier.predict_proba(emb)[0]
            if max(probs) < 0.5:
                return "Unclassified"
        pred = _classifier.predict(emb)[0]
        return pred
    except Exception as e:
        print("Error in classify_with_bert:", e)
        print(traceback.format_exc())
        return "Unclassified"

# quick manual test when running this file directly
if __name__ == "__main__":
    for s in [
        "API returned 404 not found error",
        "Multiple login failures occurred on user 6454 account",
        "System reboot initiated by user 12345."
    ]:
        print(s, "->", classify_with_bert(s))

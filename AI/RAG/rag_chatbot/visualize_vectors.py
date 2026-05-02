"""
visualize_vectors.py
───────────────────
View stored chunk embeddings in 2D (PCA) with an interactive plot.

Usage:
    # From rag_chatbot folder; use current session or specify one
    python visualize_vectors.py
    python visualize_vectors.py --session-id abc12345

    # Save plot to HTML and open in browser
    python visualize_vectors.py --session-id abc12345 --open

Requires: scikit-learn, plotly (see requirements.txt).
"""

import sys
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
if (_project_root / ".env").exists():
    import dotenv
    dotenv.load_dotenv(_project_root / ".env")
import pathsetup  # noqa: F401

from config import cfg
from embedding.vector_store import VectorStore


def _discover_sessions():
    """Return list of session_id dirs under tmp/index/ that have faiss.index."""
    index_root = Path(cfg.LOCAL_INDEX_DIR)
    if not index_root.exists():
        return []
    sessions = []
    for path in index_root.iterdir():
        if path.is_dir() and (path / "faiss.index").exists():
            sessions.append(path.name)
    return sorted(sessions)


def _reduce_2d(vectors, method="pca"):
    """Reduce (n, dim) to (n, 2). method: 'pca' or 'tsne'."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    if method == "pca":
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(vectors)
        return xy, f"PCA (explained variance: {pca.explained_variance_ratio_.sum():.2%})"
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("Install scikit-learn: pip install scikit-learn")
        # TSNE can be slow for large n
        perplexity = min(30, max(5, vectors.shape[0] - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        xy = tsne.fit_transform(vectors)
        return xy, "t-SNE"
    raise ValueError(f"Unknown method: {method}")


def build_plotly_figure(xy, metadata, title):
    """Build an interactive Plotly scatter figure (one trace per source, hover = chunk preview)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install plotly: pip install plotly")

    sources = [m.get("source", "?") for m in metadata]
    text_preview = [
        (m.get("text", "")[:120] + "…" if len(m.get("text", "")) > 120 else m.get("text", ""))
        for m in metadata
    ]
    hover = [
        f"<b>{sources[i]}</b> (page {metadata[i].get('page', '?')})<br>{text_preview[i]}"
        for i in range(len(metadata))
    ]

    # One trace per unique source so legend = document
    unique_sources = list(dict.fromkeys(sources))
    fig = go.Figure()
    for src in unique_sources:
        mask = [s == src for s in sources]
        inds = [i for i, m in enumerate(mask) if m]
        fig.add_trace(
            go.Scatter(
                x=xy[inds, 0],
                y=xy[inds, 1],
                mode="markers",
                name=src,
                text=[hover[i] for i in inds],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=8, opacity=0.8),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG chunk vectors in 2D")
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID (default: first available index)",
    )
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="2D reduction method (default: pca)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available session IDs and exit",
    )
    args = parser.parse_args()

    sessions = _discover_sessions()
    if args.list:
        if not sessions:
            print("No indexed sessions found. Upload and process documents in the app first.")
        else:
            print("Available sessions:", ", ".join(sessions))
        return 0

    session_id = args.session_id
    if not session_id:
        if not sessions:
            print("No indexed sessions found. Upload and process documents in the app first.")
            return 1
        session_id = sessions[0]
        print(f"Using session: {session_id} (use --session-id to pick another)")

    vs = VectorStore(session_id=session_id)
    if not vs.load():
        print(f"Could not load index for session '{session_id}'.")
        return 1

    vectors, metadata = vs.get_vectors_and_metadata()
    if len(vectors) == 0:
        print("No vectors in index.")
        return 1

    print(f"Loaded {len(vectors)} vectors (dim={vectors.shape[1]}). Reducing to 2D with {args.method}...")
    xy, method_label = _reduce_2d(vectors, method=args.method)
    title = f"Chunk embeddings — {session_id} — {method_label}"

    fig = build_plotly_figure(xy, metadata, title)
    out_path = _project_root / "tmp" / "vector_viz.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"Saved: {out_path}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

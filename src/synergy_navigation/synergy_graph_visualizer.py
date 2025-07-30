import os
import json
import numpy as np
import sqlite3
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
import umap
from src.utils.cards_advisor import load_lookup_cards
from src.synergy_navigation.calculate_all_synergy import filter_cards
import src.utils.conf as conf


def load_or_compute_umap(
    all_cards, card_names, umap_file_path="umap_coords.npy", n_components=2, bert=False
):
    if os.path.exists(umap_file_path):
        print(f"Loading UMAP coords from {umap_file_path}...")
        coords = np.load(umap_file_path)
    else:
        print(
            f"UMAP file not found. Computing UMAP coords with {n_components} components..."
        )
        if bert:
            embeddings = [
                np.array(all_cards[name]["emb_predicted"][0]) for name in card_names
            ]
        else:
            embeddings = [
                np.array(all_cards[name]["tags_preds_projection"][0][0])
                for name in card_names
            ]

        embeddings = np.vstack(embeddings)
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        coords = reducer.fit_transform(embeddings)
        np.save(umap_file_path, coords)
        print(f"UMAP coords computed and saved to {umap_file_path}")

    return coords


def extract_all_sets():
    return conf.mtg_sets_dict


app = Flask(__name__)

USE_BERT = True  # Set to True if you want to use BERT embeddings, False for UMAP
DB_FILE = "synergy_cache_compressed.sqlite"
BULK_EMBEDDING_FILE = "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_20250718003320.json"
UMAP_FILE = "umap_coords.npy"
UMAP_BERT_FILE = "umap_bert_coords.npy"
EDGE_LIMIT = 2000


@lru_cache(maxsize=128)
def query_synergies_cached(sets_key, min_score, max_score, scale):
    print(
        f"Querying synergies for sets: {sets_key}, score range: [{min_score}, {max_score}], scale: {scale}"
    )
    selected_sets = sets_key.split(",")
    filtered_card_names = {
        name for name, card in all_cards.items() if card.get("set") in selected_sets
    }
    selected_idxs = {name_to_idx[n] for n in filtered_card_names if n in name_to_idx}
    if not selected_idxs:
        return [], []

    placeholders = ",".join("?" for _ in selected_idxs)
    sql = f"""
        SELECT idx_a, idx_b, score FROM synergies
        WHERE idx_a IN ({placeholders}) AND idx_b IN ({placeholders})
        AND score BETWEEN ? AND ?
        ORDER BY score DESC
        LIMIT ?
    """
    params = (
        list(selected_idxs) + list(selected_idxs) + [min_score, max_score, EDGE_LIMIT]
    )

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()

    nodes = {}
    edges = []
    for a, b, score in rows:
        # Create node A if not exists
        if a not in nodes:
            nodes[a] = {
                "data": {"id": str(a), "label": card_names[a]},
                "position": {
                    "x": float(umap_coords[a][0] * scale),
                    "y": float(umap_coords[a][1] * scale),
                },
            }

        # Create node B if not exists
        if b not in nodes:
            nodes[b] = {
                "data": {"id": str(b), "label": card_names[b]},
                "position": {
                    "x": float(umap_coords[b][0] * scale),
                    "y": float(umap_coords[b][1] * scale),
                },
            }

        width = max(1, (score - min_score) * 10)

        # Add edge A -> B
        edges.append(
            {
                "data": {
                    "id": f"{a}_{b}",
                    "source": str(a),
                    "target": str(b),
                    "score": score,
                    "width": width,
                }
            }
        )

    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    return list(nodes.values()), edges


def parse_sets_param(sets_param):
    if isinstance(sets_param, list):
        return [s for s in sets_param if s in all_sets.keys()]
    elif isinstance(sets_param, str):
        s = sets_param.strip()
        return [s] if s in all_sets.keys() else []
    else:
        return []


@app.route("/")
def index():
    return render_template("graph_visualizer.html", sets=all_sets)


@app.route("/graph_data")
def graph_data():
    print(">>> /graph_data hit")
    print("Query args:", request.args)
    try:
        sets_selected = parse_sets_param(request.args.getlist("sets[]"))
        min_score = float(request.args.get("min_score", 0.9))
        max_score = float(request.args.get("max_score", 1.0))
        scale = float(request.args.get("scale", 1000))
        if not sets_selected:
            return jsonify({"nodes": [], "edges": []})

        sets_key = ",".join(sorted(sets_selected))
        nodes, edges = query_synergies_cached(sets_key, min_score, max_score, scale)
        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        print(f"Error processing graph data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/card_info")
def card_info():
    card_name = request.args.get("name")
    if not card_name or card_name not in card_info_map:
        return jsonify({"error": "Card not found"}), 404
    return jsonify(card_info_map[card_name])


if __name__ == "__main__":

    print("Loading and filtering cards...")
    all_cards_raw = load_lookup_cards(BULK_EMBEDDING_FILE)
    all_cards = filter_cards(all_cards_raw)
    card_names = list(all_cards.keys())
    name_to_idx = {name: i for i, name in enumerate(card_names)}
    all_sets = extract_all_sets()

    # Load all cards info from bulk JSON for details panel:
    with open(BULK_EMBEDDING_FILE, "r", encoding="utf-8") as f:
        bulk_cards_data = json.load(f)

    # Map card name to full info for quick lookup:
    card_info_map = {card["name"]: card for card in bulk_cards_data}

    if USE_BERT:
        print("Loading UMAP coordinates for BERT embeddings...")
        umap_coords = load_or_compute_umap(
            all_cards, card_names, umap_file_path=UMAP_BERT_FILE, bert=True
        )
    else:
        print("Loading UMAP coordinates for standard embeddings...")
        # Use the same UMAP file for both BERT and standard embeddings
        umap_coords = load_or_compute_umap(
            all_cards, card_names, umap_file_path=UMAP_FILE, bert=False
        )
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)

    app.run(debug=True, use_reloader=False)

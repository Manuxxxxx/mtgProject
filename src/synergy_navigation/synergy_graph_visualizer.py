import os
import json
import numpy as np
import sqlite3
import threading
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, current_app
import umap
from src.utils.cards_advisor import load_lookup_cards
from src.synergy_navigation.calculate_all_synergy import filter_cards
import src.utils.conf as conf
from typing import Optional


def load_or_compute_umap(
    all_cards, card_names, umap_file_path="umap_coords.npy", n_components=2, bert=False, progress: Optional[dict] = None
):
    # Ensuring consistent indentation (4 spaces) in this function
    if progress is not None:
        progress.update({"phase": "checking_files", "processed": 0, "total": len(card_names), "percent": 0.0, "message": "Looking for existing UMAP file"})
    if os.path.exists(umap_file_path):
        print(f"Loading UMAP coords from {umap_file_path}...")
        coords = np.load(umap_file_path)
        if progress is not None:
            progress.update({"phase": "loaded", "processed": len(card_names), "total": len(card_names), "percent": 100.0, "message": "Loaded cached coordinates"})
    else:
        legacy_candidates = ["umap_coords.npy", "umap_bert_coords.npy" if bert else "umap_coords.npy"]
        # for cand in legacy_candidates:
        #     if cand != umap_file_path and os.path.exists(cand):
        #         print(f"Primary UMAP file '{umap_file_path}' not found. Using existing '{cand}' instead. Set SYNERGY_UMAP_FILE to override.")
        #         coords = np.load(cand)
        #         if progress is not None:
        #             progress.update({"phase": "loaded", "processed": len(card_names), "total": len(card_names), "percent": 100.0, "message": f"Loaded legacy file {cand}"})
        #         return coords
        print(f"UMAP file not found ({umap_file_path}). Computing UMAP ({n_components} dims) for {len(card_names)} cards - may take several minutes.")
        if progress is not None:
            progress.update({"phase": "building_embeddings", "processed": 0, "total": len(card_names), "percent": 0.0, "message": "Collecting embeddings"})
        embeddings_list = []
        for i, name in enumerate(card_names):
            try:
                vec = np.array(all_cards[name]["emb_predicted"][0]) if bert else np.array(all_cards[name]["tags_predicted"][0][0])
            except Exception as e:
                if i < 5:
                    print(f"Warning: failed extracting embedding for '{name}': {e}")
                continue
            embeddings_list.append(vec)
            if (i + 1) % 1000 == 0:
                if progress is not None:
                    pct = (i + 1) / len(card_names) * 50.0
                    progress.update({"phase": "building_embeddings", "processed": i + 1, "total": len(card_names), "percent": round(pct, 2), "message": f"Collected {i+1} embeddings"})
                print(f"  Collected {i+1} embeddings...")
        embeddings = np.vstack(embeddings_list)
        print(f"Embeddings shape: {embeddings.shape}")
        if progress is not None:
            progress.update({"phase": "umap_fitting", "processed": len(card_names), "total": len(card_names), "percent": 55.0, "message": "Running UMAP"})
        reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=True)
        coords = reducer.fit_transform(embeddings)
        if progress is not None:
            progress.update({"phase": "saving", "processed": len(card_names), "total": len(card_names), "percent": 95.0, "message": "Saving coordinates"})
        np.save(umap_file_path, coords)
        print(f"UMAP coords computed and saved to {umap_file_path}")
        if progress is not None:
            progress.update({"phase": "finished", "processed": len(card_names), "total": len(card_names), "percent": 100.0, "message": "Ready"})
    return coords


def extract_all_sets():
    return conf.mtg_sets_dict


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------
DEFAULT_DB_FILE = os.environ.get(
    "SYNERGY_DB_FILE", "synergy_cache_compressed_174_sym_small_highfn2.sqlite"
)
DEFAULT_BULK_FILE = os.environ.get(
    "SYNERGY_BULK_FILE",
    "datasets/processed/embedding_predicted/joint_tag/cards_with_tags_174_20250822234013.json",
)
DEFAULT_UMAP_FILE = os.environ.get(
    "SYNERGY_UMAP_FILE", "umap_coords_174_small_highfn2.npy"
)
DEFAULT_UMAP_BERT_FILE = os.environ.get("SYNERGY_UMAP_BERT_FILE", "umap_bert_coords.npy")
USE_BERT = os.environ.get("SYNERGY_USE_BERT", "0") == "1"
EDGE_LIMIT = int(os.environ.get("SYNERGY_EDGE_LIMIT", "2000"))
DEFER_LOAD = os.environ.get("SYNERGY_DEFER_LOAD", "0") == "1"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_resources(progress: Optional[dict] = None):
    try:
        if progress is not None:
            progress.update({"phase": "loading_cards", "message": "Loading and filtering cards", "percent": 0.0})
        print("Loading and filtering cards ...")
        all_cards_raw = load_lookup_cards(DEFAULT_BULK_FILE)
        all_cards = filter_cards(all_cards_raw)
        card_names = list(all_cards.keys())
        name_to_idx = {name: i for i, name in enumerate(card_names)}
        all_sets = extract_all_sets()
        if progress is not None:
            progress.update({"phase": "loading_cards", "message": f"Loaded {len(card_names)} cards", "percent": 5.0})
        with open(DEFAULT_BULK_FILE, "r", encoding="utf-8") as f:
            bulk_cards_data = json.load(f)
        card_info_map = {card["name"]: card for card in bulk_cards_data}
        if progress is not None:
            progress.update({"phase": "umap", "message": "Preparing UMAP", "percent": 10.0})
        if USE_BERT:
            print("Loading UMAP coordinates for BERT embeddings ...")
            umap_coords = load_or_compute_umap(
                all_cards, card_names, umap_file_path=DEFAULT_UMAP_BERT_FILE, bert=True, progress=progress
            )
        else:
            print("Loading UMAP coordinates for standard embeddings ...")
            umap_coords = load_or_compute_umap(
                all_cards, card_names, umap_file_path=DEFAULT_UMAP_FILE, bert=False, progress=progress
            )
        if umap_coords.shape[0] != len(card_names):
            raise ValueError(
                f"UMAP coords count ({umap_coords.shape[0]}) != number of cards ({len(card_names)})"
            )
        if progress is not None:
            progress.update({"phase": "db", "message": "Opening database", "percent": 98.0})
        print(f"Opening DB connection {DEFAULT_DB_FILE} ...")
        conn = sqlite3.connect(DEFAULT_DB_FILE, check_same_thread=False)
        resources = {
            "ALL_CARDS": all_cards,
            "CARD_NAMES": card_names,
            "NAME_TO_IDX": name_to_idx,
            "ALL_SETS": all_sets,
            "CARD_INFO_MAP": card_info_map,
            "UMAP_COORDS": umap_coords,
            "DB_CONN": conn,
        }
        if progress is not None:
            progress.update({"phase": "finished", "message": "Resources loaded", "percent": 100.0})
        return resources
    except Exception as e:
        if progress is not None:
            progress.update({"phase": "error", "message": str(e), "percent": -1})
        raise


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------

def create_app(load_now: bool = True):
    app = Flask(__name__)
    app.config['LOAD_PROGRESS'] = {"phase": "init", "message": "App created", "percent": 0.0}

    def background_load():
        try:
            resources = _load_resources(app.config['LOAD_PROGRESS'])
            app.config.update(resources)
        except Exception:
            pass

    if load_now and not DEFER_LOAD:
        resources = _load_resources(app.config['LOAD_PROGRESS'])
        app.config.update(resources)
    else:
        # start background thread
        t = threading.Thread(target=background_load, daemon=True)
        t.start()

    @lru_cache(maxsize=128)
    def query_synergies_cached(sets_key, min_score, max_score, scale):
        # Access runtime resources from app config
        all_cards = current_app.config["ALL_CARDS"]
        name_to_idx = current_app.config["NAME_TO_IDX"]
        card_names = current_app.config["CARD_NAMES"]
        umap_coords = current_app.config["UMAP_COORDS"]
        conn = current_app.config["DB_CONN"]
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
            if a not in nodes:
                nodes[a] = {
                    "data": {"id": str(a), "label": card_names[a]},
                    "position": {
                        "x": float(umap_coords[a][0] * scale),
                        "y": float(umap_coords[a][1] * scale),
                    },
                }
            if b not in nodes:
                nodes[b] = {
                    "data": {"id": str(b), "label": card_names[b]},
                    "position": {
                        "x": float(umap_coords[b][0] * scale),
                        "y": float(umap_coords[b][1] * scale),
                    },
                }
            width = max(1, (score - min_score) * 10)
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
        all_sets = current_app.config["ALL_SETS"]
        if isinstance(sets_param, list):
            return [s for s in sets_param if s in all_sets.keys()]
        elif isinstance(sets_param, str):
            s = sets_param.strip()
            return [s] if s in all_sets.keys() else []
        else:
            return []

    @app.route("/")
    def index():
        return render_template("graph_visualizer.html", sets=current_app.config["ALL_SETS"])

    @app.route("/graph_data")
    def graph_data():
        print(">>> /graph_data hit")
        print("Query args:", request.args)
        try:
            sets_selected = parse_sets_param(request.args.getlist("sets[]") or request.args.getlist("sets"))
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
        card_info_map = current_app.config["CARD_INFO_MAP"]
        if not card_name or card_name not in card_info_map:
            return jsonify({"error": "Card not found"}), 404
        return jsonify(card_info_map[card_name])

    @app.route('/progress')
    def progress():
        return jsonify(app.config.get('LOAD_PROGRESS', {}))

    @app.teardown_appcontext
    def close_connection(exc):
        conn = current_app.config.get("DB_CONN")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return app


# Default app instance for WSGI servers (e.g. `flask run`)
app = create_app(load_now=True)


if __name__ == "__main__":
    # Running directly still works
    app.run(debug=True, use_reloader=False)

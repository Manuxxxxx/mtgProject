import sqlite3

DB_FILE = "synergy_cache copy.sqlite"  # Adjust if your DB filename is different

def visualize_top_synergies(limit=100):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT card_a, card_b, score
        FROM synergies
        ORDER BY score DESC
        LIMIT ?
    """, (limit,))

    results = cur.fetchall()
    conn.close()

    print(f"Top {limit} synergy pairs by score:")
    for i, (a, b, score) in enumerate(results, 1):
        print(f"{i:3d}. {a} + {b}: {score:.4f}")

if __name__ == "__main__":
    visualize_top_synergies()

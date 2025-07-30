import sqlite3

DB_FILE = "synergy_cache_compressed.sqlite"

def create_index():
    print("Creating index on 'score' in synergies table... MAY TAKE A WHILE")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON synergies(score);")
    conn.commit()
    conn.close()
    print("Index on 'score' created or already exists.")

def count_synergies_batch(thresholds):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Build CASE statements for each interval
    cases = []
    for i in range(len(thresholds) - 1):
        low = thresholds[i]
        high = thresholds[i+1]
        op = "<=" if i == len(thresholds) - 2 else "<"  # include upper bound in last interval
        case = f"SUM(CASE WHEN score >= {low} AND score {op} {high} THEN 1 ELSE 0 END) AS cnt_{i}"
        cases.append(case)
    sql = f"SELECT {', '.join(cases)} FROM synergies"

    cur.execute(sql)
    counts = cur.fetchone()
    conn.close()

    # Map counts back to intervals
    result = []
    for i in range(len(thresholds) - 1):
        result.append((thresholds[i], thresholds[i+1], counts[i]))
    return result

if __name__ == "__main__":
    create_index()
    # Define synergy score thresholds (edges counted in [low, high) ranges)
    thresholds = [0.97, 1.0]
    counts = count_synergies_batch(thresholds)

    print("Synergy counts per score interval:")
    for low, high, count in counts:
        print(f"  {low:.2f} ≤ score < {high:.2f}: {count:,} edges")

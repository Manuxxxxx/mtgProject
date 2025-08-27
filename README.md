# MTG Project

This repository contains scripts and tools for calculating, analyzing, and visualizing card synergies for Magic: The Gathering (MTG) decks using BERT embeddings and custom models.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd mtgProject
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Python path:

```bash
export PYTHONPATH="/home/emanuele/Documenti/Università/ANLP/mtgProject/"
```

> **Note:** Adjust the path according to the location of your project folder.

---

## Scripts Overview & Usage

### 1. Joint Training
**File:** `src/joint_train.py`  
**Purpose:** Train models jointly using configuration files.  
**Usage:**

```bash
python src/joint_train.py <config_file>
```

**Example:**

```bash
python src/joint_train.py src/config_runs/runs_configs_joint.json
```

> You can configure the JSON file or use pre-made ones.

---

### 2. Calculate Synergies
**File:** `src/synergy_navigation/calculate_all_synergy.py`  
**Purpose:** Calculate synergies between cards using models and BERT embeddings.  
**Usage:**

```bash
python src/synergy_navigation/calculate_all_synergy.py
```

> Important: Set the correct checkpoints, model settings, and store files. Can calculate synergies for all cards or a subset. Results are stored in a SQLite database.

---

### 3. Calculate BERT Embeddings
**File:** `src/synergy_navigation/embeddings_calculator.py`  
**Purpose:** Calculate BERT outputs for cards and add them to the bulk file.  
**Usage:**

```bash
python src/synergy_navigation/embeddings_calculator.py
```

> Important: Set the correct model checkpoints and store files. Outputs are saved back into the bulk file.

---

### 4. Visualize Synergy Graph
**File:** `src/synergy_navigation/synergy_graph_visualizer.py`  
**Purpose:** Launch a web interface to navigate card synergies.  
**Usage:**

```bash
python src/synergy_navigation/synergy_graph_visualizer.py
```

> Requires the SQLite database of calculated synergies.

---

### 5. Card Advisor
**File:** `src/utils/cards_advisor.py`  
**Purpose:** Recommend the best cards for a partial deck based on average synergy.  
**Usage:**

```bash
python src/utils/cards_advisor.py
```

> Optionally specify a commander to refine recommendations. Useful for quick testing.

---

### Other Scripts
There are many additional `.py` files in the repository, primarily used for **data scraping and data management**. These are more trivial utility scripts, but they support the main workflows described above.

---

## Notes

- For all scripts, make sure to first export the Python path as described above.  
- Ensure models, checkpoints, and store files are correctly set up before running the scripts.

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

## Bulk Card Data Setup

To generate all required data:

1. Download the "Oracle Cards" bulk JSON from Scryfall. It contains one card entry per Oracle ID.  
   👉 [Download Oracle Cards Bulk Data](https://scryfall.com/docs/api/bulk-data)

---

## Scripts Overview & Usage

### 1. Parse the Bulk File

**File:** `src/data_managment/json_bulk_parser.py`  
**Purpose:** Parses the downloaded bulk JSON.  
**Instructions:**

-   Modify the input bulk file path directly in the script source before running.

```bash
python src/data_managment/json_bulk_parser.py
```

---

### 2. Scrape Tag Data from Scryfall

**File:** `src/data_managment/scraping/scrapper_scryfallTagger.py`  
**Purpose:** Scrapes tag metadata from the Scryfall website.  
**Usage:**

```bash
python src/data_managment/scraping/scrapper_scryfallTagger.py
```

> **Important:** Requires copying browser cookies to enable access to Scryfall session-specific data.

---

### 3. Scrape Synergy Data from EDHREC

**File:** `src/data_managment/scraping/scrapper_edhrec.py`  
**Purpose:** Scrapes synergy data (e.g., card synergy stats) from EDHREC.  
**Usage:**

```bash
python src/data_managment/scraping/scrapper_edhrec.py
```

---

### 4. Append Tag Data to Parsed Bulk File

**File:** `src/data_managment/tag_appender.py`  
**Purpose:** Merges tag data into the parsed bulk JSON.  
**Usage:**

```bash
python src/data_managment/tag_appender.py
```

> **Outcome:** Produces two main files:
>
> -   A parsed bulk file enhanced with tag information
> -   A synergy data file  
>     These are required for model training.

---

### 5. Joint Training

**File:** `src/joint_train.py`  
**Purpose:** Trains models based on configuration files.  
**Usage:**

```bash
python src/joint_train.py <config_file>
```

**Example:**

```bash
python src/joint_train.py src/config_runs/runs_configs_joint.json
```

> Use custom or pre-made JSON configs.

---

### 6. Calculate Synergies

**File:** `src/synergy_navigation/calculate_all_synergy.py`  
**Purpose:** Computes card synergies using models and embeddings, storing results in SQLite.  
**Usage:**

```bash
python src/synergy_navigation/calculate_all_synergy.py
```

> Requires properly configured model checkpoints and store files. Supports both full and subset calculations.

---

### 7. Compute BERT Embeddings

**File:** `src/synergy_navigation/embeddings_calculator.py`  
**Purpose:** Generates BERT embeddings and appends them to the bulk file.  
**Usage:**

```bash
python src/synergy_navigation/embeddings_calculator.py
```

> Requires correct model and store configurations. Saves results back to the bulk file.

---

### 8. Visualize Synergy Graph

**File:** `src/synergy_navigation/synergy_graph_visualizer.py`  
**Purpose:** Launches a web interface to explore card synergies.  
**Usage:**

```bash
python src/synergy_navigation/synergy_graph_visualizer.py
```

> Needs the synergy SQLite database.

---

### 9. Card Advisor

**File:** `src/utils/cards_advisor.py`  
**Purpose:** Recommends top cards based on synergy for a partial deck (optional commander supported).  
**Usage:**

```bash
python src/utils/cards_advisor.py
```

> Great for quick deck tests.

---

### Other Scripts

There are additional utility `.py` files in the repository, mainly used for **data scraping and management**. These support the main tasks but are generally straightforward.

---

## Notes

-   Always run:

```bash
export PYTHONPATH="/home/emanuele/Documenti/Università/ANLP/mtgProject/"
```

before executing any scripts.

-   Ensure that **model checkpoints**, **store files**, and **cookies** (for scraping) are correctly set before running the scripts.

-   Remember to download and parse the Scryfall bulk data as your first step in the data pipeline.

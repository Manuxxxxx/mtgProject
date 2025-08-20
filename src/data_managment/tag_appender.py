import json
from typing import Counter
import os
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

import src.utils.conf as conf


def get_card_name_from_set_number(set_code, collector_number, data):

    # Iterate through the cards in the JSON data
    for card in data:
        if (
            card.get("set_code") == set_code
            and card.get("collector_number") == collector_number
        ):
            return card

    return None


def get_tags_from_set_number(set_code, collector_number, tag_data):
    # get all the tags for a specific set code and collector number
    return tag_data[set_code][collector_number]


def get_all_card_without_duplicate(input_file_processed_json):
    with open(input_file_processed_json, "r") as file:
        data = json.load(file)

    print(f"Total cards before removing duplicates: {len(data)}")
    # remove duplicates -> same name
    unique_cards = {}
    for card in data:
        name = card.get("name")
        if name not in unique_cards:
            unique_cards[name] = card
    print(f"Total cards after removing duplicates: {len(unique_cards)}")
    return list(unique_cards.values())


def add_tag_to_card(card, tag_data, include_tags=None):
    set_code = card.get("set")
    collector_number = card.get("collector_number")

    # Get the tag from the tag data
    tags = get_tags_from_set_number(set_code, collector_number, tag_data)
    if tags:
        if include_tags is not None:
            # Filter tags based on include_tags
            tags = [tag for tag in tags if tag in include_tags]
        card["tags_labels"] = tags
    else:
        card["tags_labels"] = None
    return card


def add_tags_to_all_cards(card_data, tag_data, include_tags=None):
    # Add tags to all cards
    for card in card_data:
        card = add_tag_to_card(card, tag_data, include_tags)

    return card_data


def extract_all_tags(tag_data):

    unique_tags = set()

    for set_code, data in tag_data.items():
        for key, value in data.items():
            if key == "collector_number":
                continue
            for tag_entry in value:
                tag = tag_entry.get("tag")
                if tag:
                    unique_tags.add(tag)

    return list(unique_tags)


def load_tag_data(input_file_tagger):
    # Load the tag data from the JSON file
    with open(input_file_tagger, "r") as file:
        tag_data = json.load(file)

    smaller_tag_data = {}

    for set_code, data in tag_data.items():
        for key, value in data.items():
            if key == "collector_number":
                continue
            card_tags = []
            for tag_entry in value["functional_tags"]:
                tag_name = tag_entry["name"]
                # Add the tag to the card_tags list
                card_tags.append(tag_name)
                # if the tag has a field ancestor append each ancestor to the tags
                if "ancestors" in tag_entry:
                    ancestors = tag_entry["ancestors"]
                    if isinstance(ancestors, list):
                        for ancestor in ancestors:
                            if isinstance(ancestor, str):
                                card_tags.append(ancestor)

            # Add the card_tags to the smaller_tag_data
            if set_code not in smaller_tag_data:
                smaller_tag_data[set_code] = {}

            smaller_tag_data[set_code][key] = card_tags
    return smaller_tag_data


def create_tag_dependency_graph(input_file_tagger):
    """
    Creates a directed tag dependency graph from tag data.
    Nodes are tag names, edges represent 'ancestor' relationships.

    Parameters:
        input_file_tagger (str): Path to the JSON file containing tag data.

    Returns:
        networkx.DiGraph: A directed graph with tags as nodes and edges as parent-child relationships.
    """
    with open(input_file_tagger, "r") as file:
        tag_data = json.load(file)

    graph = nx.DiGraph()

    print("Creating tag dependency graph...")

    # Track tag usage count and ancestor relationships
    tag_usage = defaultdict(int)
    tag_ancestors = defaultdict(set)
    tag_sources = defaultdict(set)

    for set_code, data in tag_data.items():
        for card_id, card_data in data.items():
            if card_id == "collector_number":
                continue

            for tag_entry in card_data.get("functional_tags", []):
                # print(f"Processing tag: {tag_entry['name']} from set {set_code}")
                tag_name = tag_entry["name"]
                tag_usage[tag_name] += 1
                tag_sources[tag_name].add(set_code)

                if not graph.has_node(tag_name):
                    graph.add_node(tag_name, label=tag_name, type="tag")

                # Process ancestors
                if "ancestors" in tag_entry:
                    # print("Processing ancestors for tag:", tag_name)
                    ancestors = tag_entry["ancestors"]
                    if isinstance(ancestors, list):
                        for ancestor in ancestors:
                            if isinstance(ancestor, str):
                                # print("Adding ancestor:", ancestor)
                                tag_ancestors[tag_name].add(ancestor)
                                tag_sources[ancestor].add(set_code)

                                # Add ancestor node if missing
                                if not graph.has_node(ancestor):
                                    graph.add_node(ancestor, label=ancestor, type="tag")

                                # Add directed edge
                                graph.add_edge(
                                    ancestor, tag_name, relationship="inherits"
                                )
                                
                                

    # Add usage and source metadata
    for tag in graph.nodes:
        graph.nodes[tag]["usage_count"] = tag_usage[tag]
        graph.nodes[tag]["source_sets"] = (
            ", ".join(sorted(tag_sources[tag])) if tag_sources[tag] else "unknown"
        )
        graph.nodes[tag]["is_root"] = str(graph.in_degree(tag) == 0).lower()
        graph.nodes[tag]["is_leaf"] = str(graph.out_degree(tag) == 0).lower()
        graph.nodes[tag]["degree"] = graph.degree(tag)
        graph.nodes[tag]["in_degree"] = graph.in_degree(tag)
        graph.nodes[tag]["out_degree"] = graph.out_degree(tag)

    return graph

def extract_hierarchy_edges(graph: nx.DiGraph, tag_to_idx: str):
    """
    Extracts hierarchy edges (child_idx, parent_idx) for a subset of tags in idx_to_tag_file,
    based on the full dependency graph.

    Args:
        graph (nx.DiGraph): Full tag dependency graph.
        idx_to_tag_file (str): Path to JSON file containing index-to-tag mapping.

    Returns:
        List[Tuple[int, int]]: List of (child_idx, parent_idx) tuples.
    """
    with open(tag_to_idx, "r") as f:
        tag_to_idx = json.load(f)

    # Convert keys to int if necessary
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    used_tags = set(tag_to_idx.keys())

    hierarchy_edges = []

    for tag in used_tags:
        if tag not in graph:
            continue  # Tag not in full graph

        for parent in graph.predecessors(tag):
            if parent in used_tags:
                child_idx = tag_to_idx[tag]
                parent_idx = tag_to_idx[parent]
                hierarchy_edges.append((child_idx, parent_idx))

    return hierarchy_edges


def extract_all_tags_with_min_freq(tag_data, min_count=20):
    tag_counter = Counter()

    for set_code, data in tag_data.items():
        for key, value in data.items():
            for tag_entry in value:
                tag_counter[tag_entry] += 1

    return {tag for tag, count in tag_counter.items() if count >= min_count}


def filter_cards_by_sets(card_data, sets_to_include):
    filtered_cards = []
    for card in card_data:
        if card.get("set") in sets_to_include:
            filtered_cards.append(card)
    return filtered_cards


if __name__ == "__main__":
    MIN_COUNT = 300
    input_file_tagger = "datasets/scryfallTagger_data/store_scrapped_ancestors.json"
    
        

    # graph = create_tag_dependency_graph(input_file_tagger)
    # print(extract_hierarchy_edges(graph, "datasets/processed/tag_included/tag_to_index_641_20250730155929.json"))
    
    # nx.draw(graph, with_labels=True)
    # plt.show()
    # save the graph to a file
    # nx.write_graphml(graph, "datasets/scryfallTagger_data/tag_dependency_graph.graphml")

    # exit(0)

    tag_data = load_tag_data(input_file_tagger)

    tags_to_include = extract_all_tags_with_min_freq(tag_data, min_count=MIN_COUNT)
    tag_to_index = {tag: idx for idx, tag in enumerate(tags_to_include)}
    # print(tag_to_index)
    print(f"Total tags to include: {len(tags_to_include)}")

    input_file_processed_json = (
        "datasets/processed/indent/commander_legal_cards20250630171009.json"
    )

    card_data = filter_cards_by_sets(
        get_all_card_without_duplicate(input_file_processed_json), conf.all_sets
    )
    print(f"Total cards after set filter: {len(card_data)}")
    card_data = add_tags_to_all_cards(card_data, tag_data, tags_to_include)
    # Save the updated card data to a new JSON file
    os.makedirs(conf.processed_tag_dir, exist_ok=True)
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(
        conf.processed_tag_dir, f"cards_with_tags_{len(tags_to_include)}_{time}.json"
    )
    output_file_tag_to_index = os.path.join(
        conf.processed_tag_dir, f"tag_to_index_{len(tags_to_include)}_{time}.json"
    )
    with open(output_file, "w") as f:
        json.dump(card_data, f, indent=2)
    with open(output_file_tag_to_index, "w") as f:
        json.dump(tag_to_index, f, indent=2)

    # all_tags = extract_all_tags(tag_data)
    # print(f"Total tags: {len(all_tags)}")
    # tags_min = extract_all_tags_with_min_freq(tag_data, min_count=20)
    # print(f"Total tags with min count 20: {len(tags_min)}")

    # print(f"Total tags with min count 40: {len(extract_all_tags_with_min_freq(tag_data, min_count=40))}") //271
    # print(f"Total tags with min count 60: {len(extract_all_tags_with_min_freq(tag_data, min_count=60))}")
    # print(f"Total tags with min count 80: {len(extract_all_tags_with_min_freq(tag_data, min_count=80))}")
    # print(f"Total tags with min count 100: {len(extract_all_tags_with_min_freq(tag_data, min_count=100))}")
    # print(f"Total tags with min count 120: {len(extract_all_tags_with_min_freq(tag_data, min_count=120))}")
    # print(f"Total tags with min count 140: {len(extract_all_tags_with_min_freq(tag_data, min_count=140))}")
    # print(f"Total tags with min count 160: {len(extract_all_tags_with_min_freq(tag_data, min_count=160))}")

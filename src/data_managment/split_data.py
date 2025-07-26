import json
import random
import mtgProject.src.utils.conf as conf


def split_data(input_file,save_dir, train_split_dim = 0.8, test_split_dim=0.1):
    """
    Split the data into train, test, and validation sets.
    
    Args:
        input_file (str): Path to input JSON file
        train_split_dim (float): Proportion of data to use for training
        test_split_dim (float): Proportion of data to use for testing
    """
    try:
        # Load the input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            cards = json.load(f)

        # Shuffle the cards
        random.shuffle(cards)

        # Calculate split indices
        total_cards = len(cards)
        train_end = int(total_cards * train_split_dim)
        test_end = int(total_cards * (train_split_dim + test_split_dim))

        # Split the data
        train_data = cards[:train_end]
        test_data = cards[train_end:test_end]
        val_data = cards[test_end:]

        # Save the split data to new files
        with open(f"{save_dir}/train_data.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"{save_dir}/test_data.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        with open(f"{save_dir}/val_data.json", "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=4, ensure_ascii=False)
        

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_file = conf.embedding_file
    save_dir = conf.split_dir
    split_data(input_file, save_dir)
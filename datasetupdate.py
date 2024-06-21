import os
from datasets import load_dataset, get_dataset_config_names
import requests
import json

# Function to get the latest version of the dataset
def get_latest_version(dataset_name):
    url = f"https://huggingface.co/api/datasets/{dataset_name}"
    response = requests.get(url)
    if response.status_code == 200:
        dataset_info = response.json()
        return dataset_info.get('lastModified')
    else:
        print(f"Error fetching dataset info: {response.status_code}")
        return None

# Function to save the latest version info to a file
def save_latest_version(dataset_name, version, save_dir="."):
    safe_name = dataset_name.replace("/", "_")
    with open(os.path.join(save_dir, f"{safe_name}_version.json"), "w") as f:
        json.dump({"latest_version": version}, f)

# Function to load the latest version info from a file
def load_latest_version(dataset_name, save_dir="."):
    safe_name = dataset_name.replace("/", "_")
    try:
        with open(os.path.join(save_dir, f"{safe_name}_version.json"), "r") as f:
            data = json.load(f)
            return data.get("latest_version")
    except FileNotFoundError:
        return None

# Function to fetch and update dataset
def fetch_and_update_dataset(dataset_name, config_name=None, save_dir="."):
    latest_version = get_latest_version(dataset_name)
    saved_version = load_latest_version(dataset_name, save_dir)

    if latest_version != saved_version:
        print(f"Fetching dataset: {dataset_name} (new version available)")
        dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
        save_latest_version(dataset_name, latest_version, save_dir)
        save_path = os.path.join(save_dir, dataset_name.replace("/", "_"))
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)
        print(f"Dataset {dataset_name} updated to version {latest_version} and saved to {save_path}")
    else:
        print(f"Dataset {dataset_name} is up to date")

# Example usage
datasets_to_check = [
    "imdb",  # Single configuration dataset
    "glue",  # Multi-configuration dataset
    "evreny/prompt_injection_tr"  # New dataset to check
]

for dataset_name in datasets_to_check:
    if dataset_name == "glue":
        config_names = get_dataset_config_names(dataset_name)
        for config_name in config_names:
            fetch_and_update_dataset(dataset_name, config_name)
    else:
        fetch_and_update_dataset(dataset_name)

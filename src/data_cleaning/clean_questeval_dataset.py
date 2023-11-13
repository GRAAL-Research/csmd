import os.path

import pandas as pd
from datasets import load_dataset

"""
This script's purpose is to remove duplicates between ASSET and QuestEval datasets.
"""

# We first load the ASSET dataset using the HuggingFace interface and hosting.

asset_dataset = load_dataset("asset", "ratings", split="full")
asset_dataset_df = asset_dataset.to_pandas()

# We load the QuestEval dataset included in this repository.
questeval_dataset = pd.read_csv(
    os.path.join("datastore", "questeval_simplification_likert_ratings.csv")
)
questeval_dataset = questeval_dataset.drop(columns=["references", "worker_id"])

# We identify the similar indexes between the two datasets based on source, simplification, rating, sentence_id,
# aspect and the system used to generate the simplification.

similar_indexes = []
for data in asset_dataset_df.iterrows():
    source = data[1]["original"]
    simplification = data[1]["simplification"]
    rating = data[1]["rating"]
    sentence_id = data[1]["original_sentence_id"]
    aspect = data[1]["aspect"]
    if aspect == 0:
        aspect = "meaning"
    elif aspect == 1:
        aspect = "fluency"
    elif aspect == 2:
        aspect = "simplicity"

    similar_index = questeval_dataset[
        (
            (questeval_dataset["source"] == source)
            & (questeval_dataset["simplification"] == simplification)
            & (questeval_dataset["rating"] == rating)
            & (questeval_dataset["sentence_id"] == sentence_id)
            & (questeval_dataset["simplification_type"] == "system")
            & (questeval_dataset["aspect"] == aspect)
        )
    ].index.tolist()

    if len(similar_index) > 0:
        similar_indexes.extend(similar_index)

# We keep only the simplification not already included in ASSET.
questeval_dataset.drop(similar_indexes).to_csv(
    "./datastore/questeval_simplification_likert_ratings_cleaned.csv", index=False
)

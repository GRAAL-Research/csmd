import os.path

import pandas as pd
from datasets import load_dataset

from dataset_analysis_tools import dataset_analysis_batch_process
from figures_generators import (
    get_dataset_analysis_table,
    get_dataset_rating_table,
)

n_cores = 3
root = ".."

saving_dir = os.path.join(root, "datastore", "figures")
os.makedirs(os.path.join(root, "datastore", "results"), exist_ok=True)

# Stats for ASSET dataset
asset_dataset = load_dataset("asset", "ratings", split="full")
asset_dataset_df = asset_dataset.to_pandas()
asset_dataset_df = asset_dataset_df[
    asset_dataset_df["aspect"] == 0
]  # 0 is the meaning criteria

# Merging the ASSET dataset into a mean per sentence
merge_asset_dataset_df = asset_dataset_df[
    ["original", "simplification", "original_sentence_id"]
].drop_duplicates()
asset_mean_rating = asset_dataset_df.groupby("original_sentence_id")["rating"].mean()
merge_asset_dataset_df["rating"] = list(asset_mean_rating)

(
    vocabulary_size_source,
    vocabulary_size_lexical_words_source,
    average_sentences_len_source,
    average_sentence_len_lexical_words_source,
    lexical_diversity_source,
) = dataset_analysis_batch_process(
    dataset=[text[1]["original"] for text in merge_asset_dataset_df.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "asset_dataset_analysis_source.txt"
    ),
    num_cores=n_cores,
)

(
    vocabulary_size_simplification,
    vocabulary_size_lexical_words_simplification,
    average_sentences_len_simplification,
    average_sentence_len_lexical_words_simplification,
    lexical_diversity_simplification,
) = dataset_analysis_batch_process(
    dataset=[text[1]["simplification"] for text in merge_asset_dataset_df.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "asset_dataset_analysis_simplification.txt"
    ),
    num_cores=n_cores,
)

data = [
    (
        vocabulary_size_source,
        vocabulary_size_lexical_words_source,
        average_sentences_len_source,
        average_sentence_len_lexical_words_source,
        lexical_diversity_source,
    ),
    (
        vocabulary_size_simplification,
        vocabulary_size_lexical_words_simplification,
        average_sentences_len_simplification,
        average_sentence_len_lexical_words_simplification,
        lexical_diversity_simplification,
    ),
]

# Stats for SimpDa_2022 dataset
simpDA_dataset = pd.read_csv(
    os.path.join(root, "datastore", "simpDA_2022.csv")
)  # https://github.com/Yao-Dou/LENS/blob/master/data/simpDA_2022.csv

# Merging the SimpDA dataset into a mean per sentence
merge_simpDA_dataset_df = simpDA_dataset[
    ["Input.original", "Input.simplified", "Input.id"]
].drop_duplicates()
simpDA_mean_rating = simpDA_dataset.groupby("Input.simplified")[
    "Answer.adequacy"
].mean()
merge_simpDA_dataset_df["Answer.adequacy"] = list(simpDA_mean_rating)

(
    vocabulary_size_source,
    vocabulary_size_lexical_words_source,
    average_sentences_len_source,
    average_sentence_len_lexical_words_source,
    lexical_diversity_source,
) = dataset_analysis_batch_process(
    dataset=[text[1]["Input.original"] for text in merge_simpDA_dataset_df.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "simpDA_dataset_analysis_source.txt"
    ),
    num_cores=n_cores,
)

(
    vocabulary_size_simplification,
    vocabulary_size_lexical_words_simplification,
    average_sentences_len_simplification,
    average_sentence_len_lexical_words_simplification,
    lexical_diversity_simplification,
) = dataset_analysis_batch_process(
    dataset=[
        text[1]["Input.simplified"] for text in merge_simpDA_dataset_df.iterrows()
    ],
    output_path=os.path.join(
        root, "datastore", "results", "simpDA_dataset_analysis_simplification.txt"
    ),
    num_cores=n_cores,
)

data.extend(
    [
        (
            vocabulary_size_source,
            vocabulary_size_lexical_words_source,
            average_sentences_len_source,
            average_sentence_len_lexical_words_source,
            lexical_diversity_source,
        ),
        (
            vocabulary_size_simplification,
            vocabulary_size_lexical_words_simplification,
            average_sentences_len_simplification,
            average_sentence_len_lexical_words_simplification,
            lexical_diversity_simplification,
        ),
    ]
)

# Stats for Simplicity-DA dataset
simplicity_DA_dataset = pd.read_csv(
    os.path.join(root, "datastore", "simplicity_DA.csv")
)  # https://github.com/feralvam/metaeval-simplification/blob/main/data/simplicity_DA.csv

(
    vocabulary_size_source,
    vocabulary_size_lexical_words_source,
    average_sentences_len_source,
    average_sentence_len_lexical_words_source,
    lexical_diversity_source,
) = dataset_analysis_batch_process(
    dataset=[text[1]["orig_sent"] for text in simplicity_DA_dataset.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "simplicityDA_dataset_analysis_source.txt"
    ),
    num_cores=n_cores,
)

(
    vocabulary_size_simplification,
    vocabulary_size_lexical_words_simplification,
    average_sentences_len_simplification,
    average_sentence_len_lexical_words_simplification,
    lexical_diversity_simplification,
) = dataset_analysis_batch_process(
    dataset=[text[1]["simp_sent"] for text in simplicity_DA_dataset.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "simplicityDA_dataset_analysis_simplification.txt"
    ),
    num_cores=n_cores,
)

data.extend(
    [
        (
            vocabulary_size_source,
            vocabulary_size_lexical_words_source,
            average_sentences_len_source,
            average_sentence_len_lexical_words_source,
            lexical_diversity_source,
        ),
        (
            vocabulary_size_simplification,
            vocabulary_size_lexical_words_simplification,
            average_sentences_len_simplification,
            average_sentence_len_lexical_words_simplification,
            lexical_diversity_simplification,
        ),
    ]
)

# Stats for the cleaned QuestEval dataset (see clean_questeval_dataset for more details).
# We use the cleaned version of QuestEval that duplicate with ASSET where remove.
questeval_dataset = pd.read_csv(
    os.path.join(
        root, "datastore", "questeval_simplification_likert_ratings_cleaned.csv"
    )
)

questeval_dataset = questeval_dataset[questeval_dataset["aspect"] == "meaning"]

merge_questeval_dataset_df = questeval_dataset[
    ["source", "simplification"]
].drop_duplicates(keep="last")
questeval_mean_rating = questeval_dataset.groupby(["source", "simplification"])[
    "rating"
].mean()
merge_questeval_dataset_df["rating"] = list(questeval_mean_rating)
merge_questeval_dataset_df["system_name"] = questeval_dataset.loc[
    questeval_dataset[["source", "simplification"]].drop_duplicates(keep="last").index
]["system_name"]

(
    vocabulary_size_source,
    vocabulary_size_lexical_words_source,
    average_sentences_len_source,
    average_sentence_len_lexical_words_source,
    lexical_diversity_source,
) = dataset_analysis_batch_process(
    dataset=[text[1]["source"] for text in merge_questeval_dataset_df.iterrows()],
    output_path=os.path.join(
        root, "datastore", "results", "questeval_dataset_analysis_source.txt"
    ),
    num_cores=n_cores,
)

(
    vocabulary_size_simplification,
    vocabulary_size_lexical_words_simplification,
    average_sentences_len_simplification,
    average_sentence_len_lexical_words_simplification,
    lexical_diversity_simplification,
) = dataset_analysis_batch_process(
    dataset=[
        text[1]["simplification"] for text in merge_questeval_dataset_df.iterrows()
    ],
    output_path=os.path.join(
        "../../..",
        "datastore",
        "results",
        "questeval_dataset_analysis_simplification.txt",
    ),
    num_cores=n_cores,
)

data.extend(
    [
        (
            vocabulary_size_source,
            vocabulary_size_lexical_words_source,
            average_sentences_len_source,
            average_sentence_len_lexical_words_source,
            lexical_diversity_source,
        ),
        (
            vocabulary_size_simplification,
            vocabulary_size_lexical_words_simplification,
            average_sentences_len_simplification,
            average_sentence_len_lexical_words_simplification,
            lexical_diversity_simplification,
        ),
    ]
)

# Stats for CSMD
renamed_merge_simpDA_dataset_df = merge_simpDA_dataset_df.rename(
    columns={
        "Input.original": "original",
        "Input.simplified": "simplification",
        "Answer.adequacy": "rating",
        "Input.id": "original_sentence_id",
    }
)
simplicity_DA_dataset_dropped = simplicity_DA_dataset.drop(
    columns=[
        "sys_name",
        "sys_type",
        "fluency",
        "fluency_zscore",
        "meaning_zscore",
        "simplicity",
        "simplicity_zscore",
    ]
)
renamed_simplicity_DA_dataset = simplicity_DA_dataset_dropped.rename(
    columns={
        "orig_sent": "original",
        "simp_sent": "simplification",
        "meaning": "rating",
        "sent_id": "original_sentence_id",
    }
)

merged_all_data_dataset = pd.concat(
    [
        merge_asset_dataset_df,
        renamed_merge_simpDA_dataset_df,
        renamed_simplicity_DA_dataset,
        merge_questeval_dataset_df,
    ]
)

merged_all_data_dataset.drop(columns=["original_sentence_id"], inplace=True)
merged_all_data_dataset.rename(columns={"rating": "label"}, inplace=True)

ratings_data = [
    merge_asset_dataset_df["rating"].describe(),
    merge_simpDA_dataset_df["Answer.adequacy"].describe(),
    simplicity_DA_dataset["meaning"].describe(),
    merge_questeval_dataset_df["rating"].describe(),
    merged_all_data_dataset["label"].describe(),
]

# We use Python2Latex to produce a Latex table.
doc = get_dataset_analysis_table(data=data, saving_dir=saving_dir)
text = doc.build()

doc = get_dataset_rating_table(data=ratings_data, saving_dir=saving_dir)
text = doc.build()

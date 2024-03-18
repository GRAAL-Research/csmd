import os

import pandas as pd
from datasets import load_dataset, Dataset
from evaluate import load
from poutyne import set_seeds
from tqdm import tqdm

rouge = load("rouge")
bleu = load("sacrebleu")

seed = 42
root = ".."

set_seeds(seed=seed)

asset_dataset = load_dataset("asset", "ratings", split="full")
asset_dataset_df = asset_dataset.to_pandas()
asset_dataset_df = asset_dataset_df[asset_dataset_df["aspect"] == 0]  # 0 is the meaning criteria

# Merging the ASSET dataset into a mean per sentence
merge_asset_dataset_df = asset_dataset_df[["original", "simplification", "original_sentence_id"]].drop_duplicates()
asset_mean_rating = asset_dataset_df.groupby("original_sentence_id")["rating"].mean()
merge_asset_dataset_df["rating"] = list(asset_mean_rating)

simpDA_dataset = pd.read_csv(os.path.join(root, "datastore", "simpDA_2022.csv"))

# Merging the SimpDA dataset into a mean per sentence
merge_simpDA_dataset_df = simpDA_dataset[["Input.original", "Input.simplified", "Input.id"]].drop_duplicates()
simpDA_mean_rating = simpDA_dataset.groupby("Input.simplified")["Answer.adequacy"].mean()
merge_simpDA_dataset_df["Answer.adequacy"] = list(simpDA_mean_rating)

simplicity_DA_dataset = pd.read_csv(os.path.join(root, "datastore", "simplicity_DA.csv"))

# We use the cleaned version of QuestEval that duplicate with ASSET where remove.
questeval_dataset = pd.read_csv(os.path.join(root, "datastore", "questeval_simplification_likert_ratings_cleaned.csv"))

questeval_dataset_df = questeval_dataset[questeval_dataset["aspect"] == "meaning"]
renamed_merge_questeval_dataset_df = questeval_dataset_df.rename(
    columns={
        "source": "original",
        "Answer.adequacy": "rating",
        "sentence_id": "original_sentence_id",
    }
)
renamed_merge_questeval_dataset_df = renamed_merge_questeval_dataset_df.drop(
    columns=["simplification_type", "system_name", "aspect"]
)

# We do some columns names cleaning and merging all the data.
merge_questeval_dataset_df = renamed_merge_questeval_dataset_df[["original", "simplification"]].drop_duplicates(
    keep="last"
)
questeval_mean_rating = renamed_merge_questeval_dataset_df.groupby(["original", "simplification"])["rating"].mean()
merge_questeval_dataset_df["rating"] = list(questeval_mean_rating)

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

# The default dataset without data augmentation (i.e. sanity checks in our article)
merged_all_data_dataset.rename(columns={"rating": "label"}, inplace=True)

# We first output the default dataset using a 60-10-30 split.
complete_dataset = Dataset.from_pandas(merged_all_data_dataset, preserve_index=False)

# 60-10-30 train-test ratio
# 30% for test
trainvalid_test = complete_dataset.train_test_split(
    test_size=0.3, shuffle=True, seed=seed
)  # 70 is the 60-10 set for train/dev.
# Split the train in 60-10
train_valid = trainvalid_test["train"].train_test_split(test_size=0.1, shuffle=True, seed=seed)

# The train set is the "train" set in the train_valid
# The dev set is the "test" set in the train_valid
# The test set is the "test" set in the trainvalid_test
train_set = train_valid["train"]
dev_set = train_valid["test"]
test_set = trainvalid_test["test"]

save_dir = os.path.join(root, "datastore", "meaning")
os.makedirs(save_dir, exist_ok=True)

# Save to disk the meaning dataset splits
train_set.to_csv(os.path.join(save_dir, "train.tsv"), sep="\t", index=False)
dev_set.to_csv(os.path.join(save_dir, "dev.tsv"), sep="\t", index=False)
test_set.to_csv(os.path.join(save_dir, "test.tsv"), sep="\t", index=False)

# Data Augmentation Dataset Creation

# We first create the identical pairs.

sentence_pairs = []
for _, sentence in merged_all_data_dataset.iterrows():
    sentence_pairs.append(
        {
            "original": sentence["original"],
            "simplification": sentence["original"],
            "label": 100.0,
        }
    )

identical_df = pd.DataFrame.from_dict(sentence_pairs, orient="columns")

asset_simplification_validation_dataset = load_dataset("asset", "simplification", split="validation").to_pandas()

sentence_pairs = []
running_idx = 0
resampled = 0
for _, sentence in tqdm(merged_all_data_dataset.iterrows(), total=len(merged_all_data_dataset)):
    valid = False
    while not valid:
        orig_sent = sentence["original"]
        selected_prediction = asset_simplification_validation_dataset["original"][running_idx]

        rouge_score = rouge.compute(predictions=[selected_prediction], references=[orig_sent])
        bleu_score = bleu.compute(predictions=[orig_sent], references=[selected_prediction])["score"]

        if (
            rouge_score["rouge1"] < 0.20
            and rouge_score["rouge2"] < 0.20
            and rouge_score["rougeL"] < 0.20
            and bleu_score < 20
        ):
            sentence_pairs.append(
                {
                    "original": orig_sent,
                    "simplification": selected_prediction,
                    "label": 0.0,
                }
            )
            valid = True
            running_idx += 1
        else:
            running_idx += 1
            resampled += 1

unrelated_dataset_df = pd.DataFrame.from_dict(sentence_pairs, orient="columns")

merged_all_data_dataset = pd.concat(
    [
        merged_all_data_dataset,
        identical_df,
        unrelated_dataset_df,
    ]
)

inverse_pair = []
for data in merged_all_data_dataset.iterrows():
    row = data[1]
    original = row["original"]
    simplification = row["simplification"]
    label = row["label"]
    inverse_sentence_document = {"original": simplification, "simplification": original, "label": label}
    inverse_pair.append(inverse_sentence_document)

inverse_pair_df = pd.DataFrame.from_dict(inverse_pair, orient="columns")
merged_all_data_dataset = pd.concat([merged_all_data_dataset, inverse_pair_df])

complete_dataset = Dataset.from_pandas(merged_all_data_dataset, preserve_index=False)

# Same splits logic as before
# The train set is the "train" set in the train_valid
# The dev set is the "test" set in the train_valid
# The test set is the "test" set in the trainvalid_test
trainvalid_test = complete_dataset.train_test_split(test_size=0.3, shuffle=True, seed=seed)
train_valid = trainvalid_test["train"].train_test_split(test_size=0.1, shuffle=True, seed=seed)
train_set = train_valid["train"]
dev_set = train_valid["test"]
test_set = trainvalid_test["test"]

save_dir = os.path.join(root, "datastore", "meaning_with_data_augmentation")
os.makedirs(save_dir, exist_ok=True)

# Save to disk the meaning dataset splits
train_set.to_csv(os.path.join(save_dir, "train_da.tsv"), sep="\t", index=False)
dev_set.to_csv(os.path.join(save_dir, "dev_da.tsv"), sep="\t", index=False)
test_set.to_csv(os.path.join(save_dir, "test_da.tsv"), sep="\t", index=False)

# Sanity Checks holdout set
# We also create a holdout set of original sentence pair (identical and irrelevant) never seen during training.
# for that we use the ASSET simplification dataset that does not include annotation.
# For the identical sentence pair, the label is 100.
# For the unrelated sentence pair, the label is 0.
asset_simplification_test_dataset = load_dataset("asset", "simplification", split="test").to_pandas()

length = len(asset_simplification_test_dataset["original"])
holdout_same_sentence_dataset = pd.DataFrame(
    {
        "original": asset_simplification_test_dataset["original"],
        "simplification": asset_simplification_test_dataset["original"],
        "label": [100.00] * length,
    }
)

sentence_pairs = []
asset_val_set_len = len(asset_simplification_validation_dataset["original"])
running_idx = 0
resampled = 0
for _, sentence in tqdm(asset_simplification_test_dataset.iterrows(), total=len(asset_simplification_test_dataset)):
    valid = False
    while not valid:
        orig_sent = sentence["original"]
        selected_prediction = asset_simplification_validation_dataset["original"][asset_val_set_len - running_idx - 1]

        rouge_score = rouge.compute(predictions=[selected_prediction], references=[orig_sent])
        bleu_score = bleu.compute(predictions=[orig_sent], references=[selected_prediction])["score"]

        if (
            rouge_score["rouge1"] < 0.25
            and rouge_score["rouge2"] < 0.25
            and rouge_score["rougeL"] < 0.25
            and bleu_score < 25
        ):
            sentence_pairs.append(
                {
                    "original": orig_sent,
                    "simplification": selected_prediction,
                    "label": 0.0,
                }
            )
            valid = True
            running_idx += 1
        else:
            running_idx += 1
            resampled += 1

holdout_irrelevant_sentence_dataset = pd.DataFrame.from_dict(sentence_pairs, orient="columns")

save_dir = os.path.join(root, "datastore", "holdout")
os.makedirs(save_dir, exist_ok=True)

# Save to disk the meaning dataset splits
holdout_same_sentence_dataset.to_csv(os.path.join(save_dir, "identical.tsv"), sep="\t", index=False)
holdout_irrelevant_sentence_dataset.to_csv(os.path.join(save_dir, "unrelated.tsv"), sep="\t", index=False)

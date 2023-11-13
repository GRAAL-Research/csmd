from python2latex import Document, Table

"""
Script to produce LateX Tables for the datasets.
"""

metric_names = [
    "Vocabulary size",
    "Vocabulary size lexical words",
    "Avg sentence length (tokens)",
    "Avg sentence length (LW)",
    "Lexical richness",
]


def get_dataset_analysis_table(data, saving_dir):
    doc = Document(
        filename="dataset_analysis",
        filepath=saving_dir,
        doc_type="article",
        border="10pt",
    )

    # Create the data
    col, row = 9, 6

    table = doc.new(
        Table(
            shape=(row + 1, col),
            as_float_env=True,
            alignment=["l"] + ["c"] * 8,
            caption=r"Dataset analysis",
            caption_pos="bottom",
        )
    )

    table[0, 1:3] = "Asset"
    table[0, 3:5] = "Simplicity-DA"
    table[0, 5:7] = r"SimpDA$_{\texttt{2022}}$"
    table[0, 7:] = "QuestEval"
    table[1, 1:] = ["Source", "Simplification"] * 4
    table[1, :].add_rule()

    for idx, metric_name in enumerate(metric_names):
        table[idx + 2, 0] = metric_name

    for col_idx, metric_data in enumerate(data):
        for row_idx, metric_value in enumerate(metric_data):
            table[row_idx + 2, col_idx + 1] = metric_value

    return doc


def get_dataset_rating_table(data, saving_dir):
    doc = Document(
        filename="dataset_rating_analysis",
        filepath=saving_dir,
        doc_type="article",
        border="10pt",
    )

    # Create the data
    col, row = 6, 7

    table = doc.new(
        Table(
            shape=(row + 1, col),
            as_float_env=True,
            alignment=["l"] + ["c"] * (col - 2) + ["|c"],
            caption=r"Datasets rating analysis",
            caption_pos="bottom",
        )
    )

    table[0, 1] = "Asset"
    table[0, 2] = "Simplicity-DA"
    table[0, 3] = r"SimpDA$_{\texttt{2022}}$"
    table[0, 4] = "QuestEval"
    table[0, 5] = "All Datasets"
    table[0, :].add_rule()

    table[:, 0] = ["", "Mean", "Std Dev", "Min", r"25\%", "Median", r"75\%", "Max"]

    for col_idx, col_data in enumerate(data):
        for row_idx, row_data in enumerate(col_data):
            if row_idx > 0:
                table[row_idx, col_idx + 1] = row_data

    return doc

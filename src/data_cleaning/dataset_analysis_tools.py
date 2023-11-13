from collections import Counter
from statistics import mean
from typing import List, Tuple

import spacy
from joblib import Parallel, delayed

nlp = spacy.load("en_core_web_sm")

"""
Script use to produce the statistic presented in the article.
"""


def tokenize(model, text):
    return [token for token in model.tokenizer(text)]


def compute_stats(data_row):
    """
    We do some symbol cleaning.
    """
    text = data_row.lower()

    tokens = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
        and not "`" in token.text
        and not " " in token.text
    ]
    tokens_len = len(tokens)

    lexical_words = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
        and not "`" in token.text
        and not " " in token.text
        and not token.is_stop
    ]
    lexical_words_len = len(lexical_words)

    vocabulary_set = set(tokens)
    vocabulary_lexical_words_set = set(lexical_words)

    doc = nlp(text)
    sentences = list(doc.sents)

    doc_sentence_len = []
    for sentence in sentences:
        doc_sentence_len.append(
            len(
                [
                    token.text
                    for token in tokenize(nlp, str(sentence))
                    if not token.is_punct
                    and not "\n" in token.text
                    and not token.is_digit
                    and not "|" in token.text
                    and not "$" in token.text
                    and not "<" in token.text
                    and not ">" in token.text
                    and not " " in token.text
                    and not "`" in token.text
                    and not " " in token.text
                ]
            )
        )
    average_sen_len = mean(doc_sentence_len)

    doc_lexical_sentence_len = []
    for sentence in sentences:
        doc_lexical_sentence_len.append(
            len(
                [
                    token.text
                    for token in tokenize(nlp, str(sentence))
                    if not token.is_punct
                    and not "\n" in token.text
                    and not token.is_digit
                    and not "|" in token.text
                    and not "$" in token.text
                    and not "<" in token.text
                    and not ">" in token.text
                    and not " " in token.text
                    and not token.is_stop
                    and not "`" in token.text
                    and not " " in token.text
                ]
            )
        )
    average_sen_len_lexical = mean(doc_lexical_sentence_len)

    lexical_words_counter = Counter(lexical_words)

    return (
        tokens_len,
        lexical_words_len,
        vocabulary_set,
        vocabulary_lexical_words_set,
        average_sen_len,
        average_sen_len_lexical,
        lexical_words_counter,
    )


def dataset_analysis_batch_process(
    dataset: List, output_path: str, num_cores: int = 10
) -> Tuple:
    """
    Batch processing the datasets.
    """
    process_data = list(
        Parallel(n_jobs=num_cores, verbose=2, backend="multiprocessing")(
            delayed(compute_stats)(data_row) for data_row in dataset
        )
    )

    number_of_tokens_per_document = []
    number_of_lexical_words_per_document = []

    vocabulary = set()
    vocabulary_lexical_words = set()

    average_sentences_len_per_document = []
    average_sentence_len_lexical_words_per_document = []

    overall_lexical_words_counter = Counter()

    for data in process_data:
        number_of_tokens_per_document.append(data[0])
        number_of_lexical_words_per_document.append(data[1])

        vocabulary.update(data[2])
        vocabulary_lexical_words.update(data[3])

        average_sentences_len_per_document.append(data[4])
        average_sentence_len_lexical_words_per_document.append(data[5])

        overall_lexical_words_counter.update(data[6])

    del process_data

    vocabulary_size = len(vocabulary)
    vocabulary_size_lexical_words = len(vocabulary_lexical_words)
    average_sentences_len = mean(average_sentences_len_per_document)
    average_sentence_len_lexical_words = mean(
        average_sentence_len_lexical_words_per_document
    )
    lexical_diversity = vocabulary_size_lexical_words / sum(
        number_of_lexical_words_per_document
    )

    with open(output_path, "w") as file:
        print(f"The number of vocabulary size is : {vocabulary_size}")
        print(
            f"The number of vocabulary size of lexical words is : {vocabulary_size_lexical_words}"
        )

        print(f"The average sentence len is: {average_sentences_len}")
        print(
            f"The average sentence len lexical words is: {average_sentence_len_lexical_words}"
        )
        print(f"The lexical richness is: {lexical_diversity}")
        print(
            f"The top 50 words frequencies are: {overall_lexical_words_counter.most_common(50)}"
        )

        print(f"The number of vocabulary size is : {vocabulary_size}", file=file)
        print(
            f"The number of vocabulary size of lexical words is : {vocabulary_size_lexical_words}",
            file=file,
        )

        print(f"The average sentence len is: {average_sentences_len}", file=file)
        print(
            f"The average sentence len lexical words is: {average_sentence_len_lexical_words}",
            file=file,
        )
        print(f"The lexical richness is: {lexical_diversity}", file=file)
        print(
            f"The top 50 words frequencies are: {overall_lexical_words_counter.most_common(50)}",
            file=file,
        )

    return (
        len(vocabulary),
        len(vocabulary_lexical_words),
        mean(average_sentences_len_per_document),
        mean(average_sentence_len_lexical_words_per_document),
        vocabulary_size_lexical_words / sum(number_of_lexical_words_per_document),
    )

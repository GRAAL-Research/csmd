# Repository for the "Continuous Scale Meaning Dataset" Proposed in "[MeaningBERT: Assessing Meaning Preservation Between Sentences](https://www.frontiersin.org/articles/10.3389/frai.2023.1223924/full)"

## About the Dataset

Aggregate statistics on textual data of the four datasets used to create "Continuous Scale Meaning Dataset".
![img_1.png](fig/img_1.png)

Aggregate statistics on meaning preservation rating data using a continuous scale (0–100) for the four datasets used to
create "Continuous Scale Meaning Dataset".
![img.png](fig/img.png)

## Dataset Structure

### Data Instances

- `Meaning` configuration: an instance consists of 1,355 meaning preservation triplets (Document, simplification,
  label).
- `meaning_with_data_augmentation` configuration: an instance consists of 1,355 meaning preservation triplets (Document,
  simplification, label) along with 1,355 data augmentation triplets (Document, Document, 1) and 1,355 data augmentation
  triplets (Document, Unrelated Document, 0) (See the sanity checks in
  our [article](https://www.frontiersin.org/articles/10.3389/frai.2023.1223924/full)).
- `meaning_holdout_identical` configuration: an instance consists of X meaning holdout preservation identical triplets (
  Document, Document, 1) based on the ASSET Simplification dataset.
- `meaning_holdout_unrelated` configuration: an instance consists of X meaning holdout preservation unrelated triplets (
  Document, Unrelated Document, 0) based on the ASSET Simplification dataset.

### Data Fields

- `original`: an original sentence from the source datasets.
- `simplification`:  a simplification of the original obtained by an automated system or a human.
- `label`: a meaning preservation rating between 0 and 100.

### Data Splits

The split statistics of CSMD are given below.

| | Train | Dev    | Test  | Total |
| ------ |-------| ------ |-------| ----- |
| Meaning | 853   | 95   | 407   | 1,355  |
| Meaning With Data Augmentation | 2,560 | 285   | 1,220 | 4,065  |
| Meaning Holdout Identical | NA    | NA   | 359   | 359 |
| Meaning Holdout Unrelated | NA    | NA   | 359   | 359  |

All the splits are randomly split using a 60-10-30 split with the seed `42`.

For more details, see our article or our [Dataset Card](https://huggingface.co/datasets/davebulaval/csmd) on HuggingFace.

## Download the dataset

You can manually download our dataset splits available in `dataset`, or you can use the HuggingFace dataset class as follows:

```python
from datasets import load_dataset

dataset = load_dataset("davebulaval/CSMD", "meaning")

# you can use any of the following config names as a second argument:
# "meaning", "meaning_with_data_augmentation", "meaning_holdout_identical", "meaning_holdout_unrelated"
```

## To Cite

```
@ARTICLE{10.3389/frai.2023.1223924,
AUTHOR={Beauchemin, David and Saggion, Horacio and Khoury, Richard},   
TITLE={{MeaningBERT: Assessing Meaning Preservation Between Sentences}},      
JOURNAL={Frontiers in Artificial Intelligence},      
VOLUME={6},           
YEAR={2023},      
URL={https://www.frontiersin.org/articles/10.3389/frai.2023.1223924},       
DOI={10.3389/frai.2023.1223924},      	
ISSN={2624-8212},   
}
```


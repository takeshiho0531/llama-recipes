import datasets
from .utils import Concatenator

def get_preprocessed_wiki40b_ja(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('range3/wiki40b-ja', split=split)
    print('range3/wiki40b-ja', dataset)
    dataset=dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    print("mapped", dataset)
    return dataset
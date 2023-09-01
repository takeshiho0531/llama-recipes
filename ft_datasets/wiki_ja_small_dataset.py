import datasets
from .utils import Concatenator

def get_preprocessed_wiki_ja_small(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("graelo/wikipedia", split=split)

    dataset=dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset
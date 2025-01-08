from datasets import Dataset, load_dataset
from pandas import DataFrame
from config import directories as Paths, Pipeline

ds = load_dataset("rajpurkar/squad")

context_qa_map : dict[str, list[str]] = {}

for example in ds["train"]:
    context = example["context"]
    question = example["question"]
    answers = example["answers"]["text"]

    context = f"{Pipeline.promptPrefix} {context} {Pipeline.promptSuffix}"
    if context not in context_qa_map:
        context_qa_map[context] = []
    context_qa_map[context].append({'question': question, 'answer': answers})

for example in ds["validation"]:
    context = example["context"]
    question = example["question"]
    answers = example["answers"]["text"]

    context = f"{Pipeline.promptPrefix} {context} {Pipeline.promptSuffix}"
    if context not in context_qa_map:
        context_qa_map[context] = []
    context_qa_map[context].append({'question': question, 'answer': answers})

dataset : DataFrame = DataFrame.from_dict({
    "modelPrompt": list(context_qa_map.keys()),
    "gold_QnA": list(context_qa_map.values())
    })

dataset.to_csv(f"{Paths.data}/squad.csv", index=False)

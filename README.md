# Introduction

Delphi was the home of a temple to Phoebus Apollo, which famously had the inscription, 'Know Thyself.' This library lets language models know themselves through automated interpretability.

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) and transcoder features. The explainer and scorer models can be run locally or accessed using API calls via OpenRouter.

The branch used for the article [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/pdf/2410.13928) is the legacy branch [article_version](https://github.com/EleutherAI/delphi/tree/article_version), that branch contains the scripts to reproduce our experiments. Note that we're still actively improving the codebase and that the newest version on the main branch could require slightly different usage.

# Installation

Install this library as a local editable installation. Run the following command from the `delphi` directory.

```pip install -e .```

# Getting Started

To run the default pipeline from the command line, use the following command:

`python -m delphi meta-llama/Meta-Llama-3-8B EleutherAI/sae-llama-3-8b-32x --explainer_model 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4' --dataset_repo 'EleutherAI/fineweb-edu-dedup-10b' --dataset_split 'train[:1%]' --n_tokens 10_000_000 --max_latents 100 --hookpoints layers.5 --filter_bos --name llama-3-8B` 

This command will:
1. Cache activations for the first 10 million tokens of EleutherAI/rpj-v2-sample.
2. Generate explanations for the first 100 features of layer 5 using the specified explainer model.
3. Score the explanations uses fuzzing and detection scorers.
4. Log summary metrics including per-scorer F1 scores and confusion matrices.

The pipeline is highly configurable and can also be called programmatically (see the [end-to-end test](https://github.com/EleutherAI/delphi/blob/main/delphi/tests/e2e.py) for an example).

To use other scorer types, instantiate a custom pipeline.

## Caching

The first step to generate explanations is to cache sparse model activations. To do so, load your sparse models into the base model, load the tokens you want to cache the activations from, create a `FeatureCache` object and run it. We recommend caching over at least 10M tokens.

```python
from sparsify.data import chunk_and_tokenize
from delphi.features import FeatureCache

data = load_dataset("EleutherAI/rpj-v2-sample", split="train[:1%]")
tokens = chunk_and_tokenize(data, tokenizer, max_seq_len=256, text_key="raw_content")["input_ids"]

cache = FeatureCache(
    model,
    submodule_dict,
    batch_size = 8
)

cache.run(n_tokens = 10_000_000, tokens=tokens)
```

Caching saves `.safetensors` of `dict["activations", "locations"]`.

```python
cache.save_splits(
    n_splits=5,
    save_dir="raw_latents"
)
```

Safetensors are split into shards over the width of the autoencoder.

## Loading Feature Records

The `.features` module provides utilities for reconstructing and sampling various statistics for sparse features. In this version of the code you needed to specify the width of the autoencoder, the minimum number examples for a feature to be included and the maximum number of examples to include, as well as the number of splits to divide the features into.

```python
from delphi.features import FeatureLoader, FeatureDataset
from delphi.config import FeatureConfig

#
cfg = FeatureConfig(width=131072, min_examples=200, max_examples=10000, n_splits=5)

dataset = FeatureDataset(
    raw_dir="feature_folder",
    modules=[".model.layer.0"], # This a list of the different caches to load from
    cfg=cfg,
)
```

The feature dataset will construct lazy loaded buffers that load activations into memory when called as an iterator object. You can iterate through the dataset using the `FeatureLoader` object. The feature loader will take in the feature dataset, a constructor and a sampler.

```python
loader = FeatureLoader(
    dataset=dataset,
    constructor = constructor,
    sampler = sampler,
)
```

We have a simple sampler and constructor that take arguments from the `ExperimentConfig` object. The constructor defines builds the context windows from the cached activations and tokens, and the sampler divides these contexts into a training and testing set, used to generate explanations and evaluate them.

```python
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from delphi.config import ExperimentConfig

cfg = ExperimentConfig(
    n_examples_train=40, # Number of examples shown to the explainer model
    n_examples_test=100, # Number of examples shown to the scorer models
    n_quantiles=10, # Number of quantiles to divide the data into
    example_ctx_len=32, # Length of each example
    n_non_activating=100, # Number of non-activating examples shown to the scorer model
    train_type="quantiles", # Type of sampler to use for training
    test_type="even", # Type of sampler to use for testing
)

constructor = partial(default_constructor, tokens=dataset.tokens, n_not_active=cfg.n_non_activating, ctx_len=cfg.example_ctx_len, max_examples=cfg.max_examples)
sampler = partial(sample, cfg=cfg)
```

## Generating Explanations

We currently support using OpenRouter's OpenAI compatible API or running locally with VLLM. Define the client you want to use, then create an explainer from the `.explainers` module.

```python
from delphi.explainers import DefaultExplainer
from delphi.clients import Offline,OpenRouter

# Run locally with VLLM
client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=1)

# Run with OpenRouter
client = OpenRouter("meta-llama/Meta-Llama-3.1-8B-Instruct",api_key=key)


explainer = DefaultExplainer(
    client,
    tokenizer = dataset.tokenizer,
)
```

The explainer should be added to a pipe, which will send the explanation requests to the client. The pipe should have a function that happens after the request is completed, to e.g. save the data, and could also have a function that happens before the request is sent, e.g to transform some of the data.

```python
from delphi.pipeline import process_wrapper

def explainer_postprocess(result):

    with open(f"{explanation_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result

explainer_pipe = process_wrapper(explainer,
    postprocess=explainer_postprocess,
)
```
The pipe should then be used in a pipeline. Running the pipeline will send requests to the client in batches of paralel requests.

```python
from delphi.pipeline import Pipeline
import asyncio

pipeline = Pipeline(
    loader,
    explainer_pipe,
)

asyncio.run(pipeline.run(n_processes))
```

## Scoring Explanations

The process of running a scorer is similar to that of an explainer. You need to have a client running, and you need to create a Scorer from the '.scorer' module. You can either load the explanations you generated earlier, or generate new ones using the explainer pipe.

```python
RecallScorer(
    client,
    tokenizer=tokenizer,
    batch_size=cfg.batch_size
)
```

You can then create a pipe to run the scorer. The pipe should have a pre-processer, that takes the results from the previous pipe and a post processor, that saves the scores. An scorer should always be run after a explainer pipe, but the explainer pipe can be used to load saved explanations.

```python
from delphi.scorers import FuzzingScorer, RecallScorer
from delphi.explainers import  explanation_loader,random_explanation_loader


# Because we are running the explainer and scorer separately, we need to add the explanation and extra examples back to the record

def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

def scorer_postprocess(result, score_dir):
    with open(f"{score_dir}/{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

# If one wants to load the explanations they generated earlier
# explainer_pipe = partial(explanation_loader, explanation_dir=EXPLAINER_OUT_DIR)

scorer_pipe = process_wrapper(
        RecallScorer(client, tokenizer=dataset.tokenizer, batch_size=cfg.batch_size),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=recall_dir),
    )

```

It is possible to have more than one scorer per pipe. One could use that to run fuzzing and detection together:

```python
scorer_pipe = Pipe(
    process_wrapper(
        RecallScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=recall_dir),
    ),
    process_wrapper(
        FuzzingScorer(client, tokenizer=tokenizer, batch_size=cfg.batch_size),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
    ),
)
```

Then the pipe should be sent to the pipeline and run:

```python
pipeline = Pipeline(
        loader.load,
        explainer_pipe,
        scorer_pipe,
)

asyncio.run(pipeline.run())
```

### Simulation

To do simulation scoring we forked and modified OpenAIs neuron explainer. The name of the scorer is `OpenAISimulator`, and it can be run with the same setup as described above.

### Surprisal

Surprisal scoring computes the loss over some examples and uses a base model. We don't use VLLM but run the model using the `AutoModelForCausalLM` wrapper from HuggingFace. The setup is similar as above but for a example check `surprisal.py` in the experiments folder.

### Embedding

Embedding scoring uses a small embedding model through `sentence_transformers` to embed the examples do retrival. It also does not use VLLM but run the model directly. The setup is similar as above but for a example check `embedding.py` in the experiments folder.

## Scripts

Example scripts can be found in `demos`. Some of these scripts can be called from the CLI, as seen in examples found in `scripts`. These baseline scripts should allow anyone to start generating and scoring explanations in any SAE they are interested in. One always needs to first cache the activations of the features of any given SAE, and then generating explanations and scoring them can be done at the same time.

## Experiments

The experiments discussed in [the blog post](https://blog.eleuther.ai/autointerp/) were mostly run in a legacy version of this code, which can be found in the [Experiments](https://github.com/EleutherAI/delphi/tree/Experiments) branch.

## License

Copyright 2024 the EleutherAI Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

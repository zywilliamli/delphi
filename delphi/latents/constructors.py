import hashlib
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from jaxtyping import Float
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..config import ConstructorConfig
from .latents import (
    ActivatingExample,
    ActivationData,
    LatentRecord,
    NonActivatingExample,
)

model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def get_model(name: str, device: str = "cuda") -> SentenceTransformer:
    global model_cache
    if (name, device) not in model_cache:
        print(f"Loading model {name} on device {device}")
        model_cache[(name, device)] = SentenceTransformer(name, device=device)
    return model_cache[(name, device)]


def prepare_non_activating_examples(
    tokens: Float[Tensor, "examples ctx_len"],
    distance: float,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> list[NonActivatingExample]:
    """
    Prepare a list of non-activating examples from input tokens and distance.

    Args:
        tokens: Tokenized input sequences.
        distance: The distance from the neighbouring latent.
    """
    return [
        NonActivatingExample(
            tokens=toks,
            activations=torch.zeros_like(toks),
            normalized_activations=None,
            distance=distance,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks in tokens
    ]


def _top_k_pools(
    max_buffer: Float[Tensor, "batch"],
    split_activations: Float[Tensor, "activations ctx_len"],
    buffer_tokens: Float[Tensor, "batch ctx_len"],
    max_examples: int,
) -> tuple[Float[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Get the top k activation pools.

    Args:
        max_buffer: The maximum buffer values.
        split_activations: The split activations.
        buffer_tokens: The buffer tokens.
        max_examples: The maximum number of examples.

    Returns:
        The token windows and activation windows.
    """
    k = min(max_examples, len(max_buffer))
    top_values, top_indices = torch.topk(max_buffer, k, sorted=True)

    activation_windows = torch.stack([split_activations[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    activations: Float[Tensor, "examples"],
    tokens: Float[Tensor, "windows seq"],
    ctx_indices: Float[Tensor, "examples"],
    index_within_ctx: Float[Tensor, "examples"],
    ctx_len: int,
    max_examples: int,
) -> tuple[Float[Tensor, "examples ctx_len"], Float[Tensor, "examples ctx_len"]]:
    """
    Pool max activation windows from the buffer output and update the latent record.

    Args:
        activations : The activations.
        tokens : The input tokens.
        ctx_indices : The context indices.
        index_within_ctx : The index within the context.
        ctx_len : The context length.
        max_examples : The maximum number of examples.
    """
    # unique_ctx_indices: array of distinct context window indices in order of first
    # appearance. sequential integers from 0 to batch_size * cache_token_length//ctx_len
    # inverses: maps each activation back to its index in unique_ctx_indices
    # (can be used to dereference the context window idx of each activation)
    # lengths: the number of activations per unique context window index
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # Get the max activation magnitude within each context window
    max_buffer = torch.segment_reduce(activations, "max", lengths=lengths)

    # Deduplicate the context windows
    new_tensor = torch.zeros(len(unique_ctx_indices), ctx_len, dtype=activations.dtype)
    new_tensor[inverses, index_within_ctx] = activations

    tokens = tokens[unique_ctx_indices]

    token_windows, activation_windows = _top_k_pools(
        max_buffer, new_tensor, tokens, max_examples
    )

    return token_windows, activation_windows


def constructor(
    record: LatentRecord,
    activation_data: ActivationData,
    constructor_cfg: ConstructorConfig,
    tokens: Float[Tensor, "batch seq"],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    all_data: Optional[dict[int, ActivationData]] = None,
    seed: int = 42,
) -> LatentRecord | None:
    cache_ctx_len = tokens.shape[1]
    example_ctx_len = constructor_cfg.example_ctx_len
    source_non_activating = constructor_cfg.non_activating_source
    n_not_active = constructor_cfg.n_non_activating
    max_examples = constructor_cfg.max_examples
    min_examples = constructor_cfg.min_examples

    # Get all positions where the latent is active
    flat_indices = (
        activation_data.locations[:, 0] * cache_ctx_len
        + activation_data.locations[:, 1]
    )
    ctx_indices = flat_indices // example_ctx_len
    index_within_ctx = flat_indices % example_ctx_len
    reshaped_tokens = tokens.reshape(-1, example_ctx_len)
    n_windows = reshaped_tokens.shape[0]

    unique_batch_pos = ctx_indices.unique()

    mask = torch.ones(n_windows, dtype=torch.bool)
    mask[unique_batch_pos] = False
    # Indices where the latent is not active
    non_active_indices = mask.nonzero(as_tuple=False).squeeze()
    activations = activation_data.activations

    # Add activation examples to the record in place
    token_windows, act_windows = pool_max_activation_windows(
        activations=activations,
        tokens=reshaped_tokens,
        ctx_indices=ctx_indices,
        index_within_ctx=index_within_ctx,
        ctx_len=example_ctx_len,
        max_examples=max_examples,
    )
    # TODO: We might want to do this in the sampler
    # we are tokenizing examples that are not going to be used
    record.examples = [
        ActivatingExample(
            tokens=toks,
            activations=acts,
            normalized_activations=None,
            str_tokens=tokenizer.batch_decode(toks),
        )
        for toks, acts in zip(token_windows, act_windows)
    ]

    if len(record.examples) < min_examples:
        # Not enough examples to explain the latent
        return None

    if source_non_activating == "random":
        # Add random non-activating examples to the record in place
        non_activating_examples = random_non_activating_windows(
            available_indices=non_active_indices,
            reshaped_tokens=reshaped_tokens,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    elif source_non_activating == "neighbours":
        assert all_data is not None, "All data is required for neighbour constructor"
        non_activating_examples = neighbour_non_activation_windows(
            record,
            not_active_mask=mask,
            tokens=tokens,
            all_data=all_data,
            ctx_len=example_ctx_len,
            n_not_active=n_not_active,
            seed=seed,
            tokenizer=tokenizer,
        )
    elif source_non_activating == "FAISS":
        non_activating_examples = faiss_non_activation_windows(
            available_indices=non_active_indices,
            record=record,
            tokens=tokens,
            ctx_len=example_ctx_len,
            tokenizer=tokenizer,
            n_not_active=n_not_active,
            embedding_model=constructor_cfg.faiss_embedding_model,
            seed=seed,
            cache_enabled=constructor_cfg.faiss_embedding_cache_enabled,
            cache_dir=constructor_cfg.faiss_embedding_cache_dir,
        )
    else:
        raise ValueError(f"Invalid non-activating source: {source_non_activating}")
    record.not_active = non_activating_examples
    return record


def create_token_key(tokens_tensor, ctx_len):
    """
    Create a file key based on token tensors without detokenization.

    Args:
        tokens_tensor: Tensor of tokens
        ctx_len: Context length

    Returns:
        A string key
    """
    h = hashlib.md5()
    total_tokens = 0

    # Process a sample of elements (first, middle, last)
    num_samples = len(tokens_tensor)
    indices_to_hash = (
        [0, num_samples // 2, -1] if num_samples >= 3 else range(num_samples)
    )

    for idx in indices_to_hash:
        if 0 <= idx < num_samples or (idx == -1 and num_samples > 0):
            # Convert tensor to bytes and hash it
            token_bytes = tokens_tensor[idx].cpu().numpy().tobytes()
            h.update(token_bytes)
            total_tokens += len(tokens_tensor[idx])

    # Add collection shape to make collisions less likely
    shape_str = f"{tokens_tensor.shape}"
    h.update(shape_str.encode())

    return f"{h.hexdigest()[:12]}_items{num_samples}_{ctx_len}"


def faiss_non_activation_windows(
    available_indices: Float[Tensor, "windows"],
    record: LatentRecord,
    tokens: Float[Tensor, "batch seq"],
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    n_not_active: int,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    cache_enabled: bool = True,
    cache_dir: str = ".embedding_cache",
) -> list[NonActivatingExample]:
    """
    Generate hard negative examples using FAISS similarity search based
    on text embeddings.

    This function builds a FAISS index over ALL examples (not just non-activating ones)
    and then filters out activating examples at search time, finding semantically
    similar examples based on text embeddings.

    Args:
        available_indices: Indices of windows where the latent is not active
        record: The latent record containing activating examples
        tokens: The input tokens
        ctx_len: The context length for examples
        tokenizer: The tokenizer to decode tokens
        n_not_active: Number of non-activating examples to generate
        embedding_model: Model used for text embeddings
        seed: Random seed
        cache_enabled: Whether to cache embeddings
        cache_dir: Directory to store cached embeddings

    Returns:
        A list of non-activating examples that are semantically similar to
            activating examples
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # Check if we have enough non-activating examples
    if available_indices.numel() < n_not_active:
        print("Not enough non-activating examples available")
        return []

    # Reshape tokens to get context windows
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    total_windows = reshaped_tokens.shape[0]

    # Define cache directory structure
    cache_dir = os.environ.get("DELPHI_CACHE_DIR", cache_dir)
    embedding_model_name = embedding_model.split("/")[-1]
    cache_path = Path(cache_dir) / embedding_model_name

    # Get activating example texts and indices
    activating_texts = [
        "".join(example.str_tokens)
        for example in record.examples[: min(10, len(record.examples))]
    ]
    
    # Extract activating indices from record
    activating_token_tensors = torch.stack(
        [example.tokens for example in record.examples[: min(10, len(record.examples))]]
    )
    
    # Create a lookup of active indices for fast filtering
    active_indices_set = set()
    for tokens_tensor in activating_token_tensors:
        # Find matching indices in reshaped_tokens
        matches = (reshaped_tokens == tokens_tensor.unsqueeze(0)).all(dim=1)
        if matches.any():
            active_indices = matches.nonzero().squeeze(1).tolist()
            active_indices_set.update(active_indices)
    
    if not activating_texts:
        print("No activating examples available")
        return []

    # Create a unique cache key for all tokens
    all_tokens_cache_key = create_token_key(reshaped_tokens, ctx_len)
    activating_cache_key = create_token_key(activating_token_tensors, ctx_len)

    # Cache files for all embeddings
    all_tokens_cache_file = cache_path / f"{all_tokens_cache_key}.faiss"
    activating_cache_file = cache_path / f"{activating_cache_key}.npy"

    # Try to load cached embeddings
    index = None
    if cache_enabled and all_tokens_cache_file.exists():
        try:
            index = faiss.read_index(str(all_tokens_cache_file), faiss.IO_FLAG_MMAP)
            print(f"Loaded all-tokens index from {all_tokens_cache_file}")
        except Exception as e:
            print(f"Error loading cached embeddings: {repr(e)}")

    if index is None:
        print("Decoding all tokens...")
        all_texts = ["".join(tokenizer.batch_decode(tokens)) for tokens in reshaped_tokens]

        print("Computing all embeddings...")
        all_embeddings = get_model(embedding_model).encode(
            all_texts, show_progress_bar=True
        )
        dim = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

        index.add(all_embeddings)  # type: ignore
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            faiss.write_index(index, str(all_tokens_cache_file))
            print(f"Cached all embeddings to {all_tokens_cache_file}")

    activating_embeddings = None
    if cache_enabled and activating_cache_file.exists():
        try:
            activating_embeddings = np.load(activating_cache_file)
            print(f"Loaded cached activating embeddings from {activating_cache_file}")
        except Exception as e:
            print(f"Error loading cached embeddings: {repr(e)}")
    
    # Compute embeddings for activating examples if not cached
    if activating_embeddings is None:
        print("Computing activating embeddings...")
        activating_embeddings = get_model(embedding_model).encode(
            activating_texts, show_progress_bar=False
        )
        # Cache the embeddings
        if cache_enabled:
            os.makedirs(cache_path, exist_ok=True)
            np.save(activating_cache_file, activating_embeddings)
            print(f"Cached activating embeddings to {activating_cache_file}")

    # Search for the nearest neighbors to each activating example, with evenly distributed k per example
    collected_indices = set()
    hard_negative_indices = []
    
    # Calculate minimum k needed per activating example to reach n_not_active
    num_activating = len(activating_embeddings)
    if num_activating == 0:
        print("No activating examples available")
        return []
        
    # Calculate k per example (ceiling division to ensure we get enough)
    k_per_example = (n_not_active + num_activating - 1) // num_activating
    
    # Calculate overfetch factor to account for filtering
    fetch_per_example = min(k_per_example * 5, total_windows)  # Overfetch by 5x
    
    # For each activating example, find exactly k non-activating examples
    for example_idx, embedding in enumerate(activating_embeddings):
        # Skip if we already have enough examples in total
        if len(hard_negative_indices) >= n_not_active:
            break
            
        # Track how many we've found for this example
        found_for_this_example = 0
        
        # Search for similar examples with overfetching
        distances, indices = index.search(embedding.reshape(1, -1), fetch_per_example)  # type: ignore
        
        # Filter out activating examples and examples not in available_indices
        for i, idx in enumerate(indices[0]):
            # Stop if we have enough for this example
            if found_for_this_example >= k_per_example:
                break
                
            # Stop if we have enough in total
            if len(hard_negative_indices) >= n_not_active:
                break
                
            # Convert idx to int to avoid any tensor/numpy type issues
            idx_int = int(idx)
            
            # Skip if this is an activating example or already collected
            if idx_int in active_indices_set or idx_int in collected_indices:
                continue
                
            # Also verify it's in our available_indices (non-activating set)
            if idx_int not in available_indices.tolist():
                continue
                
            # Add to our collection
            hard_negative_indices.append(idx_int)
            collected_indices.add(idx_int)
            found_for_this_example += 1

    # If we couldn't find enough examples, fill with random ones
    if len(hard_negative_indices) < n_not_active:
        remaining = n_not_active - len(hard_negative_indices)
        available_idx_set = set(available_indices.tolist()) - collected_indices - active_indices_set
        if available_idx_set:
            random_indices = torch.tensor(list(available_idx_set))[
                torch.randperm(len(available_idx_set))[:remaining]
            ].tolist()
            hard_negative_indices.extend(random_indices)

    # Get the token windows for the selected hard negatives
    selected_indices = torch.tensor(hard_negative_indices[:n_not_active])
    selected_tokens = reshaped_tokens[selected_indices]
    
    # Create non-activating examples
    return prepare_non_activating_examples(
        selected_tokens,
        -1.0,  # Using -1.0 as the distance since these are not neighbour-based
        tokenizer,
    )


def neighbour_non_activation_windows(
    record: LatentRecord,
    not_active_mask: Float[Tensor, "windows"],
    tokens: Float[Tensor, "batch seq"],
    all_data: dict[int, ActivationData],
    ctx_len: int,
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
):
    """
    Generate random activation windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        not_active_mask (TensorType["n_windows"]): The mask of the non-active windows.
        tokens (TensorType["batch", "seq"]): The input tokens.
        all_data (AllData): The all data containing activations and locations.
        ctx_len (int): The context length.
        n_not_active (int): The number of non-activating examples per latent.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer.
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    assert (
        record.neighbours is not None
    ), "Neighbours are not set, please precompute them"

    cache_token_length = tokens.shape[1]
    reshaped_tokens = tokens.reshape(-1, ctx_len)
    n_windows = reshaped_tokens.shape[0]
    # TODO: For now we use at most 10 examples per neighbour, we may want to allow a
    # variable number of examples per neighbour
    n_examples_per_neighbour = 10

    number_examples = 0
    all_examples = []
    for neighbour in record.neighbours:
        if number_examples >= n_not_active:
            break
        # get the locations of the neighbour
        if neighbour.latent_index not in all_data:
            continue
        locations = all_data[neighbour.latent_index].locations
        activations = all_data[neighbour.latent_index].activations
        # get the active window indices
        flat_indices = locations[:, 0] * cache_token_length + locations[:, 1]
        ctx_indices = flat_indices // ctx_len
        index_within_ctx = flat_indices % ctx_len
        # Set the mask to True for the unique locations
        unique_batch_pos_active = ctx_indices.unique()

        mask = torch.zeros(n_windows, dtype=torch.bool)
        mask[unique_batch_pos_active] = True

        # Get the indices where mask and not_active_mask are True
        mask = mask & not_active_mask

        available_indices = mask.nonzero().flatten()

        mask_ctx = torch.isin(ctx_indices, available_indices)
        available_ctx_indices = ctx_indices[mask_ctx]
        available_index_within_ctx = index_within_ctx[mask_ctx]
        activations = activations[mask_ctx]
        # If there are no available indices, skip this neighbour
        if activations.numel() == 0:
            continue
        token_windows, _ = pool_max_activation_windows(
            activations=activations,
            tokens=reshaped_tokens,
            ctx_indices=available_ctx_indices,
            index_within_ctx=available_index_within_ctx,
            max_examples=n_examples_per_neighbour,
            ctx_len=ctx_len,
        )
        # use the first n_examples_per_neighbour examples,
        # which will be the most active examples
        examples_used = len(token_windows)
        all_examples.extend(
            prepare_non_activating_examples(
                token_windows, -neighbour.distance, tokenizer
            )
        )
        number_examples += examples_used
    if len(all_examples) == 0:
        print("No examples found")
    return all_examples


def random_non_activating_windows(
    available_indices: Float[Tensor, "windows"],
    reshaped_tokens: Float[Tensor, "windows ctx_len"],
    n_not_active: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    seed: int = 42,
) -> list[NonActivatingExample]:
    """
    Generate random non-activating sequence windows and update the latent record.

    Args:
        record (LatentRecord): The latent record to update.
        available_indices (TensorType["n_windows"]): The indices of the windows where
        the latent is not active.
        reshaped_tokens (TensorType["n_windows", "ctx_len"]): The tokens reshaped
        to the context length.
        n_not_active (int): The number of non activating examples to generate.
    """
    torch.manual_seed(seed)
    if n_not_active == 0:
        return []

    # If this happens it means that the latent is active in every window,
    # so it is a bad latent
    if available_indices.numel() < n_not_active:
        print("No available randomly sampled non-activating sequences")
        return []
    else:
        random_indices = torch.randint(
            0, available_indices.shape[0], size=(n_not_active,)
        )
        selected_indices = available_indices[random_indices]

    toks = reshaped_tokens[selected_indices]

    return prepare_non_activating_examples(
        toks,
        -1.0,
        tokenizer,
    )

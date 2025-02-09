import os
from opt_einsum import contract as opt_einsum
from nnsight import NNsight
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from delphi.autoencoders.wrapper import AutoencoderLatents
import numpy as np
import torch
import torch.distributed as dist

from delphi.utils import load_tokenized_data
from delphi.features import FeatureCache
from delphi.config import CacheConfig

def to_dense(
    top_acts: torch.Tensor, top_indices: torch.Tensor, num_latents: int, instance_dims=[0, 1]
):
    instance_shape = [top_acts.shape[i] for i in instance_dims]
    dense_empty = torch.zeros(
        *instance_shape,
        num_latents,
        device=top_acts.device,
        dtype=top_acts.dtype,
        requires_grad=True,
    )
    return dense_empty.scatter(-1, top_indices.long(), top_acts)

def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--size", type=str, default="850m")
    parser.add_argument("--separate_moe", type=bool, default=True)
    args = parser.parse_args()
    model_name = f"MonetLLM/monet-vd-{args.size.upper()}-100BT-hf"
    args.tokenizer_model = model_name
    args.model_ckpt = model_name
    cfg = args.options

    return cfg, args

def main():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get('LOCAL_RANK') or 0)
    torch.cuda.set_device(rank)

    cfg, args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ckpt,
        device_map={"": f"cuda:{rank}"},
        trust_remote_code=True,
        load_in_8bit=args.size == "4.1b",
    )
    model_config = model.config

    model = NNsight(model)
    model.tokenizer = tokenizer

    submodule_dict = {}

    for layer in range(0, len(model.model.layers), model_config.moe_groups):
        @torch.compile
        def _forward(x):
            g1, g2 = x
            if args.separate_moe:
                return torch.cat([g1, g2], dim=-1).view(*g1.shape[:2], -1)
            else:
                return torch.einsum("bshx,bshy->bsxy", g1, g2).view(*g1.shape[:2], -1)

        submodule = model.model.layers[layer].router
        submodule.ae = AutoencoderLatents(
            None, _forward,
            width=model.config.moe_experts ** 2 if not args.separate_moe else model_config.moe_heads * model_config.moe_experts * 2,
            hookpoint=""
        )
        submodule_dict[submodule.path] = submodule

    with model.edit("") as edited:
        for _, submodule in submodule_dict.items():
            acts = submodule.output
            submodule.ae(acts, hook=True)

    tokens = load_tokenized_data(
        ctx_len=512,
        tokenizer=tokenizer,
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_row="text"
    )

    cache = FeatureCache(
        edited,
        submodule_dict,
        batch_size = 48,
    )

    with torch.inference_mode():
        cache.run(n_tokens = 10_000_000, tokens=tokens)

    save_dir = f"results/monet_cache{'/separate' if args.separate_moe else ''}/{args.size}"

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=save_dir
    )
    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=args.tokenizer_model
    )

if __name__ == "__main__":
    main()
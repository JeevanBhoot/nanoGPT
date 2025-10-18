"""
Sample from a trained model
"""
import argparse
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT
from tensor_parallel_model import TensorParallelGPTConfig, TensorParallelGPT


_DEFAULT_DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample from a trained model.")
    parser.add_argument(
        "--init_from",
        default="resume",
        help="Source of weights: 'resume' to load from an out_dir or a pretrained gpt2 variant.",
    )
    parser.add_argument(
        "--out_dir",
        default="out",
        help="Checkpoint directory used when init-from == 'resume'.",
    )
    parser.add_argument(
        "--start",
        default="\n",
        help="Prompt text or FILE:path to read the prompt from disk.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of completions to generate.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Tokens generated for each sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Keep only the top-k tokens during sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default=_DEFAULT_DTYPE,
        help="Computation dtype.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the loaded model.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel world size to simulate.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )

    # model
    if args.init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        if args.tp > 1:
            gptconf = TensorParallelGPTConfig(**checkpoint["model_args"])
            gptconf.tp_world_size = args.tp
            model = TensorParallelGPT(gptconf)
        else:
            gptconf = GPTConfig(**checkpoint["model_args"])
            model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif args.init_from.startswith("gpt2"):
        model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))
        checkpoint = None
    else:
        raise ValueError(f"Unsupported init_from option: {args.init_from}")

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    load_meta = False
    if (
        args.init_from == "resume"
        and checkpoint is not None
        and "config" in checkpoint
        and "dataset" in checkpoint["config"]
    ):
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    start = args.start
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    with torch.no_grad():
        with ctx:
            for _ in range(args.num_samples):
                y = model.generate(
                    x,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                print(decode(y[0].tolist()))
                print("---------------")
                if args.tp > 1:
                    print(model.get_tp_all_reduce_count())


if __name__ == "__main__":
    main()

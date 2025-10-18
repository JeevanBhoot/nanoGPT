"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import os
import time
import math
import pickle
import runpy
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tensor_parallel_model import TensorParallelGPTConfig, TensorParallelGPT
from tensor_parallel_spd_model import TensorParallelSPDGPTConfig, TensorParallelSPDGPT


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1"}:
        return True
    if value in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GPT model on OpenWebText.")
    # I/O and logging
    parser.add_argument("--out_dir", default="out", help="Output directory for checkpoints.")
    parser.add_argument("--eval_interval", type=int, default=2000, help="How often to evaluate the model.")
    parser.add_argument("--log_interval", type=int, default=1, help="How often to log training progress.")
    parser.add_argument("--eval_iters", type=int, default=200, help="Iterations per evaluation.")
    parser.add_argument("--eval_only", type=_str2bool, default=False, help="Run evaluation only and exit.")
    parser.add_argument("--always_save_checkpoint", type=_str2bool, default=True, help="Always save checkpoints on eval.")
    parser.add_argument("--init_from", default="scratch", help="Initialization method: 'scratch', 'resume', or 'gpt2*'.")
    parser.add_argument("--wandb_log", type=_str2bool, default=False, help="Enable W&B logging.")
    parser.add_argument("--wandb_project", default="owt", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", default="gpt2", help="Weights & Biases run name.")
    # data
    parser.add_argument("--dataset", default="openwebtext", help="Dataset name.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5 * 8, help="Steps to accumulate gradients.")
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size.")
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length.")
    # model
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout ratio.")
    parser.add_argument("--bias", type=_str2bool, default=False, help="Use bias in LayerNorm/Linear layers.")
    parser.add_argument("--model_impl", choices=["dense", "tensor_parallel", "tensor_parallel_spd"], default="dense", help="Model implementation to use.")
    parser.add_argument("--tp_world_size", type=int, default=2, help="Tensor-parallel world size (simulated).")
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Base learning rate.")
    parser.add_argument("--max_iters", type=int, default=600000, help="Maximum training iterations.")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    # LR schedule
    parser.add_argument("--decay_lr", type=_str2bool, default=True, help="Apply learning-rate decay.")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="Warmup iterations.")
    parser.add_argument("--lr_decay_iters", type=int, default=600000, help="Cosine decay iterations.")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate.")
    # system
    parser.add_argument("--backend", default="nccl", help="DDP backend.")
    parser.add_argument("--device", default="cuda", help="Device identifier (e.g., 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Computation dtype. Defaults to bf16 if available else fp16.",
    )
    parser.add_argument("--compile", type=_str2bool, default=True, help="Use torch.compile for speed.")
    return parser


config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument("--config", action="append", default=[], help="Path(s) to config .py files.")
config_parser.add_argument("config_files", nargs="*", help="Legacy positional config files.")
config_args, remaining_argv = config_parser.parse_known_args()

_DEFAULT_DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

parser = build_arg_parser()

config_paths = list(config_args.config_files)
config_paths.extend(config_args.config)

if config_paths:
    # We'll execute configs in order, later entries override earlier ones.
    dest_names = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    config_defaults = {}
    for path in config_paths:
        print(f"Loading config from {path}")
        overrides = runpy.run_path(path)
        for key, value in overrides.items():
            if key in dest_names:
                config_defaults[key] = value
    if config_defaults:
        parser.set_defaults(**config_defaults)

args = parser.parse_args(remaining_argv)

out_dir = args.out_dir
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_iters = args.eval_iters
eval_only = args.eval_only
always_save_checkpoint = args.always_save_checkpoint
init_from = args.init_from
wandb_log = args.wandb_log
wandb_project = args.wandb_project
wandb_run_name = args.wandb_run_name
dataset = args.dataset
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size = args.batch_size
block_size = args.block_size
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
model_impl = args.model_impl
tp_world_size = args.tp_world_size
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
grad_clip = args.grad_clip
decay_lr = args.decay_lr
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
min_lr = args.min_lr
backend = args.backend
device = args.device
dtype = args.dtype or _DEFAULT_DTYPE
compile = args.compile

if tp_world_size < 1:
    raise ValueError("tp_world_size must be >= 1")

config = vars(args).copy()
config['dtype'] = dtype
config['compile'] = compile
config['config_files'] = config_paths.copy()


def _instantiate_model(model_impl_value, model_args):
    """Create a GPT instance based on the requested implementation."""
    if model_impl_value == 'tensor_parallel':
        return TensorParallelGPT(TensorParallelGPTConfig(**model_args))
    elif model_impl_value == 'tensor_parallel_spd':
        return TensorParallelSPDGPT(TensorParallelSPDGPTConfig(**model_args))
    dense_args = {k: v for k, v in model_args.items() if k != 'world_size'}
    return GPT(GPTConfig(**dense_args))


def _existing_checkpoint_path(directory):
    if not os.path.isdir(directory):
        return None
    for root, _, files in os.walk(directory):
        if 'ckpt.pt' in files:
            return os.path.join(root, 'ckpt.pt')
    return None

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
existing_checkpoint = _existing_checkpoint_path(out_dir)
if existing_checkpoint and init_from != 'resume':
    raise RuntimeError(f"Found existing checkpoint at {existing_checkpoint}. Refusing to overwrite out_dir='{out_dir}'.")
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if model_impl == 'tensor_parallel' or model_impl == 'tensor_parallel_spd':
        model_args['world_size'] = tp_world_size
    else:
        model_args.pop('world_size', None)
    model = _instantiate_model(model_impl, model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {out_dir} to resume from.")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    checkpoint_model_impl = checkpoint.get('model_impl', 'dense')
    if model_impl != checkpoint_model_impl:
        print(f"Overriding model_impl to '{checkpoint_model_impl}' from checkpoint")
        model_impl = checkpoint_model_impl
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    if model_impl == 'tensor_parallel' or model_impl == 'tensor_parallel_spd':
        tp_world_size = checkpoint_model_args.get('world_size', tp_world_size)
        model_args['world_size'] = tp_world_size
    else:
        model_args.pop('world_size', None)
    # create the model
    model = _instantiate_model(model_impl, model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    if model_impl == 'tensor_parallel':
        model = TensorParallelGPT.from_pretrained(init_from, override_args)
    elif model_impl == 'tensor_parallel_spd':
        model = TensorParallelSPDGPT.from_pretrained(init_from, override_args)
    else:
        model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    if model_impl == 'tensor_parallel' or model_impl == 'tensor_parallel_spd':
        tp_world_size = getattr(model.config, 'world_size', tp_world_size)
        model_args['world_size'] = tp_world_size
    else:
        model_args.pop('world_size', None)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

config['model_impl'] = model_impl
config['tp_world_size'] = tp_world_size

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'model_impl': model_impl,
                    'tp_world_size': tp_world_size,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                checkpoint_dir = os.path.join(out_dir, str(iter_num))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pt')
                print(f"saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

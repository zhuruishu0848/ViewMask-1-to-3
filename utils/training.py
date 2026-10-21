"""Shared setup for pretraining, i2mv and t2mv."""
import argparse
import logging
import os
from pathlib import Path
from utils.checkpoints import get_last_checkpoint, has_training_state, resolve_checkpoint


def env_default(name, default=None):
    return os.environ.get(name, default)


def training_parser(stage):
    parser = argparse.ArgumentParser(description=f'ViewMask stage {stage}')
    parser.add_argument('--model_name_or_path', default=env_default('MODEL_NAME_OR_PATH'), help='Prepared LLaDA with visual and task tokens')
    parser.add_argument('--tokenizer_name_or_path', default=env_default('TOKENIZER_NAME_OR_PATH'), help='Defaults to model_name_or_path; must match checkpoint token IDs')
    parser.add_argument('--magvit_path', default=env_default('MAGVIT_PATH', 'showlab/magvitv2'))
    parser.add_argument('--load_from_checkpoint', default=env_default('CHECKPOINT'), help='Load weights only; start a fresh optimizer/scheduler')
    parser.add_argument('--output_dir', default=f'./outputs/stage{stage}')
    parser.add_argument('--logging_dir', default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=float, default=100)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--save_strategy', default='steps')
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--eval_num_steps', type=int, default=50)
    parser.add_argument('--eval_temperature', type=float, default=0.8)
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Must be 0: dataset image encoding runs on GPU')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--report_to', default='tensorboard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume latest complete training state in output_dir')
    parser.add_argument('--resume_from_checkpoint', default=None, help='Resume an explicit checkpoint including optimizer/scheduler')
    return parser


def custom_collate_fn(batch):
    """Keep diffusion metadata without mutating dataset-owned dictionaries."""
    from transformers import default_data_collator
    fields = ('input_ids', 'attention_mask', 'prompt_lengths', 'weights')
    collated = default_data_collator([{key: item[key] for key in fields if key in item} for item in batch])
    if any('task_name' in item for item in batch):
        collated['task_name'] = [item.get('task_name', 'unknown') for item in batch]
    return collated


def resume_checkpoint(args):
    if args.resume_from_checkpoint:
        checkpoint = resolve_checkpoint(args.resume_from_checkpoint)
        if not has_training_state(checkpoint):
            raise ValueError(f'{checkpoint} lacks complete Trainer/optimizer/scheduler/RNG state; use --load_from_checkpoint')
        return checkpoint
    if args.resume:
        checkpoint = get_last_checkpoint(args.output_dir, resumable=True)
        if not checkpoint:
            raise FileNotFoundError(f'No resumable checkpoint in {args.output_dir}; omit --resume to start training')
        return checkpoint
    return None


def validate_tokenizer(tokenizer, config, required_tokens):
    for token in required_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError(f'Tokenizer is missing the single token {token!r}; use a prepared multimodal tokenizer')
        if ids[0] >= config.vocab_size:
            raise ValueError(f'Token {token!r} ID {ids[0]} exceeds model vocab_size={config.vocab_size}')
    if len(tokenizer) > config.vocab_size:
        raise ValueError(f'Tokenizer size {len(tokenizer)} exceeds model vocab_size {config.vocab_size}')


def resize_model_for_tokenizer(model, size):
    """Expand LLaDA input/output vocabularies while retaining all existing rows."""
    if size > model.config.vocab_size:
        # LLaDA uses weight_tying, while HF resize checks tie_word_embeddings.
        model.config.tie_word_embeddings = model.config.weight_tying
        model.resize_token_embeddings(size, mean_resizing=False)
        for model_config in (model.config, model.model.config):
            model_config.vocab_size = size
            model_config.embedding_size = size
    return model


def run_training(args, build_datasets, *, multiview):
    if args.dataloader_num_workers != 0:
        raise ValueError('Use --dataloader_num_workers 0: MAGVIT runs on the GPU inside __getitem__')
    checkpoint = resume_checkpoint(args)
    source = checkpoint or args.load_from_checkpoint or args.model_name_or_path
    if not source:
        raise ValueError('Set --model_name_or_path or --load_from_checkpoint')
    if Path(source).is_dir() or str(source).startswith(('/', '.')):
        source = resolve_checkpoint(source)
    source_has_tokenizer = (Path(source) / 'tokenizer.json').is_file()
    tokenizer_source = args.tokenizer_name_or_path or (source if source_has_tokenizer else args.model_name_or_path or source)

    import torch
    from utils.checkpoint_compat import install_deepspeed_checkpoint_compatibility
    install_deepspeed_checkpoint_compatibility()
    from accelerate.utils import set_seed
    from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
    from dataset import SPECIAL_TOKENS
    from diffusion.core import DiffusionTrainer, DiffusionTrainerTask_Eval
    from tokenizer.modeling_magvitv2 import MAGVITv2

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, torch_dtype=torch.bfloat16)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    names = ('I2MI', 'SOI', 'EOI', 'SOT', 'EOT') if multiview else ('T2I', 'MMU', 'I2I', 'SOI', 'EOI', 'SOT', 'EOT')
    if not multiview and not checkpoint:
        # Original preparation order; append missing tokens without changing existing IDs.
        text_tokenizer.add_tokens([f'<|VISUAL_{i}|>' for i in range(8192)] +
                                  [SPECIAL_TOKENS[n] for n in ('SOI', 'EOI', 'SOT', 'EOT', 'MMU', 'T2I', 'I2I', 'I2MI')])
    else:
        validate_tokenizer(text_tokenizer, config, [SPECIAL_TOKENS[n] for n in names] + ['<|VISUAL_0|>', '<|VISUAL_8191|>'])
    if args.gradient_checkpointing:
        config.use_cache = False
    visual_tokenizer = MAGVITv2.from_pretrained(args.magvit_path).to(device).eval()
    visual_tokenizer.requires_grad_(False)
    model = AutoModel.from_pretrained(source, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    resize_model_for_tokenizer(model, len(text_tokenizer))
    if not multiview and not checkpoint:
        validate_tokenizer(text_tokenizer, model.config, [SPECIAL_TOKENS[n] for n in names] + ['<|VISUAL_0|>', '<|VISUAL_8191|>'])
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    train_dataset, eval_dataset = build_datasets(args, visual_tokenizer, text_tokenizer)
    if not len(train_dataset):
        raise ValueError('Training dataset is empty; check paths and data splits')
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir or str(Path(args.output_dir) / 'logs'),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        adam_beta1=0.9, adam_beta2=0.999,
        bf16=torch.cuda.is_available(), gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps, save_strategy=args.save_strategy,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        report_to=args.report_to, seed=args.seed,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    training_args.eval_num_steps = args.eval_num_steps
    training_args.eval_temperature = args.eval_temperature
    trainer_cls = DiffusionTrainerTask_Eval if multiview else DiffusionTrainer
    extra = {'eval_frequency': args.eval_steps} if multiview else {}
    trainer = trainer_cls(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=custom_collate_fn,
        processing_class=text_tokenizer, visual_tokenizer=visual_tokenizer, **extra,
    )
    logger.info('Model: %s; tokenizer: %s; resume: %s; samples: %s', source, tokenizer_source, checkpoint, len(train_dataset))
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()

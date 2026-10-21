"""Checkpoint discovery without importing GPU libraries."""
import json
import re
from pathlib import Path


def has_model_weights(path):
    path = Path(path)
    if any((path / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        return True
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path / name).is_file():
            shards = set(json.loads((path / name).read_text())['weight_map'].values())
            return bool(shards) and all((path / shard).is_file() for shard in shards)
    return False


def has_training_state(path):
    """An exported HF model alone cannot restore optimizer and random state."""
    path = Path(path)
    if not (path / 'trainer_state.json').is_file():
        return False
    optimizer = (path / 'optimizer.pt').is_file() or any(path.glob('global_step*/*optim_states.pt'))
    scheduler = (path / 'scheduler.pt').is_file()
    rng = any(path.glob('rng_state*.pth'))
    return bool(optimizer and scheduler and rng)


def get_last_checkpoint(output_dir, *, resumable=False):
    if not output_dir or not Path(output_dir).is_dir():
        return None
    candidates = []
    for path in Path(output_dir).glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d+)', path.name)
        if match and (path / 'config.json').is_file() and has_model_weights(path):
            if resumable and not has_training_state(path):
                continue
            candidates.append((int(match[1]), path))
    return str(max(candidates, key=lambda item: item[0])[1]) if candidates else None


def resolve_checkpoint(path):
    """Accept an HF model directory or an output containing numbered checkpoints."""
    path = Path(path).expanduser()
    if (path / 'config.json').is_file() and has_model_weights(path):
        return str(path)
    latest = get_last_checkpoint(path)
    if latest:
        return latest
    raise FileNotFoundError(f'No complete model checkpoint found in {path}')

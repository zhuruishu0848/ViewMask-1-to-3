import bisect

from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, datasets: dict):
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.lengths = [len(datasets[task]) for task in self.task_names]
        self.cum_lengths = [0]
        for length in self.lengths:
            self.cum_lengths.append(self.cum_lengths[-1] + length)
        self.length = self.cum_lengths[-1]

        print("\nPretraining dataset:")
        for task, length in zip(self.task_names, self.lengths):
            print(f"Task: {task} - samples: {length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        task_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        task_name = self.task_names[task_idx]
        idx_in_task = idx - self.cum_lengths[task_idx]
        assert 0 <= idx_in_task < len(self.datasets[task_name]), (
            f"idx_in_task {idx_in_task} out of range for task {task_name} "
            f"with length {len(self.datasets[task_name])}, idx={idx}, cum_lengths={self.cum_lengths}"
        )

        item = self.datasets[task_name][idx_in_task]
        if item is None:
            raise RuntimeError(f"{task_name} dataset returned None for idx {idx_in_task}")
        return item

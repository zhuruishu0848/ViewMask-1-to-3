"""Stage 3: caption-conditioned multiview training on Objaverse + HSSD."""
from utils.training import env_default, run_training, training_parser


def parse_args():
    parser = training_parser(3)
    parser.add_argument('--train_dataset_i2mi_objaverse_json', default=env_default('OBJAVERSE_CAPTION_JSON', './data/objaverse/train_caption_samples.json'))
    parser.add_argument('--train_dataset_i2mi_hssd_json', default=env_default('HSSD_TRAIN_JSON', './data/HSSD/train_samples.json'))
    parser.add_argument('--eval_dataset_i2mi_gso_json', default=env_default('GSO_EVAL_JSON'))
    parser.add_argument('--text_condition_probability', type=float, default=0.5, help='Chance of generating all four views from text, for samples with caption')
    parser.set_defaults(learning_rate=1e-5, num_train_epochs=10, gradient_accumulation_steps=2)
    return parser.parse_args()


def build_datasets(args, visual_tokenizer, text_tokenizer):
    from dataset.multiview_dataset import (
        MultiViewDatasetCollection,
        MultiViewObjaverseCaptionDataset,
        MultiViewObjaverseDataset,
    )
    common = dict(visual_tokenizer=visual_tokenizer, text_tokenizer=text_tokenizer, max_length=args.max_length)
    train = MultiViewDatasetCollection({name: MultiViewObjaverseCaptionDataset(path, phase='train', text_condition_probability=args.text_condition_probability, **common) for name, path in (
        ('objaverse', args.train_dataset_i2mi_objaverse_json),
        ('hssd', args.train_dataset_i2mi_hssd_json),
    )})
    evaluation = None
    if args.eval_dataset_i2mi_gso_json:
        evaluation = MultiViewDatasetCollection({'gso': MultiViewObjaverseDataset(args.eval_dataset_i2mi_gso_json, phase='test', **common)})
    return train, evaluation


def main():
    run_training(parse_args(), build_datasets, multiview=True)


if __name__ == '__main__':
    main()

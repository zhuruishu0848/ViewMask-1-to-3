"""Stage 1: ImageNet t2i + LLaVA understanding pretraining."""
from utils.training import env_default, run_training, training_parser


def parse_args():
    parser = training_parser(1)
    parser.add_argument('--train_dataset_ImageNet_path', default=env_default('IMAGENET_DATA', './data/Imagenet'))
    parser.add_argument('--train_dataset_LLaVA_json_path', default=env_default('LLAVA_PRETRAIN_JSON', './data/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'))
    parser.set_defaults(num_train_epochs=100)
    return parser.parse_args()


def build_datasets(args, visual_tokenizer, text_tokenizer):
    from dataset.imagenet_t2i import ImageNetTokenizedDataset
    from dataset.llava_pretrain_dataset import LLaVAPretrainDataset
    from dataset.multi_task_dataset import PretrainingDataset
    common = dict(visual_tokenizer=visual_tokenizer, text_tokenizer=text_tokenizer, max_length=args.max_length)
    datasets = {
        'imagenet': ImageNetTokenizedDataset(args.train_dataset_ImageNet_path, **common),
        'llava': LLaVAPretrainDataset(args.train_dataset_LLaVA_json_path, **common),
    }
    return PretrainingDataset(datasets), None


def main():
    run_training(parse_args(), build_datasets, multiview=False)


if __name__ == '__main__':
    main()

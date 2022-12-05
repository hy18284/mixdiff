import random
import copy

import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
    ToPILImage
)
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import numpy as np
import wandb
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)

from dataloaders.ZO_Clip_loaders import cifar10_single_isolated_class_loader
from .ood_score_calculators.ood_score_calculator import OODScoreCalculator


N = 10
M = 10
R = 3
B = 2048
SEED = 0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='mixup_v1')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--abs', action='store_true')
    parser.add_argument('--top_1', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--logit_avg', action='store_true')
    parser.add_subclass_arguments(OODScoreCalculator, 'score_calculator')
    parser.add_argument('--config', action=ActionConfigFile)

    args = parser.parse_args()
    classes = parser.instantiate_classes(args)
    return args, classes


def tokenize_for_clip(batch_sentences, tokenizer):
    default_length = 77  # CLIP default
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokenized_list = []
    for sentence in batch_sentences:
        text_tokens = [sot_token] + tokenizer.encode(sentence) + [eot_token]
        tokenized = torch.zeros((default_length), dtype=torch.long)
        tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
        tokenized_list.append(tokenized)
    tokenized_list = torch.stack(tokenized_list)
    return tokenized_list


if __name__ == '__main__':
    args, classes = parse_args()
    score_calculator: OODScoreCalculator = classes['score_calculator']

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R = args.n, args.m, args.r
    batch_size = N

    device = torch.device(args.device)
    score_calculator.load_model('ViT-B/32', device)

    splits = [
        ['airplane', 'automobile', 'bird', 'deer', 'dog', 'truck', 'cat', 'frog', 'horse', 'ship'],
        ['airplane', 'cat', 'dog', 'horse', 'ship', 'truck', 'automobile', 'bird', 'deer', 'frog'],
        ['airplane', 'automobile', 'dog', 'frog', 'horse', 'ship', 'bird', 'cat', 'deer', 'truck'],
        ['cat', 'deer', 'dog', 'horse', 'ship', 'truck', 'airplane', 'automobile', 'bird', 'frog'],
        ['airplane', 'automobile', 'bird', 'cat', 'horse', 'ship', 'deer', 'dog', 'frog', 'truck'],
    ]

    wandb.init(
        config=args,
        name=f'{args.wandb_name}_{score_calculator}',
        project='ZOC',
    )
    wandb.config.splits = splits
    
    aurocs = []
    for j, split in enumerate(splits):
        num_unknown = 4
        num_known = 6
        seen_labels = split[:num_known]

        cifar10_loaders_train = cifar10_single_isolated_class_loader(train=True)
        given_images = []
        for seen_label in seen_labels:
            loader = cifar10_loaders_train[seen_label]
            images = random.choices(loader.dataset, k=M)
            images = torch.stack(images)
            given_images.append(images)

        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        given_images = given_images.to(device)

        transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        class CIFAR10Wrapper(CIFAR10):
            def __getitem__(self, idx):
                image, label = super().__getitem__(idx)
                return transform(image), label

        cifar10 = CIFAR10Wrapper(root='./data', train=False, download=True)
        seen_idx = [cifar10.class_to_idx[seen_label] for seen_label in seen_labels]
        seen_idx = torch.tensor(seen_idx, device=device)

        r_start = 1.0 / (R + 1)
        r_end = 1 - r_start
        rates = torch.linspace(r_start, r_end, R, device=device)
        print(rates)
        targets = []
        scores = []
        loader = DataLoader(cifar10, batch_size=batch_size, num_workers=2, shuffle=True)

        score_calculator.on_eval_start(copy.deepcopy(seen_labels))

        for i, (images, labels) in enumerate(tqdm(loader)):
            N, C, H, W = images.size()
            NC = len(seen_labels)
            # (N, C, H, W)
            images = images.to(device)
            labels = labels.to(device)
            image_kwargs = score_calculator.process_images(images)

            if score_calculator.utilize_mixup:
                # (N, M, C, H, W)
                chosen_images, chosen_idx = score_calculator.select_given_images(
                    given_images, 
                    **image_kwargs
                )
                # (N, 1, C, H, W) | (N, M, C, H, W) -> (N, 1+M, C, H, W)
                input_images = torch.cat([images.unsqueeze(1), chosen_images], dim=1)
                # (N, 1+M, C, H, W, 1) * (R) -> (N, 1+M, C, H, W, R)
                input_images = input_images.unsqueeze(-1) * rates

                # (NC, M, C, H, W, 1) * (R) -> (NC, M, C, H, W, R)
                base_images = given_images.unsqueeze(-1) * (1.0 - rates)

                # (N, 1+M, 1, 1, C, H, W, R) * (1, 1, NC, M, C, H, W, R)
                # -> (N, 1+M, NC, M, C, H, W, R)
                mixed_images = input_images[:, :, None, None, ...] + base_images[None, None, ...]
                # (N, 1+M, NC, M, C, H, W, R) -> (N, 1+M, NC, M, R, C, H, W)
                mixed_images = mixed_images.permute(0, 1, 2, 3, 7, 4, 5, 6)
                # (N, 1+M, NC, M, R, C, H, W) -> (N * (1+M) * NC * M * R, C, H, W)
                mixed_images = mixed_images.reshape(-1, C, H, W)

                # (N * (1+M) * NC * M * R, C, H, W) -> (N * (1+M) * NC * M * R, NC)
                mixup_logits = score_calculator.process_mixup_images(mixed_images)
                # (N * (1+M) * NC * M * R, NC) -> (N, 1+M, NC, M, R, NC)
                mixup_logits = mixup_logits.view(N, 1+M, NC, M, R, NC)

                if args.logit_avg:
                    # (N, 1+M, NC, M, R, NC) -> (N, NC, M, R, NC)
                    known_logits = mixup_logits[:, 1, ...]
                    unknown_logits = torch.mean(mixup_logits[:, 1:, ...], dim=1)
                    # (N, NC, M, R, NC) -> (N, NC, M, R)
                    mix_diff = score_calculator.calculate_diff(known_logits, unknown_logits)
                    mix_diff[torch.arange(N), chosen_idx, ...] = 0.0

                    effective_numel = torch.numel(mix_diff[0]) * (1 - 1 / M)
                    # (N, NC, M, R) -> (N)
                    mix_diff = torch.sum(mix_diff.view(N, -1), dim=-1) / effective_numel
                
                else:
                    # (N, 1+M, NC, M, R, NC) -> (N, 1+M, NC, M, R)
                    mixup_scores = score_calculator.calculate_base_scores(mixup_logits)
                    # (N, 1, NC, M, R) - (N, M, NC, M, R) -> (N, M, NC, M, R) 
                    mix_diff = mixup_scores[:, :1, ...] - mixup_scores[:, 1:, ...]
                    # mix_diff = mixup_scores[:, 1, ...] - torch.mean(mixup_scores[:, 1:, ...], dim=1)
                    mix_diff[torch.arange(N), :, chosen_idx, ...] = 0.0

                    effective_numel = torch.numel(mix_diff[0]) * (1 - 1 / M)
                    # (N, NC, M, R) -> (N)
                    mix_diff = torch.sum(mix_diff.view(N, -1), dim=-1) / effective_numel

                if args.abs:
                    mix_diff = torch.abs(mix_diff)

                # if args.top_1:
                #     abs_dists = torch.abs(dists)
                #     abs_max_idx = torch.argmax(abs_dists, dim=-1)
                #     dists = dists[torch.arange(dists.size(0)), abs_max_idx]
                # else:
                #     dists = torch.sum(dists, dim=-1) / torch.sum(mask > 0.5, dim=-1)
            
            base_scores = score_calculator.calculate_base_scores(**image_kwargs)

            if score_calculator.utilize_mixup:
                final_scores = base_scores + args.gamma * mix_diff
            else:
                final_scores = base_scores

            targets += [int(label not in seen_idx) for label in labels]
            scores += final_scores.tolist()

            if i == 1000:
                break

        score_calculator.on_eval_end()

        ood_scores = [score for score, target in zip(scores, targets) if target == 1]
        id_scores = [score for score, target in zip(scores, targets) if target == 0]
        ood_mean = np.mean(ood_scores)
        id_mean = np.mean(id_scores)
        print(f'ood_mean: {ood_mean}')
        print(f'id_mean: {id_mean}')
        print(f'ood - id mean: {ood_mean - id_mean}')
            
        auroc = roc_auc_score(targets, scores)
        print(f'auroc: {auroc}')
        wandb.log({'auroc': auroc})
        aurocs.append(auroc)
    
    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)
    wandb.log({'avg_auroc': avg_auroc, 'auroc_std': auroc_std})

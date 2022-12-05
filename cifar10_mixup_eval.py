import random

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
from argparse import ArgumentParser

from dataloaders.ZO_Clip_loaders import cifar10_single_isolated_class_loader


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
    parser.add_argument('--b', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='mixup_v1')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--abs', action='store_true')
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--top_1', action='store_true')
    parser.add_argument('--mixup_var', type=str, default=None, choices=[
        'ls', 'sp', 'emd', None,
    ])
    parser.add_argument('--base_score', type=str, default=None, choices=[
        'msp', 'mls', 'msp+mls', None,
    ])
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    return args


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
    args = parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R, B = args.n, args.m, args.r, args.b
    batch_size = N

    device = torch.device(args.device)
    clip_model, _ = clip.load('ViT-B/32', device=device, download_root='trained_models')
    clip_model.eval()
    cliptokenizer = clip_tokenizer()

    splits = [
        ['airplane', 'automobile', 'bird', 'deer', 'dog', 'truck', 'cat', 'frog', 'horse', 'ship'],
        ['airplane', 'cat', 'dog', 'horse', 'ship', 'truck', 'automobile', 'bird', 'deer', 'frog'],
        ['airplane', 'automobile', 'dog', 'frog', 'horse', 'ship', 'bird', 'cat', 'deer', 'truck'],
        ['cat', 'deer', 'dog', 'horse', 'ship', 'truck', 'airplane', 'automobile', 'bird', 'frog'],
        ['airplane', 'automobile', 'bird', 'cat', 'horse', 'ship', 'deer', 'dog', 'frog', 'truck'],
    ]

    wandb.init(
        config=args,
        name=f'{args.wandb_name}_{args.mixup_var}+{args.base_score}',
        project='ZOC',
    )
    wandb.config.splits = splits
    
    aurocs = []
    for j, split in enumerate(splits):
        num_unknown = 4
        num_known = 6
        seen_labels = split[:num_known]
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

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

        prompts_ids = [clip.tokenize(prompt) for prompt in seen_descriptions]
        prompts_ids = torch.cat(prompts_ids, dim=0).to(device)
        prompt_ids = prompts_ids.to(device)

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
        base_scores = []
        deltas = []
        loader = DataLoader(cifar10, batch_size=batch_size, num_workers=2, shuffle=True)

        for i, (images, labels) in enumerate(tqdm(loader)):
            N, C, H, W = images.size()
            NC = prompt_ids.size(0)
            # (N, C, H, W)
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = clip_model(images, prompts_ids)

            if args.mixup_var is not None:
                # (N)
                max_indices = torch.argmax(logits, dim=-1)
                # (NC, M, C, H, W) -> (N, M, C, H, W)
                chosen_images = given_images[max_indices, ...]
                # (N, M, C, H, W) -> (N, N, M, C, H, W)
                chosen_images = chosen_images.expand(N, -1, -1, -1, -1, -1)
                # (N, N, M, C, H, W) -> (M, N, N, C, H, W)
                chosen_images = chosen_images.permute(2, 1, 0, 3, 4, 5)

                # (N, C, H, W) -> (M, N, N, C, H, W)
                images_m = images.expand(M, N, -1, -1, -1, -1)

                # (M, N, N, C, H, W, 1) * (R) -> (M, N, N, C, H, W, R)
                chosen_images = chosen_images.unsqueeze(-1) * rates 
                images_m = images_m.unsqueeze(-1) * (1 - rates)
                known_mixup = chosen_images + images_m
                known_mixup = known_mixup.permute(0, 1, 2, 6, 3, 4, 5)
                known_mixup = known_mixup.reshape(M * N * N * R, C, H, W)

                def batchfy(orig_images, input_ids):
                    # print(orig_images.size())
                    import time
                    s_time = time.time()
                    orig_images_list = torch.split(orig_images, B, dim=0)
                    logits_list = []
                    for images in orig_images_list:
                        print(images.size())
                        split_logits, _ = clip_model(images, input_ids)
                        logits_list.append(split_logits) 
                    print(orig_images.size())
                    print(input_ids.size())
                    print(f'{time.time() - s_time}')
                    return torch.cat(logits_list, dim=0)
                
                known_logits = batchfy(known_mixup, prompt_ids)

                # (M * N * N * R, NC) -> (M, N, N, R, NC)
                known_logits = known_logits.view(M, N, N, R, -1)
                # (M, N, N, R, NC) -> (N, N, R, NC)
                known_logits = torch.mean(known_logits, dim=0)
                known_probs = torch.softmax(known_logits, dim=-1)
                known_probs = known_probs.view(-1, NC)

                # (N, C, H, W) -> (N, N, C, H, W) -> (N, N, C, H, W, R)
                images_n_1 = images.expand(N, -1, -1, -1, -1)
                images_n_1 = images_n_1.permute(1, 0, 2, 3, 4).unsqueeze(-1) * rates
                images_n_2 = images.expand(N, -1, -1, -1, -1).unsqueeze(-1) * (1 - rates)
                # (N, C, H, W) -> (M, N, C, H, W)
                unknown_mixup = images_n_1 + images_n_2
                unknown_mixup = unknown_mixup.permute(0, 1, 5, 2, 3, 4)
                unknown_mixup = unknown_mixup.reshape(N * N * R, C, H, W)
                unknown_logits = batchfy(unknown_mixup, prompt_ids)
                # (N * N * R, NC) -> (N, N, R, NC)
                unknown_logits = unknown_logits.view(N, N, R, -1)
                unknown_probs = torch.softmax(unknown_logits, dim=-1)
                unknown_probs = unknown_probs.view(-1, NC)

                if args.mixup_var == 'emd':
                    dists = []
                    for known, unknown in zip(known_probs.tolist(), unknown_probs.tolist()):
                        dists.append(wasserstein_distance(known, unknown))
                    dists = torch.tensor(dists, device=device) 
                else:
                    # (N, R, N) -> (N, N, R) -> (N * N * R)
                    if args.fixed:
                        expanded_max_idx = max_indices.expand(N, R, -1)
                        expanded_max_idx = expanded_max_idx.permute(2, 0, 1)
                        expanded_max_idx = expanded_max_idx.reshape(-1)

                    if args.mixup_var in ['sp']:
                        if not args.fixed:
                            known_max, _ = torch.max(known_probs, dim=-1) 
                            unknown_max, _  = torch.max(unknown_probs, dim=-1) 
                        else: 
                            known_max = known_probs[torch.arange(N*N*R), expanded_max_idx]
                            unknown_max = unknown_probs[torch.arange(N*N*R), expanded_max_idx]

                    if args.mixup_var in ['ls']:
                        if not args.fixed:
                            known_max, _ = torch.max(known_logits, dim=-1) 
                            unknown_max, _  = torch.max(unknown_logits, dim=-1) 
                        else:
                            known_max = known_logits[torch.arange(N*N*R):, max_indices]
                            unknown_max = unknown_logits[torch.arange(N*N*R):, max_indices]

                    dists = known_max - unknown_max
                    if args.abs:
                        dists = torch.abs(dists)

                dists = dists.view(N, N, R)
                dists = torch.mean(dists, dim=-1)
                mask = torch.ones_like(dists)
                mask.fill_diagonal_(0.0)
                dists = dists * mask
                if args.top_1:
                    abs_dists = torch.abs(dists)
                    abs_max_idx = torch.argmax(abs_dists, dim=-1)
                    dists = dists[torch.arange(dists.size(0)), abs_max_idx]
                else:
                    dists = torch.sum(dists, dim=-1) / torch.sum(mask > 0.5, dim=-1)

                # max_logits, _ = torch.max(logits, dim=-1)
                # max_mask = (max_logits > 5.0).to(dtype=dists.dtype)
                # dists = dists * max_mask

                if args.base_score == 'msp':
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    deltas.append((-dists).tolist())
                    base_scores.append(max_probs.tolist())
                    dists = -max_probs + args.gamma * dists

                elif args.base_score == 'mls':
                    max_logits, _ = torch.max(logits, dim=-1)
                    deltas.append((-dists).tolist())
                    base_scores.append(max_logits.tolist())
                    dists = -max_logits + args.gamma * dists
            
            else:
                if args.base_score == 'mls':
                    max_logits, _ = torch.max(logits, dim=-1)
                    dists = -max_logits
                elif args.base_score == 'msp':
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    dists = -max_probs
                elif args.base_score == 'msp+mls':
                    max_logits, _ = torch.max(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    dists = - max_probs - args.gamma * max_logits
            
            targets += [int(label not in seen_idx) for label in labels]
            scores += dists.tolist()

        #     if i == 100:
        #         break
        # if j == 1:
        #     break

        ood_scores = [score for score, target in zip(scores, targets) if target == 1]
        id_scores = [score for score, target in zip(scores, targets) if target == 0]
        ood_mean = np.mean(ood_scores)
        id_mean = np.mean(id_scores)
        print(f'ood_mean: {ood_mean}')
        print(f'id_mean: {id_mean}')
        print(f'ood - id mean: {ood_mean - id_mean}')
        # plt.scatter(base_scores, deltas)
        # plt.savefig(f'{args.mixup_var}_{j}.png')
            
        auroc = roc_auc_score(targets, scores)
        print(f'auroc: {auroc}')
        wandb.log({'auroc': auroc})
        aurocs.append(auroc)
    
    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)
    wandb.log({'avg_auroc': avg_auroc, 'auroc_std': auroc_std})

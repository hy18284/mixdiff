import random
import copy
import itertools
from typing import (
    List,
    Union,
    Optional,
)

import torch
from sklearn.metrics import (
    roc_auc_score,
)
from tqdm import tqdm
import numpy as np
import wandb
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)

from .ood_score_calculators.ood_score_calculator import OODScoreCalculator
from .ood_datamodules.base_ood_datamodule import BaseOODDataModule
from .mixup_operators.base_mixup_operator import BaseMixupOperator
from .utils import (
    log_mixup_samples,
    calculate_fnr_at,
    calculate_fpr_at,
    calculate_ood_metrics,
)
from .adversarial.confidence_pgd_attack_label_flip import ConfidenceLinfPGDAttack


FIX_REF_MODES = ['rand_id', 'single_class', 'single_sample']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--r_ref', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inner_seeds', type=int, default=[], nargs='*')
    parser.add_argument('--wandb_name', type=str, default='mixup_v1')
    parser.add_argument('--wandb_project', type=str, default='ZOC')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--abs', action='store_true')
    parser.add_argument('--top_1', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--scale_mixdiff', type=bool, default=False)
    parser.add_argument('--max_samples', type=Optional[int], default=None, required=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ref_mode', type=str, default='in_batch')
    parser.add_argument('--fnr_at', type=float, default=0.95)
    parser.add_argument('--fpr_at', type=float, default=0.95)
    parser.add_argument('--id_as_neg', type=bool, default=True)
    parser.add_argument('--log_interval', type=Optional[int], default=None)
    parser.add_argument('--attack', type=Optional[str], default=None, choices=['id2ood', 'ood2id', 'both', 'none', 'label_flip'])
    parser.add_argument('--attack_eps', type=Optional[float], default=None)
    parser.add_argument('--attack_nb_iter', type=Optional[int], default=None)
    parser.add_argument('--trans_before_mixup', type=bool, default=True)
    parser.add_argument('--measure_time', type=bool, default=False)
    parser.add_argument('--label_flip_ratio', type=float, default=0.5)
    parser.add_subclass_arguments(OODScoreCalculator, 'score_calculator')
    parser.add_subclass_arguments(BaseOODDataModule, 'datamodule')
    parser.add_subclass_arguments(BaseMixupOperator, 'mixup_operator')
    # parser.add_subclass_arguments(ConfidenceLinfPGDAttack, 'attack', required=False, default=None)
    parser.add_argument('--config', action=ActionConfigFile)

    args = parser.parse_args()
    classes = parser.instantiate_classes(args)
    return args, classes


def calculate_metrics(
    args,
    scores,
    base_scores_log,
    mixdiff_scores_log,
    targets,
    iter_idx,
    labels,
    pred_labels,
    ignore_tgt_for_acc,
):
    targets, scores = targets[:args.max_samples], scores[:args.max_samples]

    if not args.id_as_neg: 
        scores = [-score for score in scores]
        base_scores_log = [-base_score for base_score in base_scores_log]
        mixdiff_scores_log = [-mixdiff_score for mixdiff_score in mixdiff_scores_log]
    
    table = wandb.Table(
        columns=['id', 'ood_score','base_score', 'mixdiff_score', 'is_ood', 'label', 'pred_label'],
        data=[
            (i, score, base_score, mixdiff_score, target, label, pred)
            for i, (score, base_score, mixdiff_score, target, label, pred) 
            in enumerate(zip(scores, base_scores_log, mixdiff_scores_log, targets, labels, pred_labels))
        ]
    )
    wandb.log({f'ood_scores_{iter_idx}': table})

    ood_scores = [score for score, target in zip(scores, targets) if target == 1]
    id_scores = [score for score, target in zip(scores, targets) if target == 0]
    ood_mean = np.mean(ood_scores)
    id_mean = np.mean(id_scores)
    print(f'ood_mean: {ood_mean}')
    print(f'id_mean: {id_mean}')
    print(f'ood - id mean: {ood_mean - id_mean}')

    acc = []

    for probs_dict, label, target in zip(pred_labels, labels, targets):
        if not ignore_tgt_for_acc and target == 1:
            continue

        max_val = float('-inf')
        pred = None
        for idx, prob in probs_dict.items():
            if max_val < prob:
                max_val = prob
                pred = idx

        if str(label) == pred:
            acc.append(1)
        else:
            acc.append(0)
    acc = sum(acc) / len(acc)
        
    if not args.id_as_neg: 
        results = calculate_ood_metrics(np.array(id_scores), np.array(ood_scores))
        auroc = results['AUROC']
        fpr_at = results['FPR']
        fnr_at = 100.0
    else:
        auroc = roc_auc_score(targets, scores)
        fpr_at = calculate_fpr_at(scores, targets, args.fpr_at)
        fnr_at = calculate_fnr_at(scores, targets, args.fnr_at)

    print(f'auroc: {auroc}')

    return auroc, fpr_at, fnr_at, acc


if __name__ == '__main__':
    args, classes = parse_args()
    score_calculator: OODScoreCalculator = classes['score_calculator']
    datamodule: BaseOODDataModule = classes['datamodule']
    mixup_fn: BaseMixupOperator = classes['mixup_operator']
    # attack = classes['attack']
    if args.attack in ['ood2id', 'both']:
        ood2id_attack = ConfidenceLinfPGDAttack(
            model=score_calculator, 
            eps=args.attack_eps, 
            nb_iter=args.attack_nb_iter, 
            in_distribution=False,
        )
    else:
        ood2id_attack = None

    if args.attack in ['id2ood', 'both']:
        id2ood_attack = ConfidenceLinfPGDAttack(
            model=score_calculator, 
            eps=args.attack_eps, 
            nb_iter=args.attack_nb_iter, 
            in_distribution=True,
        )
    else:
        id2ood_attack = None

    if args.attack in ['label_flip']:
        label_flip = ConfidenceLinfPGDAttack(
            model=score_calculator, 
            eps=args.attack_eps, 
            nb_iter=args.attack_nb_iter, 
            in_distribution=True,
            label_flip=True,
        )
    else:
        label_flip = None


    datamodule.ref_mode = args.ref_mode
    score_calculator.ref_mode = args.ref_mode

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R, P = args.n, args.m, args.r, args.p
    batch_size = N
    if args.ref_mode == 'oracle':
        P = M 
    elif args.ref_mode == 'in_batch':
        P = N

    device = torch.device(args.device)
    # TODO: Remove this. Use model_path from get_splits
    score_calculator.load_model(args.model_path, device)
    
    if score_calculator.utilize_mixup: 
        if args.ref_mode == 'in_batch':
            ref = 't'
        elif args.ref_mode == 'oracle':
            ref = 'o'
        elif args.ref_mode in 'rand_id':
            ref = 'i'
        elif args.ref_mode in 'single_class':
            ref = 's'
        elif args.ref_mode in 'single_sample':
            ref = 'ss'


        name=f'{args.wandb_name}_{score_calculator}_{ref}_{datamodule}_{mixup_fn}'
    else:
        name=f'{args.wandb_name}_{score_calculator}_{datamodule}'

    wandb.init(
        config=args,
        name=name,
        project=args.wandb_project,
        tags=args.wandb_tags,
    )
    wandb.config.method = str(score_calculator)
    wandb.config.dataset = str(datamodule)
     
    aurocs = []
    fprs = []
    fnrs = []
    acces = []
    known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
    unknown_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
    wandb.Table.MAX_ROWS = 200000

    if len(args.inner_seeds) > 0:
        seeds = args.inner_seeds
    else:
        seeds = [args.seed]

    iter_idx = 0
    scores_all = []
    base_scores_log_all = [] 
    mixdiff_scores_log_all = []
    targets_all = []
    labels_log_all = []
    probs_dicts_all = []
    
    if args.measure_time :
        latencies_per_sample = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_n_samples = 0

    for outer_iter, seed in enumerate(seeds):
        for inner_iter, (
            seen_labels, 
            seen_idx, 
            given_images,
            ref_images, 
            model_path,
            loader,
            few_shot_samples,
        ) in enumerate(
            datamodule.get_splits(
                n_samples_per_class=M, 
                seed=seed, 
                n_ref_samples=P if args.ref_mode in FIX_REF_MODES else 0,
                batch_size=batch_size,
                shuffle=True,
                transform=score_calculator.transform,
                n_few_shot_samples=args.k,
            )
        ):
            seen_idx = seen_idx.to(device) 

            wandb.log({
                'seen_classes': seen_labels,
                'seen_idx': seen_idx.tolist(),
            })
            
            start = args.r_ref
            end = 1.0 - start
            
            r_start = (end - start) / (R + 1)
            r_end = end - start - r_start

            rates = torch.linspace(r_start, r_end, R, device=device)
            rates += start

            print(rates)
            targets = []
            flipped_list = []
            scores = []
            mixdiff_scores_log = []
            base_scores_log = []
            labels_log = []
            probs_log = []

            mixup_fn.pre_transform = score_calculator.post_transform if args.trans_before_mixup else None
            mixup_fn.post_transform = score_calculator.post_transform if not args.trans_before_mixup else None

            score_calculator.on_eval_start(
                seen_labels=copy.deepcopy(seen_labels),
                given_images=copy.deepcopy(given_images),
                mixup_fn=mixup_fn,
                ref_images=copy.deepcopy(ref_images),
                rates=rates,
                seed=seed,
                iter_idx=iter_idx,
                model_path=model_path,
                few_shot_samples=few_shot_samples,
            )
            cur_num_samples = 0

            for i, (images, labels) in enumerate(tqdm(loader)):
                if args.measure_time:
                    starter.record()

                orig_n_samples = len(images)
                if len(images) != batch_size and score_calculator.utilize_mixup:
                    if torch.is_tensor(images):
                        dummy = images[:1].expand((batch_size - len(images)), -1, -1, -1)
                        images = torch.cat([images, dummy], dim=0)
                        dummy = torch.zeros(batch_size - len(labels), dtype=labels.dtype)
                        labels = torch.cat([labels, dummy], dim=0)
                    else:
                        dummy = [images[0]] * (batch_size - len(images))
                        images += dummy

                labels = labels.to(device)
                if torch.is_tensor(images):
                    images = images.to(device)
                if torch.is_tensor(given_images):
                    given_images = given_images.to(device)
                if torch.is_tensor(ref_images):
                    ref_images = ref_images.to(device)
                
                if id2ood_attack is not None:
                    id_mask = [label in seen_idx for label in labels]
                    if any(id_mask):
                        id_images = images[id_mask] 
                        with torch.enable_grad():
                            id_images = id2ood_attack.perturb(id_images)
                        images[id_mask] = id_images
                    
                if ood2id_attack is not None:
                    ood_mask = [label not in seen_idx for label in labels]
                    if any(ood_mask):
                        ood_images = images[ood_mask] 
                        with torch.enable_grad():
                            ood_images = ood2id_attack.perturb(ood_images)
                        images[ood_mask] = ood_images
                
                if label_flip is not None:
                    flipped_mask = [
                        label in seen_idx and (torch.rand(1) < args.label_flip_ratio).item() 
                        for label in labels
                    ]
                    if any(flipped_mask):
                        id_images = images[flipped_mask] 
                        seen2raw_label = {}
                        for i, idx in enumerate(seen_idx.tolist()):
                            seen2raw_label[idx] = i
                        raw_labels = []
                        for label, flipped in zip(labels.tolist(), flipped_mask):
                            if label in seen_idx and flipped:
                                raw_labels.append(seen2raw_label[label])
                        raw_labels = torch.tensor(raw_labels).to(device)
                            
                        with torch.enable_grad():
                            id_images = label_flip.perturb(id_images, raw_labels)
                        images[flipped_mask] = id_images

                image_kwargs = score_calculator.process_images(
                    score_calculator.post_transform(images)
                )

                if score_calculator.utilize_mixup:
                    chosen_images = score_calculator.select_given_images(
                        given_images, 
                        **image_kwargs
                    )
                    
                    if args.ref_mode == 'in_batch':
                        # O: (N, M), R: (N), T: (N)
                        # TM: (N, N, R)
                        known_mixup, unknown_mixup = mixup_fn(
                            oracle=chosen_images, 
                            references=images,
                            targets=images,
                            rates=rates,
                            seed=seed,
                        ) 
                    elif args.ref_mode == 'oracle':
                        # (N), (N, P) -> (N * P * R)
                        unknown_mixup = []
                        for image, chosen in zip(images, chosen_images):
                            # (1 * P * R)
                            unknown = mixup_fn(
                                references=chosen,
                                targets=image.unsqueeze(0) if torch.is_tensor(image) else [image],
                                rates=rates,
                                seed=seed,
                            )
                            if torch.is_tensor(unknown):
                                unknown_mixup.append(unknown)
                            else:
                                unknown_mixup += unknown
                        if torch.is_tensor(unknown_mixup[0]):
                            unknown_mixup = torch.cat(unknown_mixup, dim=0)
                        known_mixup = None
                    elif args.ref_mode in FIX_REF_MODES:
                        # (N), (P) -> (N, P, R)
                        unknown_mixup = []
                        for image in images:
                            # (1 * P * R)
                            unknown = mixup_fn(
                                references=ref_images,
                                targets=image.unsqueeze(0) if torch.is_tensor(image) else [image],
                                rates=rates,
                                seed=seed,
                            )
                            if torch.is_tensor(unknown):
                                unknown_mixup.append(unknown)
                            else:
                                unknown_mixup += unknown
                        if torch.is_tensor(unknown_mixup[0]):
                            unknown_mixup = torch.cat(unknown_mixup, dim=0)
                        known_mixup = None
                    
                    # if known_mixup is not None:
                    #     known_mixup = score_calculator.post_transform(known_mixup)
                    # if unknown_mixup is not None:
                    #     unknown_mixup = score_calculator.post_transform(unknown_mixup)

                    if args.log_interval is not None and i % args.log_interval == 0:
                        if args.ref_mode == 'oracle':
                            ref_images_log = chosen_images
                        elif args.ref_mode == 'in_batch':
                            ref_images_log = list(itertools.repeat(images, N))
                        elif args.ref_mode in FIX_REF_MODES:
                            ref_images_log = list(itertools.repeat(ref_images, P))

                        log_mixup_samples(
                            ref_images=ref_images_log,
                            known_mixup_table=known_mixup_table, 
                            unknown_mixup_table=unknown_mixup_table, 
                            known_mixup=known_mixup,
                            unknown_mixup=unknown_mixup, 
                            images=images, 
                            chosen_images=chosen_images, 
                            rates=rates, 
                            j=iter_idx,
                            N=N,
                            P=P,
                            R=R,
                            M=M,
                        )
                    
                    # (N * M * P * R * NC) -> (N, M, P, R, NC) -> (N, P, R, NC) -> (N * P * R * NC)
                    known_logits = score_calculator.process_mixed_oracle(
                        known_mixup, 
                        **image_kwargs
                    )
                    NC = known_logits.size(-1)
                    if args.ref_mode == 'in_batch' or args.ref_mode in FIX_REF_MODES:
                        known_logits = known_logits.view(N, M, P, R, -1)
                        known_logits = torch.mean(known_logits, dim=1).view(-1, NC)
                    elif args.ref_mode == 'oracle':
                        known_logits = known_logits.view(N, M, M, R, -1)
                        mask = torch.ones(
                            (M, M), 
                            device=known_logits.device
                        )
                        mask.fill_diagonal_(0)
                        mask = mask[None, :, :, None, None]
                        known_logits = known_logits * mask
                        known_logits = torch.sum(known_logits, dim=1)
                        knwon_logits = known_logits / (M - 1)
                        known_logits = knwon_logits.view(-1, NC)
                        
                    # (N * P * R) -> (N * P * R * NC)
                    unknown_logits = score_calculator.process_mixed_target(
                        unknown_mixup,
                        **image_kwargs
                    )
                    # (N * P * R * NC) -> (N * P * R)
                    dists = score_calculator.calculate_diff(known_logits, unknown_logits)

                    if args.abs:
                        dists = torch.abs(dists)

                    if args.ref_mode == 'in_batch':
                        dists = dists.view(N, N, R)
                        dists = torch.mean(dists, dim=-1)
                        mask = torch.ones_like(dists)
                        mask.fill_diagonal_(0.0)
                        dists = dists * mask
                    elif args.ref_mode == 'oracle':
                        dists = dists.view(N, M, R)
                        dists = torch.mean(dists, dim=-1)
                    elif args.ref_mode in FIX_REF_MODES:
                        dists = dists.view(N, P, R)
                        dists = torch.mean(dists, dim=-1)

                    if args.top_1:
                        abs_dists = torch.abs(dists)
                        abs_max_idx = torch.argmax(abs_dists, dim=-1)
                        dists = dists[torch.arange(dists.size(0)), abs_max_idx]
                    else:
                        if args.ref_mode == 'in_batch':
                            dists = torch.sum(dists, dim=-1) / torch.sum(mask > 0.5, dim=-1)
                        elif args.ref_mode == 'oracle' or args.ref_mode in FIX_REF_MODES:
                            dists = torch.mean(dists, dim=-1)
                
                base_scores = score_calculator.calculate_base_scores(**image_kwargs)

                if score_calculator.utilize_mixup:
                    mixdiff_scale = args.gamma
                    if 'scale' in image_kwargs and args.scale_mixdiff:
                        mixdiff_scale = image_kwargs['scale'] * mixdiff_scale

                    mixdiff_scores_log += (mixdiff_scale * dists).tolist()[:orig_n_samples]
                    base_scores_log += base_scores.tolist()[:orig_n_samples]
                    dists = base_scores + mixdiff_scale * dists
                else:
                    base_scores_log += base_scores.tolist()[:orig_n_samples]
                    dists = base_scores
                
                dists = dists[:orig_n_samples]
                
                targets += [int(label not in seen_idx) for label in labels][:orig_n_samples]
                flipped_list += [int(flipped) for flipped in flipped_mask][:orig_n_samples]
                scores += dists.tolist()
                labels_log += labels.tolist()

                cur_num_samples += N

                if args.measure_time:
                    ender.record()
                    torch.cuda.synchronize()
                    latency = starter.elapsed_time(ender)
                    latencies_per_sample.append(latency)
                    total_n_samples += len(images)

                if score_calculator.intermediate_state == 'logit':
                    probs_log += torch.softmax(image_kwargs['logits'], dim=-1).tolist()
                if score_calculator.intermediate_state == 'softmax':
                    probs_log += image_kwargs['logits'].tolist()

                if args.max_samples is not None and cur_num_samples >= args.max_samples:
                    break
                
            score_calculator.on_eval_end(iter_idx=iter_idx)

            if not score_calculator.utilize_mixup:
                mixdiff_scores_log = itertools.repeat(0.0, len(scores))
            
            probs_dicts = []
            for probs in probs_log:
                probs_dict = {}
                for label, prob in zip(seen_idx, probs):
                    probs_dict[str(label.item())] = prob
                probs_dicts.append(probs_dict)

            if label_flip is not None:
                def exclude_ood(items, targets):
                    return [item for item, target in zip(items, targets) if target == 0]

                scores = exclude_ood(scores, targets)
                base_scores_log = exclude_ood(base_scores_log, targets)
                mixdiff_scores_log = exclude_ood(mixdiff_scores_log, targets)
                labels_log = exclude_ood(labels_log, targets)
                probs_dicts = exclude_ood(probs_dicts, targets)
                flipped_list = exclude_ood(flipped_list, targets)

            if datamodule.flatten:
                scores_all += scores
                base_scores_log_all += base_scores_log_all
                mixdiff_scores_log_all += mixdiff_scores_log
                targets_all += targets
                labels_log_all += labels_log
                probs_dicts_all += probs_dicts
            else:
                auroc, fpr_at, fnr_at, acc = calculate_metrics(
                    args=args,
                    scores=scores,
                    base_scores_log=base_scores_log,
                    mixdiff_scores_log=mixdiff_scores_log,
                    targets=targets if label_flip is None else flipped_list,
                    iter_idx=iter_idx,
                    labels=labels_log,
                    pred_labels=probs_dicts,
                    ignore_tgt_for_acc=label_flip is not None,
                )

                aurocs.append(auroc)
                fprs.append(fpr_at)
                fnrs.append(fnr_at)
                acces.append(acc)

                wandb.log({
                    'auroc': auroc,
                    f'fpr{args.fpr_at}': fpr_at,
                    f'fnr{args.fnr_at}': fnr_at,
                    'acc': acc,
                })

            iter_idx += 1

    if datamodule.flatten:
        auroc, fpr_at, fnr_at = calculate_metrics(
            args,
            scores_all,
            base_scores_log_all,
            mixdiff_scores_log_all,
            targets_all,
            0,
            labels_log_all,
            probs_dicts_all,
            exclude_ood=label_flip is not None,
        )
        aurocs.append(auroc)
        fprs.append(fpr_at)
        fnrs.append(fnr_at)
        acces.append(acc)
    
    if args.log_interval is not None:
        if args.ref_mode == 'in_batch':
            wandb.log({
                'oracle_mixup': known_mixup_table,
                'target_mixup': unknown_mixup_table,
            })
        else:
            wandb.log({
                'target_mixup': unknown_mixup_table,
            })

    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)

    if args.measure_time:
        latency_per_sample = np.mean(latencies_per_sample)
        latency_per_sample_std = np.std(latencies_per_sample)

    wandb.log({
        'avg_auroc': avg_auroc, 
        'auroc_std': auroc_std,
        f'avg_fpr{round(args.fpr_at * 100)}': np.mean(fprs), 
        f'fpr{round(args.fpr_at * 100)}_std': np.std(fprs),
        f'avg_fnr{round(args.fnr_at * 100)}': np.mean(fnrs), 
        f'fnr{round(args.fnr_at * 100)}_std': np.std(fnrs),
        'avg_acc': np.mean(acces),
        'acc_std': np.std(acces)
    })


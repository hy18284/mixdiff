import itertools

import numpy as np
import torch
import wandb


def log_mixup_samples(
    ref_images,
    rates, 
    j,
    N,
    P,
    R,
    M=None,
    known_mixup_table=None, 
    unknown_mixup_table=None, 
    known_mixup=None, 
    unknown_mixup=None, 
    images=None, 
    chosen_images=None, 
):
    ref_idx = np.random.randint(0, P)

    if torch.is_tensor(images):
        images = convert_to_wandb_images(images)
    if torch.is_tensor(ref_images[0]):
        ref_images = [
            convert_to_wandb_images(ref)
            for ref in ref_images
        ]
    if torch.is_tensor(known_mixup):
        known_mixup = convert_to_wandb_images(known_mixup)
    if torch.is_tensor(unknown_mixup):
        unknown_mixup = convert_to_wandb_images(unknown_mixup)
    if torch.is_tensor(chosen_images):
        chosen_images = [
            convert_to_wandb_images(chosen)
            for chosen in chosen_images
        ]
    
    if known_mixup is not None:
        known_idx = np.random.randint(0, M)
        # (N * M * P * R)
        # [0, KI, 1, :]

        known_mixup_arr = np.empty(len(known_mixup), dtype='object')
        known_mixup_arr[:] = known_mixup
        known_mixup_arr = known_mixup_arr.reshape(N, M, P, R)

        known_rows = zip(
            itertools.repeat(chosen_images[0][known_idx], R),
            itertools.repeat(ref_images[0][ref_idx], R),
            known_mixup_arr[0, known_idx, ref_idx, :],
            rates.tolist(),
            itertools.repeat(j, R),
        )

        for row in known_rows:
            known_mixup_table.add_data(*row)

    if unknown_mixup is not None:
        # (N * P * R)
        # [0, RI, :]

        unknown_mixup_arr = np.empty(len(unknown_mixup), dtype='object')
        unknown_mixup_arr[:] = unknown_mixup
        unknown_mixup_arr = unknown_mixup_arr.reshape(N, P, R)

        unknown_rows = zip(
            itertools.repeat(images[0], R),
            itertools.repeat(ref_images[0][ref_idx], R),
            unknown_mixup_arr[0, ref_idx, :],
            rates.tolist(),
            itertools.repeat(j, R),
        )

        for row in unknown_rows:
            unknown_mixup_table.add_data(*row)


def convert_to_wandb_images(images):
    images = images.to('cpu')
    images = [
        wandb.Image(image)
        for image in images
    ]
    return images


def calculate_fnr_at(scores, targets, tpr_percent):
    scores_ood = [score for score, target in zip(scores, targets) if target == 1]
    scores_id = [score for score, target in zip(scores, targets) if target == 0]

    scores_id.sort(reverse=False)
    tpr_idx = round(len(scores_id) * tpr_percent)
    threshold = scores_id[tpr_idx - 1]

    scores_ood = np.array(scores_ood)
    false_negatives = scores_ood > threshold
    fnr = np.sum(false_negatives) / len(scores_ood)

    return fnr


def calculate_fpr_at(scores, targets, tnr_percent):
    scores_ood = [score for score, target in zip(scores, targets) if target == 1]
    scores_id = [score for score, target in zip(scores, targets) if target == 0]

    scores_ood.sort(reverse=True)
    tnr_idx = round(len(scores_ood) * tnr_percent)
    threshold = scores_ood[tnr_idx - 1]

    scores_id = np.array(scores_id)
    false_positives = scores_id > threshold
    fpr = np.sum(false_positives) / len(scores_id)

    return fpr

def calculate_ood_metrics(known, novel):
    results = cal_metric(known, novel)
    return results

def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

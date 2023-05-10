import itertools

import numpy as np


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

        # ref_arr = np.empty(len(ref_images), dtype='object')
        # ref_arr[:] = ref_images
        # ref_arr = ref_arr.reshape(N, P)
        # from pprint import pprint
        # pprint('ref')
        # pprint(ref_images)
        # pprint('unk')
        # pprint(unknown_mixup_arr)
        # pprint('####')

        unknown_rows = zip(
            itertools.repeat(images[0], R),
            itertools.repeat(ref_images[0][ref_idx], R),
            unknown_mixup_arr[0, ref_idx, :],
            rates.tolist(),
            itertools.repeat(j, R),
        )

        for row in unknown_rows:
            unknown_mixup_table.add_data(*row)

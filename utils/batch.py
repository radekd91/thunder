import torch
import numpy as np
import copy

def expand_batch(batch, num_variations):
    """
    Expands the batch along the batch dimension to have num_variations
    """
    # for key in batch.keys():
    if isinstance(batch, torch.Tensor):
        shape = list(batch.shape)
        batch = batch.expand(num_variations, *shape[1:]).contiguous()
    elif isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = expand_batch(batch[k], num_variations) 
    elif isinstance(batch, list):
        if isinstance(batch[0], str):
            batch = batch * num_variations
        else:
            batch = [expand_batch(b, num_variations) for b in batch]
    return batch


### copied from infernos MotionPrior.py
def temporal_trim_dict(sample, start_frame, end_frame, is_batched=False, keys_to_ignore=None):
    for key in sample.keys():
        if keys_to_ignore is not None and key in keys_to_ignore:
            continue
        if isinstance(sample[key], (np.ndarray, torch.Tensor)):
            # if key in ["gt_expression", "gt_exp", "gt_vertices"]:
            #     print(key)
            if len(sample[key].shape) >= 2 + int(is_batched):
                if is_batched:
                    sample[key] = sample[key][:, start_frame:end_frame]
                else:
                    sample[key] = sample[key][start_frame:end_frame]
        # elif isinstance(sample[key], str):
        #     pass
        elif isinstance(sample[key], list):
            if not isinstance(sample[key][0], (str)):
                if is_batched:
                    sample[key] = [s[:, start_frame:end_frame] for s in sample[key]]
                else:
                    sample[key] = sample[key][start_frame:end_frame]
        elif isinstance(sample[key], dict):
            sample[key] = temporal_trim_dict(sample[key], start_frame, end_frame, is_batched=is_batched, keys_to_ignore=keys_to_ignore)
        else:
            raise NotImplementedError(f"Type {type(sample[key])} not implemented")
    return sample


def temporal_split_dict(sample, max_length, num_frames=None,  is_batched=False, keys_to_ignore=None): 
    if num_frames is None:
        if is_batched:
            num_frames = sample["video"].shape[1]
        else:
            num_frames = sample["video"].shape[0]

    if num_frames < max_length:
        return [sample]

    num_splits = num_frames // max_length
    if num_frames % max_length != 0:
        num_splits += 1

    samples = []
    for i in range(num_splits):
        start_frame = i * max_length
        end_frame = min((i + 1) * max_length, num_frames)
        new_sample = temporal_trim_dict(copy.deepcopy(sample), start_frame, end_frame, is_batched=is_batched, keys_to_ignore=keys_to_ignore)
        samples.append(new_sample)
    return samples
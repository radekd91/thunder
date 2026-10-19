import pickle as pkl 
from inferno.datasets.IO import _save_hickle_file, _load_hickle_file
import h5py
import numpy as np

def save_transcript(transcript_folder, transcript, segment_dict, word_dict, language=None):
    transcript_file = transcript_folder / "transcript.txt"
    with open(transcript_file, "w") as f:
        f.write(transcript)

    if segment_dict is not None: 
        segment_dict_file = transcript_folder / "segment.pkl"
        word_dict_file = transcript_folder / "word.pkl"

        # import compress_pickle as cpkl
        with open(segment_dict_file, "wb") as f:
            # cpkl.dump(segment_dict, f)
            pkl.dump(segment_dict, f)
        
        with open(word_dict_file, "wb") as f:
            # cpkl.dump(word_dict, f)
            pkl.dump(word_dict, f)

    if language is not None:
        language_file = transcript_folder / "language.txt"
        with open(language_file, "w") as f:
            f.write(language)
    


def load_transcript(transcript_folder):
    transcript_file = transcript_folder / "transcript.txt"
    if not transcript_file.exists():
        return None, None, None, None
    with open(transcript_file, "r") as f:
        transcript = f.read()

    segment_dict_file = transcript_folder / "segment.pkl"
    word_dict_file = transcript_folder / "word.pkl"

    with open(segment_dict_file, "rb") as f:
        segment_dict = pkl.load(f)

    with open(word_dict_file, "rb") as f:
        word_dict = pkl.load(f)

    language_file = transcript_folder / "language.txt"
    if language_file.exists():
        with open(language_file, "r") as f:
            language = f.read()
    else: 
        language = None

    return transcript, segment_dict, word_dict, language


def save_speaker_embedding(filename, embedding):
    save_np_file(embedding, filename)

def load_speaker_embedding(filename):
    return load_np_file(filename)

def save_np_file(data, filename):
    np.save(filename, data)

def load_np_file(filename):
    return np.load(filename)


def save_embedding(filename, embedding): 
    _save_hickle_file(embedding, filename)


def save_unsynced_langauge_embedding(filename, embedding, non_array_data_keys=None):
    non_array_data_keys = non_array_data_keys or []
    ## split the embedding dict into array and non-array data
    array_data = {}
    non_array_data = {}
    for key, value in embedding.items():
        if key in non_array_data_keys:
            non_array_data[key] = value
        else:
            array_data[key] = value

    ## save array data with hickle
    _save_hickle_file(array_data, filename)
    ## save non-array data with pickle
    filename = filename.with_suffix('.pkl')
    with open(filename, 'wb') as f:
        pkl.dump(non_array_data, f)


def load_embedding(filename, start_frame=None, end_frame=None):
    return _load_hickle_file(filename, start_frame, end_frame)

def load_tensor_embedding(filename, start_frame=None, end_frame=None, t_dim=0):
    # data_dict = {}
    value = None
    with h5py.File(filename, 'r') as f:
        dset = f['data']
        assert isinstance(dset, h5py.Dataset)
        # for key in f.keys():
            # dset = f[key]
            # if not isinstance(dset, h5py.Dataset):
            #     continue
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            if t_dim == 0:
                end_frame = dset.shape[0]
            elif t_dim == 1:
                end_frame = dset.shape[1]
            else:
                raise NotImplementedError(f"t_dim {t_dim} not implemented")
            # for some reason we saved the data with a leading batch dimension of size 1, so let's just run with it
        if t_dim == 0:
            value = dset[start_frame:end_frame]
        elif t_dim == 1:
            value = dset[:, start_frame:end_frame]
        else: 
            raise NotImplementedError(f"t_dim {t_dim} not implemented")
    return value

def load_language_embedding(filename, start_frame=None, end_frame=None, keys_to_read=None, nested_keys_to_read=None):
    return _load_language_hickle_file(filename, keys_to_read, nested_keys_to_read)


def load_unsynced_language_embedding(filename, keys_to_read=None, nested_keys_to_read=None):
    array_data = _load_unsynced_language_hickle_file(filename, keys_to_read, nested_keys_to_read) 
    filename = filename.with_suffix('.pkl')
    with open(filename, 'rb') as f:
        non_array_data = pkl.load(f)
    array_data.update(non_array_data)
    return array_data


def _load_language_hickle_file(filename, start_frame=None, end_frame=None, keys_to_read=None, nested_keys_to_read=None):
    result_dict = {}
    keys_to_read = keys_to_read or []
    nested_keys_to_read = nested_keys_to_read or "all"
    ## these keys are synchronzied with the video frames
    keys_with_synced_frames = ['transcript_word_embeddings_time_synced', 'silent_frames']
    with h5py.File(filename, 'r') as f:
        hdf5_group = f['data'] 
        for key in hdf5_group.keys():
            key_ = key.replace('"','')
            if key_ not in keys_to_read:
                continue
            data = hdf5_group[key]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = data.shape[0]
            if key_ in keys_with_synced_frames:
                ## for things synced with the video frames, we need to slice the data
                if isinstance(data, h5py.Group):
                    result_dict[key_] = {}
                    for nested_key in data.keys():
                        nested_key_ = nested_key.replace('"','')
                        if nested_keys_to_read == "all": 
                            result_dict[key_][nested_key_] = data[nested_key][0, start_frame:end_frame]
                        elif nested_key_ in nested_keys_to_read:
                            result_dict[key_][nested_key_] = data[nested_key][0, start_frame:end_frame]
                    if len(result_dict[key_]) == 1:
                        result_dict[key_] = result_dict[key_][list(result_dict[key_].keys())[0]]

                else:
                    result_dict[key_] = data[start_frame:end_frame]
                # result_dict[key_] = data[:, start_frame:end_frame]
            else:
                ## for other things, we don't need to slice the data, return everything instead
                result_dict[key_] = data[...]
    return result_dict



def _load_unsynced_language_hickle_file(filename, keys_to_read=None, nested_keys_to_read=None):
    result_dict = {}
    keys_to_read = keys_to_read or []
    nested_keys_to_read = nested_keys_to_read or "all"
    ## these keys are synchronzied with the video frames
    # keys_with_synced_frames = ['transcript_word_embeddings_time_synced', 'silent_frames']
    with h5py.File(filename, 'r') as f:
        hdf5_group = f['data'] 
        for key in hdf5_group.keys():
            key_ = key.replace('"','')
            # if key_ not in keys_to_read:
            #     continue
            data = hdf5_group[key]
            # if key_ in keys_with_synced_frames:
                ## for things synced with the video frames, we need to slice the data
            if isinstance(data, h5py.Group):
                result_dict[key_] = {}
                for nested_key in data.keys():
                    nested_key_ = nested_key.replace('"','')
                    if nested_keys_to_read == "all": 
                        result_dict[key_][nested_key_] = data[nested_key][...]
                    elif nested_key_ in nested_keys_to_read:
                        result_dict[key_][nested_key_] = data[nested_key][...]
                # if len(result_dict[key_]) == 1:
                #     result_dict[key_] = data[list(result_dict[key_].keys())[0]][...]
                # else:
                #     result_dict[key_] = data
                # result_dict[key_] = data[:, start_frame:end_frame]
            else:
                ## for other things, we don't need to slice the data, return everything instead
                result_dict[key_] = data[...]
    return result_dict

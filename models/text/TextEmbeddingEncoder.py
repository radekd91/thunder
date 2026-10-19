from .TextBase import TextEncoderBase
import torch
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer
from inferno.utils.other import class_from_str
import timeit
import os
import string

class TextEmbeddingEncoder(TextEncoderBase): 

    def __init__(self, model_specifier, 
                 trainable, 
                 dropout_cfg=None, 
                 padding=True):
        super().__init__()
        self.trainable = trainable
        self.model_specifier = model_specifier
        access_token = None 
        if 'HF_TOKEN' in os.environ:
            access_token = os.environ['HF_TOKEN']
        self.model_cfg  =  AutoConfig.from_pretrained(model_specifier, token=access_token)
        self.model = AutoModel.from_pretrained(model_specifier, config=self.model_cfg, token=access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_specifier, token=access_token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_index_key = "all_words_frame_indices"
        self.input_text_key = "all_words_transcript"
        self.padding = padding
        if dropout_cfg is not None:
            dropout_type = dropout_cfg.pop("type") 
            assert dropout_type is not None, "audio_dropout_cfg must have a 'type' key"
            self.dropout = class_from_str(dropout_type)(**dropout_cfg)

        if not self.trainable:
            self.model.eval()
            self.model.requires_grad_(False)
    
    def freeze(self, frozen):
        self.model.train(not frozen)
        self.model.requires_grad_(not frozen)
    
    def output_feature_dim(self):
        return self.model_cfg.hidden_size

    def get_trainable_parameters(self): 
        if self.trainable:
            return [p for p in self.model.parameters() if p.requires_grad]
        return []

    def to(self, device):
        super().to(device)
        self.model.to(device)
        try:
            self.tokenizer.to(device)
        except AttributeError:
            pass
        return self

    def train(self, mode=True):
        if not self.trainable:
            return super().train(False)
        return super().train(mode)

    def forward(self, sample, train=False, desired_output_length=None, **kwargs):
        if self.trainable:
            return self.encode(sample)
        
        with torch.no_grad():
            return self.encode(sample)

    def _word_2_tokens_dict(self, input_ids): 
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        words = self.tokenizer.convert_tokens_to_string(tokens).split()

        # Create a list of dictionaries to store words and their subwords
        word_to_subwords_list = []

        # Define punctuation set
        punctuation = set(string.punctuation)

        # Initialize variables to store the current word and its subwords
        current_word = ''
        subwords = []
        token_indices = []

        # Iterate through tokens and map subwords to words
        for tok_idx, token in enumerate(tokens):
            # Check if token is punctuation or a word separator
            if token.startswith('▁') or token in punctuation:  
                if current_word:  # Add the previous word and its subwords to the list
                    word_to_subwords_list.append({
                        'word': current_word, 
                        'subwords': subwords,
                        'token_indices': token_indices,
                        })
                current_word = token.lstrip('▁')  # Remove the "▁" prefix
                subwords = [current_word]
                token_indices = [tok_idx]
            else:  # Continuation of the previous word
                current_word += token
                subwords.append(token)
                token_indices.append(tok_idx)

        # Add the last word and its subwords to the list
        if current_word:
            word_to_subwords_list.append(
                {'word': current_word, 
                 'subwords': subwords,
                 'token_indices': token_indices,
                 })

        # word_to_subwords_list_without_special_tokens = [word_dict for word_dict in word_to_subwords_list if word_dict['word'] not in self.tokenizer.all_special_tokens]
        # word_to_subwords_list_without_special_tokens = [word_dict for word_dict in word_to_subwords_list_without_special_tokens if word_dict['word'] not in punctuation]

        return word_to_subwords_list


    def encode(self, sample, desired_output_length=None, **kwargs):
        transcript_methods = list(sample["transcript"].keys())
        assert len(transcript_methods) == 1, "More transcript mehotds not supported"
        transcript_method = transcript_methods[0]
        transcript_sample = sample["transcript"][transcript_method]

        batched_words = transcript_sample[self.input_text_key]
        batched_indices = transcript_sample[self.input_index_key]

        batch_transcript = [" ".join(words[1:]) for words in batched_words] # [1:] because we skip the silence token

        B = len(batch_transcript)
        
        # start_time = timeit.default_timer()
        inputs = self.tokenizer.batch_encode_plus(
            batch_transcript,
            padding=self.padding,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        device = batched_indices.device
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device) 
        # end_time = timeit.default_timer()
        # print(f"Tokenization time: {end_time - start_time:0.3f}")

        offset_mapping = inputs.pop('offset_mapping')
        outputs = self.model(**inputs)

        # Get the token embeddings
        num_input_tokens = inputs.input_ids.shape[1]
        num_output_tokens = outputs.last_hidden_state.shape[1]

        # if num_input_tokens + 1 == num_output_tokens: 
        #     ## probably a BERT model, where the first token is a global token 
        #     global_embeddings = outputs.last_hidden_state[:,0]
        #     token_embeddings = outputs.last_hidden_state[:,1:]
        # elif num_input_tokens == num_output_tokens:
        #     ## other models
        #     global_embeddings = None 
        #     token_embeddings = outputs.last_hidden_state
        # else:
        #     raise ValueError(f"Unexpected number of tokens: {num_input_tokens} -> {num_output_tokens}. Probably an extra care has to be applied when using this model {self.model_specifier}")


        # Aggregate embeddings for each word in each sequence in the batch
        word_to_subwords_list_batch = []
        # batch_word_embeddings = []
        for batch_idx, input_words in enumerate(batched_words):

            word_to_subwords_list = self._word_2_tokens_dict(inputs['input_ids'][batch_idx].tolist())
            word_to_subwords_list_batch += [word_to_subwords_list]

            # input_words = input_words[1:]  # Skip the silence token
            # # Compute cumulative lengths for the current sequence
            # # cumulative_lengths = np.cumsum([0] + [len(word) + 1 for word in input_words])  # +1 for space
            # # cumulative_lengths = torch.cumsum(torch.tensor([0] + [len(word) + 1 for word in input_words], device=device), dim=0)  # +1 for space ... later edit: what space?!
            # cumulative_lengths = torch.cumsum(torch.tensor([0] + [len(word) for word in input_words], device=device), dim=0)  # no space
            # # cumulative_lengths = torch.cumsum(torch.tensor([len(word) + 1 for word in input_words], device=device), dim=0)  # +1 for space

            # tokens = inputs.tokens(batch_idx)
            # token2interval = dict(zip(tokens, offset_mapping[batch_idx]))
            # token_attributes = self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES
            # if 'additional_special_tokens' in token_attributes:
            #     token_attributes.remove('additional_special_tokens')
            # tokens_to_skip = [getattr(self.tokenizer, special_token) for special_token in token_attributes]
            # if hasattr(self.tokenizer, 'additional_special_tokens'):
            #     tokens_to_skip += self.tokenizer.additional_special_tokens
            # # tokens_to_skip = ['[CLS]', '[SEP]', '[PAD]']  # Skip [CLS] and [SEP] tokens
            # # Create a NumPy array of start indices
            # # start_at = 1 # to skip the [CLS] token
            # # end_at = -1 # why was this like this?
            # start_at = 0 # actually the token will be skipped anyway
            # end_at = len(offset_mapping[batch_idx]) # actually, lets end at thend
            # # start_indices = np.array(
            # # start_indices = torch.tensor(
            # #     [offset[0] for offset_idx, offset in enumerate(offset_mapping[batch_idx][start_at:-1]) \
            # #                         if tokens[offset_idx+start_at] not in tokens_to_skip ], device=device)  
            # start_indices = torch.tensor(
            #     [offset[0] for offset_idx, offset in enumerate(offset_mapping[batch_idx][start_at:end_at]) \
            #                         # if tokens[offset_idx+start_at] not in tokens_to_skip 
            #                         ], device=device)  

            # # # tokens_ = np.array(
            # tokens_ = [tokens[offset_idx+start_at] for offset_idx, offset in enumerate(offset_mapping[batch_idx][start_at:end_at]) \
            #                     # if tokens[offset_idx+start_at] not in tokens_to_skip
            #                       ]
            # # tokens = torch.tensor(tokens_, device=device)
            # # tokens_ = torch.tensor(
            # #     [tokens[offset_idx+start_at] for offset_idx, offset in enumerate(offset_mapping[batch_idx][start_at:-1]) \
            # #                     if tokens[offset_idx+start_at] not in tokens_to_skip ], device=device)  

            # # Find word index for each token using NumPy
            # # word_indices = np.searchsorted(cumulative_lengths, start_indices, side='right') - 1
            
            # word2subword = dict(zip(input_words, [[] for i in range(len(input_words))]))
            # for token_idx in range(len(tokens_)):
            #     word_idx = torch.searchsorted(cumulative_lengths, start_indices[token_idx], side='right') - 1
            #     if tokens_[token_idx] not in tokens_to_skip:
            #         word2subword[input_words[word_idx]] += [tokens_[token_idx]]
            
            # word_indices = torch.searchsorted(cumulative_lengths, start_indices, side='right') - 1

            # wordlist = [batch_transcript[batch_idx][offset_mapping[batch_idx,off_idx,0]:offset_mapping[batch_idx,off_idx,1]] for off_idx in range(len(offset_mapping[batch_idx]))]

            # # Aggregate embeddings for each word in the sequence
            # tokens2start_indices = dict(zip(tokens_, start_indices))
            # word_embeddings = [[] for i in range(len(input_words))]
            # word2subword = dict(zip(input_words, [[] for i in range(len(input_words))]))
            # for token_idx in range(len(word_indices)):
            #     word_idx = word_indices[token_idx]
            #     # word_embeddings[word_idx] += [token_embeddings[batch_idx, token_idx + start_at]]
            #     if tokens_[token_idx] not in tokens_to_skip:
            #         word2subword[input_words[word_idx]] += [tokens_[token_idx]]
            #     # word_embedding = np.mean(token_embeddings[batch_idx, tokens2start_indices[input_words[ti]], :], axis=0)
            #     # batch_word_embeddings.append(word_embedding)

            # # # Compute word-level embeddings for each word in the sequence
            # # word_embeddings = [torch.mean(torch.stack(word_embeddings[word_idx]), dim=0) if word_embeddings[word_idx] else torch.zeros(token_embeddings.size(-1)) for word_idx in range(len(input_words))]

            # # word_embeddings_oneliner = [torch.mean(torch.stack([token_embeddings[batch_idx, token_idx + 1] for token_idx in np.where(word_indices == word_idx)[0]]), dim=0) \
            # #                 if np.any(word_indices == word_idx) else torch.zeros(token_embeddings.size(-1)) for word_idx in range(len(input_words))]

            # word_embeddings_oneliner = [torch.mean(torch.stack([token_embeddings[batch_idx, token_idx + 1] for token_idx in torch.where(word_indices == word_idx)[0]]), dim=0) \
            #                 if torch.any(word_indices == word_idx) else torch.zeros(token_embeddings.size(-1), device=device) for word_idx in range(len(input_words))]

            # # word_embeddings = torch.stack(word_embeddings, dim=0)
            # if len(word_embeddings_oneliner) == 0:
            #     ## corner case for when there's only silence in the entire batch (zero words across all samples)
            #     word_embeddings_oneliner = torch.zeros(0, token_embeddings.size(-1), device=device)
            # else:
            #     word_embeddings_oneliner = torch.stack(word_embeddings_oneliner, dim=0)

            # batch_word_embeddings += [word_embeddings_oneliner]


        # batch_word_embeddings = torch.stack(batch_word_embeddings, dim=0)

        # sample["transcript_word_embeddings"] = batch_word_embeddings
        # sample["transcript_global_embeddings"] = global_embeddings

        sample["transcript_embeddings"] = outputs.last_hidden_state
        sample["word_mappings"] = word_to_subwords_list_batch
        return sample

    def has_synchronized_output(self):
        return False


class FrameSyncedTextEmbeddingEncoder(TextEmbeddingEncoder): 

    def __init__(self, silence_token_init, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        self.silence_token_init = silence_token_init
        ## add learnable tensor for silent embedding 
        silent_embedding = torch.randn(1, self.model_cfg.hidden_size)
        self.silence_embedding = torch.nn.Parameter(silent_embedding)
        if self.silence_token_init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.silence_embedding)
        elif self.silence_token_init in ["zeros", "zero"]:
            torch.nn.init.zeros_(self.silence_embedding)
        else: 
            raise NotImplementedError(f"Silence token init {self.silence_token_init} not supported")
        
    def get_trainable_parameters(self): 
        return super().get_trainable_parameters() + [self.silence_embedding]

    def to(self, device):
        super().to(device)
        self.silence_embedding.to(device)
        return self


    def _process_word_mappings(self, word_mappings): 
        B = len(word_mappings)
        ## removes special tokens, interpunction tokens and merges words with apostrophes
        word_mappings_processed_batch = []
        for bi, word_mapping in enumerate(word_mappings):
            word_mapping_processed = []
            indices_to_keep = []
            for wi, word_dict in enumerate(word_mapping):
                word = word_dict['word']
                if word in self.tokenizer.all_special_tokens:
                    continue
                if word in string.punctuation:
                    continue

                indices_to_keep += [wi]

            indices_to_merge = []
            current_merging_list = []
            for wi in indices_to_keep:
                word_mapping_processed += [word_mapping[wi]]
            
            for wi in range(len(word_mapping_processed)):
                word = word_mapping_processed[wi]['word']
                if word.startswith("'") and wi > 0:
                    current_merging_list += [wi-1, wi]
                    continue
                else:
                    if current_merging_list:
                        indices_to_merge += [sorted(list(set(current_merging_list)))]
                        current_merging_list = []
                
            ## merge words with apostrophes
            merged_word_mappings = []
            for merge_list in indices_to_merge:
                merged_word_mapping = {
                    'word': ''.join([word_mapping_processed[wi]['word'] for wi in merge_list]),
                    'subwords': sum([word_mapping_processed[wi]['subwords'] for wi in merge_list], []),
                    'token_indices': sum([word_mapping_processed[wi]['token_indices'] for wi in merge_list], []),
                }
                merged_word_mappings += [merged_word_mapping]

            ## replace the pairs to be merged with the merged ones
            for w_pair in reversed(indices_to_merge):
                for wi in w_pair[-1:0:-1]:
                    # print("Deleting", wi, word_mapping_processed[wi])
                    del word_mapping_processed[wi]

                word_mapping_processed[w_pair[0]] = merged_word_mappings[-1]
                del merged_word_mappings[-1]

            assert len(merged_word_mappings) == 0, "Merging failed"

            word_mappings_processed_batch += [word_mapping_processed]
        return word_mappings_processed_batch
    

    def _compute_word_features(self, word_mappings, word_embedings): 
        word_embedding_tensors = []
        for bi, word_mapping in enumerate(word_mappings):
            word_embedding_tensor = []
            for wi, word_dict in enumerate(word_mapping):
                # word = word_dict['word']
                # subwords = word_dict['subwords']
                token_indices = word_dict['token_indices']
                word_embedding = torch.mean(word_embedings[bi, token_indices], dim=0)
                word_dict['embedding'] = word_embedding
                word_embedding_tensor += [word_embedding]
            word_embedding_tensor = torch.stack(word_embedding_tensor, dim=0)
            word_embedding_tensors += [word_embedding_tensor]
        return word_embedding_tensors
        


    def encode(self, sample, desired_output_length=None, **kwargs):
        sample = super().encode(sample, desired_output_length=desired_output_length, **kwargs)

        transcript_methods = list(sample["transcript"].keys())
        assert len(transcript_methods) == 1, "More transcript mehotds not supported"
        transcript_method = transcript_methods[0]
        transcript_sample = sample["transcript"][transcript_method]

        batch_word_embeddings = sample["transcript_embeddings"]
        word_mappings = sample["word_mappings"]

        word_mappings_processed = self._process_word_mappings(word_mappings)

        # batch_word_embeddings = sample["transcript_word_embeddings"] 
        ## zero-th index embedding corresponds to silence in the audio so prepend the silent learned feature
        W = batch_word_embeddings.size(1) ## num words
        B = batch_word_embeddings.size(0) ## bach size 
        T = sample["frame_indices"].shape[1] ## num frames
        D = batch_word_embeddings.size(2) ## word embedding vector dim
        silent_embedding = self.silence_embedding.unsqueeze(0).repeat(B, 1, 1)
        # batch_word_embeddings_with_silence = torch.cat([silent_embedding, batch_word_embeddings], dim=1)
        

        script_1 = [word_info['word'] for word_info in word_mappings_processed[0]]
        script_2 = transcript_sample['all_words_transcript'][0][1:]

        assert len(script_1) == len(script_2), "Word mappings and transcript words are not aligned"
        

        for w1, w2 in zip(script_1, script_2):
            ## remove all punctuation from w1 and w2 
            for p in string.punctuation:
                w1 = w1.replace(p, "")
                w2 = w2.replace(p, "")

            assert w1 == w2, f"Word mappings and transcript words are not aligned: {w1} != {w2}"

        word_embedding_tensors = self._compute_word_features(word_mappings_processed, batch_word_embeddings)
        batch_word_embeddings_with_silence = []
        present_indices = transcript_sample["present_words_frame_indices"]
        word_indices = present_indices.to(torch.int64).clone() # + transcript_sample["present_word_offset"].unsqueeze(1)
        for bi, word_embedding_tensor in enumerate(word_embedding_tensors):
            # time_synced_word_embeddings = torch.zeros(T, D, device=word_embedding_tensor.device)
            word_embedding_tensor_with_silence = torch.cat([silent_embedding[bi], word_embedding_tensor], dim=0)
            batch_word_embeddings_with_silence += [word_embedding_tensor_with_silence[word_indices[bi]]]
        batch_word_embeddings_with_silence = torch.stack(batch_word_embeddings_with_silence, dim=0)

        # all_indices = transcript_sample["all_words_frame_indices"]
        present_word_offsets = transcript_sample["present_word_offset"].unsqueeze(1).repeat(1, word_indices.size(1))
        is_not_silent = word_indices != 0
        word_indices[is_not_silent] += present_word_offsets[is_not_silent]

        ## use the word_indices to access batch_word_embeddings_with_silence.
        ## This is not as straighforward as batch_word_embeddings_with_silence[word_indices]. 
        ## We need indexing for the batch dimension of the same shape as word_indices.

        ## embedding tensor is: [B, W, D], where W is number of words 
        ## index tensor is [B, T], where T is number of frames 
        ## time_synced_word_embeddings should be: [B, T, D]

        # batch_dim = torch.arange(B).view(-1, 1).expand(-1, T) ## creates the indices for the batch dimension
        # expanded_index_tensor = word_indices.unsqueeze(2).expand(-1, -1, D)
        # time_synced_word_embeddings = batch_word_embeddings_with_silence[batch_dim, word_indices, :] 
        sample["transcript_word_embeddings_time_synced"] = batch_word_embeddings_with_silence 
        sample["word_indices"] = word_indices
        sample["silent_frames"] = (word_indices == 0).any(dim=0, keepdim=True)
        del sample['word_mappings'] ## no longer need it


        ## sanity checks 
        # words = transcript_sample['all_words_transcript']
        # indices = transcript_sample['all_words_frame_indices']

        # frame_indices = sample["frame_indices"]
        # present_indices = indices[0][frame_indices[0]].to(torch.int64)
        # assert (present_indices == word_indices[0]).all(), "Frame indices and word indices are not aligned"


        # w = [words[0][int(i.item())] for i in indices[0]]
        # w_p = [words[0][int(i.item())] for i in word_indices[0]]
        return sample
    
    def has_synchronized_output(self):
        return True



    
class PrecomputedTextEmbeddingEncoder(TextEncoderBase): 

    def __init__(self, dim, silence_token_init):
        super().__init__()
        self.dim = dim 
        # self.trainable = trainable
        self.silence_token_init = silence_token_init
        ## add learnable tensor for silent embedding 
        # silent_embedding = torch.randn(1, self.model_cfg.hidden_size)
        self.silence_embedding = torch.nn.Parameter(torch.randn(dim))
        if self.silence_token_init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.silence_embedding)
        elif self.silence_token_init in ["zeros", "zero"]:
            torch.nn.init.zeros_(self.silence_embedding)
        else: 
            raise NotImplementedError(f"Silence token init {self.silence_token_init} not supported")

    def forward(self, sample, train=False, desired_output_length=None, **kwargs):
        time_synced_word_embeddings = sample["transcript_word_embeddings_time_synced"]
        silent_frames = sample["silent_frames"] 
       
        B, T = time_synced_word_embeddings.size(0), time_synced_word_embeddings.size(1)
        silent_embedding = self.silence_embedding.unsqueeze(0).repeat(B, T, 1)
        silent_embedding = silent_frames.float()[:,:T, None] * silent_embedding
        # silent_embedding = silent_frames.float()[..., None] * silent_embedding
        sample["transcript_word_embeddings_time_synced"] = time_synced_word_embeddings + silent_embedding
        return sample

    def output_feature_dim(self):
        return self.dim

    def get_trainable_parameters(self): 
        return [self.silence_embedding]
    
    def has_synchronized_output(self):
        return True

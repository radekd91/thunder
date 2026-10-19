from .TextBase import TextEncoderBase
import torch
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer
from inferno.utils.other import class_from_str
import timeit
import os
import string
import re
import unicodedata


def decode_all_hex_to_unicode(s):
    # This regex matches sequences of one or more <0xXX> patterns
    hex_pattern = re.compile(r'(<0x[0-9A-Fa-f]{2}>)+')
    
    def hex_to_unicode(match):
        # Extracting the matched string
        hex_str = match.group(0)
        # Removing the "<" and ">" and splitting into individual hex codes
        hex_codes = hex_str.replace('<', '').replace('>', '').split('0x')[1:]
        # Converting the list of hex codes into a byte array
        bytes_array = bytes(bytearray.fromhex(''.join(hex_codes)))
        # Decoding the byte array into a Unicode string
        return bytes_array.decode('utf-8')
    
    # Replacing all hex patterns in the input string with their unicode equivalents
    decoded_str = re.sub(hex_pattern, hex_to_unicode, s)
    return decoded_str


def unicode_robust_comparison(str1, str2):

    # Example usage with a string that may contain multiple encoded characters
    s1_raw = str1
    s2 = str2

    # Decode any hex sequences in s1 to unicode
    s1_decoded = decode_all_hex_to_unicode(s1_raw)

    # Normalize both strings
    s1_normalized = unicodedata.normalize('NFC', s1_decoded)
    s2_normalized = unicodedata.normalize('NFC', s2)
    return s1_normalized == s2_normalized



class TextEmbeddingEncoderV2(TextEncoderBase): 

    def __init__(self, model_specifier, 
                 trainable, 
                 dropout_cfg=None, 
                 padding=True,
                 output_hidden_states=False,
                 ):
        super().__init__()
        self.trainable = trainable
        self.model_specifier = model_specifier
        self.output_hidden_states = output_hidden_states
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
    
    def has_synchronized_output(self):
        return False

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
        if  'bert-base-' in self.model_specifier: 
            punctuation.remove("'")

        # Initialize variables to store the current word and its subwords
        current_word = ''
        subwords = []
        token_indices = []

        # Iterate through tokens and map subwords to words
        for tok_idx, token in enumerate(tokens):
            # Check if token is punctuation or a word separator

            if 'meta-llama/Llama-2' in self.model_specifier:
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
            elif 'bert-base-' in self.model_specifier: 
                # if (not token.startswith('##')) or token in punctuation:  
                #     # if current_word:  # Add the previous word and its subwords to the list
                #     current_word = token
                #     word_to_subwords_list.append({
                #         'word': current_word, 
                #         'subwords': subwords,
                #         'token_indices': token_indices,
                #         })
                #     subwords = []
                #     token_indices = [tok_idx]
                # else:  # Continuation of the previous word
                #     current_word = token.lstrip('#')  # Remove the "##" prefix
                #     current_word += token
                #     subwords.append(token)
                #     token_indices.append(tok_idx)

                # Check if token is a subword continuation
                if token.startswith('##'):
                    current_word += token[2:]  # Append to the current word
                    subwords.append(current_word)
                    token_indices.append(tok_idx)
                # Check if token is punctuation or a word separator
                elif token in punctuation:
                    if current_word:  # Add the previous word and its subwords to the list
                        word_to_subwords_list.append({'word': current_word, 'subwords': subwords, 'token_indices': token_indices})
                    current_word = ''  # Reset current_word as punctuation separates words
                    subwords = []
                    token_indices = []
                else:  # Start of a new word
                    if current_word:  # Add the previous word and its subwords to the list
                        word_to_subwords_list.append({'word': current_word, 'subwords': subwords, 'token_indices': token_indices})
                    current_word = token
                    subwords = [current_word]
                    token_indices = [tok_idx]
            else: 
                raise NotImplementedError(f"Model {self.model_specifier} not supported. One of the settings in the code above will likely work, but you should verify this first.")

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
        assert len(transcript_methods) == 1, "More transcript methods not supported"
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
            return_offsets_mapping=True, 
        )
        device = batched_indices.device
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device) 
        # end_time = timeit.default_timer()
        # print(f"Tokenization time: {end_time - start_time:0.3f}")

        offset_mapping = inputs.pop('offset_mapping')
        outputs = self.model(**inputs,              
                             output_hidden_states=self.output_hidden_states,
                             )

        # Get the token embeddings
        num_input_tokens = inputs.input_ids.shape[1]
        num_output_tokens = outputs.last_hidden_state.shape[1]

        # Aggregate embeddings for each word in each sequence in the batch
        word_to_subwords_list_batch = []
        # batch_word_embeddings = []
        for batch_idx, input_words in enumerate(batched_words):

            word_to_subwords_list = self._word_2_tokens_dict(inputs['input_ids'][batch_idx].tolist())
            word_to_subwords_list_batch += [word_to_subwords_list]

        if self.output_hidden_states:
            sample["transcript_embeddings"] = list(outputs.hidden_states)    
        else:
            sample["transcript_embeddings"] = outputs.last_hidden_state
        sample["word_mappings"] = word_to_subwords_list_batch
        return sample


class FrameSyncedTextEmbeddingEncoderV2(TextEmbeddingEncoderV2): 

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
        
    def has_synchronized_output(self):
        return True

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

        punctuation = set(string.punctuation)
        punctuation.remove("-") ## this line might not be necessary for lama (hyphenated subwords will start with '-' but the rest of the word should follow)
        punctuation.remove("&") ## ampersand is likely a word ( 'n' ) and whisperx just transcribed as ampersand
        punctuation.remove("*") ## probably indicates a beeped out word
        if  'bert-base-' in self.model_specifier:
            punctuation.remove("'")

        for bi, word_mapping in enumerate(word_mappings):
            word_mapping_processed = []
            indices_to_keep = []
            for wi, word_dict in enumerate(word_mapping):
                word = word_dict['word']
                if word in self.tokenizer.all_special_tokens:
                    continue
                if word in punctuation:
                    continue
                # if word == "****": ## beeped out word 
                #     continue

                indices_to_keep += [wi]

            indices_to_merge = []
            current_merging_list = []
            for wi in indices_to_keep:
                word_mapping_processed += [word_mapping[wi]]
            
            for wi in range(len(word_mapping_processed)):
                word = word_mapping_processed[wi]['word']
                ## if starts with punctuation but continuing with more (apostrophed, hyphenated words or acronyms with dots, etc.) 

                word_starts_with_punctuation = word[0] in punctuation and word[:3] != "..."
                ampersand_connected = word[0] == "&" and len(word) > 1

                word_length = len(word)

                if (\
                        (word.startswith("'") or word.startswith("-") \
                        or (word_starts_with_punctuation and word_length > 1) or ampersand_connected) \
                        or word == "?" \
                    )\
                    and wi > 0:
                    # :
                    if 'meta-llama/Llama-2' in self.model_specifier:
                        current_merging_list += [wi-1, wi]
                    elif 'bert-base-' in self.model_specifier:
                        current_merging_list += [wi-1, wi] 
                        if wi + 1 < len(word_mapping_processed): 
                            current_merging_list += [wi+1]
                    continue
                else:
                    if current_merging_list:
                        indices_to_merge += [sorted(list(set(current_merging_list)))]
                        current_merging_list = []
            
            if len(current_merging_list) > 0:
                indices_to_merge += [sorted(list(set(current_merging_list)))]
                
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
                    print("Deleting", wi, word_mapping_processed[wi])
                    del word_mapping_processed[wi]
                print("Deleting", w_pair[0], word_mapping_processed[w_pair[0]])
                print("And replacing with", merged_word_mappings[-1])
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
            if len(word_embedding_tensor) > 0:
                word_embedding_tensor = torch.stack(word_embedding_tensor, dim=0)
            else: 
                word_embedding_tensor = torch.zeros(0, word_embedings.shape[-1], device=word_embedings.device)
            word_embedding_tensors += [word_embedding_tensor]
        return word_embedding_tensors
        

    def encode(self, sample, desired_output_length=None, **kwargs):
        sample = super().encode(sample, desired_output_length=desired_output_length, **kwargs)

        transcript_methods = list(sample["transcript"].keys())
        assert len(transcript_methods) == 1, "More transcript mehotds not supported"
        transcript_method = transcript_methods[0]
        transcript_sample = sample["transcript"][transcript_method]

        if isinstance(sample["transcript_embeddings"], list):
            batch_word_embeddings = sample["transcript_embeddings"]
        else: 
            batch_word_embeddings = [sample["transcript_embeddings"], ]
        word_mappings = sample["word_mappings"]

        word_mappings_processed = self._process_word_mappings(word_mappings)

        # batch_word_embeddings = sample["transcript_word_embeddings"] 
        ## zero-th index embedding corresponds to silence in the audio so prepend the silent learned feature
        L = len(batch_word_embeddings) ## num used hidden language layers
        W = batch_word_embeddings[0].size(1) ## num words
        B = batch_word_embeddings[0].size(0) ## bach size 
        T = sample["frame_indices"].shape[1] ## num frames
        D = batch_word_embeddings[0].size(2) ## word embedding vector dim
        silent_embedding = self.silence_embedding.unsqueeze(0).repeat(B, 1, 1)
        # batch_word_embeddings_with_silence = torch.cat([silent_embedding, batch_word_embeddings], dim=1)
        
        ## sanity checking the scripts
        for bi in range(B):
            script_1 = [word_info['word'] for word_info in word_mappings_processed[bi]]
            script_2 = transcript_sample['all_words_transcript'][bi][1:]
            assert len(script_1) == len(script_2), "Word mappings and transcript words are not aligned"
            wi = 0
            for w1, w2 in zip(script_1, script_2):
                ## remove all punctuation from w1 and w2 
                w1_ = w1
                w2_ = w2
                for p in string.punctuation:
                    w1_ = w1_.replace(p, "").lower()
                    w2_ = w2_.replace(p, "").lower()
                if w1_ != w2_:
                    ## okay the words do not match 
                    ## either they are wrong or they might be some encoding shenanigans here (example: ♪ vs <0xE2><0x99><0xAA>) 
                    ## let's try to check for that 
                    assert unicode_robust_comparison(w1, w2) or unicode_robust_comparison(w2, w1), f"Word mappings and transcript words are not aligned: {w1} != {w2} at index {wi}"
                    # assert w1 == w2, f"Word mappings and transcript words are not aligned: {w1} != {w2} at index {wi}"
                wi += 1

        present_indices = transcript_sample["present_words_frame_indices"]
        word_indices = present_indices.to(torch.int64).clone() # + transcript_sample["present_word_offset"].unsqueeze(1)
        
        batch_word_embeddings_with_silence_list = {}
        for l in range(L):
            word_embedding_tensors = self._compute_word_features(word_mappings_processed, batch_word_embeddings[l])
            batch_word_embeddings_with_silence = []
            for bi, word_embedding_tensor in enumerate(word_embedding_tensors):
                # time_synced_word_embeddings = torch.zeros(T, D, device=word_embedding_tensor.device)
                word_embedding_tensor_with_silence = torch.cat([silent_embedding[bi], word_embedding_tensor], dim=0)
                batch_word_embeddings_with_silence += [word_embedding_tensor_with_silence[word_indices[bi]]]
            
            if len(batch_word_embeddings_with_silence) > 0:
                batch_word_embeddings_with_silence = torch.stack(batch_word_embeddings_with_silence, dim=0)
            else:
                batch_word_embeddings_with_silence = torch.zeros(1, 1, D, device=word_embedding_tensor.device)
            
            batch_word_embeddings_with_silence_list[f"layer_{l:03d}"] = batch_word_embeddings_with_silence

        # all_indices = transcript_sample["all_words_frame_indices"]
        present_word_offsets = transcript_sample["present_word_offset"].unsqueeze(1).repeat(1, word_indices.size(1))
        is_not_silent = word_indices != 0
        word_indices[is_not_silent] += present_word_offsets[is_not_silent]

        if len(batch_word_embeddings_with_silence_list) == 1:
            batch_word_embeddings_with_silence_list = batch_word_embeddings_with_silence_list["layer_000"]

        sample["transcript_word_embeddings_time_synced"] = batch_word_embeddings_with_silence_list 
        sample["word_indices"] = word_indices
        sample["silent_frames"] = (word_indices == 0).any(dim=0, keepdim=True)
        del sample['word_mappings'] ## no longer need it
        return sample



class FrameSyncedTextEmbeddingMapper(TextEncoderBase): 

    def __init__(self, silence_token_init, model_specifier, trainable, *args, **kwargs): 
        super().__init__()
        access_token = None 
        if 'HF_TOKEN' in os.environ:
            access_token = os.environ['HF_TOKEN']
        self.model_cfg = AutoConfig.from_pretrained(model_specifier, token=access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_specifier, token=access_token)
        self.silence_token_init = silence_token_init
        ## add learnable tensor for silent embedding 
        silent_embedding = torch.randn(1, self.model_cfg.hidden_size)
        self.silence_embedding = torch.nn.Parameter(silent_embedding)
        self.trainable = trainable
        self.model_specifier = model_specifier
        # self.layer_numbers = layer_numbers
        # if isinstance(self.layer_numbers, int): 
        #     self.layer_numbers = [self.layer_numbers]

        if self.silence_token_init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.silence_embedding)
        elif self.silence_token_init in ["zeros", "zero"]:
            torch.nn.init.zeros_(self.silence_embedding)
        else: 
            raise NotImplementedError(f"Silence token init {self.silence_token_init} not supported")
        self.verbose = False


    def has_synchronized_output(self):
        return True

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

        punctuation = set(string.punctuation)
        punctuation.remove("-") ## this line might not be necessary for lama (hyphenated subwords will start with '-' but the rest of the word should follow)
        punctuation.remove("&") ## ampersand is likely a word ( 'n' ) and whisperx just transcribed as ampersand
        punctuation.remove("*") ## probably indicates a beeped out word
        if  'bert-base-' in self.model_specifier:
            punctuation.remove("'")

        for bi, word_mapping in enumerate(word_mappings):
            word_mapping_processed = []
            indices_to_keep = []
            for wi, word_dict in enumerate(word_mapping):
                word = word_dict['word']
                if word in self.tokenizer.all_special_tokens:
                    continue
                if word in punctuation:
                    continue
                # if word == "****": ## beeped out word 
                #     continue

                indices_to_keep += [wi]

            indices_to_merge = []
            current_merging_list = []
            for wi in indices_to_keep:
                word_mapping_processed += [word_mapping[wi]]
            
            for wi in range(len(word_mapping_processed)):
                word = word_mapping_processed[wi]['word']
                ## if starts with punctuation but continuing with more (apostrophed, hyphenated words or acronyms with dots, etc.) 

                word_starts_with_punctuation = word[0] in punctuation and word[:3] != "..."
                ampersand_connected = word[0] == "&" and len(word) > 1

                word_length = len(word)

                if (\
                        (word.startswith("'") or word.startswith("-") \
                        or (word_starts_with_punctuation and word_length > 1) or ampersand_connected) \
                        or word == "?" \
                    )\
                    and wi > 0:
                    # :
                    if 'meta-llama/Llama-2' in self.model_specifier:
                        current_merging_list += [wi-1, wi]
                    elif 'bert-base-' in self.model_specifier:
                        current_merging_list += [wi-1, wi] 
                        if wi + 1 < len(word_mapping_processed): 
                            current_merging_list += [wi+1]
                    continue
                else:
                    if current_merging_list:
                        indices_to_merge += [sorted(list(set(current_merging_list)))]
                        current_merging_list = []
            
            if len(current_merging_list) > 0:
                indices_to_merge += [sorted(list(set(current_merging_list)))]
                
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
                    if self.verbose:
                        print("Deleting", wi, word_mapping_processed[wi])
                    del word_mapping_processed[wi]
                if self.verbose:
                    print("Deleting", w_pair[0], word_mapping_processed[w_pair[0]])
                    print("And replacing with", merged_word_mappings[-1])
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
            if len(word_embedding_tensor) > 0:
                word_embedding_tensor = torch.stack(word_embedding_tensor, dim=0)
            else: 
                word_embedding_tensor = torch.zeros(0, word_embedings.shape[-1], device=word_embedings.device)
            word_embedding_tensors += [word_embedding_tensor]
        return word_embedding_tensors
        

    def encode(self, sample, desired_output_length=None, **kwargs):
        # sample = super().encode(sample, desired_output_length=desired_output_length, **kwargs)

        transcript_methods = list(sample["transcript"].keys())
        assert len(transcript_methods) == 1, "More transcript mehotds not supported"
        transcript_method = transcript_methods[0]
        transcript_sample = sample["transcript"][transcript_method]

        if isinstance(sample["transcript_embeddings"], (list, dict)):
            batch_word_embeddings = sample["transcript_embeddings"]
        else: 
            batch_word_embeddings = [sample["transcript_embeddings"], ]
        word_mappings = sample["word_mappings"]

        word_mappings_processed = self._process_word_mappings(word_mappings)

        # batch_word_embeddings = sample["transcript_word_embeddings"] 
        ## zero-th index embedding corresponds to silence in the audio so prepend the silent learned feature
        L = len(batch_word_embeddings) ## num used hidden language layers

        key_0 = list(batch_word_embeddings.keys())[0] if isinstance(batch_word_embeddings, dict) else 0

        W = batch_word_embeddings[key_0].size(1) ## num words
        B = batch_word_embeddings[key_0].size(0) ## bach size 
        T = sample["frame_indices"].shape[1] ## num frames
        D = batch_word_embeddings[key_0].size(2) ## word embedding vector dim
        silent_embedding = self.silence_embedding.unsqueeze(0).repeat(B, 1, 1)
        # batch_word_embeddings_with_silence = torch.cat([silent_embedding, batch_word_embeddings], dim=1)
        
        ## sanity checking the scripts
        for bi in range(B):
            script_1 = [word_info['word'] for word_info in word_mappings_processed[bi]]
            script_2 = transcript_sample['all_words_transcript'][bi][1:]
            assert len(script_1) == len(script_2), "Word mappings and transcript words are not aligned"
            wi = 0
            for w1, w2 in zip(script_1, script_2):
                ## remove all punctuation from w1 and w2 
                w1_ = w1
                w2_ = w2
                for p in string.punctuation:
                    w1_ = w1_.replace(p, "").lower()
                    w2_ = w2_.replace(p, "").lower()
                if w1_ != w2_:
                    ## okay the words do not match 
                    ## either they are wrong or they might be some encoding shenanigans here (example: ♪ vs <0xE2><0x99><0xAA>) 
                    ## let's try to check for that 
                    assert unicode_robust_comparison(w1, w2) or unicode_robust_comparison(w2, w1), \
                        f"Word mappings and transcript words are not aligned: {w1} != {w2} at index {wi}"
                    # assert w1 == w2, f"Word mappings and transcript words are not aligned: {w1} != {w2} at index {wi}"
                wi += 1

        present_indices = transcript_sample["present_words_frame_indices"]
        # word_indices = present_indices.to(torch.int64).clone() + transcript_sample["present_word_offset"].unsqueeze(1)
        word_indices = present_indices.to(torch.int64).clone() 
        is_not_silent = word_indices != 0
        is_silent = word_indices == 0
        word_indices += transcript_sample["present_word_offset"].unsqueeze(1)
        word_indices[is_silent] = 0

        batch_word_embeddings_with_silence_list = {}
        for l in range(L):
            embeddings_l = batch_word_embeddings[l] if isinstance(batch_word_embeddings, list) else batch_word_embeddings[list(batch_word_embeddings.keys())[l]]
            word_embedding_tensors = self._compute_word_features(word_mappings_processed, embeddings_l)
            batch_word_embeddings_with_silence = []
            batch_words_with_silence = []
            for bi, word_embedding_tensor in enumerate(word_embedding_tensors):
                # time_synced_word_embeddings = torch.zeros(T, D, device=word_embedding_tensor.device)
                word_embedding_tensor_with_silence = torch.cat([silent_embedding[bi], word_embedding_tensor], dim=0)
                batch_word_embeddings_with_silence += [word_embedding_tensor_with_silence[word_indices[bi]]]

                batch_words_with_silence += [[]]

                for ti in range(len(word_indices[bi])):
                    wi = word_indices[bi][ti]
                    if wi == 0: ## silence 
                        batch_words_with_silence[bi] += ["[SILENCE]"]
                        continue
                    wi -= 1 ## account for the prepended silence token
                    # word_dict = word_mappings_processed[bi][wi + transcript_sample["present_word_offset"][bi] - 1]
                    word_dict = word_mappings_processed[bi][wi ]
                    batch_words_with_silence[bi] += [word_dict['word']]
                    assert  (word_dict["embedding"] - batch_word_embeddings_with_silence[bi][ti]).abs().sum() == 0., \
                        f"The embeddings of the word '{word_dict['word']}' that should be in frame {ti} and the embedding "\
                        "that actually is in that frame do not match. Something is wrong."
            
            if len(batch_word_embeddings_with_silence) > 0:
                batch_word_embeddings_with_silence = torch.stack(batch_word_embeddings_with_silence, dim=0)
            else:
                batch_word_embeddings_with_silence = torch.zeros(1, 1, D, device=word_embedding_tensor.device)
            
            key_l = list(embeddings_l.keys())[l] if isinstance(embeddings_l, dict) else f"layer_{l:03d}"
            batch_word_embeddings_with_silence_list[key_l] = batch_word_embeddings_with_silence

        # # all_indices = transcript_sample["all_words_frame_indices"]
        # present_word_offsets = transcript_sample["present_word_offset"].unsqueeze(1).repeat(1, word_indices.size(1))

        if len(batch_word_embeddings_with_silence_list) == 1:
            batch_word_embeddings_with_silence_list = batch_word_embeddings_with_silence_list["layer_000"]

        sample["transcript_word_embeddings_time_synced"] = batch_word_embeddings_with_silence_list 
        sample["word_indices"] = word_indices
        sample["silent_frames"] = is_silent.any(dim=0, keepdim=True)
        del sample['word_mappings'] ## no longer need it
        return sample

    def forward(self, sample, train=False, desired_output_length=None, **kwargs):
        if self.trainable:
            return self.encode(sample)
        
        with torch.no_grad():
            return self.encode(sample)
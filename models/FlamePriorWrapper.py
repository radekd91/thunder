from pathlib import Path
import torch 
import numpy as np
from inferno.models.DecaFLAME import FLAME 
from inferno.models.temporal.Preprocessors import FlamePreprocessor
from inferno.layers.losses.RotationLosses import compute_rotation_loss, convert_rot
from inferno.models.temporal.motion_prior.MotionPrior import rotation_dim
import torch.nn.functional as F


class FlameMotionPrior(torch.nn.Module): 
    """
    Just a simple wrapper around FLAME that is compatible with the MotionPrior interface. Some functions were copy-pasted from MotionPrior.py
    """

    def __init__(self, 
                 cfg, 
                 ):
        super().__init__()
        self.config = cfg
        self.flame_processor =  FlamePreprocessor(cfg)

        if cfg.get("normalizing_stats", None) is not None:
            self.normalizing_stats = cfg.normalizing_stats
            assert Path(cfg.normalizing_stats).is_file(), f"Normalizing stats file {cfg.normalizing_stats} not found. Check the inferno/assets folder"\
                " or the dataset folder. If it's there, replace the path in the config with the correct one."
            self.normalizing_stats = dict(np.load( cfg.normalizing_stats))
            self.normalizing_stats = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0).unsqueeze(0) for key, value in self.normalizing_stats.items()}
            if 'jaw_mean' not in self.normalizing_stats.keys():
                self.normalizing_stats['jaw_mean'] = self.normalizing_stats['pose_mean'][..., 3:]  # Assuming jaw is the last part of pose
            if 'jaw_std' not in self.normalizing_stats.keys():
                self.normalizing_stats['jaw_std'] = self.normalizing_stats['pose_std'][..., 3:]
            if 'global_pose_mean' not in self.normalizing_stats.keys():
                self.normalizing_stats['global_pose_mean'] = self.normalizing_stats['pose_mean'][..., :3]
            if 'global_pose_std' not in self.normalizing_stats.keys():
                self.normalizing_stats['global_pose_std'] = self.normalizing_stats['pose_std'][..., :3]
        else: 
            self.normalizing_stats = None

    @property 
    def sequence_components(self):
        return self.config.sequence_components
    
    @property 
    def has_global_pose(self):
        return "global_pose" in self.config.sequence_components
    
    @property 
    def has_neck_pose(self):
        return "neck" in self.config.sequence_components
    
    @property
    def forward_with_global(self): 
        ## by default True, global pose forward pass will be done if global or neck poses are present
        return self.config.get('forward_with_global', True) 

    @property
    def detach_posed_expression(self):
        # return self.config.detach_posed_expression
        return self.config.get('detach_posed_expression', False)

    @property
    def detach_posed_jaw(self):
        return self.config.get('detach_posed_jaw', False)

    @property
    def replace_gt_exp_posed(self):
        return self.config.get('replace_gt_exp_posed', False)

    @property
    def replace_gt_jaw_posed(self):
        return self.config.get('replace_gt_jaw_posed', False)
    
    @property
    def replace_gt_neck_posed(self):
        return self.config.get('replace_gt_neck_posed', False)
    
    @property
    def clean_rotations(self): 
        """
        If true, the rotations will be cleaned up by converting them to a matrix and back to the original representation
        """
        return self.config.get("clean_rotations", False)

    def _rotation_representation(self):
        return self.config.rotation_representation

    def to(self, *args, **kwargs):
        self.flame_processor = self.flame_processor.to(*args, **kwargs)
        if self.normalizing_stats is not None:
            self.normalizing_stats = {key: value.to(*args, **kwargs) for key, value in self.normalizing_stats.items()}
        return super().to(*args, **kwargs)

    def get_trainable_parameters(self): 
        return []

    def forward(self, input): 
        return self.flame(input)
    
    def bottleneck_dim(self):
        dim = 0
        if "exp" in self.config.sequence_components:
            dim += self.config.flame.n_exp 
        if "shape" in self.config.sequence_components:
            dim += self.config.flame.n_shape
        if "jaw" in self.config.sequence_components:
            dim += rotation_dim(self._rotation_representation()) if self.config.sequence_components["jaw"] == "rot" else  self.config.sequence_components["jaw"]
        if "neck" in self.config.sequence_components:
            dim += rotation_dim(self._rotation_representation()) if self.config.sequence_components["neck"] == "rot" else  self.config.sequence_components["neck"]
        if "global_pose" in self.config.sequence_components:
            dim += rotation_dim(self._rotation_representation()) if self.config.sequence_components["global_pose"] == "rot" else  self.config.sequence_components["global_pose"]
        if "cam" in self.config.sequence_components:
            dim += 3
        if "trans" in self.config.sequence_components:
            dim += 3
        return dim
    
    
    def latent_temporal_factor(self): 
        return 1

    def quant_factor(self): 
        return 0
    
    def preprocess(self, batch, training_or_validation=True):
        if self.flame_processor is not None:
            if (self.has_global_pose or self.has_neck_pose) and (self.forward_with_global or not training_or_validation):
                batch = self.flame_processor(batch, input_key=None, output_prefix="gt_", with_grad=False)
                batch["gt_vertices_posed"] = batch["gt_vertices"]
                if "gt_trans" in batch.keys():
                    B, T = batch["gt_trans"].shape[:2]
                    batch["gt_vertices_posed"] = (batch["gt_vertices_posed"].view(B,T,-1, 3) + batch["gt_trans"][..., None, :]).view(B,T,-1)
                if self.has_global_pose:
                    global_pose = batch.pop("gt_global_pose")
                if self.has_neck_pose:
                    neck_pose = batch.pop("gt_neck")
                batch = self.flame_processor(batch, input_key=None, output_prefix="gt_", with_grad=False)
                if self.has_global_pose:
                    batch["gt_global_pose"] = global_pose
                if self.has_neck_pose:
                    batch["gt_neck"] = neck_pose
            else:
                # assert "gt_global_pose" not in batch.keys(), "Global pose is present in the batch but the class is configured to not use it"
                # if self._rotation_representation() != "aa_1": 
                #     batch["gt_jaw"][..., 1:] = 0.0  # DiffPoseTalk uses 1D joint rotation hack, so we zero out the second and third dimension
                ## if the global pose is not used, we don't want it in the batch
                if self.has_global_pose or "gt_global_pose" in batch.keys():
                    global_pose = batch.pop("gt_global_pose")
                if self.has_neck_pose or "gt_neck" in batch.keys():
                    neck_pose = batch.pop("gt_neck")
                batch = self.flame_processor(batch, input_key=None, output_prefix="gt_", with_grad=False) 
                ## put the global pose back if it was present
                if self.has_global_pose:
                    batch["gt_global_pose"] = global_pose
                if self.has_neck_pose:
                    batch["gt_neck"] = neck_pose
        # if "template" in batch.keys():
        #     batch["gt_vertex_offsets"] = batch["gt_vertices"] - batch["template"][:,None, ...]
        return batch
    

    def compose_sequential_input(self, batch, ):
        prefix = "gt_"
        if "reconstruction" in batch.keys():
            assert len(batch["reconstruction"]) <= 1, "Only one type of reconstruction is supported for now."
            reconstruction_key = list(batch["reconstruction"].keys())[0]
            rec_dict = batch["reconstruction"][reconstruction_key]
        else:
            rec_dict = batch
        # self._input_sequence_keys = [prefix + key for key in sequence_components.keys()]
        # self._input_feature_dims = {key: rec_dict[key].shape[2] for key in self._input_sequence_keys}
        # self._input_feature_dims = sequence_components
        input_sequences = [] 
        for key in self.sequence_components.keys():
            if self.sequence_components[key] == "rot":
                rotation = rec_dict[prefix + key][..., :rotation_dim('aa')]
                if self._rotation_representation() != "aa":
                    ## convert from aa to another representation if need be
                    B, T = rotation.shape[:2]
                    rotation = convert_rot(rotation.view(B*T, -1), "aa", self._rotation_representation()).view(B,T, -1)
                if self.clean_rotations:
                    if self._rotation_representation() in ["6d", "6dof"]:
                        rotation = self.clean_up_rotation(rotation)
                if self.normalizing_stats is not None:
                    norm_mean = self.normalizing_stats.get(key + "_mean", 0.0) if self.normalizing_stats else 0.0
                    norm_std = self.normalizing_stats.get(key + "_std", 1.0) if self.normalizing_stats else 1.0
                    norm_mean = convert_rot(norm_mean.view(1, -1), "aa", self._rotation_representation()).view(1,1, -1)
                    norm_std = convert_rot(norm_std.view(1, -1), "aa", self._rotation_representation()).view(1,1, -1)
                    rotation = (rotation - norm_mean) / norm_std
                input_sequences += [rotation]
            elif self.sequence_components[key] == "flame_verts":
                input_sequences += [rec_dict[prefix + key][..., :5023 * 3]]
            else:
                dim = self.sequence_components[key]
                seq = rec_dict[prefix + key][..., :dim]
                if seq.shape[-1] < dim and key in ['shape', 'exp']: # pad with zeros if the not using all dimensions
                    difference = dim - seq.shape[-1]
                    seq = F.pad(seq, (0, difference), mode='constant', value=0)
                if self.normalizing_stats is not None:
                    norm_mean = self.normalizing_stats.get(key + "_mean", 0.0) if self.normalizing_stats else 0.0
                    norm_std = self.normalizing_stats.get(key + "_std", 1.0) if self.normalizing_stats else 1.0
                    seq = (seq - norm_mean) / norm_std
                input_sequences += [seq]
        batch["input_sequence"] = torch.cat(input_sequences, dim=2)
        return batch

    def encode(self, batch):
        batch["encoded_features"] = batch["input_sequence"]
        return batch
    
    def quantize(self, batch, training_or_validation=False):
        return batch

    def decode(self, batch, assert_size=False):
        batch["decoded_sequence"] = batch["encoded_features"]
        if self.clean_rotations:
            batch["decoded_sequence"] = self.clean_up_rotation_in_sequence(batch["decoded_sequence"])
        return batch

    def decompose_sequential_output(self, batch):
        output_sequence_key = "decoded_sequence"
        prefix = "reconstructed_"
        start_idx = 0
        for i, key in enumerate(self.sequence_components.keys()):
            # dim = self._input_feature_dims[key]
            dim = self.sequence_components[key]
            if dim == "rot":
                dim = rotation_dim(self._rotation_representation()) 
            elif dim == "flame_verts":
                dim = 5023 * 3
            
            if self.sequence_components[key] == "rot": 
                ## convert from the used rotation representation to aa if need be 
                rotation = batch[output_sequence_key][:, :, start_idx:start_idx + dim]
                # B, T = rotation.shape[:2]
                # if self._rotation_representation() != "aa":
                #     rotation = convert_rot(rotation.view(B*T, -1), 
                #                            self._rotation_representation(), 
                #                            "aa").view(B,T, -1)
                if self.clean_rotations:
                    if self._rotation_representation() in ["6d", "6dof"]:
                        rotation = self.clean_up_rotation(rotation)
                if self.normalizing_stats is not None:
                    norm_mean = self.normalizing_stats.get(key + "_mean", 0.0) if self.normalizing_stats else 0.0
                    norm_std = self.normalizing_stats.get(key + "_std", 1.0) if self.normalizing_stats else 1.0
                    norm_mean = convert_rot(norm_mean.view(1, -1), "aa", self._rotation_representation()).view(1,1, -1)
                    assert self._rotation_representation() in ["aa", "aa_1"] , "If you want to fiddle with std of rotations, only aa or aa_1 is supported"
                    norm_std = convert_rot(norm_std.view(1, -1), "aa", self._rotation_representation()).view(1,1, -1)
                    rotation = rotation * norm_std + norm_mean
                batch[prefix + key] = rotation
            else:    
                seq = batch[output_sequence_key][:, :, start_idx:start_idx + dim]
                if self.normalizing_stats is not None:
                    norm_mean = self.normalizing_stats.get(key + "_mean", 0.0) if self.normalizing_stats else 0.0
                    norm_std = self.normalizing_stats.get(key + "_std", 1.0) if self.normalizing_stats else 1.0
                    seq = seq * norm_std + norm_mean
                batch[prefix + key] = seq
            start_idx += dim
        return batch


    def clean_up_rotation_in_sequence(self, decoded_sequence):
        input_sequences = [] 
        start_idx = 0
        for key in self.sequence_components.keys():
            dim = self.sequence_components[key]
            if dim == "rot":
                dim = rotation_dim(self._rotation_representation()) 
            elif dim == "flame_verts":
                dim = 5023 * 3
            
            vec = decoded_sequence[:, :, start_idx:start_idx + dim]
            if self.sequence_components[key] == "rot":
                if self._rotation_representation() in ["6d", "6dof"]:
                    B, T = vec.shape[:2]
                    rotrep = vec
                    rotrep_fixed = self.clean_up_rotation(rotrep)
                    vec = rotrep_fixed.view(B, T, -1)

            start_idx += dim
            input_sequences += [vec]
        decoded_sequence_clean = torch.cat(input_sequences, dim=2)
        return decoded_sequence_clean 
    
    def clean_up_rotation(self, rotation):
        B, T = rotation.shape[:2]
        ## 6D vectors are allowed to be messy but this can propagate through many diffusion steps and cause problems
        ## so we clean them up here by converting them to a matrix and back to 6D 
        rotation_fixed = convert_rot(rotation.view(B*T, -1), 
                                                self._rotation_representation(), 
                                                "matrix")
        rotation_fixed = convert_rot(rotation_fixed, "matrix", self._rotation_representation())
        return rotation_fixed.view(B, T, -1)

    def postprocess(self, batch, training_or_validation=True):
        if "reconstruction" in batch.keys(): # is the pseudo-GT reconstruction fed to the model?
            reconstruction_key = list(batch["reconstruction"].keys())[0]
            rec_batch = batch["reconstruction"][reconstruction_key].copy() 
        else: 
            rec_batch = batch.copy()
        
        rec_batch_backup = rec_batch.copy()

        for key in self.sequence_components.keys():
            if "gt_" + key in rec_batch.keys():
                del rec_batch["gt_" + key]
            if "reconstructed_" + key in rec_batch.keys():
                del rec_batch["reconstructed_" + key]
            rec_batch["gt_" + key] = batch["reconstructed_" + key].contiguous()
            if self.sequence_components[key] == "rot":
                if self._rotation_representation() != "aa":
                    B, T = rec_batch["gt_" + key].shape[:2]
                    rec_batch["gt_" + key] = convert_rot(rec_batch["gt_" + key].view(B*T, -1), 
                                                         self._rotation_representation(), 
                                                         "aa").view(B, T, -1)
        if "gt_vertices" in rec_batch.keys():
            del rec_batch["gt_vertices"]

        # if self.flame_processor is not None:

        if (self.has_global_pose or self.has_neck_pose) and (self.forward_with_global or not training_or_validation):
            rec_batch_unposed = rec_batch.copy()
            ## zero out the global pose
            if self.has_global_pose:
                rec_batch_unposed["reconstructed_global_pose"] = torch.zeros_like(rec_batch_unposed["gt_global_pose"])
            
                rec_batch_unposed["gt_global_pose"] = rec_batch_unposed["reconstructed_global_pose"]
            if self.has_neck_pose:
                rec_batch_unposed["reconstructed_neck"] = torch.zeros_like(rec_batch_unposed["gt_neck"])
                rec_batch_unposed["gt_neck"] = rec_batch_unposed["reconstructed_neck"]
            ## detach expression and jaw if desired (probably you don't want the gradient to flow from globally posed to exp and jaw )
            if self.detach_posed_expression:
                # rec_batch_unposed["reconstructed_exp"] = rec_batch["reconstructed_exp"].detach()
                # rec_batch_unposed["gt_exp"] = rec_batch_unposed["reconstructed_exp"]
                # rec_batch_unposed["gt_exp"] = rec_batch["gt_exp"].detach()

                ## NOTE TO SELF, MAY 10th, 2024. the above line appears to be wrong and should probably detach the posed, not the unposed, fixing 
                rec_batch["gt_exp"] = rec_batch["gt_exp"].detach() 


                # rec_batch_unposed["reconstructed_shape"] = rec_batch["reconstructed_shape"].detach()
            if self.detach_posed_jaw:
                # rec_batch_unposed["reconstructed_jaw"] = rec_batch["reconstructed_jaw"].detach()
                # rec_batch_unposed["gt_jaw"] = rec_batch_unposed["reconstructed_jaw"] 
                # rec_batch_unposed["gt_jaw"] = rec_batch["gt_jaw"].detach()
                ## NOTE TO SELF, MAY 10th, 2024. the above line appears to be wrong and should probably detach the posed, not the unposed, fixing 
                rec_batch["gt_jaw"] = rec_batch["gt_jaw"].detach()


            if self.replace_gt_exp_posed and "gt_exp" in rec_batch_backup.keys(): 
                ## flips in gt exp 
                rec_batch["gt_exp"] = rec_batch_backup["gt_exp"][..., :rec_batch["gt_exp"].shape[-1]]

            if self.replace_gt_jaw_posed and "gt_jaw" in rec_batch_backup.keys(): 
                ## flips in gt jaw 
                rec_batch["gt_jaw"] = rec_batch_backup["gt_jaw"]


            if self.replace_gt_neck_posed and "gt_neck" in rec_batch_backup.keys(): 
                ## flips in gt jaw 
                rec_batch["gt_neck"] = rec_batch_backup["gt_neck"]


            rec_batch_unposed = self.flame_processor(rec_batch_unposed, input_key=None, output_prefix="reconstructed_unposed_", with_grad=True)
            # rec_batch["gt_global_pose"] = torch.zeros_like(rec_batch_unposed["gt_global_pose"]) ### REMOVE THIS DEBUG LINE!!
            rec_batch = self.flame_processor(rec_batch, input_key=None, output_prefix="reconstructed_", with_grad=True)
            batch["reconstructed_vertices"] = rec_batch_unposed["reconstructed_unposed_vertices"]
            batch["reconstructed_vertices_posed"] = rec_batch["reconstructed_vertices"]

            # if "reconstructed_trans" in rec_batch.keys():
            #     B, T = rec_batch["reconstructed_trans"].shape[:2]
                # batch["reconstructed_vertices_posed"] = (batch["reconstructed_vertices_posed"].view(B,T,-1, 3) + rec_batch["reconstructed_trans"][..., None, :]).view(B,T,-1)

            if "gt_trans" in rec_batch.keys():
                B, T = rec_batch["gt_trans"].shape[:2]
                batch["reconstructed_vertices_posed"] = (batch["reconstructed_vertices_posed"].view(B,T,-1, 3) + rec_batch["gt_trans"][..., None, :]).view(B,T,-1)
            else: 
                assert "trans" not in self.sequence_components.keys(), "Global translation is configured to be used but not present in the"

        else:
            # assert "gt_global_pose" not in rec_batch.keys(), "Global pose is present in the batch but the class is configured to not use it"
            if self.has_global_pose or "gt_global_pose" in rec_batch.keys():
                global_pose = rec_batch.pop("gt_global_pose")
            if self.has_neck_pose or "gt_neck" in rec_batch.keys():
                neck_pose = rec_batch.pop("gt_neck")
            rec_batch = self.flame_processor(rec_batch, input_key=None, output_prefix="reconstructed_", with_grad=True)
        
            # if self.has_global_pose:
            #     rec_batch["reconstructed_global_pose"] = global_pose
            # if self.has_neck_pose:
            #     rec_batch["reconstructed_neck"] = neck_pose

            # if "template" in batch.keys() and "reconstructed_vertex_offsets" in rec_batch.keys():
            #     rec_batch["reconstructed_vertices"] = rec_batch["reconstructed_vertex_offsets"] + batch["template"][:, None, ...]
            
            batch["reconstructed_vertices"] = rec_batch["reconstructed_vertices"]
        return batch
    

    def encoding_step(self, batch, train=False, validation=False): 
        batch = self.preprocess(batch)
        batch = self.compose_sequential_input(batch)
        batch = self.encode(batch)
        batch = self.quantize(batch, training_or_validation=train or validation)
        return batch
    

    def decoding_step(self, batch, train=False, validation=False):
        batch = self.decode(batch, assert_size=False)
        batch = self.decompose_sequential_output(batch)
        batch = self.postprocess(batch)
        return batch

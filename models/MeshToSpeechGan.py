from .MeshToSpeech import *


class MeshToSpeechGan(MeshToSpeech): 

    def __init__(self,
                 cfg,
                 preprocessors: Optional[Dict] = None,
                 input_encoder: Mesh4DEncoderBase = None,
                 sequence_encoder: SequenceEncoder = None, 
                #  transcript_encoder: Optional[nn.Module] = None,
                #  prior: Optional[nn.Module] = None,
                #  renderer: Optional[Renderer] = None, 
                 speech_prediction_heads: Dict = None,
                 vocoder: Optional[Mesh2SpeechVocoderBase] = None,
                 postprocessors: Optional[Dict] = None,
                 discriminators: Optional[Dict] = None,
                 **kwargs):
        
        super().__init__(
            cfg,
            preprocessors = preprocessors,
            input_encoder = input_encoder,
            sequence_encoder = sequence_encoder, 
            speech_prediction_heads = speech_prediction_heads,
            vocoder = vocoder,
            postprocessors = postprocessors,
            **kwargs
            )
        self.discriminators = torch.nn.ModuleDict(discriminators)
        self._automatic_optimization = False

    def get_discriminator_trainable_parameters(self):
        trainable_params = []
        for key, discriminator in self.discriminators.items():
            trainable_params += discriminator.get_trainable_parameters()
        return trainable_params
    
    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_discriminator_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        elif self.cfg.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.cfg.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.cfg.learning.optimizer}'")

        schedulers = []

        discriminator_opt_dict = {}
        discriminator_opt_dict['optimizer'] = opt
        if 'learning_rate_patience' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                                   patience=self.cfg.learning.learning_rate_patience,
                                                                   factor=self.cfg.learning.learning_rate_decay,
                                                                   mode=self.cfg.learning.lr_sched_mode)
            schedulers += [scheduler]
            discriminator_opt_dict['lr_scheduler'] = scheduler
            discriminator_opt_dict['monitor'] = 'val_loss_total'
        elif 'learning_rate_decay' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.learning.learning_rate_decay)
            discriminator_opt_dict['lr_scheduler'] = scheduler
            schedulers += [scheduler]

        generator_opt_dict = super().configure_optimizers()
        return [generator_opt_dict, discriminator_opt_dict]
    

    def discriminator_forward(self, batch, input_prefix="predicted_", output_prefix="", train=False, validation=False, detach_inputs=False, **kwargs):
        for key, discriminator in self.discriminators.items():
            batch = discriminator(batch, input_prefix=input_prefix, output_prefix=output_prefix + f"{key}", 
                                  train=train, validation=validation, 
                                  detach_inputs=detach_inputs, 
                                  **kwargs)
        return batch


    def _step(self, training, validation, batch, batch_idx, dataloader_idx=None, *args, **kwargs):

        assert not (training and validation), "Only one of training and validation can be True."

        ## 1) GENERATOR
        if training:
            optimizer_g, optimizer_d = self.optimizers()
            # optimizer_g_idx = self.optimizers().index(optimizer_g)
            # optimizer_d_idx = self.optimizers().index(optimizer_d)
            # self.toggle_optimizer(optimizer_g, optimizer_g_idx)

        batch = self.forward(batch, train=training, validation=validation, **kwargs)

        real_batch = self.discriminator_forward(batch, input_prefix="predicted", output_prefix="adv_", training=training, validation=validation, **kwargs)
        
        # calculate loss
        generator_total_loss, generator_losses, generator_metrics = self.compute_loss(batch, training=training, validation=validation, **kwargs)

        if training:
            self.manual_backward(generator_total_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()

        prefix = "train" if training else "val" if validation else "test"
        # if dataloader_idx is not None and dataloader_idx > 0:
        #     prefix = prefix + f"_{dataloader_idx}"

        losses_and_metrics_to_log = {**generator_losses, **generator_metrics}
        losses_and_metrics_to_log = {f"{prefix}/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        # if training:
        #     self.untoggle_optimizer(optimizer_g_idx)

        ## 2) DISCRIMINATOR
        # if training:
        #     self.toggle_optimizer(optimizer_d, optimizer_d_idx)

        real_batch["gt_mel"] = real_batch["gt_mel"].clone()
        real_batch["gt_mel"].requires_grad = True

        real_batch = self.discriminator_forward(
            batch, input_prefix="gt", output_prefix="real_", 
            training=training, validation=validation, **kwargs)
        fake_batch = self.discriminator_forward(batch, input_prefix="predicted", output_prefix="fake_",
                                                training=training, validation=validation,
                                                detach_inputs=True,
                                                **kwargs)

        discriminator_real_total_loss,  discriminator_real_losses,  discriminator_real_metrics = self.compute_discriminator_loss(
            real_batch, training=training, validation=validation, **kwargs)
        discriminator_fake_total_loss,  discriminator_fake_losses,  discriminator_fake_metrics = self.compute_discriminator_loss(
            fake_batch, training=training, validation=validation, **kwargs)

        real_losses_and_metrics_to_log = {**discriminator_real_losses, **discriminator_real_metrics}
        real_losses_and_metrics_to_log = {f"{prefix}_real/" + k: v.item() if isinstance(v, (torch.Tensor)) else v \
                                          if isinstance(v, float) else 0. for k, v in real_losses_and_metrics_to_log.items()}
        
        fake_losses_and_metrics_to_log = {**discriminator_fake_losses, **discriminator_fake_metrics}
        fake_losses_and_metrics_to_log = {f"{prefix}_fake/" + k: v.item() if isinstance(v, (torch.Tensor)) else v \
                                          if isinstance(v, float) else 0. for k, v in fake_losses_and_metrics_to_log.items()}

        discriminator_loss = (discriminator_real_total_loss + discriminator_fake_total_loss) 
        if training:
            self.manual_backward(discriminator_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            # self.untoggle_optimizer(optimizer_d_idx)

        losses_and_metrics_to_log = {**losses_and_metrics_to_log, **real_losses_and_metrics_to_log, **fake_losses_and_metrics_to_log}

        return losses_and_metrics_to_log



    def training_step(self, batch, batch_idx):
        training = True 
        validation = False

        # forward pass
        losses_and_metrics_to_log = self._step(training, validation, batch, batch_idx)

        self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended


    def validation_step(self, batch, batch_idx, dataloader_idx=None,  *args, **kwargs): 
        training = False 
        validation = True 

        # forward pass
        losses_and_metrics_to_log = self._step(training, validation, batch, batch_idx, dataloader_idx=dataloader_idx)

        # if self.logger is not None:
        self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
    

    def compute_discriminator_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 

        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.discriminator_losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self.compute_discriminator_loss_term(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.discriminator_metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self.compute_discriminator_loss_term(sample, training, validation, metric_name, metric_cfg)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.discriminator_losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
                if isinstance(term, torch.Tensor) and term.isnan().any():
                    print(f"[WARNING]: loss '{loss_name}' is NaN. Skipping this term.")
                    continue
                if total_loss is None: 
                    total_loss = 0.
                weighted_term =  (term * loss_cfg["weight"])
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
        return total_loss, losses, metrics

    def compute_discriminator_loss_term(self, sample, training, validation, loss_name, loss_cfg): 
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        
        mask_invalid = loss_cfg.get('mask_invalid', False) # mask invalid frames 
        if mask_invalid:
            if mask_invalid == "mediapipe_landmarks": 
                # frames with invalid mediapipe landmarks will be masked for loss computation
                mask = sample["landmarks_validity"]["mediapipe"].to(dtype=torch.bool)
            else: 
                raise ValueError(f"mask_invalid value of '{mask_invalid}' not supported")
        else:
            mask = torch.ones((*sample["gt_exp"].shape[:2], 1), device=sample["gt_exp"].device, dtype=torch.bool)

        # mask = torch.zeros_like(mask)

        # if "disentangled" in loss_type and not validation and not training:
        #     # disentangled losses are only computed during training
        #     return 0.

        B = sample["gt_exp"].shape[0] # total batch size
        T = sample["gt_exp"].shape[1] # sequence size
       
        mask_sum = mask.sum().item()

        if mask_invalid and mask_sum == 0:
            return 0.

        if loss_type in ["softplus"]:
            pred_key = loss_cfg.input_key
            value = sample[pred_key]
            l = -1 if loss_cfg.label else 1
            loss_value = F.softplus(l*value).mean()
                 
        elif loss_type in ["grad_penalty"]:
            if not training: 
                loss_value = 0.
            else:
                input_key = loss_cfg.input_key
                pred_key = loss_cfg.output_key
                output_value = sample[pred_key]
                input_value = sample[input_key]
                from torch.autograd import grad
                grad_ = grad(outputs=output_value.sum(), inputs=input_value, create_graph=True)[0]
                grad_v = grad_.reshape(grad_.size(0), -1).norm(2, dim=1)
                if loss_cfg.metric == 'l1': 
                    pass 
                elif loss_cfg.metric == 'l2':
                    grad_v = grad_v ** 2
                loss_value = grad_v.mean()

        else: 
            raise ValueError(f"loss_type '{loss_type}' not supported")

        return loss_value


    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'MeshToSpeechGan':
        """
        Function that instantiates the model from checkpoint or config
        """
        kwargs = instantiate_components_gan(cfg)
        if checkpoint is None:
            model = MeshToSpeechGan(cfg, **kwargs)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = MeshToSpeechGan.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **kwargs,
                **checkpoint_kwargs
            )
        return model


def discriminators_from_cfg(cfg):
    discriminators = {}
    for key, disc_cfg in cfg.items():
        if disc_cfg.type == "VcaDiscriminator":
            from .mesh2speech.Mesh2SpeechDiscriminator import VcaDiscriminator
            discriminator = VcaDiscriminator(disc_cfg)
        elif disc_cfg.type == "VcaUnconditionedDiscriminator":
            from .mesh2speech.Mesh2SpeechDiscriminator import VcaUnconditionedDiscriminator
            discriminator = VcaUnconditionedDiscriminator(disc_cfg)
        else: 
            raise NotImplementedError(f"Discriminator head type '{disc_cfg.type}' not implemented")
        discriminators[key] = discriminator 
    return discriminators


def instantiate_components_gan(cfg): 
    kwargs = instantiate_components(cfg)
    discriminators = discriminators_from_cfg(cfg.model.discriminators)
    kwargs["discriminators"] = discriminators
    return kwargs
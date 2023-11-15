"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, \
                       count_params, instantiate_from_config, \
                       ortho_subtract, ortho_l2loss, normalized_l2loss, gen_gradient_scaler, \
                       convert_attn_to_spatial_weight, calc_delta_loss, \
                       save_grid, chunk_list, patch_multi_embeddings, fix_emb_scales, \
                       halve_token_indices, double_token_indices, normalize_dict_values, masked_mean, \
                       resize_mask_for_feat_or_attn, mix_static_vk_embeddings, repeat_selected_instances, \
                       anneal_t, rand_annealed, calc_layer_subj_comp_k_or_v_ortho_loss

from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from evaluation.clip_eval import CLIPEvaluator, NoisedCLIPEvaluator
import copy
from functools import partial
import random
from safetensors.torch import load_file as safetensors_load_file
import sys

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,    # clip the range of denoised variables, not the CLIP model.
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 unfreeze_model=False,
                 model_lr=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 recon_loss_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 optimizer_type='AdamW',
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 use_layerwise_embedding=False,
                 use_ada_embedding=False,
                 composition_regs_iter_gap=-1,
                 static_embedding_reg_weight=0.,
                 ada_embedding_reg_weight=0.,
                 prompt_emb_delta_reg_weight=0.,
                 subj_comp_key_ortho_loss_weight=0.,
                 subj_comp_value_ortho_loss_weight=0.,
                 subj_comp_attn_complementary_loss_weight=0.,
                 mix_prompt_distill_weight=0.,
                 subj_attn_norm_distill_loss_scale=0.,
                 comp_fg_bg_preserve_loss_weight=0.,
                 fg_bg_complementary_loss_weight=0.,
                 fg_bg_mask_align_loss_weight=0.,
                 fg_bg_xlayer_consist_loss_weight=0.,
                 do_clip_teacher_filtering=False,
                 distill_deep_neg_prompt=None,
                 distill_deep_cfg_scale=0.,
                 use_background_token=False,
                 use_conv_attn=False,
                 # 'face portrait' is only valid for humans/animals. On objects, use_fp_trick will be ignored.
                 use_fp_trick=True,      
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings

        self.use_layerwise_embedding = use_layerwise_embedding
        self.N_LAYERS = 16 if self.use_layerwise_embedding else 1

        self.use_ada_embedding = (use_layerwise_embedding and use_ada_embedding)

        self.static_embedding_reg_weight = static_embedding_reg_weight
        self.ada_embedding_reg_weight    = ada_embedding_reg_weight

        self.composition_regs_iter_gap          = composition_regs_iter_gap
        self.prompt_emb_delta_reg_weight        = prompt_emb_delta_reg_weight
        self.subj_comp_key_ortho_loss_weight    = subj_comp_key_ortho_loss_weight
        self.subj_comp_value_ortho_loss_weight  = subj_comp_value_ortho_loss_weight
        self.subj_comp_attn_complementary_loss_weight = subj_comp_attn_complementary_loss_weight
        self.mix_prompt_distill_weight          = mix_prompt_distill_weight
        self.subj_attn_norm_distill_loss_scale  = subj_attn_norm_distill_loss_scale
        self.comp_fg_bg_preserve_loss_weight    = comp_fg_bg_preserve_loss_weight
        self.fg_bg_complementary_loss_weight    = fg_bg_complementary_loss_weight
        self.fg_bg_mask_align_loss_weight       = fg_bg_mask_align_loss_weight
        self.fg_bg_xlayer_consist_loss_weight   = fg_bg_xlayer_consist_loss_weight
        self.do_clip_teacher_filtering          = do_clip_teacher_filtering
        self.distill_deep_neg_prompt            = distill_deep_neg_prompt
        self.distill_deep_cfg_scale             = distill_deep_cfg_scale
        self.prompt_mix_scheme                  = 'mix_hijk'

        self.use_conv_attn                   = use_conv_attn
        self.use_background_token            = use_background_token
        # If use_conv_attn, the subject is well expressed, and use_fp_trick is unnecessary 
        # (actually harmful).
        self.use_fp_trick                    = use_fp_trick
        self.fg_mask_avail_ratio = 0

        self.cached_inits_available          = False
        self.cached_inits                    = None
        self.init_iter_flags()

        # Training flags. 
        # No matter wheter the scheme is layerwise or not,
        # as long as composition_regs_iter_gap > 0 and prompt_emb_delta_reg_weight > 0, 
        # do static comp delta reg.
        self.do_static_prompt_delta_reg = self.composition_regs_iter_gap > 0 \
                                            and self.prompt_emb_delta_reg_weight > 0
        # Is this for DreamBooth training? Will be overwritten in LatentDiffusion ctor.
        self.is_dreambooth                  = False

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.optimizer_type = optimizer_type
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
            self.warm_up_steps = scheduler_config.params.warm_up_steps
        else:
            self.scheduler = None
            self.warm_up_steps = 500

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.recon_loss_weight = recon_loss_weight
        self.unfreeze_model = unfreeze_model
        self.model_lr = model_lr

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # create_clip_evaluator() is called in main.py, so that we can specify device as cuda device.
    # We couldn't create clip_evaluator on cpu and then move it to cuda device, because 
    # NoisedCLIPEvaluator is not properly implemented to support this.
    def create_clip_evaluator(self, device):        
        self.clip_evaluator = CLIPEvaluator(device=device)
        for param in self.clip_evaluator.model.parameters():
            param.requires_grad = False

        self.num_total_teacher_filter_iters = 0
        self.num_teachable_iters = 0
        # A tiny number to avoid division by zero.
        self.num_total_reuse_filter_iters = 0.001
        self.num_reuse_teachable_iters = 0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        if path.endswith(".ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
        elif path.endswith(".safetensors"):
            sd = safetensors_load_file(path, device="cpu")
        else:
            print(f"Unknown checkpoint format: {path}")
            sys.exit(1)

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # self.loss_type: default 'l2'.
    def get_loss(self, pred, target, mean=True, loss_type=None):
        if loss_type is None:
            loss_type = self.loss_type

        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_recon': loss.mean().detach()})
        loss_recon = loss.mean() * self.recon_loss_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb.mean().detach()})

        loss = loss_recon + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss.mean().detach()})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def init_iter_flags(self):
        self.iter_flags = { 'calc_clip_loss':               False,
                            'do_normal_recon':              True,
                            'is_compos_iter':               False,
                            'do_mix_prompt_distillation':   False,
                            'do_ada_emb_delta_reg':         False,
                            # 'do_teacher_filter':        False,
                            # 'is_teachable':             False,
                            'use_background_token':         False,
                            'use_fp_trick':                 False,
                            'reuse_init_conds':             False,
                            'comp_init_with_fg_area':       False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    #LINK #shared_step
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.init_iter_flags()

        self.training_percent = self.global_step / self.trainer.max_steps

        # How many regularizations are done intermittently during the training iterations?
        cand_reg_types = []
        cand_reg_probs = []
        # do_mix_prompt_distillation implies do_ada_emb_delta_reg.
        # So if do_mix_prompt_distillation is enabled (mix_prompt_distill_weight > 0),
        # then no need to put do_ada_emb_delta_reg in cand_reg_types.
        # Otherwise, we need to put do_ada_emb_delta_reg in cand_reg_types.
        # There's only one reg type: 'do_mix_prompt_distillation' in cand_reg_types.
        # This structure is kept for possible future extensions.
        if self.mix_prompt_distill_weight > 0:
            cand_reg_types.append('do_mix_prompt_distillation')
            cand_reg_probs.append(1.)
        else:
            cand_reg_types.append('do_ada_emb_delta_reg')
            cand_reg_probs.append(1.)

        # NOTE: No need to have standalone ada prompt delta reg, 
        # since each prompt mix reg iter will also do ada prompt delta reg.

        N_CAND_REGS = len(cand_reg_types)
        cand_reg_probs = np.array(cand_reg_probs) / np.sum(cand_reg_probs)

        # If N_CAND_REGS == 0, then no intermittent regularizations, set the two flags to False.
        if N_CAND_REGS > 0 and self.composition_regs_iter_gap > 0 \
            and self.global_step % self.composition_regs_iter_gap == 0:
            # Alternate among the regularizations in cand_reg_types. 
            # If both do_ada_emb_delta_reg and do_mix_prompt_distillation,
            # then alternate between do_ada_emb_delta_reg and do_mix_prompt_distillation.
            # The two regularizations cannot be done in the same batch, as they require
            # different handling of prompts and are calculated by different loss functions.
            # reg_type_idx = (self.global_step // self.composition_regs_iter_gap) % N_CAND_REGS
            reg_type_idx = np.random.choice(N_CAND_REGS, p=cand_reg_probs)
            iter_reg_type     = cand_reg_types[reg_type_idx]
            if iter_reg_type   == 'do_ada_emb_delta_reg':
                self.iter_flags['do_mix_prompt_distillation']  = False
                self.iter_flags['do_ada_emb_delta_reg'] = True
            # do_mix_prompt_distillation => do_ada_emb_delta_reg = True.
            elif iter_reg_type == 'do_mix_prompt_distillation':
                self.iter_flags['do_mix_prompt_distillation']  = True
                self.iter_flags['do_ada_emb_delta_reg'] = True

            self.iter_flags['is_compos_iter'] = True
            # Always calculate clip loss during comp reg iterations, even if self.iter_flags['do_teacher_filter'] is False.
            # This is to monitor how well the model performs on compositionality.
            self.iter_flags['calc_clip_loss'] = True
            self.iter_flags['do_normal_recon']    = False

        if self.is_dreambooth:
            # DreamBooth uses ConcatDataset to make batch a tuple of train_batch and reg_batch.
            # train_batch: normal subject image recon. reg_batch: general class regularization.
            train_batch = batch[0]
            reg_batch   = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_reg, _ = self.shared_step(reg_batch)
            loss = loss_train + self.db_reg_weight * loss_reg
        else:            
            loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

# LatentDiffusion inherits from DDPM. So:
# LatentDiffusion.model = DiffusionWrapper(unet_config, conditioning_key, use_layerwise_embedding)
class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 personalization_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 # To do DreamBooth training, set is_dreambooth=True.
                 is_dreambooth=False,
                 *args, **kwargs):

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # conditioning_key: crossattn
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        # cond_stage_config is a dict:
        # {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'}
        # Not sure why it's compared with a string
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


        if not self.unfreeze_model:
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

            self.model.eval()
            self.model.train = disabled_train
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.is_dreambooth = is_dreambooth
        self.db_reg_weight  = 1.
        if not is_dreambooth:
            self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
            # embedding_manager.optimized_parameters(): string_to_static_embedder_dict, 
            # which maps custom tokens to embeddings
            for param in self.embedding_manager.optimized_parameters():
                param.requires_grad = True
            self.num_vectors_per_token = max(self.embedding_manager.token2num_vectors.values())
        else:
            # For DreamBooth.
            self.embedding_manager = None
            self.num_vectors_per_token = 1

        self.generation_cache = []
        self.generation_cache_img_colors = []
        self.cache_start_iter = 0
        self.num_cached_generations = 0
        
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.global_step == 0:
            self.create_clip_evaluator(next(self.parameters()).device)
            with torch.no_grad():
                self.uncond_context             = self.get_learned_conditioning([""] * 2)
                if (self.distill_deep_cfg_scale > 1) and (self.distill_deep_neg_prompt is not None):
                    # distill_deep_neg_context is generated with a batch size of 1. 
                    # Need to repeat it BLOCK_SIZE times to match the distillation batch size later.
                    distill_deep_neg_context   = self.get_learned_conditioning([self.distill_deep_neg_prompt])
                    # Only use the static embeddings of distill_deep_neg_prompt.
                    self.distill_deep_neg_context = distill_deep_neg_context[0]
                else:
                    self.distill_deep_neg_context   = None

        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")


    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        # num_timesteps_cond: 1
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        # cond_stage_trainable = True
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
            self.cond_stage_model = model
            
    
    def instantiate_embedding_manager(self, config, text_embedder):
        model = instantiate_from_config(config, text_embedder=text_embedder)

        if config.params.get("embedding_manager_ckpt", None): # do not load if missing OR empty string
            model.load(config.params.embedding_manager_ckpt)
        
        return model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    # cond_in: a batch of prompts like ['an illustration of a dirty z', ...]
    def get_learned_conditioning(self, cond_in, img_mask=None, randomize_clip_weights=True):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                # cond_in: a list of prompts: ['an illustration of a dirty z', 'an illustration of the cool z']
                # each prompt in c is encoded as [1, 77, 768].
                # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
                if randomize_clip_weights:
                    self.cond_stage_model.sample_last_layers_skip_weights()

                # c: [128, 77, 768]
                c = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)
                # c is tensor. So the following statement is False.
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
                
                c = fix_emb_scales(c, self.embedding_manager.placeholder_indices_fg, num_layers=self.N_LAYERS)
                # c = fix_emb_scales(c, self.embedding_manager.placeholder_indices_bg, num_layers=self.N_LAYERS)

                extra_info = { 
                                'use_layerwise_context': self.use_layerwise_embedding, 
                                'use_ada_context':       self.use_ada_embedding,
                                'use_conv_attn':         self.use_conv_attn,
                                # Default is False. Will be changed to True in forward() if necessary.
                                'ada_bp_to_unet':        False, 
                                # Setting up 'subj_indices' here is necessary for inference.
                                # During training, 'subj_indices' will be overwritten in p_losses().
                                # Although 'bg_indices' is usually not used during inference,
                                # we also set it up here just in case.
                                'subj_indices':          copy.copy(self.embedding_manager.placeholder_indices_fg),
                                'bg_indices':            copy.copy(self.embedding_manager.placeholder_indices_bg),
                             }
                
                if self.use_ada_embedding:
                    ada_embedder = self.get_layer_ada_conditioning
                    # Initialize the ada embedding cache, so that the subsequent calls to 
                    # self.get_layer_ada_embedding() will store the ada embedding 
                    # for each layer into the cache. 
                    # The cache will be used in calc_prompt_emb_delta_loss().
                    self.embedding_manager.reset_prompt_embedding_caches()
                    # The image mask here is used when computing Ada embeddings in embedding_manager.
                    # Do not consider mask on compositional reg iterations, as in such iters, 
                    # the original pixels (out of the fg_mask) do not matter and can freely take any values.
                    if self.iter_flags['is_compos_iter']:
                        img_mask = None

                    # img_mask is used by the ada embedding generator. 
                    # So we pass img_mask to embedding_manager here.
                    self.embedding_manager.set_img_mask(img_mask)
                    extra_info['ada_embedder'] = ada_embedder

                c = (c, cond_in, extra_info)
            else:
                c = self.cond_stage_model(cond_in)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(cond_in)

        return c

    # get_layer_ada_conditioning() is a callback function called iteratively by each layer in UNet.
    # It returns the conditioning embedding (ada embedding & other token embeddings -> clip encoder) 
    # for the current layer to UNet.
    def get_layer_ada_conditioning(self, c_in, layer_idx, layer_attn_components, time_emb, ada_bp_to_unet):
        # We don't want to mess with the pipeline of cond_stage_model.encode(), so we pass
        # c_in, layer_idx and layer_infeat directly to embedding_manager. They will be used implicitly
        # when embedding_manager is called within cond_stage_model.encode().
        self.embedding_manager.cache_layer_features_for_ada(layer_idx, layer_attn_components, time_emb, ada_bp_to_unet)
        # DO NOT call sample_last_layers_skip_weights() here, to make the ada embeddings are generated with 
        # CLIP skip weights consistent with the static embeddings.
        cond = self.cond_stage_model.encode(c_in, embedding_manager=self.embedding_manager)
        cond = fix_emb_scales(cond, self.embedding_manager.placeholder_indices_fg)
        # Cache the computed ada embedding of the current layer for delta loss computation.
        # Before this call, reset_prompt_embedding_caches() should have been called somewhere.
        self.embedding_manager.cache_ada_prompt_embedding(layer_idx, cond)
        return cond, self.embedding_manager.get_ada_emb_weight() #, self.embedding_manager.token_attn_weights

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    # k: key for the images, i.e., 'image'. k is not a number.
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if 'fg_mask' in batch:
            fg_mask = batch['fg_mask']
            fg_mask = fg_mask.unsqueeze(1).to(x.device)
            #fg_mask = F.interpolate(fg_mask, size=x.shape[-2:], mode='nearest')
        else:
            fg_mask = None

        if 'aug_mask' in batch:
            aug_mask = batch['aug_mask']
            aug_mask = aug_mask.unsqueeze(1).to(x.device)
            #img_mask = F.interpolate(img_mask, size=x.shape[-2:], mode='nearest')
        else:
            aug_mask = None

        if fg_mask is not None or aug_mask is not None:
            mask_dict = {'fg_mask': fg_mask, 'aug_mask': aug_mask}
        else:
            mask_dict = None

        encoder_posterior = self.encode_first_stage(x, mask_dict)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # conditioning_key: 'crossattn'.
        if self.model.conditioning_key is not None:
            if cond_key is None:
                # cond_stage_key: 'caption'.
                cond_key = self.cond_stage_key
            # first_stage_key: 'image'.
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    # batch.keys(): 'image', 'caption'.
                    # batch['caption']: 
                    # ['an illustration of a dirty z', 'an illustration of the cool z']
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            # cond_stage_trainable: True. force_c_encode: False.
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            #if bs is not None:
            #    c = c[:bs]
            #if bs is not None and c.shape[0] != bs:
            #    breakpoint()

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    # output: -1 ~ 1.
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as decode_first_stage() but without torch.no_grad() decorator
    # output: -1 ~ 1.
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, mask=None):
        if hasattr(self, "split_input_params"):     # False
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i], mask)
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x, mask)
        else:
            # Execute this statement only.
            return self.first_stage_model.encode(x, mask)

    # LatentDiffusion.shared_step() overloads DDPM.shared_step().
    # shared_step() is called in training_step() and (no_grad) validation_step().
    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # batch: { 'caption':               ['an illustration of a dirty z',                    
    #                                    'a depiction of a z'], 
    #          'subj_prompt_comp':     ['an illustration of a dirty z dancing with a boy', 
    #                                    'a depiction of a z kicking a punching bag'],
    #          'cls_prompt_single':     ['an illustration of a dirty person',          
    #                                    'a depiction of a person'],
    #                                    'a depiction of a person kicking a punching bag']
    #          'cls_prompt_comp'  :    ['an illustration of a dirty person dancing with a boy', 
    #                                    'a depiction of a person kicking a punching bag'],
    #          'image':   [2, 512, 512, 3] }
    # 'caption' is not named 'subj_prompt_single' to keep it compatible with older code.
    # ANCHOR[id=shared_step]
    def shared_step(self, batch, **kwargs):
        # captions = batch["caption"]
        # Encode noise as 4-channel latent features. Get prompts from batch. No gradient into here.
        # NOTE: captions (batch["caption"] or batch["caption_bg"])
        # are only for image reconstruction iterations.
        x, captions = self.get_input(batch, self.first_stage_key)

        batch_have_fg_mask       = batch['has_fg_mask']
        self.fg_mask_avail_ratio = batch_have_fg_mask.sum() / batch_have_fg_mask.shape[0]

        # If cached_inits_available, cached_inits are only used if do_mix_prompt_distillation = True.
        self.iter_flags['reuse_init_conds']  = (self.do_clip_teacher_filtering and self.iter_flags['do_mix_prompt_distillation'] \
                                                and self.cached_inits_available)

        # do_teacher_filter: If not reuse_init_conds and do_teacher_filtering, then we choose the better instance 
        # between the two in the batch, if it's above the usable threshold.
        # do_teacher_filtering and reuse_init_conds are mutually exclusive.
        self.iter_flags['do_teacher_filter'] = (self.do_clip_teacher_filtering and self.iter_flags['do_mix_prompt_distillation'] \
                                                and not self.iter_flags['reuse_init_conds'])

        # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
        # or traditional TI.        
        # do_ada_emb_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
        if self.do_static_prompt_delta_reg or self.iter_flags['do_mix_prompt_distillation']:
            # *_fp prompts are like "a face portrait of ...". They are advantageous over "a photo of ..."
            # when doing compositional mix regularization on humans/animals.
            # For objects (broad_class != 1), even if use_fp_trick = True, 
            # *_fp prompts are not available in batch, so fp_trick won't be used.
            # 'subj_prompt_single_fp' in batch is True <=> (broad_class == 1 and use_fp_trick).
            # If do_mix_prompt_distillation but broad_class == 0 or 2, this statement is False, and 
            # used prompts will be 'subj_prompt_comp', 'cls_prompt_single', 'cls_prompt_comp'...
            p_use_fp_trick = 0.9
            self.iter_flags['use_fp_trick'] = self.iter_flags['do_mix_prompt_distillation'] and self.use_fp_trick \
                                                and 'subj_prompt_single_fp' in batch \
                                                and random.random() < p_use_fp_trick

            # Slightly larger than 0.5, since comp_init_with_fg_area is disabled under reuse_init_conds.
            # So in all distillation iterations, comp_init_with_fg_area percentage will be around 0.5.
            p_comp_init_with_fg_area = 0.7
            # If reuse_init_conds, comp_init_with_fg_area may be set to True later
            # if the previous iteration has comp_init_with_fg_area = True.
            self.iter_flags['comp_init_with_fg_area'] = self.iter_flags['do_mix_prompt_distillation'] \
                                                          and not self.iter_flags['reuse_init_conds'] \
                                                          and self.fg_mask_avail_ratio > 0 \
                                                          and random.random() < p_comp_init_with_fg_area

            # Mainly use background token on recon iters.
            # To avoid the backgound token taking too much of the foreground, 
            # we only use the background token on 90% of the training images, to 
            # force the foreground token to focus on the whole image.
            if not self.iter_flags['is_compos_iter']:
                p_use_background_token  = 0.9
            else:
                # When do_mix_prompt_distillation, we don't use background token.
                p_use_background_token  = 0

            # Only use_background_token on recon iters.
            # No need to check do_mix_prompt_distillation, because if do_mix_prompt_distillation,
            # in most iterations p_use_background_token = 0, except for 50% of the iterations when
            # comp_init_with_fg_area = True.
            self.iter_flags['use_background_token'] = self.use_background_token \
                                                        and random.random() < p_use_background_token
                        
            if self.iter_flags['use_fp_trick'] and self.iter_flags['use_background_token']:
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_fp_bg'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_fp_bg'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_fp_bg'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_fp_bg'
            # use_fp_trick but not use_background_token.
            elif self.iter_flags['use_fp_trick']:
                # Never use_fp_trick for recon iters. So no need to have "caption_fp" or "caption_fp_bg".
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_fp'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_fp'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_fp'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_fp'
            # not use_fp_trick and use_background_token.
            elif self.iter_flags['use_background_token']:
                captions = batch["caption_bg"]
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_bg'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_bg'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_bg'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_bg'
            # Either do_mix_prompt_distillation but not (use_fp_trick_iter and broad_class == 1), 
            # or recon iters (not do_mix_prompt_distillation) and not use_background_token 
            # We don't use_fp_trick on training images. use_fp_trick is only for compositional regularization.
            else:
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp'
                CLS_PROMPT_COMP    = 'cls_prompt_comp'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single'

            # Each prompt_comp consists of multiple prompts separated by "|".
            # Split them into a list of subj_comp_prompts/cls_comp_prompts.
            subj_single_prompts = batch[SUBJ_PROMPT_SINGLE]
            cls_single_prompts  = batch[CLS_PROMPT_SINGLE]
            subj_comp_prompts = []
            for prompt_comp in batch[SUBJ_PROMPT_COMP]:
                subj_comp_prompts.append(prompt_comp.split("|"))
            cls_comp_prompts = []
            for prompt_comp in batch[CLS_PROMPT_COMP]:
                cls_comp_prompts.append(prompt_comp.split("|"))
            # REPEATS: how many prompts correspond to each image.
            REPEATS = len(subj_comp_prompts[0])
            if REPEATS == 1 or self.iter_flags['do_mix_prompt_distillation'] or self.iter_flags['do_ada_emb_delta_reg']:
                # When this iter computes ada prompt delta loss / prompt mixing loss, 
                # only use the first of the composition prompts (in effect num_compositions_per_image=1),
                # otherwise it will use more than 40G RAM.
                subj_comp_prompts = [ prompts[0] for prompts in subj_comp_prompts ]
                cls_comp_prompts  = [ prompts[0] for prompts in cls_comp_prompts ]
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
            else:
                subj_comp_prompts2 = []
                cls_prompt_comp2   = []
                # Suppose R = num_compositions_per_image, and B the batch size.
                # Each of subj_comp_prompts, cls_comp_prompts is like [ (p1_1,..., p1_R), ..., (pB_1,..., pB_R) ].
                # Interlace the list of composition prompt lists into one list:
                # [ p1_1, p2_1, ..., pB_1, p1_2, p2_2, ..., pB_2, ..., p1_R, p2_R, ..., pB_R ].
                # Interlacing makes it easy to choose the first B prompts (just as for a normal batch). 
                # Do not simply concatenate along B.
                for prompts in zip(*subj_comp_prompts):
                    subj_comp_prompts2 += prompts
                for prompts in zip(*cls_comp_prompts):
                    cls_prompt_comp2 += prompts
                subj_comp_prompts = subj_comp_prompts2
                cls_comp_prompts  = cls_prompt_comp2
                captions = captions * REPEATS
                cls_single_prompts = cls_single_prompts * REPEATS
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
        else:
            delta_prompts = None
            if self.iter_flags['use_background_token']:
                captions = batch["caption_bg"]

        if 'aug_mask' in batch:
            img_mask = batch['aug_mask']
            img_mask = img_mask.unsqueeze(1).to(x.device)
            img_mask = F.interpolate(img_mask, size=x.shape[-2:], mode='nearest')
        else:
            img_mask = None

        if 'fg_mask' in batch:
            fg_mask = batch['fg_mask']
            fg_mask = fg_mask.unsqueeze(1).to(x.device)
            fg_mask = F.interpolate(fg_mask, size=x.shape[-2:], mode='nearest')
        else:
            assert self.fg_mask_avail_ratio == 0
            fg_mask = None

        self.iter_flags['img_mask'] = img_mask
        self.iter_flags['fg_mask']  = fg_mask
        self.iter_flags['batch_have_fg_mask']   = batch_have_fg_mask
        self.iter_flags['delta_prompts']        = delta_prompts

        # reuse_init_conds, discard the prompts offered in shared_step().
        if self.iter_flags['reuse_init_conds']:
            # cached_inits['delta_prompts'] is a tuple of 4 lists. No need to split them.
            delta_prompts = self.cached_inits['delta_prompts']
            self.iter_flags['img_mask']                 = self.cached_inits['img_mask']
            self.iter_flags['fg_mask']                  = self.cached_inits['fg_mask']
            self.iter_flags['batch_have_fg_mask']       = self.cached_inits['batch_have_fg_mask']
            self.iter_flags['filtered_fg_mask']         = self.cached_inits['filtered_fg_mask']
            self.iter_flags['use_background_token']     = self.cached_inits['use_background_token']
            self.iter_flags['comp_init_with_fg_area']   = self.cached_inits['comp_init_with_fg_area']

        loss = self(x, captions, **kwargs)

        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x, captions, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        # Use >=, i.e., assign decay in all iterations after the first 100.
        # This is in case there are skips of iterations of global_step 
        # (shouldn't happen but just in case).

        if self.model.conditioning_key is not None:
            assert captions is not None
            # get_learned_conditioning(): convert captions to a [B, 77, 768] tensor.
            if self.cond_stage_trainable:
                # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
                # or traditional TI.
                # do_ada_emb_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
                # captions: plain prompts like ['an illustration of a dirty z', 'an illustration of the cool z']
                if self.do_static_prompt_delta_reg or self.iter_flags['do_mix_prompt_distillation']:
                    # reuse_init_conds, discard the prompts offered in shared_step().
                    if self.iter_flags['reuse_init_conds']:
                        # cached_inits['delta_prompts'] is a tuple of 4 lists. No need to split them.
                        delta_prompts = self.cached_inits['delta_prompts']
                        # cached_inits will be used in p_losses(), 
                        # so don't set cached_inits_available to False yet.
                    else:
                        # iter_flags['delta_prompts'] is a tuple of 4 lists. No need to split them.
                        delta_prompts = self.iter_flags['delta_prompts']

                    subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = delta_prompts

                    #if self.iter_flags['use_background_token']:
                    #    print(subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

                    ORIG_BS  = len(x)
                    N_EMBEDS = ORIG_BS * self.N_LAYERS
                    
                    if self.iter_flags['do_mix_prompt_distillation'] or self.iter_flags['do_ada_emb_delta_reg']:
                        # In distillation iterations, the full batch size cannot fit in the RAM.
                        # So we have to halve the batch size. 
                        # BLOCK_SIZE is at least 1. So if ORIG_BS == 1, then BLOCK_SIZE = 1.
                        BLOCK_SIZE  = max(ORIG_BS // 2, 1)
                        # Only keep the first half of batched prompts to save RAM.
                        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                            subj_single_prompts[:BLOCK_SIZE], subj_comp_prompts[:BLOCK_SIZE], \
                            cls_single_prompts[:BLOCK_SIZE],  cls_comp_prompts[:BLOCK_SIZE]
                    else:
                        BLOCK_SIZE = ORIG_BS

                    # Otherwise, do_static_prompt_delta_reg but not do_ada_emb_delta_reg.
                    # Do not halve the batch. BLOCK_SIZE = ORIG_BS = 2.
                    # 8 prompts will be fed into get_learned_conditioning().
                                            
                    # If not do_ada_emb_delta_reg, we still compute the static embeddings 
                    # of the 4 types of prompts, to compute static delta loss. 
                    # But now there are 8 prompts (4 * ORIG_BS = 8), as the batch is not halved.
                    delta_prompts = subj_single_prompts + subj_comp_prompts \
                                    + cls_single_prompts + cls_comp_prompts
                    #print(delta_prompts)
                    # breakpoint()
                    # c_static_emb: the static embeddings [4 * N_EMBEDS, 77, 768], 
                    # 4 * N_EMBEDS = 4 * ORIG_BS * N_LAYERS,
                    # whose layer dimension (N_LAYERS) is tucked into the batch dimension. 
                    # delta_prompts: the concatenation of
                    # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
                    # extra_info: a dict that contains extra info.
                    c_static_emb, _, extra_info = self.get_learned_conditioning(delta_prompts, 
                                                                                img_mask=self.iter_flags['img_mask'])
                    subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
                        c_static_emb.chunk(4)

                    # By default, ada_bp_to_unet is False. Will be changed to True if do_mix_prompt_distillation.
                    extra_info['ada_bp_to_unet'] = False

                    # *_2b: two sub-blocks of the batch (e.g., subj single prompts and subj comp prompts).
                    # *_1b: one sub-block  of the batch (e.g., only subj single prompts).
                    extra_info['subj_indices_2b'] = copy.copy(self.embedding_manager.placeholder_indices_fg)
                    # Only keep the first half (for single prompts), as the second half is the same 
                    # (for comp prompts, differs at the batch index, but the token index is identical).
                    # placeholder_indices_fg is only for (subj_single_prompts, subj_comp_prompts), since
                    # the placeholder token doesn't appear in the class prompts. 
                    # Now we take the first half of placeholder_indices_fg, so that 
                    # they only account for the subject single prompt, but they are also 
                    # applicable to the other 3 types of prompts as they are all aligned 
                    # at the beginning part of the prompts.
                    extra_info['subj_indices_1b'] = halve_token_indices(extra_info['subj_indices_2b'])
                    subj_indices_half_N  = extra_info['subj_indices_1b'][1]
                    # The subject is represented with a multi-embedding token. The corresponding tokens
                    # in the class prompts are "class , , ,", 
                    # therefore the embeddings of "," need to be patched.
                    # BUG: if the batch size of a mix batch > 4, then the subj_indices_half_N
                    # corresponds to the indices in more than one instance. But patch_multi_embeddings()
                    # treat the indices as if they are always in the same instance.
                    # len(subj_indices_half_N): embedding number of the subject token.
                    if len(subj_indices_half_N) > 1:
                        cls_single_emb = patch_multi_embeddings(cls_single_emb, subj_indices_half_N)
                        cls_comp_emb   = patch_multi_embeddings(cls_comp_emb,   subj_indices_half_N)

                    # In mix reg iters, background tokens only appear 10% of the time 
                    # to provide delta reg on ada embeddings of bg tokens.
                    if self.iter_flags['use_background_token']:
                        # Sometimes (when bg_init_words is not speicified), all 4 types of prompts 
                        # have the same background token. Then placeholder_indices_bg == bg_indices_4b.
                        # Otherwise, 'bg_indices_4b' is absent in extra_info.
                        if self.embedding_manager.placeholder_indices_bg[0].unique().numel() == 4 * BLOCK_SIZE:
                            extra_info['bg_indices_4b'] = copy.copy(self.embedding_manager.placeholder_indices_bg)
                            extra_info['bg_indices_2b'] = halve_token_indices(extra_info['bg_indices_4b'])
                        else:
                            extra_info['bg_indices_2b'] = copy.copy(self.embedding_manager.placeholder_indices_bg)

                        extra_info['bg_indices_1b'] = halve_token_indices(extra_info['bg_indices_2b'])
                        # bg_indices_half_N: the first half batch of background token indices (the token dim)
                        bg_indices_half_N = extra_info['bg_indices_2b'][1]
                        # Patch background embeddings when the number of background embeddings > 1.
                        if len(bg_indices_half_N) > 1:
                            cls_single_emb = patch_multi_embeddings(cls_single_emb, bg_indices_half_N)
                            cls_comp_emb   = patch_multi_embeddings(cls_comp_emb,   bg_indices_half_N)
                    else:
                        extra_info['bg_indices_4b'] = None
                        extra_info['bg_indices_2b'] = None
                        extra_info['bg_indices_1b'] = None

                    # These embeddings are patched. So combine them back into c_static_emb.
                    c_static_emb = torch.cat([subj_single_emb, subj_comp_emb, 
                                              cls_single_emb, cls_comp_emb], dim=0)
                    extra_info['c_static_emb_4b'] = c_static_emb

                    # if do_ada_emb_delta_reg, then do_mix_prompt_distillation 
                    # may be True or False, depending whether mix reg is enabled.
                    if self.iter_flags['do_mix_prompt_distillation']:
                        # c_in2 = delta_prompts is used to generate ada embeddings.
                        # c_in2: subj_single_prompts + subj_comp_prompts + cls_single_prompts + cls_comp_prompts
                        # The cls_single_prompts/cls_comp_prompts within c_in2 will only be used to 
                        # generate ordinary prompt embeddings, i.e., 
                        # it doesn't contain subject token, and no ada embedding will be injected by embedding manager.
                        # Instead, subj_single_emb, subj_comp_emb and subject ada embeddings 
                        # are manually mixed into their embeddings.
                        c_in2 = delta_prompts
                        extra_info['iter_type']      = self.prompt_mix_scheme   # 'mix_hijk'
                        # Set ada_bp_to_unet to False will reduce performance.
                        extra_info['ada_bp_to_unet'] = True

                        # The prompts are either (subj single, subj comp, cls single, cls comp) or
                        # (subj comp, subj comp, cls comp, cls comp) if do_teacher_filter. 
                        # So the first 2 sub-blocks always contain the subject/background tokens, and we use *_2b.
                        extra_info['subj_indices'] = extra_info['subj_indices_2b']
                        extra_info['bg_indices']   = extra_info['bg_indices_2b']                            

                        # In distillation iters, at 50% chance, apply positive prompts and deep_neg_context. 
                        #              At the other 50% chance, apply only the positive prompts.
                        if (self.distill_deep_neg_context is not None) and (random.random() < 0.5):
                            extra_info['deep_neg_context'] = self.distill_deep_neg_context.repeat(BLOCK_SIZE * 4, 1, 1)
                            extra_info['deep_cfg_scale']   = self.distill_deep_cfg_scale

                    # This iter is a simple ada prompt delta loss iter, without prompt mixing loss. 
                    # This branch is reached only if prompt mixing is not enabled.
                    # "and not self.iter_flags['do_mix_prompt_distillation']" is redundant, because it's at an "elif" branch.
                    # Kept for clarity. 
                    elif self.iter_flags['do_ada_emb_delta_reg'] and not self.iter_flags['do_mix_prompt_distillation']:
                        # Do ada prompt delta loss in this iteration. 
                        c_in2         = delta_prompts
                        # c_in2 consists of four types of prompts: 
                        # subj_single, subj_comp, cls_single, cls_comp.
                        extra_info['iter_type']     = 'do_ada_emb_delta_reg'
                        # The prompts are either (subj single, subj comp, cls single, cls comp) or
                        # (subj comp, subj comp, cls comp, cls comp) if do_teacher_filter. 
                        # So the first 2 sub-blocks always contain the subject/background tokens, and we use *_2b.
                        extra_info['subj_indices']  = extra_info['subj_indices_2b']
                        extra_info['bg_indices']    = extra_info['bg_indices_2b']    

                    else:
                        # do_normal_recon. The original scheme. 
                        extra_info['iter_type']      = 'normal_recon'
                        # Use the original "captions" prompts and embeddings.
                        # When num_compositions_per_image > 1, subj_single_prompts contains repeated prompts,
                        # so we only keep the first N_EMBEDS embeddings and the first ORIG_BS prompts.
                        c_in2         = captions
                        # subj_single_emb has been patched above.
                        c_static_emb  = subj_single_emb

                        # The prompts used to compute the static embeddings are 
                        # (subj single, subj comp, cls single, cls comp).
                        # But only the subj single block is used for recon.
                        # The blocks as input to get_learned_conditioning() are not halved. 
                        # So BLOCK_SIZE = ORIG_BS = 2. Therefore, for the two instances, we use *_1b.
                        extra_info['subj_indices'] = extra_info['subj_indices_1b']
                        extra_info['bg_indices']   = extra_info['bg_indices_1b']

                        # In normal_recon iters, at 50% chance, apply positive prompts and deep_neg_context. 
                        #              At the other 50% chance, apply only the positive prompts.
                        if (self.distill_deep_neg_context is not None) and (random.random() < 0.5):
                            extra_info['deep_neg_context'] = self.distill_deep_neg_context.repeat(ORIG_BS, 1, 1)
                            extra_info['deep_cfg_scale']   = self.distill_deep_cfg_scale

                    extra_info['cls_comp_prompts']   = cls_comp_prompts
                    extra_info['cls_single_prompts'] = cls_single_prompts
                    # 'delta_prompts' is only used in comp_prompt_mix_reg iters. 
                    # Keep extra_info['delta_prompts'] and iter_flags['delta_prompts'] the same structure.
                    # (Both are tuples of 4 lists. But iter_flags['delta_prompts'] may contain more prompts
                    # than those actually used in this iter.)
                    # iter_flags['delta_prompts'] is not used in p_losses(). Keep it for debugging purpose.
                    extra_info['delta_prompts']      = (subj_single_prompts, subj_comp_prompts, \
                                                        cls_single_prompts,  cls_comp_prompts)

                    # Restore the placeholder_indces of the embedding_manager according to the original prompts.
                    self.embedding_manager.set_placeholder_indices(extra_info['subj_indices'], 
                                                                   extra_info['bg_indices'])
                    
                    # c_static_emb is the full set of embeddings of subj_single_prompts, subj_comp_prompts, 
                    # cls_single_prompts, cls_comp_prompts. 
                    # c_static_emb: [64, 77, 768]                    
                    cond = (c_static_emb, c_in2, extra_info)
                else:
                    # Not (self.do_static_prompt_delta_reg or 'do_mix_prompt_distillation').
                    # That is, non-compositional iter, or recon iter without static delta loss. 
                    # Keep the tuple cond unchanged. prompts: subject single.
                    cond = self.get_learned_conditioning(captions)
                    # cond[2]: extra_info. Here is only reached when do_static_prompt_delta_reg = False.
                    # Either prompt_emb_delta_reg_weight == 0 (ablation) or 
                    # it's called by self.validation_step().
                    assert self.iter_flags['do_normal_recon']
                    cond[2]['iter_type'] = 'normal_recon'

                    # In recon iter without static delta loss, 
                    # At 50% chance,           apply positive prompts and deep_neg_context. 
                    # At the other 50% chance, apply only the positive prompts.
                    if (self.distill_deep_neg_context is not None) and (random.random() < 0.5):
                        extra_info['deep_neg_context'] = self.distill_deep_neg_context.repeat(ORIG_BS, 1, 1)
                        extra_info['deep_cfg_scale']   = self.distill_deep_cfg_scale

                # cond[2]: extra_info. 
                cond[2]['use_background_token'] = self.iter_flags['use_background_token']

            # shorten_cond_schedule: False. Skipped.
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                # q_sample() is only called during training. 
                # q_sample() calls apply_model(), which estimates the latent code of the image.
                cond = self.q_sample(x_start=captions, t=tc, noise=torch.randn_like(captions.float()))

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        return self.p_losses(x, cond, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    # apply_model() is called both during training and inference.
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # has split_input_params: False.
        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            # Only execute this sentence.
            # self.model: DiffusionWrapper -> 
            # self.model.diffusion_model: ldm.modules.diffusionmodules.openaimodel.UNetModel
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # subj_indices, bg_indices: the indices of the subject/background tokens in the prompts.
    # Sometimes the prompts changed after generating the static embeddings, 
    # so in such cases we need to manually specify these indices. If they are not provided (None),
    # then the indices stored in embedding_manager are not updated.
    # do_pixel_recon: return denoised images. This is not the iter_type 'do_normal_recon'.
    # if do_pixel_recon and cfg_scale > 1, apply classifier-free guidance. 
    # unet_has_grad: when returning do_pixel_recon (e.g. to select the better instance by smaller clip loss), 
    # to speed up, no BP is done on these instances, so unet_has_grad=False.
    def guided_denoise(self, x_start, noise, t, cond, 
                       subj_indices=None, bg_indices=None,
                       inj_noise_t=None, 
                       unet_has_grad=True, do_pixel_recon=False, cfg_scales=None):
        
        if inj_noise_t is not None:
            # We can choose to add amount of noises different from t.
            x_noisy = self.q_sample(x_start=x_start, t=inj_noise_t, noise=noise)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        self.embedding_manager.set_placeholder_indices(subj_indices, bg_indices)
        # model_output is the predicted noise.
        # if not unet_has_grad, we save RAM by not storing the computation graph.
        if not unet_has_grad:
            with torch.no_grad():
                model_output = self.apply_model(x_noisy, t, cond)
        else:
            # if unet_has_grad, we don't have to take care of embedding_manager.force_grad.
            # Subject embeddings will naturally have gradients.
            model_output = self.apply_model(x_noisy, t, cond)

        # Save ada embeddings generated during apply_model(), to be used in delta loss. 
        # Otherwise it will be overwritten by uncond denoising.
        if self.embedding_manager.ada_prompt_embedding_cache is not None:
            # ada_embeddings: [4, 16, 77, 768]
            ada_embeddings = torch.stack(self.embedding_manager.ada_prompt_embedding_cache, dim=1)
        else:
            ada_embeddings = None

        # Get model output of both conditioned and uncond prompts.
        # Unconditional prompts and reconstructed images are never involved in optimization.
        if do_pixel_recon:
            with torch.no_grad():
                x_start_ = x_start.chunk(2)[0]
                noise_   = noise.chunk(2)[0]
                t_       = t.chunk(2)[0]
                # For efficiency, x_start_, noise_ and t_ are of half-batch size,
                # and only compute model_output_uncond on half of the batch, 
                # as the second half (mix single, mix comp) is generated under the same initial conditions 
                # (only differ on prompts, but uncond means no prompts).
                x_noisy_ = self.q_sample(x_start=x_start_, t=t_, noise=noise_)
                # Clear the cached placeholder indices, as they are for conditional embeddings.
                self.embedding_manager.clear_placeholder_indices()
                # Do not apply deep neg prompts on unconditional embeddings. 
                # It will cancel out the effect of unconditional embeddings.
                cond[2]['disable_deep_neg_context'] = True
                # self.uncond_context: precomputed unconditional embeddings.
                model_output_uncond = self.apply_model(x_noisy_, t_, self.uncond_context)
                # model_output_uncond: [2, 4, 64, 64] -> [4, 4, 64, 64]
                model_output_uncond = model_output_uncond.repeat(2, 1, 1, 1)
                cond[2]['disable_deep_neg_context'] = False
            # Classifier-free guidance to make the contents in the 
            # generated images more pronounced => smaller CLIP loss.
            if cfg_scales is not None:
                cfg_scales = cfg_scales.view(-1, 1, 1, 1)
                pred_noise = model_output * cfg_scales - model_output_uncond * (cfg_scales - 1)
            else:
                pred_noise = model_output

            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=pred_noise)
        else:
            x_recon = None
        
        return model_output, x_recon, ada_embeddings

    # Release part of the computation graph on unused instances to save RAM.
    def release_plosses_intermediates(self, local_vars):
        for k in ('model_output', 'x_recon', 'clip_images_code'):
            if k in local_vars:
                del local_vars[k]

        cond = local_vars['cond']
        for k in ('unet_feats', 'unet_attns', 'unet_attnscores', 'unet_ks', 'unet_vs'):
            if k in cond[2]:
                del cond[2][k]

    # t: steps.
    # cond: (c_static_emb, c_in, extra_info). c_in is the textual prompts. 
    # extra_info: a dict that contains 'ada_embedder' and other fields. 
    # ada_embedder: a function to convert c_in to ada embeddings.
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, cond, t, noise=None):
        # If noise is not None, then use the provided noise.
        # Otherwise, generate noise randomly.
        noise = default(noise, lambda: torch.randn_like(x_start))
        #print(cond[1])

        img_mask            = self.iter_flags['img_mask']
        fg_mask             = self.iter_flags['fg_mask']
        batch_have_fg_mask  = self.iter_flags['batch_have_fg_mask']
        filtered_fg_mask    = self.iter_flags.get('filtered_fg_mask', None)

        cfg_scales_for_clip_loss = None
        c_static_emb, c_in, extra_info = cond

        # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
        # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
        # (img_mask is not used in the prompt-guided cross-attention layers).
        # Don't consider img_mask in compositional iterations. Because in compositional iterations,
        # the original images don't play a role (even if comp_init_with_fg_area,
        # we still don't consider the actual pixels out of the subject areas, so img_mask doesn't matter).
        extra_info['img_mask']  = img_mask if not self.iter_flags['is_compos_iter'] else None

        inj_noise_t = None

        if self.iter_flags['is_compos_iter']:
            # Only reuse_init_conds if do_mix_prompt_distillation.
            if self.iter_flags['reuse_init_conds']:
                # If self.iter_flags['reuse_init_conds'], we use the cached x_start and cond.
                # cond is already organized as (subj single, subj comp, mix single, mix comp). 
                # No need to manipulate.
                # noise will be kept as the sampled random noise at the beginning of p_losses(). 
                x_start = self.cached_inits['x_start']
                prev_t  = self.cached_inits['t']
                # Avoid the next mix iter to still use the cached inits.
                self.cached_inits_available = False
                self.cached_inits = None
                # reuse init iter takes a smaller cfg scale, as in the second denoising step, 
                # a particular scale tend to make the cfg-denoised mixed images more dissimilar 
                # to the subject images than in the first denoising step. 

                # x_start is like (x1, x2, x1, x2) (x1, x2 repeated in the previous distillation iter).
                # x1, x2 are different, but they are generated with the same initial noise in the 
                # previous reconstruction. This is desired as a simulation of a multi-step inference process.
                BLOCK_SIZE  = max(x_start.shape[0] // 4, 1)
                # Randomly choose t from the middle 400-700 timesteps, 
                # so as to match the once-denoised x_start.
                # generate the full batch size of t, but actually only use the first block of BLOCK_SIZE.
                # This is to make the code consistent with the non-comp case and avoid unnecessary confusion.
                t_mid = torch.randint(int(self.num_timesteps * 0.4), int(self.num_timesteps * 0.7), 
                                      (x_start.shape[0],), device=x_start.device)
                # t_upperbound: old t - 150. That is, at least 150 steps away from the previous t.
                t_upperbound = prev_t - int(self.num_timesteps * 0.15)
                # t should be at least 150 steps away from the previous t, 
                # so that the noise level is sufficiently different.
                t = torch.minimum(t_mid, t_upperbound)
                # Decrease t slightly to decrease noise amount and preserve more semantics.
                inj_noise_t = anneal_t(t, self.training_percent, self.num_timesteps, ratio_range=(0.8, 1.0))

            else:
                # Fresh compositional iter. May do teacher filtering.
                t_tail = torch.randint(int(self.num_timesteps * 0.8), self.num_timesteps, (x_start.shape[0],), device=x_start.device)
                t = t_tail
                # x_start is of ORIG_BS = 2. So BLOCK_SIZE=1.
                # Randomly choose t from the largest 150 timesteps, so as to match the completely noisy x_start.
                BLOCK_SIZE  = max(x_start.shape[0] // 2, 1)
                # At 60% of the chance, randomly initialize x_start and t. Note the batch size is still 2 here.
                # At 40% of the chance, use a noisy x_start based on the training images. 
                # This may help the model ignore the background in the training images given prompts, 
                # i.e., give prompts higher priority over the background.

                # If reuse_init_conds, and the previous iter has comp_init_with_fg_area=True, then 
                # the current iter will also have comp_init_with_fg_area=True.
                if self.iter_flags['comp_init_with_fg_area'] and self.fg_mask_avail_ratio > 0 \
                  and not self.iter_flags['reuse_init_conds']:
                    # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
                    # Therefore, using fg_mask for comp_init_with_fg_area will force the model remember 
                    # the background in the training images, which is not desirable.
                    # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0.
                    # fg_mask is 4D. So expand batch_have_fg_mask to 4D.
                    filtered_fg_mask    = fg_mask.to(x_start.dtype) * batch_have_fg_mask.view(-1, 1, 1, 1)
                    # At background, fill x_start with random values (100% noise).
                    # If no fg_mask is available, then fg_mask_or_allone is all 1, i.e., 
                    # use the whole image to initialize x_start.
                    x_start_origsize = torch.where(filtered_fg_mask.bool(), x_start, torch.randn_like(x_start))
                    # Scale down the fg area to 60%-100% of the original size, to avoid it dominating the whole image.
                    fg_rand_scale = np.random.uniform(0.6, 1.0)
                    # Resize x_start_origsize and filtered_fg_mask by rand_scale. They have different numbers of channels,
                    # so we need to concatenate them at dim 1, and then resize.
                    x_mask = torch.cat([x_start_origsize, filtered_fg_mask], dim=1)
                    x_mask_scaled = F.interpolate(x_mask, scale_factor=fg_rand_scale, mode='bilinear', align_corners=False)

                    # Pad filtered_fg_mask_scaled to the original size, with left/right padding roughly equal
                    pad_w1 = int((x_start.shape[3] - x_mask_scaled.shape[3]) / 2)
                    pad_w2 =      x_start.shape[3] - x_mask_scaled.shape[3] - pad_w1
                    pad_h1 = int((x_start.shape[2] - x_mask_scaled.shape[2]) / 2)
                    pad_h2 =      x_start.shape[2] - x_mask_scaled.shape[2] - pad_h1
                    x_mask_scaled_padded = F.pad(x_mask_scaled, 
                                                 (pad_w1, pad_w2, pad_h1, pad_h2),
                                                 mode='constant', value=0)

                    x_start_scaled_padded, filtered_fg_mask = x_mask_scaled_padded[:, :4], x_mask_scaled_padded[:, 4:]

                    x_start = torch.where(filtered_fg_mask.bool(), x_start_scaled_padded, torch.randn_like(x_start))
                    # Gradually increase the noise amount from 0.25 to 0.5.
                    fg_noise_amount = rand_annealed(self.training_percent, final_percent=1, mean_range=(0.25, 0.5))
                    # At foreground, keep 50% of the original x_start values and add 50% noise. 
                    x_start = torch.randn_like(x_start) * fg_noise_amount + x_start * (1 - fg_noise_amount)
                else:
                    x_start.normal_()

            if not self.iter_flags['do_mix_prompt_distillation']:
                # Only do ada delta loss. This usually won't happen unless mix_prompt_distill_weight = 0.
                # Generate a batch of 4 instances with the same initial x_start, noise and t.
                # This doubles the batch size to 4, if bs=2.
                x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                t       = t[:BLOCK_SIZE].repeat(4)
                # Calculate CLIP score only for image quality evaluation
                cfg_scales_for_clip_loss = torch.ones_like(t) * 5

                # Update masks to be a 4-fold structure.
                img_mask, fg_mask, batch_have_fg_mask = \
                    repeat_selected_instances(slice(0, BLOCK_SIZE), 4, img_mask, fg_mask, batch_have_fg_mask)
                self.fg_mask_avail_ratio = batch_have_fg_mask.float().mean()

            else:
                # do_mix_prompt_distillation. We need to compute CLIP scores for teacher filtering.
                # Set up cfg configs for guidance to denoise images, which are input to CLIP.
                # Teachers are slightly more aggressive, to increase the teachable fraction.                
                cfg_scale_for_teacher  = 6
                cfg_scale_for_student  = 5
                cfg_scales_for_teacher   = torch.ones(BLOCK_SIZE*2, device=x_start.device) * cfg_scale_for_teacher
                cfg_scales_for_student   = torch.ones(BLOCK_SIZE*2, device=x_start.device) * cfg_scale_for_student
                cfg_scales_for_clip_loss = torch.cat([cfg_scales_for_student, cfg_scales_for_teacher], dim=0)

                # First iteration of a two-iteration do_mix_prompt_distillation.
                # Generate a batch of 4 instances in *two* sets, each set with the *2* instances. 
                # Within each set, the same initial x_start, noise and t are used.
                # Then filter, find the best teachable set (if any) and pass to the recursive iteration.
                # If no teachable set is found, skip recursion and the prompt mix reg.
                # Note x_start[0] = x_start[2] != x_start[1] = x_start[3].
                # That means, instances are arranged as: 
                # (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
                # This doubles the batch size to 4, if bs=2.
                if self.iter_flags['do_teacher_filter']:
                    x_start = x_start.repeat(2, 1, 1, 1)
                    # noise and t are repeated in the same way as x_start for two sets. 
                    # Set 1 is for two subj comp instances, set 2 is for two mix comp instances.
                    # Noise and t are different between the two instances within one set.
                    noise   = noise.repeat(2, 1, 1, 1)
                    t       = t.repeat(2)

                    # Make two identical sets of c_static_emb2 and c_in2 (first half batch and second half batch).
                    # The two sets are applied on different initial x_start, noise and t (within each half batch).
                    subj_single_emb, subj_comp_emb, mix_single_emb, mix_comp_emb = \
                        c_static_emb.chunk(4)
                    # Only keep *_comp_emb, but repeat them to form twin comp sets.
                    c_static_emb2 = torch.cat([ subj_comp_emb, subj_comp_emb, 
                                                mix_comp_emb,  mix_comp_emb ], dim=0)
                    
                    subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                        chunk_list(c_in, 4)
                    # Only keep *comp_prompts, but repeat them to form twin comp sets.
                    c_in2 = subj_comp_prompts + subj_comp_prompts + cls_comp_prompts + cls_comp_prompts
                    # Back up cond as cond_orig. Replace cond with the cond for the twin comp sets.
                    cond_orig = cond
                    cond = (c_static_emb2, c_in2, extra_info)
                    # subj_indices and bg_indices are the same as the conventional 4-fold structure.
                    # Since, subj_single_emb, subj_comp_emb, mix_single_emb, mix_comp_emb 
                    # and    subj_comp_emb,   subj_comp_emb, mix_comp_emb,   mix_comp_emb, 
                    # have the same subj_indices and bg_indices respectively.
                    # So we don't need to change subj_indices and bg_indices.

                    # Update masks to be a two-fold * 2 structure.
                    # Before repeating, img_mask, fg_mask, batch_have_fg_mask should all 
                    # have a batch size of 2*BLOCK_SIZE. So repeat_selected_instances() 
                    # won't discard part of them, but simply repeat them twice.
                    img_mask, fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances(slice(0, 2 * BLOCK_SIZE), 2, img_mask, fg_mask, batch_have_fg_mask)
                    self.fg_mask_avail_ratio = batch_have_fg_mask.float().mean()

                # Not self.iter_flags['do_teacher_filter']. This branch is do_mix_prompt_distillation.
                # So it's either reuse_init_conds, or not do_clip_teacher_filtering (globally).
                # In any case, we do not need to change the prompts and static embeddings 
                # and simply do mix reg.
                else:
                    if (not self.do_clip_teacher_filtering) and (not self.iter_flags['reuse_init_conds']):
                        # Usually we shouldn't go here, as do_clip_teacher_filtering is always True.
                        x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)

                    # If reuse_init_conds, prev_t is already 1-repeat-4, and 
                    # x_start is denoised from a 1-repeat-4 x_start in the previous iteration 
                    # (precisely speaking, a 1-repeat-2 x_start that's repeated again to 
                    # approximate a 1-repeat-4 x_start).
                    # But noise, t and inj_noise_t are not 1-repeat-4. 
                    # So we still need to make them 1-repeat-4.
                    noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                    t       = t[:BLOCK_SIZE].repeat(4)
                    if inj_noise_t is not None:
                        inj_noise_t = inj_noise_t[:BLOCK_SIZE].repeat(4)

                    # Update masks to be a 1-repeat-4 structure.
                    img_mask, fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances(slice(0, BLOCK_SIZE), 4, img_mask, fg_mask, batch_have_fg_mask)
                    self.fg_mask_avail_ratio = batch_have_fg_mask.float().mean()

                    # use cached x_start and cond. cond already has the 4-type structure. 
                    # No change to cond here.
                    # NOTE cond is mainly determined by the prompts c_in. Since c_in is inherited from
                    # the previous iteration, cond is also almost the same.
            
            # The prompts are either 2-repeat-2 (do_teacher_filter) or 1-repeat-4 (distillation) structure.
            # Use cond[1] instead of c_static_emb as input, since cond[1] is updated as 2-repeat-2 
            # in the 'do_teacher_filter' branch. We need to do mixing on the c_static_emb 
            # to be used for denoising.
            # In either case, c_static_emb is of (subject embeddings, class embeddings) structure.
            # Therefore, we don't need to deal with the two cases separately.
            # No matter whether t is 2-repeat-2 or 1-repeat-4 structure, 
            # t.chunk(2)[0] always corresponds to the first two blocks of instances.
            t_frac = t.chunk(2)[0] / self.num_timesteps
            # The subject indices are applied to every of the half-batch instances, 
            # so extra_info['subj_indices_1b'] is enough.
            # (extra_info['subj_indices_2b'][1] just repeats extra_info['subj_indices_1b'][1] twice.)
            c_static_emb_vk, emb_v_mixer, emb_v_layers_cls_mix_scales = \
                mix_static_vk_embeddings(cond[0], extra_info['subj_indices_1b'][1], 
                                         self.training_percent,
                                         t_frac = t_frac, 
                                         use_layerwise_embedding = self.use_layerwise_embedding,
                                         N_LAYERS = self.N_LAYERS,
                                         CLS_E_SCALE_LAYERWISE_RANGE=[1.0, 0.7])
          
            # Update cond[0] to c_static_emb_vk.
            # Use cond[1] instead of c_in as part of the tuple, since c_in is changed in the
            # 'do_teacher_filter' branch.
            cond = (c_static_emb_vk, cond[1], extra_info)
            extra_info['emb_v_mixer']                   = emb_v_mixer
            # emb_v_layers_cls_mix_scales: [2, 16]. Each set of scales (for 16 layers) is for an instance.
            extra_info['emb_v_layers_cls_mix_scales']  = emb_v_layers_cls_mix_scales            
            
        # Otherwise, it's a recon iter (attentional or unweighted).
        else:
            assert self.iter_flags['do_normal_recon']
            BLOCK_SIZE = x_start.shape[0]
            # Increase t slightly to increase noise amount and increase robustness.
            inj_noise_t = anneal_t(t, self.training_percent, self.num_timesteps, ratio_range=(1, 1.2))
            # No need to update masks.

        subj_indices, bg_indices = extra_info['subj_indices'], extra_info['bg_indices']
        if not self.iter_flags['do_teacher_filter']:
            extra_info['capture_distill_attn'] = True

        # There are always some subj prompts in this batch. So if self.use_conv_attn,
        # then extra_info['use_conv_attn'] = True, it will inform U-Net to do conv attn.
        # Disabling unet_has_grad for recon iters will greatly reduce performance,
        # although ada_bp_to_unet = False for recon iters.
        # (probably because gradients of static embeddings still need to go through UNet)
        model_output, x_recon, ada_embeddings = \
            self.guided_denoise(x_start, noise, t, cond, 
                                subj_indices=subj_indices, bg_indices=bg_indices,
                                inj_noise_t=inj_noise_t,
                                unet_has_grad=not self.iter_flags['do_teacher_filter'], 
                                # Reconstruct the images at the pixel level for CLIP loss.
                                # do_pixel_recon is not the iter_type 'do_normal_recon'.
                                do_pixel_recon=self.iter_flags['calc_clip_loss'],
                                # cfg_scales: classifier-free guidance scales.
                                cfg_scales=cfg_scales_for_clip_loss)

        extra_info['capture_distill_attn'] = False

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        # default is "eps", i.e., the UNet predicts noise.
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # Do not apply img_mask when it's a compositional iter.
        if img_mask is not None and self.iter_flags['do_normal_recon']:
            target       = target       * img_mask
            model_output = model_output * img_mask

        loss = 0
        twin_comp_ada_embeddings = None

        ###### do_normal_recon ######
        if self.iter_flags['do_normal_recon']:
            if self.iter_flags['use_background_token']:
                # extra_info['subj_indices'] and extra_info['bg_indices'] are used, instead of
                # extra_info['subj_indices_1b'] and extra_info['bg_indices_1b']. 
                if self.fg_bg_complementary_loss_weight > 0:
                    self.fg_bg_comple_attn_uses_scores = True
                    fg_bg_comple_attn_key = 'unet_attnscores' if self.fg_bg_comple_attn_uses_scores \
                                            else 'unet_attns'
                    loss_fg_bg_complementary, loss_fg_mask_align, loss_bg_mask_align, loss_fg_bg_mask_contrast = \
                                self.calc_fg_bg_complementary_loss(extra_info[fg_bg_comple_attn_key], 
                                                                   extra_info['unet_attnscores'],
                                                                   extra_info['subj_indices'],
                                                                   extra_info['bg_indices'],
                                                                   x_start.shape[0],
                                                                   img_mask,
                                                                   fg_grad_scale=0.1,
                                                                   fg_mask=fg_mask,
                                                                   batch_have_fg_mask=batch_have_fg_mask
                                                                  )

                    loss_dict.update({f'{prefix}/fg_bg_complem': loss_fg_bg_complementary.mean().detach()})
                    # If fg_mask is None, then loss_fg_mask_align = loss_bg_mask_align = 0.
                    if loss_fg_mask_align > 0:
                        loss_dict.update({f'{prefix}/fg_mask_align': loss_fg_mask_align.mean().detach()})
                    if loss_bg_mask_align > 0:
                        loss_dict.update({f'{prefix}/bg_mask_align': loss_bg_mask_align.mean().detach()})
                    if loss_fg_bg_mask_contrast > 0:
                        loss_dict.update({f'{prefix}/fg_bg_mask_contrast': loss_fg_bg_mask_contrast.mean().detach()})

                    loss += self.fg_bg_complementary_loss_weight * loss_fg_bg_complementary \
                            + self.fg_bg_mask_align_loss_weight * \
                              (loss_fg_mask_align + loss_bg_mask_align + loss_fg_bg_mask_contrast)

            # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
            loss_recon, loss_recon_pixels = self.calc_recon_loss(model_output, target, fg_mask, 
                                                                 self.iter_flags['use_background_token'])
            loss_dict.update({f'{prefix}/loss_recon': loss_recon.detach()})

            logvar_t = self.logvar.to(self.device)[t].reshape(-1, 1, 1, 1)
            # In theory, the loss can be weighted according to t. However in practice,
            # all the weights are the same, so this step is useless.
            loss_recon_gamma_pixels = loss_recon_pixels / torch.exp(logvar_t) + logvar_t

            if self.iter_flags['use_background_token']:
                loss_recon_gamma = loss_recon_gamma_pixels.mean()
            else:
                # When background token is not use, only evaluate recon loss on the foreground pixels.
                loss_recon_gamma = masked_mean(loss_recon_gamma_pixels, fg_mask)

            # loss = loss_recon / torch.exp(self.logvar) + self.logvar
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_recon_gamma': loss_recon_gamma.detach()})
                loss_dict.update({'logvar': self.logvar.data.mean().detach()})

            loss += self.recon_loss_weight * loss_recon_gamma

            loss_vlb_pixels = self.get_loss(model_output, target, mean=False)
            loss_vlb_pixels = self.lvlb_weights[t].reshape(-1, 1, 1, 1) * loss_vlb_pixels

            if self.iter_flags['use_background_token']:
                loss_vlb = loss_vlb_pixels.mean()
            else:
                # Only evaluate loss_vlb on the foreground pixels.
                loss_vlb = masked_mean(loss_vlb_pixels, fg_mask)

            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.detach()})
            # original_elbo_weight = 0, so that loss_vlb is disabled.
            loss += (self.original_elbo_weight * loss_vlb)
        ###### end of do_normal_recon ######

        ###### begin of preparation for is_compos_iter ######
        # is_compos_iter <=> calc_clip_loss. But we keep this redundancy for possible flexibility.
        if self.iter_flags['is_compos_iter'] and self.iter_flags['calc_clip_loss']:
            # Images generated both under subj_comp_prompts and cls_comp_prompts 
            # are subject to the CLIP text-image matching evaluation.
            # If self.iter_flags['do_teacher_filter'] (implying do_mix_prompt_distillation), 
            # the batch is (subj_comp_emb, subj_comp_emb, mix_comp_emb,  mix_comp_emb).
            # So cls_comp_prompts is used to compute the CLIP text-image matching loss on
            # images guided by the subject or mixed embeddings.
            if self.iter_flags['do_teacher_filter']:
                clip_images_code  = x_recon
                # 4 sets of cls_comp_prompts for (subj comp 1, subj comp 2, mix comp 1, mix comp 2).                
                clip_prompts_comp = extra_info['cls_comp_prompts'] * 4            
            else:
                # Either self.iter_flags['reuse_init_conds'], or a pure ada delta loss iter.
                # A batch of either type has the (subj_single, subj_comp, mix_single, mix_comp) structure.
                # Only evaluate the CLIP loss on the comp images. 
                # So the batch size of clip_images_code is halved.
                # If teachable, the whole batch of x_recon is still used for distillation.
                x_recon_subj_single, x_recon_subj_comp, x_recon_mix_single, x_recon_mix_comp = \
                    x_recon.chunk(4)
                clip_images_code = torch.cat([x_recon_subj_comp, x_recon_mix_comp], dim=0)
                clip_prompts_comp = extra_info['cls_comp_prompts'] * 2

            # Use CLIP loss as a metric to evaluate the compositionality of the generated images 
            # and do distillation selectively.
            # DO NOT use CLIP loss to optimize the model. It will hurt the performance.
            with torch.no_grad():
                clip_images = self.decode_first_stage(clip_images_code)
                losses_clip_comp   = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_comp,   
                                                                                     clip_images,  
                                                                                     reduction='diag')
                #print(clip_prompts_comp)

            # Instances are arranged as: 
            # (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
            losses_clip_subj_comp, losses_clip_mix_comp \
                = losses_clip_comp.chunk(2)
            
            if not self.iter_flags['reuse_init_conds']:
                loss_dict.update({f'{prefix}/loss_clip_subj_comp': losses_clip_subj_comp.mean().detach()})
                loss_dict.update({f'{prefix}/loss_clip_cls_comp':  losses_clip_mix_comp.mean().detach()})
            else:
                loss_dict.update({f'{prefix}/reuse_loss_clip_subj_comp': losses_clip_subj_comp.mean().detach()})
                loss_dict.update({f'{prefix}/reuse_loss_clip_cls_comp':  losses_clip_mix_comp.mean().detach()})

            if self.iter_flags['do_teacher_filter'] or self.iter_flags['reuse_init_conds']:
                # Discard instances that seem to be too far from the text 
                # (it may be a bit premature to make this decision now, as the images are only denoised once).
                # 0.35/0.006: 30%-40% instances will meet these thresholds.
                # 0.33/0.008: 15% instances will meet these thresholds.
                clip_loss_thres             = 0.28
                cls_subj_clip_margin        = 0.002
                clip_loss_thres_base        = 0.26
                cls_subj_clip_margin_base   = 0.003

                # are_insts_teachable: The teacher instances are only teachable if both 
                # the teacher and student are qualified (<= clip_loss_thres), 
                # and the compositional clip loss is smaller than the student.
                # If the student is qualified (<= clip_loss_thres), and the 
                # teacher is teachable, then the teacher is also qualified, 
                # as it has to have a smaller loss to be teachable. 
                # So only need to check losses_clip_subj_comp against clip_loss_thres.
                loss_diffs_subj_mix = losses_clip_subj_comp - losses_clip_mix_comp
                are_insts_teachable = (losses_clip_subj_comp <= clip_loss_thres) & (loss_diffs_subj_mix > cls_subj_clip_margin)
                # print(losses_clip_subj_comp, losses_clip_mix_comp)
                # If any of the two instances is teachable, we consider it as a teachable iteration,
                # and select the better one (with larger loss diff) to teach.
                self.iter_flags['is_teachable']  = are_insts_teachable.sum() > 0
                # Number of filtered pairs is half of the batch size. 
                # Repeat to match the number of instances in the batch.
                # log_image_colors: take values in {0, 1, 2, 3}, 
                # i.e., no box, green, red, purple boxes respectively.
                # no box: not teachable;                 green:  teachable in the first iter and not reused;
                # red: teachable in the first iter but not in the second iter; purple: teachable in both iters.
                # If self.iter_flags['do_teacher_filter'] and an instance is teachable, then in the log image file, 
                # log_image_flag = 1, so the instance has a green bounary box in the logged image.
                log_image_colors = are_insts_teachable.repeat(2).int()
                if self.iter_flags['reuse_init_conds']:
                    # If reuse_init_conds and an instance is teachable, then in the log image file,
                    # log_image_flag = 3, so the instance has a purple bounary box in the logged image.
                    log_image_colors += 2

                self.num_total_teacher_filter_iters += 1
                self.num_teachable_iters += int(self.iter_flags['is_teachable'])
                teachable_frac = self.num_teachable_iters / self.num_total_teacher_filter_iters
                loss_dict.update({f'{prefix}/teachable_frac': teachable_frac})
                self.num_total_reuse_filter_iters += int(self.iter_flags['reuse_init_conds'])
                self.num_reuse_teachable_iters += int(self.iter_flags['reuse_init_conds'] and self.iter_flags['is_teachable'])
                reuse_teachable_frac = self.num_reuse_teachable_iters / self.num_total_reuse_filter_iters 
                loss_dict.update({f'{prefix}/reuse_teachable_frac': reuse_teachable_frac})

                # Only do distillation if at least one of teacher instances is teachable.
                if self.iter_flags['do_teacher_filter'] and self.iter_flags['is_teachable']:
                    # No need the intermediates of the twin-comp instances. Release them to save RAM.
                    self.release_plosses_intermediates(locals())
                    # reset_prompt_embedding_caches() will implicitly clear the cache.
                    self.embedding_manager.reset_prompt_embedding_caches()

                    better_cand_idx = torch.argmax(loss_diffs_subj_mix)

                    # Choose the x_start, noise, and t of the better candidate. 
                    # Repeat 4 times and use them as the condition to do denoising again.
                    x_start_sel = x_start[better_cand_idx].repeat(4, 1, 1, 1)
                    noise_sel   = noise[better_cand_idx].repeat(4, 1, 1, 1)
                    t_sel       = t[better_cand_idx].repeat(4)
                    t_frac      = t_sel.chunk(2)[0] / self.num_timesteps
                    # Mix embeddings to get c_static_emb_orig_vk for cond_orig.
                    # Do mixing on saved cond_orig instead of the updated "cond".
                    # cond_orig is the 4-type prompt embeddings (subj single, subj comp, mix single, mix comp).
                    # but cond  has been re-organized as (subj comp, subj comp, mix comp, mix comp). 
                    # So we use cond_orig.
                    c_static_emb_orig_vk, emb_v_mixer, emb_v_layers_cls_mix_scales = \
                        mix_static_vk_embeddings(cond_orig[0], extra_info['subj_indices_1b'][1], 
                                                 self.training_percent,
                                                 t_frac = t_frac, 
                                                 use_layerwise_embedding = self.use_layerwise_embedding,
                                                 N_LAYERS = self.N_LAYERS,
                                                 CLS_E_SCALE_LAYERWISE_RANGE=[1.0, 0.7])
                    
                    extra_info['emb_v_mixer']                   = emb_v_mixer
                    # emb_v_layers_cls_mix_scales: [2, 16]. Each set of scales (for 16 layers) is for an instance.
                    # Different from the emb_v_layers_cls_mix_scales above, which is for the twin comp instances.
                    extra_info['emb_v_layers_cls_mix_scales']   = emb_v_layers_cls_mix_scales  
                    # Update c_static_emb.
                    cond_orig_qv = (c_static_emb_orig_vk, cond_orig[1], extra_info)

                    # unet_has_grad has to be enabled here. Here is the actual place where the computation graph 
                    # on mix reg and ada embeddings is generated for the delta loss. 
                    # (The previous call to guided_denoise() didn't enable gradients, 
                    # as it's only used to filter a teacher.)
                    # If unet_has_grad=False, the gradients of the ada delta loss
                    # couldn't pass through UNet, reducing the performance.
                    # do_pixel_recon=True: return denoised images x_recon. If cfg_scale > 1, 
                    # do classifier-free guidance, so that x_recon are better instances
                    # to be used to initialize the next reuse_init comp iteration.
                    # student prompts are subject prompts.  
                    model_output, x_recon, ada_embeddings = \
                        self.guided_denoise(x_start_sel, noise_sel, t_sel, cond_orig_qv, 
                                            subj_indices=extra_info['subj_indices_2b'],
                                            bg_indices=extra_info['bg_indices_2b'],
                                            unet_has_grad=True, 
                                            do_pixel_recon=True, cfg_scales=cfg_scales_for_clip_loss)

                    # Update masks according to x_start_sel. Select the masks corresponding to 
                    # the better candidate, indexed by [better_cand_idx] (Keep it as a list).
                    img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances([better_cand_idx], 4, img_mask, fg_mask, 
                                                  filtered_fg_mask, batch_have_fg_mask)
                    self.fg_mask_avail_ratio = batch_have_fg_mask.float().mean()
                    # Cache x_recon for the next iteration with a smaller t.
                    # Note the 4 types of prompts have to be the same as this iter, 
                    # since this x_recon was denoised under this cond.
                    # Use the subject half of the batch, chunk(2)[0], instead of the mix half, chunk(2)[1], as 
                    # the subject half is better at subject authenticity (but may be worse on composition).
                    x_recon_sel_rep = x_recon.detach().chunk(2)[0].repeat(2, 1, 1, 1)
                    # We cannot simply use cond_orig[1], as they are (subj single, subj comp, mix single, mix comp).
                    # mix single = class single, but under some settings, maybe mix comp = subj comp.
                    # cached_inits['x_start'] has a BS of 4.
                    # x_recon_sel_rep doesn't have the 1-repeat-4 structure, instead a 
                    # 1-repeat-2 structure that's repeated twice.
                    # But it approximates a 1-repeat-4 structure, so the distillation should still work.
                    # NOTE: no need to update masks to correspond to x_recon_sel_rep, as x_recon_sel_rep
                    # is half-repeat-2 of (the reconstructed images of) x_start_sel. 
                    # Doing half-repeat-2 on masks won't change them, as they are 1-repeat-4.
                    self.cached_inits = { 'x_start':                x_recon_sel_rep, 
                                          'delta_prompts':          cond_orig[2]['delta_prompts'],
                                          't':                      t_sel,
                                          'img_mask':               img_mask,
                                          'fg_mask':                fg_mask,
                                          'batch_have_fg_mask':     batch_have_fg_mask,
                                          'filtered_fg_mask':       filtered_fg_mask,
                                          'use_background_token':   self.iter_flags['use_background_token'],
                                          'comp_init_with_fg_area': self.iter_flags['comp_init_with_fg_area'],
                                        }
                    
                    # Do not put cached_inits_available on self.iter_flags, as iter_flags will be reset
                    # in the beginning of the next iteration.
                    self.cached_inits_available = True
                    
                elif not self.iter_flags['is_teachable']:
                    # If not is_teachable, do not do distillation this time 
                    # (since both instances are not good teachers), 
                    # NOTE: thus we can only compute emb reg and delta loss.

                    # In an self.iter_flags['do_teacher_filter'], and not teachable instances are found.
                    # guided_denoise() above is done on twin comp instances, 
                    # and we've computed twin_comp_ada_embeddings.
                    # ada_embeddings = None => loss_ada_delta = 0.
                    if self.iter_flags['do_teacher_filter']:
                        ada_embeddings = None
                    # Otherwise, it's an reuse_init_conds iter, and no teachable instances are found.
                    # We've computed the ada embeddings for the 4-type instances, 
                    # so just use the existing ada_embeddings (but avoid distillation).

                # Otherwise, it's an reuse_init_conds iter, and a teachable instance is found.
                # Do nothing.

            else:
                # Not do_teacher_filter, nor reuse_init_conds. 
                # So only one possibility: not do_clip_teacher_filtering.
                # The teacher instance is always teachable as teacher filtering is disabled.
                self.iter_flags['is_teachable'] = True
                # Since not self.iter_flags['do_teacher_filter']/reuse_init_conds, log_image_colors are all 0,
                # i.e., no box to be drawn on the images in cache_and_log_generations().
                log_image_colors = torch.zeros(clip_images.shape[0], device=x_start.device)

            self.cache_and_log_generations(clip_images, log_image_colors)

        else:
            # Not is_compos_iter. Distillation won't be done in this iter, so is_teachable = False.
            self.iter_flags['is_teachable'] = False
        ###### end of preparation for is_compos_iter ######

        if self.static_embedding_reg_weight + self.ada_embedding_reg_weight > 0:
            loss_emb_reg = self.embedding_manager.embedding_reg_loss()
            if self.use_layerwise_embedding:
                loss_static_emb_reg, loss_ada_emb_reg = loss_emb_reg
            else:
                loss_static_emb_reg = loss_emb_reg
                loss_ada_emb_reg    = 0

            if loss_static_emb_reg > 0:
                loss_dict.update({f'{prefix}/loss_static_emb_reg': loss_static_emb_reg.mean().detach()})
            if loss_ada_emb_reg > 0:
                loss_dict.update({f'{prefix}/loss_ada_emb_reg':    loss_ada_emb_reg.mean().detach()})

            loss += loss_static_emb_reg * self.static_embedding_reg_weight \
                    + loss_ada_emb_reg  * self.ada_embedding_reg_weight

        if self.do_static_prompt_delta_reg:
            # do_ada_emb_delta_reg controls whether to do ada comp delta reg here.
            # Use subj_indices_1b here, since this index is used to extract 
            # subject embeddings from each block, and compare two such blocks.
            loss_static_delta, loss_ada_delta \
            = self.embedding_manager.calc_prompt_emb_delta_loss( 
                                    extra_info['c_static_emb_4b'], ada_embeddings,
                                    self.iter_flags['do_ada_emb_delta_reg']
                                    )

            loss_dict.update({f'{prefix}/static_delta': loss_static_delta.mean().detach()})
            if loss_ada_delta != 0:
                loss_dict.update({f'{prefix}/ada_delta': loss_ada_delta.mean().detach()})

            # The prompt delta loss for ada embeddings is only applied 
            # every self.composition_regs_iter_gap iterations. So the ada loss 
            # should be boosted proportionally to composition_regs_iter_gap. 
            # Divide it by 2 to reduce the proportion of ada emb loss relative to 
            # static emb loss in the total loss.                
            ada_comp_loss_boost_ratio = self.composition_regs_iter_gap / 2
            loss_prompt_delta_reg = loss_static_delta + loss_ada_delta * ada_comp_loss_boost_ratio
            
            loss += (self.prompt_emb_delta_reg_weight * loss_prompt_delta_reg)
        
        if self.fg_bg_xlayer_consist_loss_weight > 0:
            # SSB_SIZE: subject sub-batch size.
            # If do_normal_recon, then both instances are subject instances. 
            # The subject sub-batch size SSB_SIZE = 2 (1 * BLOCK_SIZE).
            # If is_compos_iter, then subject sub-batch size SSB_SIZE = 2 * BLOCK_SIZE. 
            # (subj-single and subj-comp instances).
            SSB_SIZE = BLOCK_SIZE if self.iter_flags['do_normal_recon'] else 2 * BLOCK_SIZE
            loss_fg_xlayer_consist, loss_bg_xlayer_consist = \
                self.calc_fg_bg_xlayer_consist_loss(extra_info['unet_attnscores'],
                                                    extra_info['subj_indices'],
                                                    extra_info['bg_indices'],
                                                    SSB_SIZE, 
                                                    self.embedding_manager.delta_loss_emb_mask,                                                    
                                                    img_mask)
            if loss_fg_xlayer_consist > 0:
                loss_dict.update({f'{prefix}/fg_xlayer_consist': loss_fg_xlayer_consist.mean().detach()})
            if loss_bg_xlayer_consist > 0:
                loss_dict.update({f'{prefix}/bg_xlayer_consist': loss_bg_xlayer_consist.mean().detach()})

            # bg attention is more flexible and could be less consistent across layers.
            bg_xlayer_consist_loss_scale = 0.2
            loss += self.fg_bg_xlayer_consist_loss_weight * \
                            (loss_fg_xlayer_consist + loss_bg_xlayer_consist * bg_xlayer_consist_loss_scale)

        if self.iter_flags['do_mix_prompt_distillation']:
            # unet_feats is a dict as: layer_idx -> unet_feat. 
            # It contains the 6 specified conditioned layers of UNet features.
            # i.e., layers 7, 8, 12, 16, 17, 18.
            # Similar are unet_attns and unet_attnscores.
            unet_feats  = extra_info['unet_feats']
            self.subj_attn_delta_distill_uses_scores    = True
            self.subj_comp_attn_comple_loss_uses_scores = True
            self.comp_bg_attn_suppress_uses_scores      = True

            if self.iter_flags['is_teachable']:
                # subj_indices_2b is used here, as it's used to index subj single and subj comp embeddings.
                # The indices will be shifted along the batch dimension (size doubled) within calc_prompt_mix_loss()
                # to index all the 4 blocks.

                subj_attn_delta_distill_key = 'unet_attnscores' if self.subj_attn_delta_distill_uses_scores \
                                            else 'unet_attns'
                loss_subj_attn_delta_distill, loss_subj_attn_norm_distill, loss_feat_delta_distill = \
                                    self.calc_prompt_mix_loss(unet_feats, extra_info[subj_attn_delta_distill_key], 
                                                              extra_info['subj_indices_2b'],
                                                              BLOCK_SIZE)

                if loss_feat_delta_distill > 0:
                    loss_dict.update({f'{prefix}/feat_delta_distill':       loss_feat_delta_distill.mean().detach()})
                if loss_subj_attn_delta_distill > 0:
                    loss_dict.update({f'{prefix}/subj_attn_delta_distill':  loss_subj_attn_delta_distill.mean().detach()})
                if loss_subj_attn_norm_distill > 0:
                    loss_dict.update({f'{prefix}/subj_attn_norm_distill':   loss_subj_attn_norm_distill.mean().detach()})

                loss_subj_comp_key_ortho = 0
                if self.subj_comp_key_ortho_loss_weight > 0:
                    subj_comp_attn_comple_key = 'unet_attnscores' if self.subj_comp_attn_comple_loss_uses_scores \
                                                else 'unet_attns'
                    loss_subj_comp_key_ortho, loss_subj_comp_value_ortho, loss_subj_comp_attn_comple = \
                        self.calc_subj_comp_ortho_loss(extra_info['unet_ks'], extra_info['unet_vs'], 
                                                        extra_info[subj_comp_attn_comple_key],
                                                        extra_info['subj_indices_2b'],
                                                        self.embedding_manager.delta_loss_emb_mask,
                                                        BLOCK_SIZE, cls_grad_scale=0.05)

                    if loss_subj_comp_key_ortho != 0:
                        loss_dict.update({f'{prefix}/subj_comp_key_ortho':   loss_subj_comp_key_ortho.mean().detach()})
                    if loss_subj_comp_value_ortho != 0:
                        loss_dict.update({f'{prefix}/subj_comp_value_ortho': loss_subj_comp_value_ortho.mean().detach()})
                    if loss_subj_comp_attn_comple != 0:
                        loss_dict.update({f'{prefix}/subj_comp_attn_comple': loss_subj_comp_attn_comple.mean().detach()})

                subj_attn_delta_distill_loss_scale = 0.5
                if self.subj_attn_delta_distill_uses_scores:
                    subj_attn_norm_distill_loss_scale = self.subj_attn_norm_distill_loss_scale
                else:
                    # loss_subj_attn_norm_distill uses L1 loss, which tends to be in 
                    # smaller magnitudes than the delta loss. So we scale it up by 4x.
                    subj_attn_norm_distill_loss_scale = self.subj_attn_norm_distill_loss_scale * 4
                # If deep_neg_context is used 
                # (If self.distill_deep_neg_context is not None, 50% of the chance we will use deep_neg_context 
                # by assigning it to extra_info['deep_neg_context']), 
                # the attn magnitude may become slightly larger. So we scale up delta loss by 2x.
                if 'deep_neg_context' in extra_info:
                    subj_attn_norm_distill_loss_scale *= 2

                loss_mix_prompt_distill =  loss_subj_attn_delta_distill   * subj_attn_delta_distill_loss_scale \
                                            + loss_subj_attn_norm_distill * subj_attn_norm_distill_loss_scale \
                                            + loss_feat_delta_distill 
                                            
                if loss_mix_prompt_distill > 0:
                    loss_dict.update({f'{prefix}/mix_prompt_distill':  loss_mix_prompt_distill.mean().detach()})
  
                # mix_prompt_distill_weight: 2e-4.
                loss += loss_mix_prompt_distill      * self.mix_prompt_distill_weight \
                        + loss_subj_comp_key_ortho   * self.subj_comp_key_ortho_loss_weight \
                        + loss_subj_comp_value_ortho * self.subj_comp_value_ortho_loss_weight \
                        + loss_subj_comp_attn_comple * self.subj_comp_attn_complementary_loss_weight
                
                # NOTE: loss_comp_fg_bg_preserve is applied only when this 
                # iteration is teachable, because at such iterations the unet gradient is enabled.
                # The current iteration may be a fresh iteration or a reuse_init_conds iteration.
                # In both cases, if comp_init_with_fg_area, then we need to preserve the fg/bg areas.
                # Although fg_mask_avail_ratio > 0 when comp_init_with_fg_area,
                # fg_mask_avail_ratio may have been updated after doing teacher filtering 
                # (since x_start has been filtered, masks are also filtered accordingly, 
                # and the same as to fg_mask_avail_ratio). So we need to check it here.
                if self.iter_flags['comp_init_with_fg_area'] and self.fg_mask_avail_ratio > 0 \
                  and self.comp_fg_bg_preserve_loss_weight > 0:
                    # By default, attns_or_scores = 'unet_attnscores'
                    attns_or_scores = 'unet_attnscores' if self.comp_bg_attn_suppress_uses_scores \
                                    else 'unet_attns'
                    # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
                    # Therefore, using fg_mask for comp_init_with_fg_area will force the model remember 
                    # the background in the training images, which is not desirable.
                    # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0, 
                    # excluding the instance from the fg_bg_preserve_loss.
                    loss_comp_subj_fg_feat_preserve, loss_comp_subj_fg_attn_preserve, loss_comp_subj_bg_attn_suppress = \
                        self.calc_comp_fg_bg_preserve_loss(unet_feats, extra_info[attns_or_scores], 
                                                            filtered_fg_mask, batch_have_fg_mask,
                                                            extra_info['subj_indices_1b'], BLOCK_SIZE)
                    if loss_comp_subj_fg_feat_preserve > 0:
                        loss_dict.update({f'{prefix}/comp_subj_fg_feat_preserve': loss_comp_subj_fg_feat_preserve.mean().detach()})
                    if loss_comp_subj_fg_attn_preserve > 0:
                        loss_dict.update({f'{prefix}/comp_subj_fg_attn_preserve': loss_comp_subj_fg_attn_preserve.mean().detach()})
                    if loss_comp_subj_bg_attn_suppress > 0:
                        loss_dict.update({f'{prefix}/comp_subj_bg_attn_suppress': loss_comp_subj_bg_attn_suppress.mean().detach()})
                    bg_attn_suppress_loss_scale = 0.2
                    loss_comp_fg_bg_preserve = loss_comp_subj_fg_feat_preserve \
                                                + loss_comp_subj_fg_attn_preserve \
                                                + loss_comp_subj_bg_attn_suppress * bg_attn_suppress_loss_scale
                else:
                    loss_comp_fg_bg_preserve = 0

                # Scale down loss_comp_fg_bg_preserve if reuse_init_conds.
                comp_fg_bg_preserve_loss_scale = 0.25 if self.iter_flags['reuse_init_conds'] else 1
                loss += loss_comp_fg_bg_preserve * self.comp_fg_bg_preserve_loss_weight \
                        * comp_fg_bg_preserve_loss_scale

        self.release_plosses_intermediates(locals())
        loss_dict.update({f'{prefix}/loss': loss.mean().detach()})

        return loss, loss_dict

    # pixel-wise recon loss. If batch_indices is not None, 
    # only compute the recon loss on selected instances.
    def calc_recon_loss(self, model_output, target, fg_mask, use_background_token, batch_indices=None):
        if batch_indices is not None:
            model_output = model_output[batch_indices]
            target       = target[batch_indices]
            fg_mask      = fg_mask[batch_indices]

        # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
        loss_recon_pixels = self.get_loss(model_output, target, mean=False)
        if use_background_token:
            # recon loss on the whole image.
            loss_recon = loss_recon_pixels.mean()
        else:
            # recon loss only on the foreground pixels.
            loss_recon = masked_mean(loss_recon_pixels, fg_mask)

        return loss_recon, loss_recon_pixels
    
    def calc_prompt_mix_loss(self, unet_feats, unet_attns_or_scores, placeholder_indices, BLOCK_SIZE):
        # do_mix_prompt_distillation iterations. No ordinary image reconstruction loss.
        # Only regularize on intermediate features, i.e., intermediate features generated 
        # under subj_comp_prompts should satisfy the delta loss constraint:
        # F(subj_comp_prompts)  - F(mix(subj_comp_prompts, cls_comp_prompts)) \approx 
        # F(subj_single_prompts) - F(cls_single_prompts)

        # Avoid doing distillation on top layers (too detailed) and 
        # the first few bottom layers (little difference).
        # distill_layer_weights: relative weight of each distillation layer. 
        # distill_layer_weights are normalized using distill_overall_weight.
        # Most important conditioning layers are 7, 8, 12, 16, 17. All the 5 layers have 1280 channels.
        # But intermediate layers also contribute to distillation. They have small weights.

        # feature map distillation only uses delta loss on the features to reduce the 
        # class polluting the subject features.
        feat_distill_layer_weights = { # 7:  1., 8: 1.,   
                                        12: 1.,
                                        16: 1., 17: 1.,
                                        18: 1.,
                                        19: 1., 20: 1., 
                                        21: 1., 22: 1., 
                                        23: 1., 24: 1., 
                                     }

        # attn delta loss is more strict and could cause pollution of subject features with class features.
        # so top layers layers 21, 22, 23, 24 are excluded by setting their weights to 0.
        attn_delta_distill_layer_weights = { # 7:  1., 8: 1.,
                                            12: 1.,
                                            16: 1., 17: 1.,
                                            18: 1.,
                                            19: 1., 20: 1., 
                                            21: 1., 22: 1., 
                                            23: 1., 24: 1.,                            
                                            }
        # DISABLE attn delta loss.
        # attn_delta_distill_layer_weights = {}

        # attn norm distillation is applied to almost all conditioning layers.
        attn_norm_distill_layer_weights = { # 7:  1., 8: 1.,
                                            12: 1.,
                                            16: 1., 17: 1.,
                                            18: 1.,
                                            19: 1., 20: 1., 
                                            21: 1., 22: 1., 
                                            23: 1., 24: 1.,                                   
                                           }

        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weights          = normalize_dict_values(feat_distill_layer_weights)
        attn_norm_distill_layer_weights     = normalize_dict_values(attn_norm_distill_layer_weights)
        attn_delta_distill_layer_weights    = normalize_dict_values(attn_delta_distill_layer_weights)

        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(placeholder_indices[0]) // len(torch.unique(placeholder_indices[0]))
        placeholder_indices = double_token_indices(placeholder_indices, BLOCK_SIZE * 2)

        mix_feat_grad_scale = 0.1
        mix_feat_grad_scaler = gen_gradient_scaler(mix_feat_grad_scale)
        # mix_attn_grad_scale = 0.05, almost zero, effectively no grad to teacher attn. 
        # Setting to 0 may prevent the graph from being released and OOM.
        mix_attn_grad_scale  = 0.05  
        mix_attn_grad_scaler = gen_gradient_scaler(mix_attn_grad_scale)

        loss_subj_attn_delta_distill    = 0
        loss_subj_attn_norm_distill     = 0
        loss_feat_delta_distill = 0

        for unet_layer_idx, unet_feat in unet_feats.items():
            if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_norm_distill_layer_weights):
                continue

            # each is [1, 1280, 16, 16]
            subj_single_feat, subj_comp_feat, mix_single_feat, mix_comp_feat \
                = unet_feat.chunk(4)
            
            # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
            # We don't need BP through attention into UNet.
            attn_mat = unet_attns_or_scores[unet_layer_idx].permute(0, 3, 1, 2)
            # subj_attn: [4, 8, 256] (1 embedding  for 1 token)  => [4, 1, 8, 256] => [4, 8, 256]
            # or         [16, 8, 256] (4 embeddings for 1 token) => [4, 4, 8, 256] => [4, 8, 256]
            # BLOCK_SIZE*4: this batch contains 4 blocks. Each block should have one instance.
            subj_attn = attn_mat[placeholder_indices].reshape(BLOCK_SIZE*4, K_fg, *attn_mat.shape[2:]).sum(dim=1)
            # subj_single_subj_attn, ...: [1, 8, 256] (1 embedding  for 1 token) 
            # or                          [1, 8, 256] (4 embeddings for 1 token)
            subj_single_subj_attn, subj_comp_subj_attn, mix_single_subj_attn,  mix_comp_subj_attn \
                = subj_attn.chunk(4)

            # attn_delta_distill_layer_weights is a subset of attn_norm_distill_layer_weights. 
            # So here we only check attn_norm_distill_layer_weights.
            if unet_layer_idx in attn_norm_distill_layer_weights:
                attn_norm_distill_layer_weight     = attn_norm_distill_layer_weights[unet_layer_idx]
                attn_delta_distill_layer_weight    = attn_delta_distill_layer_weights.get(unet_layer_idx, 0)

                # mix_attn_grad_scale = 0.05, almost zero, effectively no grad to mix_comp_subj_attn/mix_single_subj_attn. 
                # Use this scaler to release the graph and avoid OOM.
                mix_comp_subj_attn_gs   = mix_attn_grad_scaler(mix_comp_subj_attn)
                mix_single_subj_attn_gs = mix_attn_grad_scaler(mix_single_subj_attn)

                if attn_delta_distill_layer_weight > 0:
                    # No need to use *_gs version here, as gradient scaling is done in calc_delta_loss().
                    single_attn_delta = ortho_subtract(subj_single_subj_attn, mix_single_subj_attn)
                    comp_attn_delta   = ortho_subtract(subj_comp_subj_attn,   mix_comp_subj_attn)

                    # Setting exponent as 2 seems to push too hard restriction on subject embeddings 
                    # towards class embeddings, hurting authenticity.
                    loss_layer_subj_delta_attn = calc_delta_loss(single_attn_delta, comp_attn_delta, 
                                                                 exponent=3,
                                                                 do_demean_first=False,
                                                                 first_n_dims_to_flatten=2, 
                                                                 ref_grad_scale=1)
                    
                    loss_subj_attn_delta_distill  += loss_layer_subj_delta_attn * attn_delta_distill_layer_weight
                
                # Align the attention corresponding to each embedding individually.
                # Note mix_*subj_attn use *_gs versions.
                loss_layer_subj_comp_attn_norm   = (subj_comp_subj_attn.mean(dim=-1)   - mix_comp_subj_attn_gs.mean(dim=-1)).abs().mean()
                loss_layer_subj_single_attn_norm = (subj_single_subj_attn.mean(dim=-1) - mix_single_subj_attn_gs.mean(dim=-1)).abs().mean()
                # loss_subj_attn_norm_distill uses L1 loss, which tends to be in 
                # smaller magnitudes than the delta loss. So it will be scaled up later in p_losses().
                loss_subj_attn_norm_distill   += ( loss_layer_subj_comp_attn_norm + loss_layer_subj_single_attn_norm ) \
                                                  * attn_norm_distill_layer_weight

            if unet_layer_idx not in feat_distill_layer_weights:
                continue

            use_subj_attn_as_spatial_weights = True
            if use_subj_attn_as_spatial_weights:
                feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

                # subj_single_feat, ...: [1, 1280, 16, 16]
                subj_single_feat, subj_comp_feat, mix_single_feat, mix_comp_feat \
                    = unet_feat.chunk(4)

                # convert_attn_to_spatial_weight() will detach attention weights to 
                # avoid BP through attention.
                # reversed=True: larger subject attention => smaller spatial weight, i.e., 
                # pay more attention to the context.
                spatial_weight_mix_comp, spatial_attn_mix_comp   = convert_attn_to_spatial_weight(mix_comp_subj_attn, BLOCK_SIZE, 
                                                                                                  mix_comp_feat.shape[2:],
                                                                                                  reversed=True)

                spatial_weight_subj_comp, spatial_attn_subj_comp = convert_attn_to_spatial_weight(subj_comp_subj_attn, BLOCK_SIZE,
                                                                                                  subj_comp_feat.shape[2:],
                                                                                                  reversed=True)
                spatial_weight = (spatial_weight_mix_comp + spatial_weight_subj_comp) / 2

                # spatial_attn_mix_comp, spatial_attn_subj_comp are returned for debugging purposes. 
                # Delete them to release RAM.
                del spatial_attn_mix_comp, spatial_attn_subj_comp

                # Use mix single/comp weights on both subject-only and mix features, 
                # to reduce misalignment and facilitate distillation.
                # The multiple heads are aggregated by mean(), since the weighted features don't have multiple heads.
                subj_single_feat = subj_single_feat * spatial_weight
                subj_comp_feat   = subj_comp_feat   * spatial_weight
                mix_single_feat  = mix_single_feat  * spatial_weight
                mix_comp_feat    = mix_comp_feat    * spatial_weight

            do_feat_pooling = True
            feat_pool_kernel_size = 4
            feat_pool_stride      = 2
            # feature pooling: allow small perturbations of the locations of pixels.
            if do_feat_pooling:
                pooler = nn.AvgPool2d(feat_pool_kernel_size, stride=feat_pool_stride)
            else:
                pooler = nn.Identity()

            # Pool the H, W dimensions to remove spatial information.
            # After pooling, subj_single_feat, subj_comp_feat, 
            # mix_single_feat, mix_comp_feat: [1, 1280] or [1, 640], ...
            subj_single_feat = pooler(subj_single_feat).reshape(subj_single_feat.shape[0], -1)
            subj_comp_feat   = pooler(subj_comp_feat).reshape(subj_comp_feat.shape[0], -1)
            mix_single_feat  = pooler(mix_single_feat).reshape(mix_single_feat.shape[0], -1)
            mix_comp_feat    = pooler(mix_comp_feat).reshape(mix_comp_feat.shape[0], -1)

            # mix_feat_grad_scale = 0.1.
            mix_single_feat  = mix_feat_grad_scaler(mix_single_feat)
            mix_comp_feat    = mix_feat_grad_scaler(mix_comp_feat)

            # ortho_subtract() is done on the last dimension. 
            # So we flatten the spatial dimensions first as above.
            # NOTE: use normalized_ortho_subtract() will reduce performance.
            comp_feat_delta   = ortho_subtract(subj_comp_feat,   mix_comp_feat)
            single_feat_delta = ortho_subtract(subj_single_feat, mix_single_feat)
                
            # single_feat_delta, comp_feat_delta: [1, 1280], ...
            # Pool the spatial dimensions H, W to remove spatial information.
            # The gradient goes back to single_feat_delta -> subj_comp_feat,
            # as well as comp_feat_delta -> mix_comp_feat.
            # If stop_single_grad, the gradients to subj_single_feat and mix_single_feat are stopped, 
            # as these two images should look good by themselves (since they only contain the subject).
            # Note the learning strategy to the single image features should be different from 
            # the single embeddings, as the former should be optimized to look good by itself,
            # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
            # and 2) the embeddings are amendable to composition.
            loss_layer_feat_delta_distill = ortho_l2loss(comp_feat_delta, single_feat_delta, mean=True)
            
            # print(f'layer {unet_layer_idx} loss: {loss_layer_prompt_mix_reg:.4f}')
            loss_feat_delta_distill += loss_layer_feat_delta_distill * feat_distill_layer_weight

        return loss_subj_attn_delta_distill, loss_subj_attn_norm_distill, loss_feat_delta_distill

    def calc_fg_bg_complementary_loss(self, unet_attns_or_scores, unet_attnscores,
                                      subj_indices, 
                                      bg_indices, 
                                      BS, img_mask, fg_grad_scale=0.1,
                                      fg_mask=None, batch_have_fg_mask=None):
        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        attn_align_layer_weights = { #7:  1., 8: 1.,
                                    12: 1.,
                                    16: 1., 17: 1.,
                                    18: 1.,
                                    19: 1., 20: 1., 
                                    21: 1., 22: 1., 
                                    23: 1., 24: 1., 
                                   }
        
        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)

        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # K_bg: 1 or 2, number of embeddings per background token.
        K_bg = len(bg_indices[0]) // len(torch.unique(bg_indices[0]))

        loss_fg_bg_complementary = 0
        loss_fg_mask_align = 0
        loss_bg_mask_align = 0
        loss_fg_bg_mask_contrast = 0

        emb_mfmb_contrast_scale         = 0.01
        fgbg_emb_contrast_scale         = 0.05
        mfmb_contrast_score_margin            = 0.4
        subj_bg_contrast_at_mf_score_margin   = 0.4 * K_fg / K_bg     # 0.8
        bg_subj_contrast_at_mb_score_margin   = 0.4 * K_bg / K_fg     # 0.2

        # In each instance, subj_indices has K_fg times as many elements as bg_indices.
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                          [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # bg_indices: ([0, 1, 2, 3, 4, 5, 6, 7], [11, 12, 34, 29, 11, 12, 34, 29]).
        # BS = 2, so we only keep instances indexed by [0, 1].
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:BS*K_fg], subj_indices[1][:BS*K_fg])
        # bg_indices: ([0, 1], [11, 12]).
        bg_indices = (bg_indices[0][:BS*K_bg], bg_indices[1][:BS*K_bg])

        #fg_attn_grad_scale  = 0.5
        #fg_attn_grad_scaler = gen_gradient_scaler(fg_attn_grad_scale)

        for unet_layer_idx, unet_attn in unet_attns_or_scores.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            # [2, 8, 256, 77] / [2, 8, 64, 77] =>
            # [2, 77, 8, 256] / [2, 77, 8, 64]
            attn_mat = unet_attn.permute(0, 3, 1, 2)
            # subj_attn: [8, 8, 64] -> [2, 4, 8, 64] sum among K_fg embeddings -> [2, 8, 64]
            subj_attn = attn_mat[subj_indices].reshape(BS, K_fg, *attn_mat.shape[2:]).sum(dim=1)

            # bg_attn:   [4, 8, 64] -> [2, 2, 8, 64] sum among K_bg embeddings -> [2, 8, 64]
            # 8: 8 attention heads. Last dim 64: number of image tokens.
            bg_attn   = attn_mat[bg_indices].reshape(BS, K_bg, *attn_mat.shape[2:]).sum(dim=1)

            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]
            
            if img_mask is not None:
                # img_mask: [2, 1, 64, 64] -> [2, 1, 8, 8]. subj_attn: [2, 8, 64]
                img_mask2 = resize_mask_for_feat_or_attn(subj_attn, img_mask, "img_mask", mode="nearest|bilinear")
                # img_mask2: [2, 1, 8, 8] -> [2, 1, 64] -> [2, 8, 64].
                img_mask2 = img_mask2.reshape(BS, 1, -1).repeat(1, subj_attn.shape[1], 1)
                bg_attn   = bg_attn   * img_mask2
                subj_attn = subj_attn * img_mask2
            else:
                img_mask2 = torch.ones_like(subj_attn)

            # aim_to_align=False: push bg_attn to be orthogonal with subj_attn, 
            # so that the two attention maps are complementary.
            # exponent = 2: exponent is 3 by default, which lets the loss focus on large activations.
            # But we don't want to only focus on large activations. So set it to 2.
            # ref_grad_scale = 0.05: small gradients will be BP-ed to the subject embedding,
            # to make the two attention maps more complementary (expect the loss pushes the 
            # subject embedding to a more accurate point).
            loss_layer_fg_bg_comple = calc_delta_loss(bg_attn, subj_attn, 
                                                      exponent=2,    
                                                      do_demean_first=False,
                                                      first_n_dims_to_flatten=2, 
                                                      ref_grad_scale=fg_grad_scale,
                                                      aim_to_align=False,
                                                      debug=False)
            
            loss_fg_bg_complementary += loss_layer_fg_bg_comple * attn_align_layer_weight

            if (fg_mask is not None) and (batch_have_fg_mask.sum() > 0):
                attnscore_mat = unet_attnscores[unet_layer_idx].permute(0, 3, 1, 2)
                # subj_score: [8, 8, 64] -> [2, 4, 8, 64] sum among K_fg embeddings -> [2, 8, 64]
                subj_score = attnscore_mat[subj_indices].reshape(BS, K_fg, *attnscore_mat.shape[2:]).sum(dim=1)
                # bg_score:   [4, 8, 64] -> [2, 2, 8, 64] sum among K_bg embeddings -> [2, 8, 64]
                bg_score   = attnscore_mat[bg_indices].reshape(BS, K_bg, *attnscore_mat.shape[2:]).sum(dim=1)

                fg_mask2 = resize_mask_for_feat_or_attn(subj_attn, fg_mask, "fg_mask", mode="nearest|bilinear")
                # Repeat 8 times to match the number of attention heads.
                fg_mask2 = fg_mask2.reshape(BS, 1, -1).repeat(1, subj_attn.shape[1], 1)
                fg_mask3 = torch.zeros_like(fg_mask2)
                fg_mask3[fg_mask2 >  1e-6] = 1.

                # The ones in fg_mask should always be a subset of ones in img_mask. 
                # However, it's possible that after scale, some elements in img_mask2 become 0.5.
                # So this equality doesn't always hold: fg_mask3 == fg_mask3 * img_mask2.
                #if (fg_mask3 != fg_mask3 * img_mask2).any():
                #    breakpoint()

                bg_mask3 = (1 - fg_mask3) * img_mask2

                if (fg_mask3.sum(dim=(1, 2)) == 0).any():
                    # Very rare cases. Safe to skip.
                    print("WARNING: fg_mask3 has all-one masks.")
                    continue
                if (bg_mask3.sum(dim=(1, 2)) == 0).any():
                    # Should never happen.
                    print("WARNING: bg_mask3 has all-one masks.")
                    breakpoint()

                # subj, bg: subject embedding,         background embedding.
                # mf,   mb: mask foreground locations, mask background locations.
                # sum(dim=(1,2)): avoid summing across the batch dimension. 
                # It's meaningless to average among the instances.
                subj_score_at_mf = subj_score * fg_mask3
                subj_score_at_mb = subj_score * bg_mask3
                bg_score_at_mf   = bg_score   * fg_mask3
                bg_score_at_mb   = bg_score   * bg_mask3

                avg_subj_score_at_mf = subj_score_at_mf.sum(dim=(1,2), keepdim=True)  / fg_mask3.sum(dim=(1,2), keepdim=True)
                avg_subj_score_at_mb = subj_score_at_mb.sum(dim=(1,2), keepdim=True)  / bg_mask3.sum(dim=(1,2), keepdim=True)
                avg_bg_score_at_mf   = bg_score_at_mf.sum(dim=(1,2), keepdim=True)    / fg_mask3.sum(dim=(1,2), keepdim=True)
                avg_bg_score_at_mb   = bg_score_at_mb.sum(dim=(1,2), keepdim=True)    / bg_mask3.sum(dim=(1,2), keepdim=True)

                if 'DEBUG' in os.environ:
                    print(f'layer {unet_layer_idx}')
                    print(f'avg_subj_score_at_mf: {avg_subj_score_at_mf.mean():.4f}, avg_subj_score_at_mb: {avg_subj_score_at_mb.mean():.4f}')
                    print(f'avg_bg_score_at_mf:   {avg_bg_score_at_mf.mean():.4f},   avg_bg_score_at_mb:   {avg_bg_score_at_mb.mean():.4f}')
                
                # Encourage avg_subj_score_at_mf (subj_score averaged at foreground locations) 
                # to be at least larger by mfmb_contrast_score_margin = 1 than 
                # subj_score_at_mb at any background locations.
                # If not, clip() > 0, incurring a loss.
                loss_layer_subj_mfmb_contrast      = torch.clip(subj_score_at_mb + mfmb_contrast_score_margin   - avg_subj_score_at_mf,   min=0)
                # Compared to masked_mean(), mean() is like dynamically reducing the loss weight when more and more 
                # activations conform to the margin restrictions.
                loss_layer_subj_mfmb_contrast      = loss_layer_subj_mfmb_contrast.mean()
                # Encourage avg_bg_score_at_mb (bg_score averaged at background locations)
                # to be at least larger by mfmb_contrast_score_margin = 1 than
                # bg_score_at_mf at any foreground locations.
                # If not, clip() > 0, incurring a loss.
                loss_layer_bg_mfmb_contrast        = torch.clip(bg_score_at_mf   + mfmb_contrast_score_margin   - avg_bg_score_at_mb,     min=0)
                loss_layer_bg_mfmb_contrast        = loss_layer_bg_mfmb_contrast.mean()
                # Encourage avg_subj_score_at_mf (subj_score averaged at foreground locations)
                # to be at least larger by subj_bg_contrast_at_mf_score_margin = 0.8 than
                # bg_score_at_mf at any foreground locations.
                # loss_layer_subj_bg_contrast_at_mf is usually 0, as avg_bg_score_at_mf 
                # usually takes a much smaller value than avg_subj_score_at_mf.
                loss_layer_subj_bg_contrast_at_mf  = torch.clip(bg_score_at_mf    + subj_bg_contrast_at_mf_score_margin - avg_subj_score_at_mf,  min=0)
                loss_layer_subj_bg_contrast_at_mf  = loss_layer_subj_bg_contrast_at_mf.mean()
                # Encourage avg_bg_score_at_mb (bg_score averaged at background locations)
                # to be at least larger by subj_bg_contrast_at_mf_score_margin = 0.2 than
                # subj_score_at_mb at any background locations.
                loss_layer_bg_subj_contrast_at_mb  = torch.clip(subj_score_at_mb   + bg_subj_contrast_at_mb_score_margin - avg_bg_score_at_mb,   min=0)
                loss_layer_bg_subj_contrast_at_mb  = loss_layer_bg_subj_contrast_at_mb.mean()
                # loss_layer_subj_bg_contrast_at_mf is usually 0, 
                # so loss_fg_mask_align is much smaller than loss_bg_mask_align.
                loss_fg_mask_align          += loss_layer_subj_mfmb_contrast \
                                                * attn_align_layer_weight * emb_mfmb_contrast_scale
                loss_bg_mask_align          += loss_layer_bg_mfmb_contrast \
                                                * attn_align_layer_weight * emb_mfmb_contrast_scale
                loss_fg_bg_mask_contrast    += (loss_layer_subj_bg_contrast_at_mf + loss_layer_bg_subj_contrast_at_mb) \
                                                * attn_align_layer_weight * fgbg_emb_contrast_scale
                #print(f'layer {unet_layer_idx}')
                #print(f'subj_contrast: {loss_layer_subj_contrast:.4f}, subj_bg_contrast_at_mf: {loss_layer_subj_bg_contrast_at_mf:.4f},')
                #print(f"bg_contrast:   {loss_layer_bg_contrast:.4f},   subj_bg_contrast_at_mb: {loss_layer_subj_bg_contrast_at_mb:.4f}")

        if batch_have_fg_mask.any():
            loss_fg_mask_align          = masked_mean(loss_fg_mask_align,       loss_fg_mask_align > 0)
            loss_bg_mask_align          = masked_mean(loss_bg_mask_align,       loss_bg_mask_align > 0)
            loss_fg_bg_mask_contrast    = masked_mean(loss_fg_bg_mask_contrast, loss_fg_bg_mask_contrast > 0)
        # Otherwise, loss_fg_mask_align, loss_bg_mask_align, loss_fg_bg_mask_contrast are all initial 0.
        # loss_fg_bg_complementary doesn't need fg_mask.

        return loss_fg_bg_complementary, loss_fg_mask_align, loss_bg_mask_align, loss_fg_bg_mask_contrast

    # SSB_SIZE: subject sub-batch size.
    # bg_indices could be None if iter_flags['use_background_token'] = False.
    def calc_fg_bg_xlayer_consist_loss(self, unet_attns_or_scores,
                                       subj_indices, 
                                       bg_indices, 
                                       SSB_SIZE, 
                                       delta_loss_emb_mask, 
                                       img_mask):
        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        attn_align_layer_weights = { #7:  1., 8: 1.,
                                    #12: 1.,
                                    16: 1., 17: 1.,
                                    18: 1.,
                                    19: 1.,   20: 0.5, 
                                    21: 0.5,  22: 0.5, 
                                    23: 0.25, 24: 0.25, 
                                   }
        # 16-18: feature maps 16x16.
        # 19-21: feature maps 32x32.
        # 22-24: feature maps 64x64.
        # The weight is inversely proportional to the feature map size of the layer 
        # below each layer, which the current layer is resized to.
        # The larger the feature map, the more details the layer captures, and 
        # fg/bg loss hurts more high-frequency details, therefore it has a smalll weight.
                        
        attn_align_xlayer_maps = { 16: 12, 17: 16, 18: 17, 19: 18, 
                                   20: 19, 21: 20, 22: 21, 23: 22, 24: 23 }

        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)

        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # In each instance, subj_indices has K_fg elements, 
        # and bg_indices has K_bg elements or 0 elements 
        # (if iter_flags['use_background_token'] = False)
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                          [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # bg_indices: ([0, 1, 2, 3, 4, 5, 6, 7], [11, 12, 34, 29, 11, 12, 34, 29]).
        # SSB_SIZE = 2, so we only keep instances indexed by [0, 1].
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:SSB_SIZE*K_fg], subj_indices[1][:SSB_SIZE*K_fg])

        if bg_indices is not None:
            # K_bg: 1 or 2, number of embeddings per background token.
            K_bg = len(bg_indices[0]) // len(torch.unique(bg_indices[0]))
            # bg_indices: ([0, 1], [11, 12]).
            bg_indices = (bg_indices[0][:SSB_SIZE*K_bg], bg_indices[1][:SSB_SIZE*K_bg])

        # align_weight_mask: [SSB_SIZE, 1, 77, 1] (recon iters) or [2*SSB_SIZE, 1, 77, 1] (comp iters)
        # => [SSB_SIZE, 77]
        xlayer_align_weight_mask = delta_loss_emb_mask[:SSB_SIZE, 0, :, 0].clone()
        # Set higher weights for the foreground and background embeddings.
        xlayer_align_weight_mask[subj_indices] = 4
        if bg_indices is not None:
            xlayer_align_weight_mask[bg_indices] = 4

        loss_fg_xlayer_consist = 0
        loss_bg_xlayer_consist = 0

        for unet_layer_idx, unet_attn in unet_attns_or_scores.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            # [2, 8, 256, 77] => [2, 77, 8, 256]
            attn_mat        = unet_attn.permute(0, 3, 1, 2)
            # [2, 8, 64, 77]  => [2, 77, 8, 64]
            attn_mat_xlayer = unet_attns_or_scores[attn_align_xlayer_maps[unet_layer_idx]].permute(0, 3, 1, 2)
            
            # Make sure attn_mat_xlayer is always smaller than attn_mat.
            # So we always scale down attn_mat to match attn_mat_xlayer.
            if attn_mat_xlayer.shape[-1] > attn_mat.shape[-1]:
                attn_mat, attn_mat_xlayer = attn_mat_xlayer, attn_mat

            # H: 16, Hx: 8
            H  = int(np.sqrt(attn_mat.shape[-1]))
            Hx = int(np.sqrt(attn_mat_xlayer.shape[-1]))

            # Why taking mean over 8 heads: In a CA layer, FFN features added to the CA features. 
            # If there were no FFN pathways, the CA features of the previous CA layer would be better aligned
            # with the current CA layer (ignoring the transformations by the intermediate conv layers).
            # Therefore, the CA features in the current CA layer may be barely aligned 
            # with the CA features in the L_xlayer-th CA layer. So it's of little benefit
            # to align the corresponding heads across CA layers.
            # subj_attn:        [8, 8, 256] -> [2, 4, 8, 256] -> mean among 8 heads -> [2, 4, 256] 
            subj_attn        = attn_mat[subj_indices].reshape(       SSB_SIZE, K_fg, *attn_mat.shape[2:]).mean(dim=2)
            # subj_attn_xlayer: [8, 8, 64]  -> [2, 4, 8, 64]  -> mean among 8 heads -> [2, 4, 64]
            subj_attn_xlayer = attn_mat_xlayer[subj_indices].reshape(SSB_SIZE, K_fg, *attn_mat_xlayer.shape[2:]).mean(dim=2)

            # subj_attn: [2, 4, 256] -> [2, 4, 16, 16] -> [2, 4, 8, 8] -> [2, 4, 64]
            subj_attn = subj_attn.reshape(SSB_SIZE, -1, H, H)
            subj_attn = F.interpolate(subj_attn, size=(Hx, Hx), mode="bilinear", align_corners=False)
            subj_attn = subj_attn.reshape(SSB_SIZE, -1, Hx*Hx)

            if bg_indices is not None:
                # bg_attn:   [4, 8, 256] -> [2, 2, 8, 256] -> sum over 8 attention heads -> [2, 2, 256]
                # 8: 8 attention heads. Last dim 256: number of image tokens.
                bg_attn   = attn_mat[bg_indices].reshape(SSB_SIZE, K_bg, *attn_mat.shape[2:]).mean(dim=2)
                bg_attn_xlayer   = attn_mat_xlayer[bg_indices].reshape(SSB_SIZE, K_bg, *attn_mat_xlayer.shape[2:]).mean(dim=2)
                # bg_attn: [2, 2, 256] -> [2, 2, 16, 16] -> [2, 2, 8, 8] -> [2, 2, 64]
                bg_attn   = bg_attn.reshape(SSB_SIZE, -1, H, H)
                bg_attn   = F.interpolate(bg_attn, size=(Hx, Hx), mode="bilinear", align_corners=False)
                bg_attn   = bg_attn.reshape(SSB_SIZE, -1, Hx*Hx)
            
            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]

            if img_mask is not None:
                # img_mask: [4, 1, 64, 64] -> [2, 1, 64, 64]
                img_mask = img_mask[:SSB_SIZE]
                # img_mask2: [2, 1, 64, 64] -> [2, 1, 8, 8] -> [2, 1, 64]
                # "1" will be broadcasted to 2: subj_attn_xlayer: [2, 2, 64]
                img_mask2 = resize_mask_for_feat_or_attn(subj_attn_xlayer, img_mask, "img_mask", mode="nearest|bilinear")
                img_mask2 = img_mask2.reshape(SSB_SIZE, 1, -1)
                subj_attn           = subj_attn         * img_mask2
                subj_attn_xlayer    = subj_attn_xlayer  * img_mask2

                if bg_indices is not None:
                    bg_attn             = bg_attn           * img_mask2
                    bg_attn_xlayer      = bg_attn_xlayer    * img_mask2

            # exponent = 2: exponent is 3 by default, which lets the loss focus on large activations.
            # But we don't want to only focus on large activations. So set it to 2.
            # ref_grad_scale = 1: No gradient scaling will be applied to the reference embedding, e.g.,
            # subj_attn_xlayer and bg_attn_xlayer, so that they will be optimized as well.
            # do_demean_first=True: If not do_demean_first, then at low-attention locations of one attention map,
            # the loss on the other attention map is small (the cosine loss is weighted by 
            # both attention values, so if one of them is small, the loss at this location is small).
            # After demeaning, there are no low-attention locations, so all locations are better aligned.
            '''
            loss_layer_fg_xlayer_consist = calc_delta_loss(subj_attn, subj_attn_xlayer, 
                                                            exponent=2,    
                                                            do_demean_first=True,
                                                            first_n_dims_to_flatten=2, 
                                                            ref_grad_scale=1,
                                                            debug=False)
            '''
            loss_layer_fg_xlayer_consist = ortho_l2loss(subj_attn, subj_attn_xlayer, mean=True)
            loss_fg_xlayer_consist += loss_layer_fg_xlayer_consist * attn_align_layer_weight
            
            if bg_indices is not None:
                # do_demean_first=False: Regularizations are more relaxed on the background embeddings.
                '''
                loss_layer_bg_xlayer_consist = calc_delta_loss(bg_attn, bg_attn_xlayer,
                                                                exponent=2,    
                                                                do_demean_first=True,
                                                                first_n_dims_to_flatten=2, 
                                                                ref_grad_scale=1,
                                                                debug=False)
                '''
                loss_layer_bg_xlayer_consist = ortho_l2loss(bg_attn, bg_attn_xlayer, mean=True)
                loss_bg_xlayer_consist += loss_layer_bg_xlayer_consist * attn_align_layer_weight

        return loss_fg_xlayer_consist, loss_bg_xlayer_consist

    # BS: BLOCK_SIZE, not batch size.    
    def calc_subj_comp_ortho_loss(self, unet_ks, unet_vs, unet_attns_or_scores,
                                  subj_indices, delta_loss_emb_mask, 
                                  BS, cls_grad_scale=0.05):

        # Discard the first few bottom layers from the orthogonal loss.
        k_ortho_layer_weights = { #7:  1., 8: 1.,
                                12: 1.,
                                16: 1., 17: 1.,
                                18: 1.,
                                19: 1., 20: 1., 
                                21: 1., 22: 1., 
                                23: 1., 24: 1.,     
                                }
        
        v_ortho_layer_weights = { #7:  1., 8: 1.,
                                12: 1.,
                                16: 1., 17: 1.,
                                18: 0.5,
                                19: 0.5, 20: 0.5, 
                                21: 0.25, 22: 0.25, 
                                23: 0.25, 24: 0.25,                                
                               }
        
        # Normalize the weights above so that each set sum to 1.
        k_ortho_layer_weights = normalize_dict_values(k_ortho_layer_weights)
        v_ortho_layer_weights = normalize_dict_values(v_ortho_layer_weights)
        # K_fg: 4, number of embeddings per subject token.
        K_fg   = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # In each instance, subj_indices has K_fg times as many elements as bg_indices.
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # ind_subj_subj_B, ind_subj_subj_N: [1, 1, 1, 1], [5, 6, 7, 8].
        # Shift ind_subj_subj_B by BS, so that it points to subj comp embeddings.
        ind_subj_subj_B, ind_subj_subj_N = subj_indices[0][:BS*K_fg] + BS, subj_indices[1][:BS*K_fg]
        # Shift ind_subj_subj_B by 2 * BS, so that it points to cls comp embeddings.
        ind_cls_subj_B,  ind_cls_subj_N  = ind_subj_subj_B + 2 * BS, ind_subj_subj_N

        # delta_loss_emb_mask: [4, 1, 77, 1] => [4, 77]
        comp_extra_emb_mask = delta_loss_emb_mask[:, 0, :, 0].clone()
        # Mask out the foreground embeddings.
        comp_extra_emb_mask[subj_indices] = 0
        # comp_extra_emb_mask: subj single, subj comp, cls single, cls comp extra emb mask.
        # subj_comp_extra_emb_mask: [1, 77].
        subj_comp_extra_emb_mask = comp_extra_emb_mask.chunk(4)[1]
        # subj_comp_extra_emb_indices: ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        #                               [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        subj_comp_extra_emb_indices = torch.where(subj_comp_extra_emb_mask > 0)

        # K_comp: 18.
        K_comp = len(subj_comp_extra_emb_indices[0]) // len(torch.unique(subj_comp_extra_emb_indices[0]))
        # BS = 1, so we only keep instances indexed by [0].
        # subj_comp_extra_emb_indices: ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        #                               [ 1,  2,  3,  4,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]).
        # Shift batch indices by BS, so that it points to subj comp embeddings.
        # ind_subj_comp_B, ind_subj_comp_N: ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        #                                    [ 1,  2,  3,  4,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]).
        ind_subj_comp_B, ind_subj_comp_N = subj_comp_extra_emb_indices[0] + BS, subj_comp_extra_emb_indices[1]
        # Shift ind_subj_comp_B by 2 * BS, so that it points to cls comp embeddings.
        ind_cls_comp_B,  ind_cls_comp_N  = ind_subj_comp_B + 2 * BS, ind_subj_comp_N

        loss_subj_comp_key_ortho   = 0
        loss_subj_comp_value_ortho = 0
        loss_subj_comp_attn_comple = 0

        emb_kq_do_demean_first = False

        for unet_layer_idx, unet_seq_k in unet_ks.items():
            if (unet_layer_idx not in k_ortho_layer_weights):
                continue

            loss_layer_subj_comp_key_ortho = \
                calc_layer_subj_comp_k_or_v_ortho_loss(unet_seq_k, K_fg, K_comp, BS,
                                                        ind_subj_subj_B, ind_subj_subj_N, 
                                                        ind_cls_subj_B,  ind_cls_subj_N, 
                                                        ind_subj_comp_B, ind_subj_comp_N, 
                                                        ind_cls_comp_B,  ind_cls_comp_N,
                                                        do_demean_first=emb_kq_do_demean_first, 
                                                        cls_grad_scale=cls_grad_scale)
            loss_layer_subj_comp_value_ortho = \
                calc_layer_subj_comp_k_or_v_ortho_loss(unet_vs[unet_layer_idx], K_fg, K_comp, BS,
                                                        ind_subj_subj_B, ind_subj_subj_N, 
                                                        ind_cls_subj_B,  ind_cls_subj_N, 
                                                        ind_subj_comp_B, ind_subj_comp_N, 
                                                        ind_cls_comp_B,  ind_cls_comp_N,
                                                        do_demean_first=emb_kq_do_demean_first, 
                                                        cls_grad_scale=cls_grad_scale)
                    
            k_ortho_layer_weight = k_ortho_layer_weights[unet_layer_idx]
            v_ortho_layer_weight = v_ortho_layer_weights[unet_layer_idx]
            loss_subj_comp_key_ortho   += loss_layer_subj_comp_key_ortho   * k_ortho_layer_weight
            loss_subj_comp_value_ortho += loss_layer_subj_comp_value_ortho * v_ortho_layer_weight

            ###########   loss_subj_comp_attn_comple   ###########
            # attn_mat: [4, 8, 64, 77] => [4, 77, 8, 64]
            attn_mat = unet_attns_or_scores[unet_layer_idx]
            attn_mat = attn_mat.permute(0, 3, 1, 2)
            # subj_subj_attn: [4, 8, 64] -> [1, 4, 8, 64]
            subj_subj_attn = attn_mat[ind_subj_subj_B, ind_subj_subj_N].reshape(BS, K_fg, *attn_mat.shape[2:])
            cls_subj_attn  = attn_mat[ind_cls_subj_B,  ind_cls_subj_N].reshape(BS,  K_fg,   *attn_mat.shape[2:])
            # subj_comp_attn: [18, 8, 64] -> [1, 18, 8, 64] sum among K_comp embeddings -> [1, 1, 8, 64]
            # 8: 8 attention heads. Last dim 64: number of image tokens.
            subj_comp_attn = attn_mat[ind_subj_comp_B, ind_subj_comp_N].reshape(BS, K_comp, *attn_mat.shape[2:]).sum(dim=1, keepdim=True)
            cls_comp_attn  = attn_mat[ind_cls_comp_B,  ind_cls_comp_N].reshape(BS,  K_comp, *attn_mat.shape[2:]).sum(dim=1, keepdim=True)

            # The orthogonal projection of subj_subj_attn against subj_comp_attn.
            # subj_comp_attn will broadcast to the K_fg dimension.
            # ortho_subtract() is scale-invariant w.r.t. subj_comp_attn. So no need to normalize it.
            subj_comp_attn_diff = ortho_subtract(subj_subj_attn, subj_comp_attn)
            # The orthogonal projection of cls_subj_attn against cls_comp_attn.
            # cls_comp_attn will broadcast to the K_fg dimension.
            # ortho_subtract() is scale-invariant w.r.t. cls_comp_attn.  So no need to normalize it.
            cls_comp_attn_diff  = ortho_subtract(cls_subj_attn,  cls_comp_attn)
            # The two orthogonal projections should be aligned. That is, subj_subj_attn is allowed to
            # vary only along the direction of the orthogonal projections of class attention.

            # exponent = 2: exponent is 3 by default, which lets the loss focus on large activations.
            # But we don't want to only focus on large activations. So set it to 2.
            # ref_grad_scale = 0.05: small gradients will be BP-ed to the subject embedding,
            # to make the two attention maps more complementary (expect the loss pushes the 
            # subject embedding to a more accurate point).
            loss_layer_comp_attn_comple = calc_delta_loss(subj_comp_attn_diff, cls_comp_attn_diff, 
                                                          exponent=2,    
                                                          do_demean_first=False,
                                                          first_n_dims_to_flatten=3, 
                                                          ref_grad_scale=cls_grad_scale)
            
            loss_subj_comp_attn_comple += loss_layer_comp_attn_comple * k_ortho_layer_weight   

        return loss_subj_comp_key_ortho, loss_subj_comp_value_ortho, loss_subj_comp_attn_comple

    # Intuition: In distillation iterations, if comp_init_with_fg_area, then at fg_mask areas, x_start is initialized with 
    #            the noisy input images. (Usually in distillation iterations, x_start is initialized as pure noise.)
    #            Essentially, it's to mask the background out of the input images. 
    #            Therefore, features under single prompts should be close to the foreground of the original images (features not computed yet).
    #            features under comp prompts should align with the foreground of the original images as well.
    #            So features under comp prompts should be close to features under single prompts, at fg_mask areas.
    #            (The features at background areas under comp prompts are the compositional contents, which shouldn't be regularized.) 
    # BS: block size, not batch size.
    def calc_comp_fg_bg_preserve_loss(self, unet_feats, unet_attns_or_scores, 
                                      fg_mask, batch_have_fg_mask, subj_indices, BS):
        # No masks available. loss_comp_subj_fg_feat_preserve, loss_comp_subj_bg_attn_suppress are both 0.
        if fg_mask is None or batch_have_fg_mask.sum() == 0:
            return 0, 0, 0

        feat_distill_layer_weights = { # 7:  1., 8: 1.,   
                                        12: 1.,
                                        16: 1., 17: 1.,
                                        18: 1.,
                                        19: 1., 20: 1., 
                                        21: 1., 22: 1., 
                                        23: 1., 24: 1., 
                                     }

        # fg_mask is 4D. So expand batch_have_fg_mask to 4D.
        # *_4b means it corresponds to a 4-block batch (batch size = 4 * BS).
        fg_mask_4b = fg_mask            * batch_have_fg_mask.view(-1, 1, 1, 1)
        bg_mask_4b = (1 - fg_mask_4b)   * batch_have_fg_mask.view(-1, 1, 1, 1)

        # K_fg: 4, number of embeddings per subject token.
        K_fg   = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # ind_subj_subj_B_1b, ind_subj_subj_N_1b: [0, 0, 0, 0], [5, 6, 7, 8].
        ind_subj_subj_B_1b, ind_subj_subj_N_1b = subj_indices[0][:BS*K_fg], subj_indices[1][:BS*K_fg]
        ind_subj_B = torch.cat([ind_subj_subj_B_1b,            ind_subj_subj_B_1b + BS,
                                ind_subj_subj_B_1b + 2 * BS,   ind_subj_subj_B_1b + 3 * BS], dim=0)
        ind_subj_N = ind_subj_subj_N_1b.repeat(4)

        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weights  = normalize_dict_values(feat_distill_layer_weights)
        single_feat_or_attn_grad_scale  = 0.02
        single_feat_or_attn_grad_scaler = gen_gradient_scaler(single_feat_or_attn_grad_scale)

        loss_comp_subj_fg_feat_preserve = 0
        loss_comp_subj_fg_attn_preserve = 0
        loss_comp_subj_bg_attn_suppress = 0

        for unet_layer_idx, unet_feat in unet_feats.items():
            if unet_layer_idx not in feat_distill_layer_weights:
                continue
            feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

            # fg_feat_mask_4b: [1, 1, 64, 64] => [1, 1, 8, 8]
            fg_feat_mask_4b = resize_mask_for_feat_or_attn(unet_feat, fg_mask_4b, "fg_mask_4b", 
                                                           mode="nearest|bilinear", warn_on_all_zero=False)

            # Mask out the background features, and only keep the foreground features.
            fg_feat = unet_feat * fg_feat_mask_4b
            # each is [1, 1280, 16, 16]
            subj_single_fg_feat, subj_comp_fg_feat, mix_single_fg_feat, mix_comp_fg_feat \
                = fg_feat.chunk(4)

            do_feat_pooling = True
            feat_pool_kernel_size = 4
            feat_pool_stride      = 2
            # feature pooling: allow small perturbations of the locations of pixels.
            if do_feat_pooling:
                pooler = nn.AvgPool2d(feat_pool_kernel_size, stride=feat_pool_stride)
            else:
                pooler = nn.Identity()

            subj_single_fg_feat = pooler(subj_single_fg_feat)
            subj_comp_fg_feat   = pooler(subj_comp_fg_feat)
            mix_single_fg_feat  = pooler(mix_single_fg_feat)
            mix_comp_fg_feat    = pooler(mix_comp_fg_feat)

            # single_feat_or_attn_grad_scale = 0.02. 
            # feat_*_single are used as references, so their gradients are reduced to very small.
            subj_single_fg_feat_gs = single_feat_or_attn_grad_scaler(subj_single_fg_feat)
            mix_single_fg_feat_gs  = single_feat_or_attn_grad_scaler(mix_single_fg_feat)
            # normalized_l2loss() or L2 loss (get_loss()) perform worse than ortho_l2loss().
            loss_layer_subj_fg_feat_preserve = ortho_l2loss(subj_comp_fg_feat, subj_single_fg_feat_gs, mean=True)
            loss_layer_mix_fg_feat_preserve  = ortho_l2loss(mix_comp_fg_feat,  mix_single_fg_feat_gs,  mean=True)
            # A small weight to the preservation loss on mix instances. 
            # The requirement of preserving foreground features is not as strict as that of preserving
            # subject features, as the former is only used to facilitate composition.
            mix_fg_preserve_loss_scale = 0.1
            loss_comp_subj_fg_feat_preserve += (loss_layer_subj_fg_feat_preserve 
                                                + loss_layer_mix_fg_feat_preserve * mix_fg_preserve_loss_scale) \
                                                * feat_distill_layer_weight

            ##### unet_attn fg preservation loss & bg suppression loss #####
            unet_attn = unet_attns_or_scores[unet_layer_idx]
            # attn_mat: [4, 8, 64, 77] => [4, 77, 8, 64] => sum over 8 attention heads => [4, 77, 64]
            attn_mat = unet_attn.permute(0, 3, 1, 2).sum(dim=2)
            # subj_subj_attn: [4, 77, 64] -> [4 * K_fg, 64] -> [4, K_fg, 64]
            subj_attn = attn_mat[ind_subj_B, ind_subj_N].reshape(BS * 4, K_fg, -1)

            # fg_attn_mask_4b: [1, 1, 64, 64] => [1, 1, 8, 8]
            fg_attn_mask_4b = resize_mask_for_feat_or_attn(attn_mat, fg_mask_4b, "fg_mask_4b", 
                                                           mode="nearest|bilinear", warn_on_all_zero=False)
            # fg_attn_mask_flat_4b: [1, 1, 8, 8] => [4, 1, 64]
            fg_attn_mask_flat_4b  = fg_attn_mask_4b.reshape(BS * 4, 1, -1)

            subj_fg_attn = subj_attn * fg_attn_mask_flat_4b

            subj_single_subj_fg_attn, subj_comp_subj_fg_attn, mix_single_subj_fg_attn, mix_comp_subj_fg_attn \
                = subj_fg_attn.chunk(4)
            
            subj_single_subj_fg_attn_gs = single_feat_or_attn_grad_scaler(subj_single_subj_fg_attn)
            mix_single_subj_fg_attn_gs  = single_feat_or_attn_grad_scaler(mix_single_subj_fg_attn)

            ##### loss_comp_subj_fg_attn_preserve #####
            loss_layer_subj_subj_fg_attn_contrast = ortho_l2loss(subj_comp_subj_fg_attn, subj_single_subj_fg_attn_gs)
            loss_layer_mix_subj_fg_attn_contrast  = ortho_l2loss(mix_comp_subj_fg_attn,  mix_single_subj_fg_attn_gs)
            loss_comp_subj_fg_attn_preserve += (loss_layer_subj_subj_fg_attn_contrast 
                                                + loss_layer_mix_subj_fg_attn_contrast * mix_fg_preserve_loss_scale) \
                                                * feat_distill_layer_weight
            
            ##### loss_comp_subj_bg_attn_suppress #####
            bg_attn_mask_4b = resize_mask_for_feat_or_attn(attn_mat, bg_mask_4b, "bg_mask_4b",
                                                             mode="nearest|bilinear", warn_on_all_zero=True)
            # bg_attn_mask_flat_4b: [4, 1, 8, 8] => [4, 1, 64]
            bg_attn_mask_flat_4b  = bg_attn_mask_4b.reshape(BS * 4, 1, -1)
            if bg_attn_mask_flat_4b.sum() == 0:
                breakpoint()
            
            subj_bg_attn = subj_attn * bg_attn_mask_flat_4b
            # subj_subj_bg_attn: attention of subj embeddings in subj instances on background areas.
            # It's not the attention of background embeddings.
            # mix_subj_bg_attn:  attention of mix embeddings (corresponding to the subj embeddings) in class instances 
            # on background areas.
            subj_subj_bg_attn, mix_subj_bg_attn = subj_bg_attn.chunk(2)

            # Simply suppress the subj attention on background areas. 
            # No need to use attn_*_single as references.
            loss_layer_subj_bg_attn_suppress = masked_mean(subj_subj_bg_attn, subj_subj_bg_attn > 0)
            loss_layer_mix_bg_attn_suppress  = masked_mean(mix_subj_bg_attn,  mix_subj_bg_attn  > 0)
            mix_bg_attn_suppress_loss_scale = 0.2
            loss_comp_subj_bg_attn_suppress += (loss_layer_subj_bg_attn_suppress 
                                                + loss_layer_mix_bg_attn_suppress * mix_bg_attn_suppress_loss_scale) \
                                                * feat_distill_layer_weight
                    
        return loss_comp_subj_fg_feat_preserve, loss_comp_subj_fg_attn_preserve, loss_comp_subj_bg_attn_suppress

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    def cache_and_log_generations(self, samples, img_colors, max_cache_size=48):
        self.generation_cache.append(samples)
        self.generation_cache_img_colors.append(img_colors)
        self.num_cached_generations += len(samples)

        if self.num_cached_generations >= max_cache_size:
            grid_folder = self.logger._save_dir + f'/samples'
            os.makedirs(grid_folder, exist_ok=True)
            grid_filename = grid_folder + f'/{self.cache_start_iter:04d}-{self.global_step:04d}.png'
            save_grid(self.generation_cache, self.generation_cache_img_colors, 
                      grid_filename, 12, do_normalize=True)
            print(f"{self.num_cached_generations} generations saved to {grid_filename}")
            
            # Clear the cache. If num_cached_generations > max_cache_size,
            # some samples at the end of the cache will be discarded.
            self.generation_cache = []
            self.generation_cache_img_colors = []
            self.num_cached_generations = 0
            self.cache_start_iter = self.global_step + 1

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
            
            uc = self.get_learned_conditioning(N * [""])
            sample_scaled, _ = self.sample_log(cond=c, 
                                               batch_size=N, 
                                               ddim=use_ddim, 
                                               ddim_steps=ddim_steps,
                                               eta=ddim_eta,                                                 
                                               unconditional_guidance_scale=5.0,
                                               unconditional_conditioning=uc)
            log["samples_scaled"] = self.decode_first_stage(sample_scaled)

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    # configure_optimizers() is called later as a hook function by pytorch_lightning.
    # call stack: main.py: trainer.fit()
    # ...
    # pytorch_lightning/core/optimizer.py:
    # optim_conf = model.trainer._call_lightning_module_hook("configure_optimizers", pl_module=model)
    def configure_optimizers(self):
        # self.learning_rate and self.weight_decay are set in main.py.
        # self.learning_rate = base_learning_rate * 2, 2 is the batch size.
        lr = self.learning_rate
        model_lr = self.model_lr
        weight_decay = self.weight_decay

        OptimizerClass = torch.optim.AdamW

        # If using textual inversion, then embedding_manager is not None.
        if self.embedding_manager is not None: 
            embedding_params = list(self.embedding_manager.optimized_parameters())
            # unfreeze_model:
            # Are we allowing the base model to train? If so, set two different parameter groups.
            if self.unfreeze_model: 
                model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
                opt = OptimizerClass([{"params": embedding_params, "lr": lr}, {"params": model_params}], lr=model_lr)
            # Otherwise, train only embedding
            else:
                opt = OptimizerClass(embedding_params, lr=lr, weight_decay=weight_decay)
        else:
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.ap

            opt = OptimizerClass(params, lr=lr, weight_decay=weight_decay)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            
            self.scheduler = scheduler[0]['scheduler']
            return [opt], scheduler
        
        return opt

    # configure_opt_embedding() is never called.
    def configure_opt_embedding(self):
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.embedding_manager.optimized_parameters():
            param.requires_grad = True

        lr = self.learning_rate
        params = list(self.embedding_manager.optimized_parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    # configure_opt_model() is never called.
    def configure_opt_model(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.embedding_manager.optimized_parameters():
            param.requires_grad = True

        model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
        embedding_params = list(self.embedding_manager.optimized_parameters())
        return torch.optim.AdamW([{"params": embedding_params, "lr": self.learning_rate}, {"params": model_params}], lr=self.model_lr)

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):

        if not self.unfreeze_model: # If we are not tuning the model itself, zero-out the checkpoint content to preserve memory.
            checkpoint.clear()
        
        if os.path.isdir(self.trainer.checkpoint_callback.dirpath) and self.embedding_manager is not None:
            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, "embeddings.pt"))
            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, f"embeddings_gs-{self.global_step}.pt"))


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # diffusion_model: UNetModel
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    # t: a 1-D batch of timesteps (during training: randomly sample one timestep for each instance).
    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            # For textual inversion, there's usually only one element tensor in c_crossattn.
            # So we take c_crossattn[0] directly, instead of torch.cat() on a list of tensors.
            if isinstance(c_crossattn[0], tuple):
                c_static_emb, c_in, extra_info = c_crossattn[0]
            else:
                c_static_emb    = c_crossattn[0]
                c_in            = None
                extra_info      = None
            # c_static_emb = torch.cat(c_crossattn, 1)
            # self.diffusion_model: UNetModel.
            #if 'iter_type' in extra_info:
            #    print(extra_info['iter_type'], c_in)
            out = self.diffusion_model(x, t, context=c_static_emb, context_in=c_in, extra_info=extra_info)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs

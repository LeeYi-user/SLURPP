# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from slurpp.diffusers_utils import _replace_unet_conv_out, _replace_unet_conv_in
from slurpp.slurpp_pipeline import SlurppPipeline
from src.util import metric
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence

from diffusers import DDIMScheduler

import time
from termcolor import colored
from src.util.myutils import *

from src.trainer.trainer_util import  get_predicted_original_sample
from torch.nn import L1Loss
import piq.ssim as piq_ssim
from src.util.metric import lpips_loss


class SlurppTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: SlurppPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        real_vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: SlurppPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.real_vis_loaders: List[DataLoader] = real_vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        self.input_fields = getattr(self.cfg.trainer, 'inputs', ['diff'])
        print("input fields")
        print(self.input_fields)
        self.output_fields = getattr(self.cfg.trainer, 'output', ['sfo'])
        print("output fields")
        print(self.output_fields)
        self.out_imgs = len(self.output_fields)
        self.in_imgs = len(self.output_fields) + len(self.input_fields)
        self.in_channels = 4 * (self.in_imgs)
        self.out_channels = 4 * (len(self.output_fields))
        self.dual = getattr(self.cfg, "dual", False)
        # Adapt input layers
        self.all_inputs = self.input_fields.copy()
        self.all_outputs = self.output_fields.copy() 

        if self.dual:
            self.input_fields2 = getattr(self.cfg.trainer, 'inputs2', ['u'])
            self.output_fields2 = getattr(self.cfg.trainer, 'output2', ['bc','ill'])
            self.all_inputs += self.input_fields2
            self.all_outputs += self.output_fields2
            if self.in_channels != self.model.unet.unet1.config["in_channels"]:
                print("load unet")
                self.model.unet.unet1 = _replace_unet_conv_in(self.model.unet.unet1 , len(self.input_fields), len(self.output_fields))
                self.model.unet.unet1 = _replace_unet_conv_out(self.model.unet.unet1 , len(self.output_fields))
                self.model.unet.unet2  = _replace_unet_conv_in(self.model.unet.unet2 , len(self.input_fields2), len(self.output_fields2))
                self.model.unet.unet2  = _replace_unet_conv_out(self.model.unet.unet2 , len(self.output_fields2))
                print(f"self.model.unet.unet1.config['in_channels']: {self.model.unet.unet1.config['in_channels']}")
                print(f"self.model.unet.unet1.config['out_channels']: {self.model.unet.unet1.config['out_channels']}")
                print(f"self.model.unet.unet2.config['in_channels']: {self.model.unet.unet2.config['in_channels']}")
                print(f"self.model.unet.unet2.config['out_channels']: {self.model.unet.unet2.config['out_channels']}")
            self.model.unet.add_additional_params()

            if hasattr(self.cfg, "dual_load"):
                self.model.unet.unet1.load_state_dict(torch.load(os.path.join(self.cfg.dual_load.unet1, 'diffusion_pytorch_model.bin'), weights_only=False))
                self.model.unet.unet2.load_state_dict(torch.load(os.path.join(self.cfg.dual_load.unet2, 'diffusion_pytorch_model.bin'), weights_only=False))

            self.dual_loss_weight_1 = getattr(self.cfg, 'dual_loss_weight_1', 0.6)
        else:
            self.model.unet = _replace_unet_conv_in(self.model.unet , len(self.input_fields), len(self.output_fields))
            self.model.unet = _replace_unet_conv_out(self.model.unet , len(self.output_fields))


        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)
        self.model.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        if getattr(self.cfg, "dual_cross_only", False):
            paramters = list(self.model.unet.unet1.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross.parameters()) + \
                list(self.model.unet.unet2.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross.parameters()) 
            self.optimizer = Adam(paramters, lr=lr)
        else:
            self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        scheduler_path = os.path.join(base_ckpt_dir,cfg.trainer.training_noise_scheduler.pretrained_path,"scheduler",)
        print(f"loading scheduler from {scheduler_path}")
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        self.train_metrics = MetricTracker(*["loss"])


        all_metrics = []
        for out_field in self.all_outputs:
            all_metrics += [f"{m.__name__}_{out_field}" for m in self.metric_funcs]
        self.train_vis_metrics = MetricTracker(*all_metrics)

        all_metrics = []
        for out_field in self.all_outputs:
            all_metrics += [f"{m.__name__}_{out_field}" for m in self.metric_funcs]
        self.val_metrics = MetricTracker(*all_metrics)

        # main metric for best checkpoint saving
        self.main_val_metric = f"{cfg.validation.main_val_metric}_{self.output_fields[0]}"
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        # assert (
        #     self.main_val_metric in cfg.eval.eval_metrics
        # ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.num_hour_between_val = self.cfg.trainer.num_hour_between_val
        self.num_hour_between_real_vis = self.cfg.trainer.num_hour_between_real_vis
        self.has_been_valed_at_begining = False
        self.has_been_real_vised_at_begining = False
        
        self.realvis = hasattr(self.cfg, 'real_data_visulization')

        # Multi-resolution noise
        self.apply_multi_res_noise = hasattr(self.cfg, 'multi_res_noise') 
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )
        else:
            pass

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        self.upsample = hasattr(self.cfg.trainer, 'upsample') 
        if self.upsample:
            print(f"upsampling images by {self.cfg.trainer.upsample} times ")
            self.up = torch.nn.Upsample(scale_factor=self.cfg.trainer.upsample, mode='bilinear')
            self.downsample_factor = 1/self.cfg.trainer.upsample
        else:
            print("no upsampling")

        self.timestamp_begining = time.time()
        self.alpha_schedule = torch.sqrt(self.training_noise_scheduler.alphas_cumprod).to(self.device)
        self.sigma_schedule = torch.sqrt(1 - self.training_noise_scheduler.alphas_cumprod).to(self.device)
        self.one_step = getattr(self.cfg, 'one_step', False)
        if self.one_step:
            print("one step")
            self.cfg.validation.denoising_steps = 1
        if self.cfg.validation.denoising_steps == 1:
            self.model.scheduler.config.timestep_spacing = 'trailing'
        
        self.rgb_loss = getattr(self.cfg, 'rgb_loss', False)
        self.gt_label_loss = getattr(self.cfg, 'gt_label_loss', True)

        if self.rgb_loss:
            print("rgb loss")

            train_metrics = ["loss"]
            if self.gt_label_loss:
                for out_field in self.all_outputs:
                    train_metrics += [f"ssim_{out_field}", f"l1_{out_field}", f"lpips_{out_field}"]
                

            self.reconstruction_loss = getattr(self.cfg, 'reconstruction_loss', False)
            if self.reconstruction_loss:
                train_metrics += [f"ssim_u", f"l1_u", f"lpips_u"]

            self.train_metrics = MetricTracker(*train_metrics)


    def normalize_imgs(self, rgb, gamma = None):
        rgb = torch.clamp(rgb, 0, 1)
        if gamma is not None:
            rgb = rgb ** gamma
        rgb_norm: torch.Tensor = rgb * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.device).to(self.model.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        return rgb_norm
    
    def process_batch(self, batch, fields, latent=False):
        res = []
        for field in fields:
            res.append(batch["imgs"][field])
        if self.upsample:
            for i in range(len(res)):
                res[i] = self.up(res[i])
        for i in range(len(res)):
            res[i] = self.normalize_imgs(res[i])    
            
        if latent:
            res_latent = [self.model.encode_rgb(r) for r in res]
            res_latent = torch.cat(res_latent, dim=1)
            return res, res_latent
        
        return res

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0
        self.timestamp_last_val = time.time()
        self.timestamp_last_real_vis = time.time()
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch_id, batch in enumerate( tqdm(self.train_loader, desc=f"training epoch {epoch}")):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # Get data
                with torch.no_grad():
                    batch_size = batch["imgs"][self.input_fields[0]].shape[0]

                    input_rgbs, input_latents = self.process_batch(batch, self.all_inputs, latent=True)

                    if self.gt_label_loss:
                        output_rgbs, output_latent = self.process_batch(batch, self.all_outputs, latent=True)
                    else:
                        output_rgbs = None
                        output_latent = torch.ones_like(input_latents[:,:4].repeat(1, len(self.all_outputs), 1, 1))


                # Sample a random timestep for each image
                if self.one_step:
                    timesteps = torch.ones((batch_size,), device=device) * (self.scheduler_timesteps - 1) # 999
                    timesteps = timesteps.long()
                else:
                    timesteps = torch.randint(
                        0,
                        self.scheduler_timesteps,
                        (batch_size,),
                        device=device,
                        generator=rand_num_generator,).long()  # [B]

                if self.one_step:
                    noisy_latents = torch.zeros_like(output_latent)
                else:
                    if self.apply_multi_res_noise:
                        strength = self.mr_noise_strength
                        if self.annealed_mr_noise:
                            # calculate strength depending on t
                            strength = strength * (timesteps / self.scheduler_timesteps)
                        noise = multi_res_noise_like(
                            output_latent,
                            strength=strength,
                            downscale_strategy=self.mr_noise_downscale_strategy,
                            generator=rand_num_generator,
                            device=device,
                        )
                    else:
                        noise = torch.randn(
                            output_latent.shape,
                            device=device,
                            generator=rand_num_generator,
                        )  # [B, 4, h, w]

                    noisy_latents = self.training_noise_scheduler.add_noise(
                        output_latent, noise, timesteps
                    )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]
                
                cat_latents = torch.cat(
                    [input_latents, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()


                # Predict the noise residual
                model_pred = self.model.unet(
                    cat_latents, timesteps, text_embed
                ).sample  # [B, 4, h, w]
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                
                if self.one_step:
                    prediction = get_predicted_original_sample(model_pred, 
                                                            timesteps, 
                                                            noisy_latents, 
                                                            self.prediction_type, 
                                                            self.alpha_schedule, 
                                                            self.sigma_schedule)
                    
                    target = output_latent

                    if self.rgb_loss:
                        prediction = prediction/ self.model.rgb_latent_scale_factor
                        decoded_rgbs =[]
                        for i in range(prediction.shape[1]//4):
                            pred_x0_i = prediction[:, i*4:(i+1)*4, :, :]
                            pred_x0_i = self.model.vae.post_quant_conv(pred_x0_i)
                            pred_x0_i = self.model.vae.decoder(pred_x0_i)
                            decoded_rgbs.append(pred_x0_i)

                        decoded_rgbs = torch.cat(decoded_rgbs, dim=1)


                        decoded_rgbs = decoded_rgbs * 0.5 + 0.5
                        decoded_rgbs = torch.clamp(decoded_rgbs, 0, 1)
                        def compute_loss(recon, target):
                            ssim_img = piq_ssim(recon, target).mean()   
                            l1_img = L1Loss()(recon, target).mean()
                            lpips_img = lpips_loss(recon, target).mean()
                            img_loss = 0.5 * (1 - ssim_img) + l1_img + 0.5 * lpips_img
                            return img_loss, ssim_img.item(), l1_img.item(), lpips_img.item()

                        img_loss_total = 0
                        weights = [0.5, 0.25, 0.25]

                        if self.gt_label_loss:
                            output_rgbs = torch.cat(output_rgbs, dim=1)
                            output_rgbs = output_rgbs * 0.5 + 0.5
                            all_ssim = []
                            all_l1 = []
                            all_lpips = []
                                    
                            for i in range(decoded_rgbs.shape[1]//3):
                                img_loss, ssim_img, l1_img, lpips_img = compute_loss(decoded_rgbs[:, i*3:(i+1)*3, :, :].float(), output_rgbs[:, i*3:(i+1)*3, :, :].float())
                                img_loss_total += weights[i] * img_loss
                                all_ssim.append(ssim_img)
                                all_l1.append(l1_img)
                                all_lpips.append(lpips_img)

                        if self.reconstruction_loss:
                            
                            u_image = ((decoded_rgbs[:, 0:3, :, :])) * (decoded_rgbs[:, 6:9, :, :]) + (decoded_rgbs[:, 3:6, :, :])
                            u_image = torch.clamp(u_image, 0, 1)
                            u_loss, ssim_u, l1_u, lpips_u = compute_loss(u_image, (input_rgbs[0].float() * 0.5 + 0.5))

                            if self.gt_label_loss:
                                loss = img_loss_total + 0.2 * u_loss
                            else:
                                loss = u_loss
                            del u_image

                        else:
                            loss = img_loss_total

                        del decoded_rgbs, output_rgbs

                else:
                    prediction = model_pred

                    if "sample" == self.prediction_type:
                        target = output_latent
                    elif "epsilon" == self.prediction_type:
                        target = noise
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            output_latent, noise, timesteps
                        )  # [B, 4, h, w]
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")
                
                if not self.rgb_loss:

                    if self.dual:
                        model_pred1, model_pred2 = torch.split(prediction, [4 * len(self.output_fields), 4 * len(self.output_fields2)], dim=1)
                        target1, target2 = torch.split(target, [4 * len(self.output_fields), 4 * len(self.output_fields2)], dim=1)
                        latent_loss1 = self.loss(model_pred1.float(), target1.float()).mean()
                        latent_loss2 = self.loss(model_pred2.float(), target2.float()).mean()
                        loss = self.dual_loss_weight_1 * latent_loss1 + (1 - self.dual_loss_weight_1) * latent_loss2
                    else:
                        latent_loss = self.loss(prediction.float(), target.float())
                        loss = latent_loss.mean()
                    
                    

                self.train_metrics.update("loss", loss.item())

                if self.rgb_loss:
                    if self.gt_label_loss:
                        for i, out_field in enumerate(self.all_outputs):
                            self.train_metrics.update(f"ssim_{out_field}", all_ssim[i])
                            self.train_metrics.update(f"l1_{out_field}", all_l1[i])
                            self.train_metrics.update(f"lpips_{out_field}", all_lpips[i])
                    if self.reconstruction_loss:
                        self.train_metrics.update("ssim_u", ssim_u)
                        self.train_metrics.update("l1_u", l1_u)
                        self.train_metrics.update("lpips_u", lpips_u)

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.cuda.empty_cache()

                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    self.num_data_visited = self.effective_iter * self.cfg.dataloader.effective_batch_size

                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.num_data_visited,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.num_data_visited,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.num_data_visited,
                    )

                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=False)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    
    def _train_step_callback(self):
        """Executed after every iteration"""
        # modified by XMY

        num_data_visited = self.effective_iter * self.cfg.dataloader.effective_batch_size
        self.num_data_visited = num_data_visited

        if (time.time() - self.timestamp_last_val) / 3600 > self.num_hour_between_val or num_data_visited > 400 and not self.has_been_valed_at_begining:

            self.has_been_valed_at_begining = True
            self.timestamp_last_val = time.time()


            num_hours_elapsed_since_begining = (time.time() - self.timestamp_begining) / 3600
            print(colored(f"Validation at {self.effective_iter} iteration", 'red'))
            print(colored(f"Elapsed time: {num_hours_elapsed_since_begining} hours", 'red'))

            tb_logger.writer.add_scalar(
                "hours",
                num_hours_elapsed_since_begining,
                global_step=num_data_visited,
            )
            # Save backup (with a larger interval, without training states)
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

            # Validation
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=False)
            if self.gt_label_loss:
                self.validate(num_data_visited=num_data_visited)
            self.in_evaluation = False
        
        if (time.time() - self.timestamp_last_real_vis) / 3600 > self.num_hour_between_real_vis or num_data_visited > 400 and not self.has_been_real_vised_at_begining:
            self.has_been_real_vised_at_begining = True
            self.timestamp_last_real_vis = time.time()
            print(colored(f"Real data visualization at {self.effective_iter} iteration", 'red'))
            num_hours_elapsed_since_begining = (time.time() - self.timestamp_begining) / 3600
            print(colored(f"Elapsed time: {num_hours_elapsed_since_begining} hours", 'red'))
            self.visualize(num_data_visited=num_data_visited)
            
        torch.cuda.empty_cache()

    def validate(self, num_data_visited=None):
        for i, val_loader in enumerate(self.val_loaders):
            # val_dataset_name = val_loader.dataset.disp_name
            val_dataset_name = "val"
            val_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), val_dataset_name
            )
            val_out_dir_all_iters = f'{self.out_dir_vis}/all_iters_val'
            os.makedirs(val_out_dir_all_iters, exist_ok=True)
            os.makedirs(val_out_dir, exist_ok=True)
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, 
                metric_tracker=self.val_metrics,
                save_to_dir=val_out_dir,
                save_to_dir_all_iters=val_out_dir_all_iters,
                dataset_type="val",
                vis_frequency=1,
                num_data_visited=num_data_visited,
                # vis_frequency=self.cfg.trainer.val_vis_frequency 
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=num_data_visited,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path="",
            )

            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )


        
        for i, vis_loader in enumerate(self.vis_loaders):  # visualize a subset of training data
            vis_dataset_name = "vis"
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            vis_out_dir_all_iters = f'{self.out_dir_vis}/all_iters_vis'
            os.makedirs(vis_out_dir_all_iters, exist_ok=True)
            os.makedirs(vis_out_dir, exist_ok=True)

            print("visualize selected train")
            train_vis_metric_dic = self.validate_single_dataset(data_loader=vis_loader, 
                                                        metric_tracker=self.train_vis_metrics,
                                                        dataset_type="train_vis",
                                                        save_to_dir=vis_out_dir,
                                                        save_to_dir_all_iters=vis_out_dir_all_iters,
                                                        num_data_visited=num_data_visited,  
                                                        val_frequency=1)
            
            tb_logger.log_dic(
                {f"train/flash2d_train/{k}": v for k, v in train_vis_metric_dic.items()},
                global_step=num_data_visited,
            )
    
    @torch.no_grad()
    def visualize(self, num_data_visited=None):
        self.model.vae.enable_tiling()
        for data_loader in self.real_vis_loaders:
            vis_dataset_name = "real_vis"
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            vis_out_dir_all_iters = f'{self.out_dir_vis}/all_iters_real_vis'
            os.makedirs(vis_out_dir_all_iters, exist_ok=True)
            os.makedirs(vis_out_dir, exist_ok=True)

            
            for batch_id, batch in enumerate(
                tqdm(data_loader, desc=f"visualizing real data at iteration {self.effective_iter}")
            ):
                inputs  = []
                inputs = [batch["imgs"][field] for field in self.all_inputs]
                inputs = [self.normalize_imgs(i) for i in inputs]

                output_pred = self.model(
                    inputs,
                    denoising_steps=self.cfg.validation.denoising_steps,
                    show_progress_bar=True,
                    is_dual=self.dual
                )

                output_pred_gc =  output_pred
                
                vis_image_list = []
                label_list = []

                vis_fields = self.input_fields

                for field in vis_fields:
                    img = batch["imgs"][field]
                    # img_gc =  self.real_vis_resize(torch.clamp(img, 0, 1)) 
                    vis_image_list.append(img[0].detach().cpu().float())
                    label_list.append(f'input_{field}')

                for i, field in enumerate(self.all_outputs):
                    vis_image_list.append(output_pred_gc[i].detach().cpu().float())
                    label_list.append(f'pred_{field}')

                concat_image = concat_images_with_labels(vis_image_list, label_list, font_size=20)
                concat_image.save(f'{vis_out_dir}/{batch_id:04d}.png')
                concat_image.save(f'{vis_out_dir_all_iters}/{batch_id:04d}_{self.effective_iter:06d}.png')

                concat_image_torch = torch.tensor(np.array(concat_image)).permute(2, 0, 1)
                tb_logger.writer.add_image(f"real_vis/{batch_id:04d}", concat_image_torch, global_step=num_data_visited)
                
                torch.cuda.empty_cache()
        self.model.vae.disable_tiling()

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
        save_to_dir_all_iters: str = None,
        dataset_type: str = None,
        val_frequency = 1,
        vis_frequency = 1,
        num_data_visited = None
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for batch_id, batch in enumerate(
            tqdm(data_loader, desc=f"validating {dataset_type} at iteration {self.effective_iter}")
        ):
            if batch_id % val_frequency != 0:
                continue

            inputs = self.process_batch(batch, self.all_inputs, latent=False)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            output_pred = self.model(
                inputs,
                denoising_steps=self.cfg.validation.denoising_steps,
                generator=generator,
                show_progress_bar=False,
                is_dual=self.dual
            )

            # Evaluate
            output_pred = output_pred.to(self.device)
            
            output_pred_gc = self.process_output_pred(output_pred)


            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                for field_idx, field in enumerate(self.all_outputs):
                    metric_value = met_func(torch.clamp(output_pred_gc[field_idx: field_idx + 1], 0, 1), torch.clamp(batch["imgs"][field].to(self.device), 0, 1))
                    metric_tracker.update(f"{_metric_name}_{field}", metric_value.item())

            if (batch_id % vis_frequency== 0):
                vis_image_list = []
                label_list = []
                vis_fields = self.input_fields
                if self.dual:
                    vis_fields = list(set(vis_fields + self.input_fields2))
                for field in vis_fields:
                    img = batch["imgs"][field].to(self.device)
                    img_gc =  (torch.clamp(img, 0, 1))
                    vis_image_list.append(img_gc[0].detach().cpu().float())
                    label_list.append(f'input_{field}')
                
                for  field_idx, field in enumerate(self.all_outputs):
                    output_gt = batch["imgs"][field]
                    output_gt = (torch.clamp(output_gt, 0, 1))
                    output_pred = output_pred_gc[field_idx:  field_idx + 1]
                    vis_image_list.append(output_pred[0].detach().cpu().float())
                    label_list.append(f'pred_{field}')
                    
                    vis_image_list.append(output_gt[0].detach().cpu().float())
                    label_list.append(f'Target_{field}')


                concat_image = concat_images_with_labels(vis_image_list, label_list, font_size=20)
                concat_image.save(f'{save_to_dir}/{batch_id:04d}.png')
                concat_image.save(f'{save_to_dir_all_iters}/{batch_id:04d}_{self.effective_iter:06d}.png')
                # convert to torch tensor
                concat_image_torch = torch.tensor(np.array(concat_image)).permute(2, 0, 1)
                tb_logger.writer.add_image(f"{dataset_type}_vis/{batch_id:04d}", concat_image_torch, global_step=num_data_visited)

        return metric_tracker.result()

    def process_output_pred(self, output_pred):
        if self.upsample:
            output_pred = torch.nn.functional.interpolate(output_pred, scale_factor=self.downsample_factor, mode='bilinear',antialias=True)
        output_pred_gc =  output_pred
        return output_pred_gc

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        if self.dual:
            self.model.unet.save_pretrained(ckpt_dir, safe_serialization=False)
            unet_path = ckpt_dir
        else:
            unet_path = os.path.join(ckpt_dir, "unet")
            self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device, weights_only=False)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"), weights_only=False)
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
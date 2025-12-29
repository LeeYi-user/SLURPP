# limitations under the License.
import os 
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import _chunked_feed_forward
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

import xformers

class DualUNetCondition( ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self, unet1 = None, unet2 = None, unet_path1 = None, unet_path2 = None):
        super().__init__()
        if unet1 is None:
            self.unet1 = UNet2DConditionModel.from_pretrained(unet_path1, subfolder="unet")
        else:
            self.unet1 = unet1
        if unet2 is None:
            self.unet2 = UNet2DConditionModel.from_pretrained(unet_path2, subfolder="unet")
        else:
            self.unet2 = unet2

        self.mid_block_transformer = UniDrectionTransformer()
        # self.add_additional_params()
    
    def save_pretrained(self, ckpt_path, safe_serialization=True):
        unet_path1 = os.path.join(ckpt_path, "unet1")
        unet_path2 = os.path.join(ckpt_path, "unet2")
        self.unet1.save_pretrained(unet_path1, safe_serialization = safe_serialization)
        self.unet2.save_pretrained(unet_path2, safe_serialization = safe_serialization)
    
    def add_additional_params(self):
        attn1 = self.unet1.mid_block.attentions[0].transformer_blocks[0].attn1
        if not hasattr(attn1, 'to_q_cross'):
            self.unet1.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross= nn.Linear(attn1.query_dim, attn1.inner_dim, bias=attn1.use_bias)
            nn.init.constant_(self.unet1.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross.weight, 0)
        attn2 = self.unet2.mid_block.attentions[0].transformer_blocks[0].attn1
        if not hasattr(attn2, 'to_q_cross'):
            self.unet2.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross= nn.Linear(attn2.query_dim, attn2.inner_dim, bias=attn2.use_bias)
            nn.init.constant_(self.unet2.mid_block.attentions[0].transformer_blocks[0].attn1.to_q_cross.weight, 0)

    def load_checkpoint(self, model_path1, model_path2, device="cuda"):
        # _model_path1 = os.path.join(ckpt_path, "unet1", "diffusion_pytorch_model.bin")
        self.model.unet1.load_state_dict(
            torch.load(model_path1, map_location=device, weights_only=False)
        )
        self.model.unet1.to(device)

        # _model_path2 = os.path.join(ckpt_path, "unet2", "diffusion_pytorch_model.bin")
        self.model.unet2.load_state_dict(
            torch.load(model_path2, map_location=device, weights_only=False)
        )
        self.model.unet2.to(device)
    
    def breakup_sample(self, sample):
        offset = 0

        unet1_out = self.unet1.config["out_channels"]
        unet2_out = self.unet2.config["out_channels"]
        unet1_in = self.unet1.config["in_channels"] - unet1_out
        unet2_in = self.unet2.config["in_channels"] - unet2_out

        sample_1_in = sample[:, offset:offset + unet1_in, :, :]
        offset += unet1_in
        # print(f"offset after unet1_in: {offset}")
        sample_2_in = sample[:, offset:offset + unet2_in, :, :]
        offset += unet2_in
        # print(f"offset after unet2_in: {offset}")
        sample_1_out = sample[:, offset:offset + unet1_out, :, :]
        offset += unet1_out
        # print(f"offset after unet1_out: {offset}")
        sample_2_out = sample[:, offset:offset + unet2_out, :, :]
        # print(f"{sample_1_in.shape}, {sample_2_in.shape}, {sample_1_out.shape}, {sample_2_out.shape}")

        sample_1 = torch.cat([sample_1_in, sample_1_out], dim=1)
        sample_2 = torch.cat([sample_2_in, sample_2_out], dim=1)
        return sample_1, sample_2
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.unet1.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
        
        # sample1, sample2 = sample.chunk(2, dim=0)
        sample1, sample2 = self.breakup_sample(sample)
        batch_size = sample1.shape[0]

        # 0. center input if necessary
        if self.unet1.config.center_input_sample:
            sample1 = 2 * sample1 - 1.0
            sample2 = 2 * sample2 - 1.0

        # 1. time
        #branch 1
        t_emb1 = self.unet1.get_time_embed(sample=sample1, timestep=timestep)
        emb1 = self.unet1.time_embedding(t_emb1, timestep_cond)
        #branch 2
        t_emb2 = self.unet2.get_time_embed(sample=sample2, timestep=timestep)
        emb2 = self.unet2.time_embedding(t_emb2, timestep_cond)

        encoder_hidden_states = self.unet1.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample1 = self.unet1.conv_in(sample1)
        sample2 = self.unet2.conv_in(sample2)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None

        #downblock branch 1 
        down_block_res_samples1 = (sample1,)
        for downsample_block in self.unet1.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}

                sample1, res_samples = downsample_block(
                    hidden_states=sample1,
                    temb=emb1,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=None,
                    **additional_residuals,
                )
            else:
                sample1, res_samples = downsample_block(hidden_states=sample1, temb=emb1)

            down_block_res_samples1 += res_samples
        
        #downblock branch 2

        down_block_res_samples2 = (sample2,)
        for downsample_block in self.unet2.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}

                sample2, res_samples = downsample_block(
                    hidden_states=sample2,
                    temb=emb2,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    **additional_residuals,
                )
            else:
                sample2, res_samples = downsample_block(hidden_states=sample2, temb=emb2)

            down_block_res_samples2 += res_samples


        # 4. mid
        # sample1 = self.unet1.mid_block(
        #     sample1,
        #     emb1,
        #     encoder_hidden_states=encoder_hidden_states,
        #     cross_attention_kwargs=cross_attention_kwargs,
        # )
        # sample2 = self.unet2.mid_block(
        #     sample2,
        #     emb2,
        #     encoder_hidden_states=encoder_hidden_states,
        #     cross_attention_kwargs=cross_attention_kwargs,
        # )


        sample1 = self.unet1.mid_block.resnets[0](sample1, emb1)
        sample2 = self.unet2.mid_block.resnets[0](sample2, emb2)   
        
        batch_size, _, height, width = sample1.shape
        residual1 = sample1
        residual2 = sample2
        sample1, inner_dim1 = self.unet1.mid_block.attentions[0]._operate_on_continuous_inputs(sample1)
        sample2, inner_dim2 = self.unet2.mid_block.attentions[0]._operate_on_continuous_inputs(sample2)


        sample1, sample2 = self.mid_block_transformer(self.unet1.mid_block.attentions[0].transformer_blocks[0],
                                                      self.unet2.mid_block.attentions[0].transformer_blocks[0], 
                                                      sample1, 
                                                      sample2, 
                                                      encoder_hidden_states = encoder_hidden_states,
                                                      cross_attention_kwargs=cross_attention_kwargs)

        sample1 = self.unet1.mid_block.attentions[0]._get_output_for_continuous_inputs(
            hidden_states=sample1,
            residual=residual1,
            batch_size=batch_size,
            height=height,
            width=width,
            inner_dim=inner_dim1,
        )

        sample2 = self.unet2.mid_block.attentions[0]._get_output_for_continuous_inputs(
            hidden_states=sample2,
            residual=residual2,
            batch_size=batch_size,
            height=height,
            width=width,
            inner_dim=inner_dim2,
        )

        sample1 =  self.unet1.mid_block.resnets[1](sample1, emb1)
        sample2 =  self.unet2.mid_block.resnets[1](sample2, emb2)
        #end of mid block

        # 5. up
        #branch 1
        for i, upsample_block in enumerate(self.unet1.up_blocks):
            is_final_block = i == len(self.unet1.up_blocks) - 1

            res_samples1 = down_block_res_samples1[-len(upsample_block.resnets) :]
            down_block_res_samples1 = down_block_res_samples1[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples1[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample1 = upsample_block(
                    hidden_states=sample1,
                    temb=emb1,
                    res_hidden_states_tuple=res_samples1,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                )
            else:
                sample1 = upsample_block(
                    hidden_states=sample1,
                    temb=emb1,
                    res_hidden_states_tuple=res_samples1,
                    upsample_size=upsample_size,
                )
        #branch 2
        for i, upsample_block in enumerate(self.unet2.up_blocks):
            is_final_block = i == len(self.unet2.up_blocks) - 1

            res_samples2 = down_block_res_samples2[-len(upsample_block.resnets) :]
            down_block_res_samples2 = down_block_res_samples2[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples2[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample2 = upsample_block(
                    hidden_states=sample2,
                    temb=emb2,
                    res_hidden_states_tuple=res_samples2,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                )
            else:
                sample2 = upsample_block(
                    hidden_states=sample2,
                    temb=emb2,
                    res_hidden_states_tuple=res_samples2,
                    upsample_size=upsample_size,
                )
        ### end of up block

        # 6. post-process
        #branch 1
        if self.unet1.conv_norm_out:
            sample1 = self.unet1.conv_norm_out(sample1)
            sample1 = self.unet1.conv_act(sample1)
        sample1 = self.unet1.conv_out(sample1)

        #branch 2
        if self.unet2.conv_norm_out:
            sample2 = self.unet2.conv_norm_out(sample2)
            sample2 = self.unet2.conv_act(sample2)
        sample2 = self.unet2.conv_out(sample2)


        sample = torch.cat([sample1, sample2], dim=1)

        if not return_dict:
            return (sample,)   
        
                
        return UNet2DConditionOutput(sample=sample)

class UniDrectionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.uni_direction_attention = AttentionUniDirectionProcessor()  
    def __call__(
        self,
        transformer1: nn.Module,
        transformer2: nn.Module,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        #layer norm
        #branch 1
        norm_hidden_states1 = transformer1.norm1(hidden_states1)
        if transformer1.pos_embed is not None:
            norm_hidden_states1 = transformer1.pos_embed(norm_hidden_states1)
        #branch 2
        norm_hidden_states2 = transformer2.norm1(hidden_states2)
        if transformer2.pos_embed is not None:
            norm_hidden_states2 = transformer2.pos_embed(norm_hidden_states2)
        


        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output1, attn_output2 = self.uni_direction_attention(transformer1.attn1, 
                                                                  transformer2.attn1,
                                                                  norm_hidden_states1,
                                                                  norm_hidden_states2)
        
        #branch 1
        hidden_states1 = attn_output1 + hidden_states1
        if hidden_states1.ndim == 4:
            hidden_states1 = hidden_states1.squeeze(1)
        #branch 2
        hidden_states2 = attn_output2 + hidden_states2
        if hidden_states2.ndim == 4:
            hidden_states2 = hidden_states2.squeeze(1)
        
        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states1 = transformer1.fuser(hidden_states1, gligen_kwargs["objs"])
            hidden_states2 = transformer2.fuser(hidden_states2, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if transformer1.attn2 is not None:
            #layer norm
            norm_hidden_states1 = transformer1.norm2(hidden_states1)
            norm_hidden_states2 = transformer2.norm2(hidden_states2)

            if transformer1.pos_embed is not None:
                norm_hidden_states1 = transformer1.pos_embed(norm_hidden_states1)

            if transformer2.pos_embed is not None:
                    norm_hidden_states2 = transformer2.pos_embed(norm_hidden_states2)
            
            attn_output1 = transformer1.attn2(norm_hidden_states1,
                                        encoder_hidden_states=encoder_hidden_states,
                                        **cross_attention_kwargs)
            attn_output2 = transformer2.attn2(norm_hidden_states2,
                            encoder_hidden_states=encoder_hidden_states,
                            **cross_attention_kwargs)
            hidden_states1 = attn_output1 + hidden_states1
            hidden_states2 = attn_output2 + hidden_states2

        # 4. Feed-forward
        #branch 1
        if transformer1._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output1 = _chunked_feed_forward(transformer1.ff, norm_hidden_states1, transformer1._chunk_dim, transformer1._chunk_size)
        else:
            ff_output1 = transformer1.ff(norm_hidden_states1)
        
        #branch 2
        if transformer2._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output2 = _chunked_feed_forward(transformer2.ff, norm_hidden_states2, transformer2._chunk_dim, transformer2._chunk_size)
        else:
            ff_output2 = transformer2.ff(norm_hidden_states2)

        #branch 1
        hidden_states1 = ff_output1 + hidden_states1
        if hidden_states1.ndim == 4:
            hidden_states1 = hidden_states1.squeeze(1)

        #branch 2
        hidden_states2 = ff_output2 + hidden_states2
        if hidden_states2.ndim == 4:
            hidden_states2 = hidden_states2.squeeze(1)
        
        return hidden_states1, hidden_states2



class AttentionUniDirectionProcessor(nn.Module):
    def __call__(self,
        attn1: Attention,
        attn2: Attention,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        **kwargs,):
        
        residual1 = hidden_states1
        residual2 = hidden_states2
        
        input_ndim = hidden_states1.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states1.shape
            hidden_states1 = hidden_states1.view(batch_size, channel, height * width).transpose(1, 2)
            hidden_states2 = hidden_states2.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states1.shape[0]
        head_size = attn1.heads

        q1 = attn1.to_q(hidden_states1)
        k1 = attn1.to_k(hidden_states1)
        v1 = attn1.to_v(hidden_states1)
        q1_cross = attn1.to_q_cross(hidden_states1)

        q2 = attn2.to_q(hidden_states2)
        k2 = attn2.to_k(hidden_states2)
        v2 = attn2.to_v(hidden_states2)
        q2_cross = attn2.to_q_cross(hidden_states2)

        q1 = self.head_to_batch_dim(q1, head_size).contiguous()  #[batch_size * heads, seq_len, dim // heads]
        k1 = self.head_to_batch_dim(k1, head_size).contiguous()
        v1 = self.head_to_batch_dim(v1, head_size).contiguous()
        q1_cross = self.head_to_batch_dim(q1_cross, head_size).contiguous()

        q2 = self.head_to_batch_dim(q2, head_size).contiguous()  #[batch_size * heads, seq_len, dim // heads]
        k2 = self.head_to_batch_dim(k2, head_size).contiguous()
        v2 = self.head_to_batch_dim(v2, head_size).contiguous()
        q2_cross = self.head_to_batch_dim(q2_cross, head_size).contiguous()


        #Using the k and q from unet 1 for both unet 1 and unet 2
        # q = torch.cat([q1, q2], dim=1).contiguous()  #[batch_size * heads, 2 * seq_len, dim // heads]

        # hidden_states = xformers.ops.memory_efficient_attention(
        #     q, k1,v1, scale=attn1.scale
        # )        
        # hidden_states = hidden_states.to(q1.dtype)
        # hidden_states1, hidden_states2 = hidden_states.chunk(2, dim=1) #[batch_size * heads, seq_len, dim // heads]

        hidden_states1 = xformers.ops.memory_efficient_attention(q1, k1, v1, scale=attn1.scale) 
        hidden_states2 = xformers.ops.memory_efficient_attention(q2, k2,v2, scale=attn2.scale)
        hidden_states1_cross = xformers.ops.memory_efficient_attention(q1_cross, k2, v2, scale=attn2.scale)     
        hidden_states2_cross = xformers.ops.memory_efficient_attention(q2_cross, k1, v1, scale=attn1.scale)

        hidden_states1 =   hidden_states1 + hidden_states1_cross
        hidden_states2 =   hidden_states2 + hidden_states2_cross

        hidden_states1 = self.batch_to_head_dim(hidden_states1, head_size) #[batch_size // heads, seq_len, dim * heads]
        hidden_states2 = self.batch_to_head_dim(hidden_states2, head_size) #[batch_size // heads, seq_len, dim * heads]

        # linear proj
        hidden_states1 = attn1.to_out[0](hidden_states1)
        # dropout
        hidden_states1 = attn1.to_out[1](hidden_states1)
        
        # linear proj
        hidden_states2 = attn2.to_out[0](hidden_states2)
        # dropout
        hidden_states2 = attn2.to_out[1](hidden_states2)

        if input_ndim == 4:
            hidden_states1 = hidden_states1.transpose(-1, -2).reshape(batch_size, channel, height, width)
            hidden_states2 = hidden_states2.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn1.residual_connection:
            hidden_states1 = hidden_states1 + residual1
        
        if attn2.residual_connection:
            hidden_states2 = hidden_states2 + residual2

        hidden_states1 = hidden_states1 / attn1.rescale_output_factor

        hidden_states2 = hidden_states2 / attn2.rescale_output_factor

        return hidden_states1, hidden_states2   
    
    def batch_to_head_dim(self, tensor: torch.Tensor, head_size) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, head_size, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import logging
import os

import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from .slurpp_pipeline import SlurppPipeline

from diffusers import DDIMScheduler

def _replace_unet_conv_out(unet, output_imgs=1):
    out_channels = output_imgs * 4
    if out_channels == unet.config["out_channels"]:
        return unet
    _weight = unet.conv_out.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_out.bias.clone()  # [320]
    _bias = _bias.repeat((output_imgs,))  # Keep selected channel(s)
    _weight = _weight.repeat((output_imgs, 1, 1, 1))  # Keep selected channel(s)

    # new conv_in channel
    _n_convout_in_channel = unet.conv_out.in_channels
    _new_conv_out = Conv2d(
        _n_convout_in_channel, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_out.weight = Parameter(_weight)
    _new_conv_out.bias = Parameter(_bias)
    unet.conv_out = _new_conv_out
    logging.info("Unet conv_out layer is replaced")
    # replace config
    unet.config["out_channels"] = out_channels
    logging.info("Unet out config is updated")
    return unet

def _replace_unet_conv_in(unet, input_imgs=1, output_imgs=1):
    in_imgs = input_imgs + output_imgs
    in_channels = in_imgs * 4
    if in_channels == unet.config["in_channels"]:
        return unet
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    in_channels_ori = unet.conv_in.in_channels
    if in_channels_ori  == 4:
        _weight = _weight.repeat((1, in_imgs, 1, 1))  # Keep selected channel(s)
        _weight *= (1/in_imgs)
    else:
        _weight_input = _weight[:, :4].repeat((1, input_imgs, 1, 1))  # Keep selected channel(s)
        _weight_input = _weight_input * (1/input_imgs)
        _weight_output = _weight[:, 4:8].repeat((1, output_imgs, 1, 1))  # Keep selected channel(s)
        _weight_output = _weight_output * (1/output_imgs)
        _weight = torch.cat((_weight_input, _weight_output), dim=1)
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = in_channels
    logging.info("Unet config is updated")
    return unet


def load_stage1(model_path, checkpoint_path, cfg):
    pipe = SlurppPipeline.from_pretrained(
        model_path
    )


    inputs_fields = getattr(cfg.trainer, 'inputs', ['diff'])
    outputs_fields = getattr(cfg.trainer, 'output', ['bc', 'ill'])
    dual = getattr(cfg, 'dual', False)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if dual:
        from my_diffusers.dual_unet_condition import DualUNetCondition
        print(f"using dual unet")
        pipe.unet = DualUNetCondition(unet_path1 = model_path, unet_path2 = model_path)
        pipe.unet.unet1 = _replace_unet_conv_in(pipe.unet.unet1 , len(inputs_fields), len(outputs_fields))
        pipe.unet.unet1 = _replace_unet_conv_out(pipe.unet.unet1 , len(outputs_fields))
        inputs_fields2 = getattr(cfg.trainer, 'inputs2', ['diff'])
        outputs_fields2 = getattr(cfg.trainer, 'output2', ['bc', 'ill'])
        pipe.unet.unet2 = _replace_unet_conv_in(pipe.unet.unet2 , len(inputs_fields2), len(outputs_fields2))
        pipe.unet.unet2 = _replace_unet_conv_out(pipe.unet.unet2 , len(outputs_fields2))
        pipe.unet.add_additional_params()
        pipe.unet.unet1.load_state_dict(torch.load(os.path.join(checkpoint_path, 'unet1', 'diffusion_pytorch_model.bin'), weights_only=False))
        pipe.unet.unet2.load_state_dict(torch.load(os.path.join(checkpoint_path, 'unet2', 'diffusion_pytorch_model.bin'), weights_only=False))
        inputs_fields = inputs_fields + inputs_fields2
        outputs_fields = outputs_fields + outputs_fields2 
    else:

        pipe.unet = _replace_unet_conv_in(pipe.unet, len(inputs_fields), len(outputs_fields))
        pipe.unet = _replace_unet_conv_out(pipe.unet, len(outputs_fields))

        pipe.unet.load_state_dict(torch.load(os.path.join(checkpoint_path, 'unet', 'diffusion_pytorch_model.bin'), weights_only=False))

    return pipe, inputs_fields, outputs_fields, dual

# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from slurpp import load_stage1
from src.util.seeding import seed_all

from src.util.config_util import (
    recursive_load_config,
)

from slurpp.io import save_image, normalize_imgs

from torch.utils.data import DataLoader

from datasets.UR_real_data import UnderwaterRealDataset


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image underwater restoration with SLURPP."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None, 
        help="path to the data directory containg underwater images.",
    )


    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    
    parser.add_argument(
        "--inference_resolution",
        type=int,
        default=512,  # quantitative evaluation uses 50 steps
        help="Resolution for inference, default is 512.",
    )

    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")

    args = parser.parse_args()

    cfg = args.config
    cfg = recursive_load_config(args.config)
    checkpoint_path = args.checkpoint

    output_dir = args.output_dir

    denoise_steps = args.denoise_steps

    seed = args.seed


    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    # image_size = getattr(cfg.dataloader, 'image_size', 256)
    # image_size= 512
    image_size = args.inference_resolution

    data_dir = args.data_dir
    
    val_ds = UnderwaterRealDataset(root_dir = data_dir, image_size= image_size)
    dataloader = DataLoader(val_ds, 
                                num_workers=1, 
                                batch_size=1, 
                                shuffle=False)


    # -------------------- Model --------------------
    dtype = torch.float32

    base_ckpt_dir = os.environ["BASE_CKPT_DIR"]
    model_path = f"{base_ckpt_dir}/stable-diffusion-2"


    print(f"LOADING STAGE 1")

    pipe, inputs_fields, outputs_fields, dual = load_stage1(model_path, checkpoint_path, cfg)
    
    stage2 = args.stage2_checkpoint is not None
    if stage2:
        from stage2 import  CrossLatentUNet
        model = CrossLatentUNet(config_path = f"{model_path}/vae/config.json")
        checkpoint_path = args.stage2_checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print(f"          ===> Checkpoint Loaded From: {checkpoint_path} ...")
        del checkpoint
        # faster inference
        pipe.skip_connection = True
        pipe.vae_cld = model
        stage2 = False # in this case stage2 inference is integrated into stage1
    else:
        print(f"          ===> No stage2 checkpoint provided ...")


    print(f"inputs_fields {inputs_fields}")
    print(f"outputs_fields {outputs_fields}")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    

    # -------------------- Inference and saving --------------------
    
    one_step = getattr(cfg, 'one_step', False)
    if one_step:
        print(f"using one step inference")
        denoise_steps = 1
    if denoise_steps == 1:
        pipe.scheduler.config.timestep_spacing = 'trailing'
    

    for epoch in range(1):
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(dataloader, disable=None)):

                count = batch_id
                save_to_dir = f"{output_dir}/"

                os.makedirs(save_to_dir, exist_ok=True)
                
                inputs  = []
                for field in inputs_fields:
                    inputs.append(batch["imgs"][field])
                save_name = batch["imgs"]["name"][0]
                original_size = batch["imgs"].get("original_size", None)
                if original_size is not None:
                    original_size = (original_size[0].item(), original_size[1].item())  # Convert from tensor to tuple (width, height)

                for i in range(len(inputs)):
                    inputs[i] = normalize_imgs(inputs[i])
                
                output_pred = pipe(
                    inputs,
                    denoising_steps=denoise_steps,
                    show_progress_bar=False,
                    return_latent = True,
                    is_dual = dual,
                )

                
                output_pred_latent = output_pred[1]
                output_pred_stage_1 = output_pred[0]
                if stage2:
                    output_pred = output_pred_stage_1
                    composite_img = inputs[0]
                    output_pred_stage_2 = model(output_pred_latent[:, :4],composite_img )
                    output_pred[0:1] = (output_pred_stage_2 / 2 + 0.5).clamp(0, 1)
                else:
                    output_pred = output_pred_stage_1

                sample_metric = []
                output_pred = output_pred.to(device)
                
                #save images
                for i in range(len(outputs_fields)):
                    output_pred_gc = output_pred[i:i+1].clone()
                    save_image(f"{save_to_dir}/{save_name}_{outputs_fields[i]}.png", output_pred_gc, original_size=original_size)

                composite_img = batch["imgs"][inputs_fields[0]]
                composite_img = torch.clamp(composite_img, 0, 1).to(device)

                save_image(f"{save_to_dir}/{save_name}_composite_img.png", composite_img, original_size=original_size)
                
# Authors: Mingyang Xie, Tianfu Wang

from utils import *
from data import *
from network.CLUNet import  CrossLatentUNet
from data import ImageDataset_Decoder
from diffusers import AutoencoderKL
import piq.psnr as piq_psnr
import piq.ssim as piq_ssim
import math
from lr_scheduler import IterExponential
from torch.optim.lr_scheduler import LambdaLR
import lpips
from time import time
import argparse
from pathlib import Path
import tqdm as tqdm
import glob
from tqdm import tqdm
from pathlib import Path
from piq import ssim, psnr
import torch.nn as nn
import wandb

parser = argparse.ArgumentParser(
        description=""
)

parser.add_argument(
    "--field", type=str, required=True, help="which field to train"

)

parser.add_argument(
    "--output_dir", type=str, required=True, help="Output directory."
)

parser.add_argument(
    "--data_dir", type=str, required=True, help="Data directory."
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
    default=None,
    help="Checkpoint path or hub name.",
)

parser.add_argument(
    "--preserve_encoder",
    action="store_true",
    help="Whether to preserve the encoder.",
)

args = parser.parse_args()
print(args.config)
cfg = recursive_load_config(args.config)


def save_image(path, tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)

    image = None

    if tensor.size(0) == 3:
        # Convert 3-channel tensor (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor)  # Create an RGB image
        image.save(path)

    elif tensor.size(0) == 1:
        # Convert 1-channel tensor to grayscale (H, W)
        tensor = tensor.squeeze(0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor, mode='L')  # Create a grayscale image
        image.save(path)
    else:
        pass

root_dir = args.output_dir
data_dir = args.data_dir
base_ckpt_dir = os.environ["BASE_CKPT_DIR"]

field= args.field
img_size_out = cfg.dataloader.image_size
img_size_out = 512
model_path = os.path.join(base_ckpt_dir, cfg.model.pretrained_path)


# exp_name = f'debug_lpips_ssim_{field}' # name for this specific experiment
exp_name = f'debug_lpips_ssim_debug' # name for this specific experiment
save_frequency = 1
num_epochs = 50
init_lr=1e-5
final_lr=3e-8

#---------------------------------------------------------------------------------------
#--- environment
#---------------------------------------------------------------------------------------
seed_torch(42)
init_env()
device = torch.device('cuda') # device for training
save_dir = os.path.join(root_dir, exp_name)
num_workers = 8
run = wandb.init(project="AttUNet", name=exp_name, reinit=True, dir=f'{save_dir}/wandb', settings=wandb.Settings(start_method="fork"))
create_save_folder(save_dir, verbose=True)
for temp_name in ['checkpoints', 'codes', 'results', 'iters']:
    Path(f'{save_dir}/{temp_name}').mkdir(parents=True, exist_ok=True)
for file_path in glob.iglob(f'{os.getcwd()}/**/*.py', recursive = True): # https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/
    if 'backup' not in file_path:
        shutil.copy(file_path, f'{save_dir}/codes')

#-------------------------------------------------------------------------------------
#--- load data
#-------------------------------------------------------------------------------------
test_size = 20
batch_size = 4
upsample_factor = 1
# transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(int(img_size))])



places_dataset = ImageDataset_Decoder(data_dir, field, img_size_out = img_size_out)
test_dataset = ImageDataset_Decoder(data_dir, field,img_size_out = img_size_out)

img_inds = np.arange(len(places_dataset))
seed_torch(0)
np.random.shuffle(img_inds)
train_inds = img_inds[int(test_size):]
test_inds = img_inds[:int(test_size)]
train_dataset = torch.utils.data.Subset(places_dataset, train_inds)
test_dataset = torch.utils.data.Subset(test_dataset, test_inds)
train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=1, drop_last=True)

train_size = len(train_inds) 
print("train size: ", train_size)
total_iter_length = math.ceil(train_size / batch_size) * num_epochs

#-------------------------------------------------------------------------------------
#--- prepare for training
#-------------------------------------------------------------------------------------

model = CrossLatentUNet(config_path = f"{model_path}/vae/config.json")
autoencoder = AutoencoderKL.from_pretrained(model_path, subfolder='vae').to(device)
model.load_vae(autoencoder)
#load checkpoint 
checkpoint_path = args.checkpoint
if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"          ===> Checkpoint Loaded From: {checkpoint_path} ...")
    if args.preserve_encoder:
        print("preserving encoder")
        model.load_vae(autoencoder, encoder_only=True)
        model.encode = autoencoder.encode
    del checkpoint
    #done loading checkpoint
else:
    print(f"          ===> No checkpoint found, starting from scratch ...")

model = model.to(device)

if args.preserve_encoder:
    print("preserving encoder, only training the decoder and the zero convs")
    del autoencoder

    autoencoder = model
    

    param_list = [param for param in model.zero_conv_0.parameters()] + \
        [param for param in model.zero_conv_1.parameters()] + \
        [param for param in model.zero_conv_2.parameters()] + \
        [param for param in model.zero_conv_3.parameters()] + \
        [param for param in model.Decoder.parameters()] + \
        [param for param in model.post_quant_conv.parameters()]

    optimizer = torch.optim.Adam(param_list, lr=init_lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

lr_func = IterExponential(
    total_iter_length=total_iter_length,
    final_ratio=0.01,
    warmup_steps=100,
)
scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

last_val_time = time()
num_hour_between_val =2
total_hours = 0
not_valed = True

lpips_fn = lpips.LPIPS(net='alex').to("cuda")
def lpips_loss(output, target, valid_mask=None):
    return lpips_fn(target * 2.0 -1.0, output * 2.0 - 1.0)

def encode_rgb(rgb):
    rgb = (rgb * 2.0 - 1.0)
    rgb = autoencoder.encode(rgb).latent_dist.sample()
    rgb *= 0.18215
    return rgb

model.train()
for epoch in range(num_epochs):
    model.train()
    train_bar = tqdm(train_loader, desc='', disable=False)   
    for train_iter, train_batch in enumerate(train_bar, 1):

        torch.cuda.empty_cache()
        
        if (time() - last_val_time) / 3600 > num_hour_between_val or not_valed:
            not_valed = False
            print(f"          ===> Saving Results ...")

            last_val_time = time()
            total_hours += num_hour_between_val

            model.eval()
            with torch.no_grad():
                total_psnr = 0
                total_ssim = 0
                total_l1 = 0
                total_lpips = 0
                test_bar = tqdm(test_loader, desc='', disable=False)
                for test_iter, test_batch in enumerate(test_bar, 1):
                    target = test_batch['target_img'].to(device)
                    blurry_rgb = test_batch['blurry_img'].to(device)
                    composite_img = test_batch['composite_img'].to(device)

                    composite_img = composite_img * 2.0 - 1.0

                    blurry_latent = encode_rgb(blurry_rgb)
                    # blurry_latent = test_batch['latent'].to(device)
                    recon = model(blurry_latent, composite_img)
                    recon = (recon / 2 + 0.5).clamp(0, 1)

                    l1_loss = nn.L1Loss()(recon, target)
                    ssim = piq_ssim(recon, target)
                    
                    psnr = piq_psnr(recon.detach(), target.detach())
                    
                    lpips_l = lpips_loss(recon.detach(), target.detach()).mean()

                    total_psnr += psnr.item()
                    total_ssim += ssim.item()
                    total_l1 += l1_loss.item()
                    total_lpips += lpips_l.item()

                    test_bar.set_description(f"[Epoch {epoch}] ===> Test : PSNR: {psnr:.4f}, ssim: {ssim:.4f}, l1: {l1_loss:.4f}, lpips: {lpips_l:.4f}")
                    test_bar.refresh()
                    
                    torch.cuda.empty_cache()
                
                total_psnr /= test_size
                total_ssim /= test_size
                total_l1 /= test_size
                total_lpips /= test_size
                
                label_list = ["blurry", "composite", "target", "recon"]
                vis_image_list = [blurry_rgb[0].detach().cpu(), composite_img[0].detach().cpu(), target[0].detach().cpu(), recon[0].detach().cpu()]
                concat_image = concat_images_with_labels(vis_image_list, label_list, font_size=20)
                os.makedirs(f'{save_dir}/vis', exist_ok=True)
                concat_image.save(f'{save_dir}/vis/test_epoch_{epoch}_total_hours_{total_hours}.png')                

                wandb.log({ 'test psnr': total_psnr, 'test ssim': total_ssim,"test l1": total_l1, "test lpips": total_lpips})

                #SAVE MODEL
                save_model_name = f'{exp_name}_total_hours_{total_hours}'
                model_out_path = f'{save_dir}/checkpoints/{save_model_name}.pth'
                outs = {'total_hours': total_hours , 'state_dict': model.state_dict(),
                        'name': 0, 'optimizer' : optimizer.state_dict()}
                torch.save(outs, model_out_path)
                print(f"          ===> Checkpoint Saved At: {model_out_path} ...")
            
            model.train()
        
        
        # print(train_iter)
        optimizer.zero_grad()
        with torch.no_grad():
            target = train_batch['target_img'].to(device)
            blurry_rgb = train_batch['blurry_img'].to(device)
            composite_img = train_batch['composite_img'].to(device)

            composite_img = composite_img * 2.0 - 1.0

            blurry_latent = encode_rgb(blurry_rgb)
        
        recon = model(blurry_latent, composite_img)
        recon = (recon / 2 + 0.5).clamp(0, 1)

        lbda=0.5
        l1_loss = nn.L1Loss()(recon, target)
        ssim = piq_ssim(recon, target)
        lpips_l = lpips_loss(recon, target).mean()
        # loss = (1 - lbda) * l1_loss + lbda * (1 - ssim) + lpips_l * 0.5
        loss = 1 - ssim + lpips_l + l1_loss * 0.5
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        psnr = piq_psnr(recon.detach(), target.detach())

        # lpips_l = lpips_loss(recon.detach(), target.detach()).mean()
        train_bar.set_description(f"[Epoch {epoch}] ===> Train : PSNR: {psnr:.4f}, loss: {loss:.4f}, ssim: {ssim:.4f}, l1: {l1_loss:.4f}, lpips: {lpips_l:.4f}")
        train_bar.refresh()
        wandb.log({'train loss': loss , 'train psnr': psnr , 'train ssim': ssim ,"train l1": l1_loss , "train lpips": lpips_l}) 
        
# run.finish()

import os 
from huggingface_hub import snapshot_download

repo_id = "Tianfwang/SLURPP"

save_path = "./models/slurpp"
os.makedirs(save_path, exist_ok=True)

snapshot_download(repo_id=repo_id, local_dir=save_path)

repo_id = "sd2-community/stable-diffusion-2"
save_path = "./models/stable-diffusion-2"
os.makedirs(save_path, exist_ok=True)

snapshot_download(repo_id=repo_id, local_dir=save_path)


#DOWNLOAD THIS IF YOU NEED TO LOAD  FOR TRAINING
# repo_id = "prs-eth/marigold-depth-v1-0"
# save_path = "./models/marigold-depth-v1-0"
# os.makedirs(save_path, exist_ok=True)
# snapshot_download(repo_id=repo_id, local_dir=save_path)
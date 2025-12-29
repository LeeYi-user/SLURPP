import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

SCRATCH_DATA_DIR = os.environ.get('SCRATCH_DATA_DIR', "./")

class UnderwaterRealDataset(Dataset):
    def __init__(self, root_dir= f"{SCRATCH_DATA_DIR}/train/underwater_datasets/USOD10k", image_size=256):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])
        self.transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor()])

        self.image_files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if os.path.isfile(os.path.join(subdir, file)):
                    self.image_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            u_image = Image.fromarray(np.zeros((256, 256, 3), dtype = np.uint8))  # Placeholder for underwater image
            bc_image =  Image.fromarray(np.zeros((256, 256, 3), dtype = np.uint8))  # Placeholder for backscattered image
            ill_image =  Image.fromarray(np.zeros((256, 256, 3), dtype = np.uint8)) # Placeholder for illuminated image
            clear_image =  Image.fromarray(np.zeros((256, 256, 3), dtype = np.uint8))  # Placeholder for clear image
            if self.transform:
                u_image = self.transform(u_image) # underwater image
                bc_image = self.transform(bc_image) # backscattered image
                ill_image = self.transform(ill_image) # illumiated image
                clear_image = self.transform(clear_image) # clear image

            print(u_image.shape, u_image.dtype)
            out = {"imgs":{}}
            out['imgs']['u'] = u_image
            out['imgs']['bc'] = bc_image
            out['imgs']['ill'] = ill_image
            out['imgs']['clear'] = clear_image
            
            return out

    def getitem(self, idx):
        u_img_name = self.image_files[idx]
        u_image = Image.open(u_img_name)
        
        # Store original size
        original_size = u_image.size  # (width, height)

        if self.transform:
            u_image = self.transform(u_image) # underwater image

        out = {"imgs":{}}


        out['imgs']['u'] = u_image
        out['imgs']['name'] = os.path.splitext(os.path.basename(u_img_name))[0]
        out['imgs']['original_size'] = original_size  # Store original size as (width, height)

        return out

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(root_dir=f'{SCRATCH_DATA_DIR}/train/U_imgs', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    import matplotlib.pyplot as plt

    # Read the first set of images in the dataset
    u_image, bc_image, ill_image, clear_image = dataset[0]

    # Ensure the output directory exists
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)

    # Change the order of dimensions for plotting and save the images
    plt.imsave(os.path.join(output_dir, 'u_image.png'), u_image.permute(1, 2, 0).numpy())
    plt.imsave(os.path.join(output_dir, 'bc_image.png'), bc_image.permute(1, 2, 0).numpy())
    plt.imsave(os.path.join(output_dir, 'ill_image.png'), ill_image.permute(1, 2, 0).numpy())
    plt.imsave(os.path.join(output_dir, 'clear_image.png'), clear_image.permute(1, 2, 0).numpy())


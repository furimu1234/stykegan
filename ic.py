import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys
import glob
from colorama import Fore, Back, Style
from datetime import datetime

def main():
    # Initialize TensorFlow.
    tflib.init_tf()
 
    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    with open("./data.pkl", "rb") as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
 
    # Print network details.
    Gs.print_layers()
 
 
    
    #----------------------------------------------------------------------------------------------
    for i in range(int(e_num)):
        print(f"{Back.LIGHTCYAN_EX}{i+1}枚目の画像を生成します。{Style.RESET_ALL}")
        # Pick latent vector.
        rnd = np.random.RandomState()
        latents = rnd.randn(1, Gs.input_shape[1])
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        
        # Save image.
        file_length = len(glob.glob("results/*.png"))
        new_img_name = f"results/{file_length}.png"

        PIL.Image.fromarray(images[0], 'RGB').save(new_img_name)
        print(f"{Fore.LIGHTCYAN_EX}{i+1}/{e_num}枚目の画像を生成しました。\nfile name: {new_img_name}{Style.RESET_ALL}")
 
 
if __name__ == "__main__":
    args = sys.argv
    e_num = int(args[1])
    start_now = datetime.now()
    print(f"{Fore.CYAN} {e_num}枚の画像を生成します\n{start_now}{Style.RESET_ALL}")
    main()
    end_now = datetime.now()
    print(f"{Back.RED}指定された枚数の画像を生成しました。\n{end_now}\n生成にかかった時間: {end_now - start_now}{Style.RESET_ALL}")

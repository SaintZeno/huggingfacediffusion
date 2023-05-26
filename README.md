# huggingfacediffusion
Various Hugging Face Diffusion pipelines creating some cool images from NLP models

The embbeded latent images coming from NLP models are funky and throwing diffusion models ontop of them generates some neat pictures. This repo attempts to run parameterized diffusion pipelines via pipeline config. Torch is required for this project as it uses the Hugging Face API. 


# Quick install

##create your virtual environment (maintained in python 3.10)\n
python3.10 -m venv venv
##activate the env
venv\scripts\activate
##install pytorch https://pytorch.org/
##can copy/paste the command from torch site
##pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#pip install the requirements
pip install -r requirements.txt


# Usage 
Execute the `main` script which can be executed to generate images for the `prompt` in the `pipeline` configuration file (pipeline.yaml)
python main.py

The config file can be edited to adjust how the diffusion process runs; even allowing the ablility to change the Hugging Face transformers/diffusers used to construct the diffusion pipeline (still in beta for some). See below for `params` section of config.
params:
  seed: 1 ## random seed - change to get a different latent image from the encoder
  num_inference_steps: 50 ## number of iterations in the denoising loop
  guidance_scale_min: 10 ## learning rate min (if min<max then a file for each will be created)
  guidance_scale_max: 10 ## learning rate max 
  guidance_scale_random: False ## use a random learning rate for each denoising iteration
  height: 512 ## height of the output image; recommend multiples of 8
  width: 512 ## width of the output image; recommend multiples of 8
  show_image: False ## show the denoised image while running
  output_dir: location1/location2/location3 ## output directory
  save_denoising_iterations: True ## True/False - save every iteration of the denoising loop



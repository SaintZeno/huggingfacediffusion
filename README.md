# huggingfacediffusion
Various Hugging Face Diffusion pipelines creating some cool images from NLP models

The embbeded latent images coming from NLP models are funky and throwing diffusion models ontop of them generates some neat pictures. This repo attempts to run parameterized diffusion pipelines via pipeline config. Torch is required for this project as it uses the Hugging Face API. 


# Quick install
```
##create your virtual environment (maintained in python 3.10)
python3.10 -m venv venv
##activate the env
venv\scripts\activate
##install pytorch https://pytorch.org/
##can copy/paste the command from torch site
##pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#pip install the requirements
pip install -r requirements.txt
```


# Usage 
`python main.py` 
Runs the main script which will generate images corresponding to the `prompt` in the pipeline configuration file (`pipeline.yml`)
The config file can be edited to adjust how the diffusion process runs; even allowing the ablility to change the Hugging Face transformers/diffusers used to construct the diffusion pipeline (still in beta for some). See below for `params` section of config.
```
params:
  seed: ## random seed - change to get a different latent image from the encoder
  num_inference_steps: 55 ## number of iterations in the denoising loop
  guidance_scale_min: 10 ## learning rate min (if min<max then a file for each will be created)
  guidance_scale_max: 10 ## learning rate max (if min<max then a file for each will be created)
  guidance_scale_random: False ## use a random learning rate for each denoising iteration (guidance min and max are ignored if this is True)
  height: 512 ## height of the output image; recommend multiples of 8
  width: 512 ## width of the output image; recommend multiples of 8
  show_image: False ## show the denoised image while running
  #output_dir: location1/location2 ## output directory; if missing then images will be save to "results" directory
  output_dirname_type: pipeline ## pipeline or generic; generic creates no additional subfolders to output_dir
  save_denoising_iterations: False ## True/False - save every iteration of the denoising loop
```

# Example: Change scheduler
Simply adjust the `scheduler` section of pipeline config to grab whatever scheduler you want. See the following for an example using the LMSDiscrete Scheduler and local model config.
```
scheduler:
  name: LMSDiscreteScheduler #UniPCMultistepScheduler #PNDMScheduler #LMSDiscreteScheduler #EulerDiscreteScheduler 
  module: diffusers
  pretrained_dir: ./config/LMSDiscreteScheduler_config.json #runwayml/stable-diffusion-v1-5 #./config/LMSDiscreteScheduler_config.json #./config/EulerDiscreteScheduler_config
  subfolder: scheduler
```


# TODO:: 
-improve config explaination in readme
-testing suite



import importlib
import os
import yaml
import transformers
import diffusers
import torch
from tqdm.auto import tqdm ## fancy loading bar!
import random
from PIL import Image

def instantiate_hf_object(module, params):
    #print(f'instantiating {params["name"]} from {module}')
    func = getattr(module, params['name'])
    return func.from_pretrained(params['pretrained_dir'], subfolder=params['subfolder'])

def instantiate_object_choice(params):
    if params['module'] == 'diffusers':
        return instantiate_hf_object(diffusers, params)
    elif params['module'] == 'transformers':
        return instantiate_hf_object(transformers, params)


class Diffuse():
    def __init__(self):
        self.config = self.get_config()
        self.prompt=[self.config['prompt']]
        self.guidance_scale=None
        self.image=None
        self.output_path=None
        

    def get_config(self):
        return yaml.safe_load(open('pipeline.yml', 'r'))
    
    def run(self):
        ## loop thru each guidance scale!
        for i in range(self.config['params']['guidance_scale_min'], self.config['params']['guidance_scale_max'] + 1, 1):
            print(f'scale: {i}')
            ## if we want to run a random adjustment then set to 'random'
            if self.config['params']['guidance_scale_random'] == True:
                i='random'
        
            self.guidance_scale=i
            ## create output directory 
            self.create_output_dirs()
            ## run the pipeline
            self.run_pipeline()
            self.show_denoised_image()
            self.store_denoised_image()
        

    def show_denoised_image(self):
        if self.config['params']['show_image']:
            self.image.show()

    def store_denoised_image(self):
        self.image.save(self.output_path + f'complete_denoised_image.png')
        
    def create_output_dirs(self):
        alnum_prompt=''.join(e for e in self.prompt[0] if e.isalnum())
        self.output_path = f'{self.config["params"]["output_dir"]}\\{alnum_prompt}\\'
        granular_path = self.output_path + f'iterations\\'
        if not os.path.exists(granular_path):
            os.makedirs(granular_path)
        
    def run_pipeline(self):
        vae, tokenizer, text_encoder, model, scheduler = self.create_pipeline_components()

        height = self.config['params']['height']
        width = self.config['params']['width']
        generator = torch.manual_seed(self.config['params']['seed'])  # Seed generator to create the inital latent noise
        batch_size = len(self.prompt)

        ## tokenize the prompt
        ## this is just a dict w/ keys input_ids and attention_mask
        ## and both have tensors as values
        text_input = tokenizer(
            self.prompt, 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt")

        ## create embeddings for text input
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(self.config['torch_device']))[0]
        ## text_embeddings is just a tensor
        ## Recall: stable diffusion U-Net is able to condition 
        ## its output on text-embeddings via cross-attention 
        ## layers. The cross-attention layers are added to both the encoder and decoder part of the U-Net algo.

        ## next! 
        ## we create embeddings for uncoditioned input; encode a tokenized empty/blank string (kinda like an intercept)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        ## run unconditioned input thru text encoder
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.config['torch_device']))[0]
        ## Concate the uncoditioned and text emedings for model and avoid extra forward pass.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        ## time for some noise!!
        ## initialize our random noise latent
        ## divide by 8 b/c of reasons
        latents = torch.randn(
            (batch_size, model.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.config['torch_device'])

        ## denoise the initial noise; we just need to scale the input noise real quick 
        ## this is a requirement of the scheduler chosen. 
        latents = latents * scheduler.init_noise_sigma

        ## set the timesteps on the scheduler
        scheduler.set_timesteps(self.config['params']['num_inference_steps'])

        ## Time to create the Denoising algorithm 
        ## loop thru each timestep and...
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = model(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if self.config['params']['guidance_scale_random']:
                scale = random.randrange(1, 100, 1)
            else: 
                scale = self.guidance_scale
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            if self.config['params']['save_denoising_iterations']:
                self.latents_to_image(latents, vae=vae).save(self.output_path + f'iterations\\iteration_{t}.png')
        ## save image on self
        self.image = self.latents_to_image(latents=latents, vae=vae)
        
    def latents_to_image(self, latents, vae):
        # scale and decode the image latents with vae
        
        latents_scale = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents_scale).sample

        image = (image / 2 + 0.5).clamp(0, 1) ## again, random constants!
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images[0]
    
    def create_pipeline_components(self):
        ## get our config
        config = self.config
        if config:
            ## instantiate diffusion pipeline components
            vae = instantiate_object_choice(config['vae'])
            tokenizer = instantiate_object_choice(config['tokenizer'])
            text_encoder = instantiate_object_choice(config['text_encoder'])
            model = instantiate_object_choice(config['model'])
            scheduler = instantiate_object_choice(config['scheduler'])
            ## move to gpu; hope u can 2.
            torch_device = config['torch_device']
            vae.to(torch_device)
            text_encoder.to(torch_device)
            model.to(torch_device)
            
            return vae, tokenizer, text_encoder, model, scheduler 
        else: 
            raise Exception('No config set!!')


def main():
    d = Diffuse()
    d.run()

if __name__=='__main__':
    main()



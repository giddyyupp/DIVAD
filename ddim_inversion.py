import os
import argparse

import torch
import requests
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


# Useful function for later
def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img


## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


parser = argparse.ArgumentParser(description="DDIM Inversion for AD")
parser.add_argument("--data_set", default='visa', help="choices are btad|mpdd|mvtec|visa")
parser.add_argument("--data_path", default='/mnt/isilon/shicsonmez/ad/data/visa_dataset_processed')
parser.add_argument("--nis", default=50, type=int, help='num_inference_steps')
parser.add_argument("--inf_step", default=100, type=int, help='inference_steps')
parser.add_argument("--ss", default=40, type=int, help='start_step')
parser.add_argument("--sd_version", default="21", type=str, help='stable diff version, 15 or 21')
parser.add_argument('--save_inverted_image', action='store_true', help='save the inverted image')

args = parser.parse_args()

# DDIM params
num_inference_steps = args.nis
inference_steps = args.inf_step
# The reason we want to be able to specify start step
start_step = args.ss
sd_version = args.sd_version
dataset = args.data_set
data_dir = args.data_path

if sd_version == '15':
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
elif sd_version == '21':
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
else:
    raise Exception("Not supported SD version!")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# create save folder
save_dir = f'./{dataset}_ddim_inversion_results_sd{sd_version}_ss{start_step}_is{inference_steps}_nis{num_inference_steps}'
os.makedirs(save_dir, exist_ok=True)

all_objects = os.listdir(data_dir)

# MVTEC and VISA dataset loops.
for cl in all_objects:
    if not os.path.isdir(os.path.join(data_dir, cl)):
        continue
    os.makedirs(f"{save_dir}/{cl}/", exist_ok=True)
    input_image_prompt = f"An image of a {cl}"
    subfolders = os.listdir(os.path.join(data_dir, cl, 'test'))
    for sf in subfolders:
        test_ims = os.listdir(os.path.join(data_dir, cl, 'test', sf))
        os.makedirs(f"{save_dir}/{cl}/{sf}", exist_ok=True)

        for test_im in test_ims:
            image_path = os.path.join(data_dir, cl, 'test', sf, test_im)
            print(image_path)
            input_image = load_image(image_path, size=(512, 512))
            input_image.save(f"./{save_dir}/{cl}/{sf}/{test_im}")

            # Encode with VAE
            with torch.no_grad():
                latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
            l = 0.18215 * latent.latent_dist.sample()
            inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_inference_steps)
            inverted_latents.shape

            # Decode the final inverted latents
            if args.save_inverted_image:
                with torch.no_grad():
                    im = pipe.decode_latents(inverted_latents[-(start_step + 1)].unsqueeze(0))
                noisy_latent = pipe.numpy_to_pil(im)[0]
                noisy_latent.save(f"./{save_dir}/{cl}/{sf}/{test_im}_noisy_latent_{start_step}.png")

            sampled_orj_image = sample(
                input_image_prompt,
                start_latents=inverted_latents[-(start_step + 1)][None],
                start_step=start_step,
                num_inference_steps=inference_steps,
            )[0]

            sampled_orj_image.save(f"./{save_dir}/{cl}/{sf}/{test_im}_sampled.png")

            # # Sampling with a new prompt
            # start_step = 40
            # new_prompt = input_image_prompt + " without anomaly."
            # sampled_new_image = sample(
            #     new_prompt,
            #     start_latents=inverted_latents[-(start_step + 1)][None],
            #     start_step=start_step,
            #     num_inference_steps=inference_steps,
            # )[0]
            # sampled_new_image.save(f"./{save_dir}/{cl}/{sf}/{test_im}_sampled_new.png")

        print('')

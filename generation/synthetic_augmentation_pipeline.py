import argparse
import os
import pickle
import random
import torch
import torchvision.transforms as tfms
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import open_clip
import numpy as np

from llava.llava_processor import LlaVaProcessor
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def pad_image(image):
    """
    A function to pad an image to square and resize to SD size (512x512)
    """
    # pad to square using PIL
    width, height = image.size
    if width > height:
        new_width = width
        new_height = width
    else:
        new_width = height
        new_height = height
    new_im = Image.new("RGB", (new_width, new_height))
    new_im.paste(image, ((new_width - width) // 2, (new_height - height) // 2))

    return new_im.resize((512, 512))


def load_image(p):
    """     
    Function to load images from a defined path     
    """
    img = Image.open(p).convert("RGB")
    img = pad_image(img)
    return img


@torch.no_grad()
def pil_to_latents(image, vae):
    """     
    Function to convert image to latents     
    """
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist


@torch.no_grad()
def path_to_latents(p, vae, mixup):
    """     
    Function to convert path to latents     
    """
    if isinstance(p, list):
        images = []
        latents = []
        for p_ in p:
            image = load_image(p_)
            images.append(image)

        if mixup:
            # sample lambda from a beta distribution with beta = 0.2
            lambdas = np.random.beta(0.2, 0.2, size=len(images))

            mixed_images = []
            for i in range(len(images)):
                # pick two random images
                idx1 = random.randint(0, len(images) - 1)
                idx2 = random.randint(0, len(images) - 1)

                # mix them up
                mixed_image = Image.blend(images[idx1], images[idx2], lambdas[i])
                mixed_images.append(mixed_image)

            latents = [pil_to_latents(image, vae) for image in mixed_images]
        else:
            latents = [pil_to_latents(image, vae) for image in images]
        return torch.cat(latents)
    else:
        image = load_image(p)
        return pil_to_latents(image, vae)


@torch.no_grad()
def latents_to_pil(latents, vae):
    """     
    Function to convert latents to images     
    """
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


@torch.no_grad()
def text_enc(prompts, tokenizer, text_encoder, maxlen=None):
    """
    A function to take a texual promt and convert it into embeddings
    """
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(
        prompts,
        padding="max_length",
        max_length=maxlen,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(inp.input_ids.to("cuda"))[0].half()


def load_llava(llava_model_path):
    print("Loading LLAVA")

    disable_torch_init()

    model_name = get_model_name_from_path(llava_model_path)
    (
        llava_tokenizer,
        llava_model,
        llava_image_processor,
        context_len,
    ) = load_pretrained_model(llava_model_path, None, model_name, False, False)

    llava_processor = LlaVaProcessor(
        llava_tokenizer, llava_image_processor, llava_model.config.mm_use_im_start_end
    )

    print("Loaded LLAVA")

    return (
        llava_tokenizer,
        llava_model,
        llava_image_processor,
        context_len,
        llava_processor,
    )


def load_clip(clip_model="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
    print("Loading CLIP")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model)
    clip_tokenizer = open_clip.get_tokenizer(clip_model)

    print("Loaded CLIP")

    return clip_model, clip_tokenizer, clip_preprocess


sd_models = {
    "stable_diffusion": "stabilityai/stable-diffusion-2-1-base",
    "realistic": "SG161222/Realistic_Vision_V2.0",
}


def load_stable_diffusion(model_name):
    print("Loading Stable Diffusion Pipeline")

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_models[model_name], torch_dtype=torch.float16, local_files_only=True
    )
    pipe = pipe.to("cuda")
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(20)
    pipe.scheduler = scheduler

    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    print("Loaded Stable Diffusion Pipeline")

    return vae, unet, scheduler, tokenizer, text_encoder


def get_prompt(dataset, class_):
    """
    Get the prompt for the class, given the dataset
    """
    if dataset == "dtd":
        return f"a photo of a {class_} texture."
    elif dataset == "eurosat":
        return f"a centered satellite photo of a {class_}."
    elif dataset == "cars":
        return f"a photo of a {class_}, a type of car."
    elif dataset == "fgvca":
        return f"a photo of a {class_}, a type of aircraft."
    elif dataset == "flowers":
        return f"a photo of a {class_} flower."
    elif dataset == "pets":
        return f"a photo of a {class_}, a type of pet."
    else:
        return f"a photo of a {class_}."


def caption_image(
    images_path, class_name, llava_tokenizer, llava_model, llava_processor
):
    """
    Caption the image using LLAVA
    """

    # including the class name in the query makes LLaVA perform better
    # final caption will almost surely include the class name
    query = f"Provide a detailed caption for the image focusing on the object, knowing it's a {class_name}."
    batch_size = len(images_path)

    batch_images, batch_text = llava_processor.get_processed_tokens_batch([query] * batch_size, images_path)

    # run batch inference
    conv = conv_templates[llava_processor.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = batch_text
    image_tensor = batch_images
    input_ids = input_ids.cuda()

    with torch.inference_mode():
        output_ids = llava_model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    generated_outputs = llava_tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    generated_outputs = [out.strip() for out in generated_outputs]
    generated_outputs = [
        out[: -len(stop_str)] if out.endswith(stop_str) else out
        for out in generated_outputs
    ]
    generated_outputs = [out.strip() for out in generated_outputs]

    return generated_outputs


def build_zeroshot_weights(dataset, class_names, clip_model, clip_tokenizer):
    """
    Build the zero-shot weights for the given dataset, using the corresponding caption
    """
    zeroshot_weights = []
    for class_ in class_names:
        caption = get_prompt(dataset, class_)
        with torch.no_grad(), torch.autocast("cuda"):
            text_embed = (
                clip_model.encode_text(clip_tokenizer([caption]).to("cpu"))
                .float()
                .cpu()
            )
            text_embed /= text_embed.norm(dim=-1, keepdim=True)
            
            zeroshot_weights.append(text_embed.squeeze())

    zeroshot_weights = torch.stack(zeroshot_weights)

    return zeroshot_weights


def main(args):
    datasets_root = args.datasets_root
    dataset = args.dataset
    # starting step is the step at which we start generating images, the closer to 0, the higher the noise
    starting_step = args.steps
    batch_size = args.batch_size
    cfg_strength = args.cfg_strength
    seed = args.seed
    images_per_class = args.images_per_class

    dataset_root = f"{datasets_root}/{dataset}-by-seed"

    class_names_ori = os.listdir(f"{dataset_root}/seed_1")
    class_names = class_names_ori.copy()

    # CLIP
    clip_model, clip_tokenizer, clip_preprocess = load_clip()

    # SD
    vae, unet, scheduler, tokenizer, text_encoder = load_stable_diffusion(args.model_name)

    # LLAVA
    if args.use_llava:
        llava_tokenizer, llava_model, llava_image_processor, context_len, llava_processor = load_llava(args.llava_model_path)
    else:
        # in case of multiple runs, we don't want to load the LLAVA model every time, so we just load it once
        available_captions = pickle.load(open(f"captions/{dataset}.pkl", "rb"))

    if dataset == "imagenet":
        # imagenet class names are from wordnet, we need to map them to the names used in the dataset
        imagenet_map_file = "imagenet_map.txt"

        inmap = {}
        with open(imagenet_map_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                inmap[line[0]] = line[2]

        class_names = [inmap[_] for _ in class_names]

    # Zero-shot weights
    zeroshot_weights = build_zeroshot_weights(dataset, class_names, clip_model, clip_tokenizer)

    data_root = f"{dataset_root}/seed_{seed}"
    out_root = f"{args.output_dir}/{dataset}-{args.model_name}-step{starting_step}/seed_{seed}"
    os.makedirs(out_root, exist_ok=True)

    # disable tqdm if you don't want to see the progress bar (especially for SLURM jobs)
    for idx, class_ in enumerate(tqdm(class_names_ori, disable=True)):
        class_dir = os.path.join(data_root, class_)
        output_dir = os.path.join(out_root, class_)
        os.makedirs(output_dir, exist_ok=True)

        # get the shots for this class
        available_files = os.listdir(class_dir)

        # useful in case of multiple parallel runs
        generated_this_class = len(os.listdir(output_dir))
        # defines a sort of timeout for each class, if we can't generate an image for a class in 100 cycles, we move on with whatever the model generates
        cycles_spent_this_class = 0

        prompt_cache = {}

        while generated_this_class < images_per_class:
            class_name = class_names[idx]
            cycles_spent_this_class += 1
            # pick random samples
            file_names = random.choices(available_files, k=batch_size)
            file_paths = [os.path.join(class_dir, _) for _ in file_names]

            # load the latents
            latents = path_to_latents(file_paths, vae, args.mixup)
            noise = torch.randn_like(latents)

            noised_latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[starting_step]]))

            prompts = []

            if args.use_llava:
                # check if we have the prompt in the cache for all the shots in the class
                missing_file_names_idx = [p for p, f in enumerate(file_names) if f not in prompt_cache]

                if len(missing_file_names_idx) > 0:
                    missing_file_names = [file_names[_] for _ in missing_file_names_idx]
                    missing_file_paths = [file_paths[_] for _ in missing_file_names_idx]

                    missing_prompts = caption_image(missing_file_paths, class_name, llava_tokenizer, llava_model, llava_processor)

                    for i, _ in enumerate(missing_file_names_idx):
                        prompt_cache[missing_file_names[i]] = missing_prompts[i]

                for _ in range(batch_size):
                    prompts.append(prompt_cache[file_names[_]])

                # shuffle the prompts
                prompts = random.sample(prompts, len(prompts))
            else:
                # if we don't use LLAVA on-the-fly, we just pick a random prompt from the available ones for this class
                prompts_this_class = available_captions[class_]
                try:
                    prompts = random.choices(prompts_this_class, k=batch_size)
                except:
                    # if we don't have enough captions for this class (or any error arises during picking), we just pick the generic prompt
                    prompts = [f"a photo of a {class_name}."] * batch_size

            # embed the text with the tokenizer
            text_embed = torch.cat(
                [text_enc([prompt], tokenizer, text_encoder) for prompt in prompts]
            )
            uncond = text_enc([""] * 1, tokenizer, text_encoder, text_embed.shape[1])
            uncond = uncond.repeat(batch_size, 1, 1)
            emb = torch.cat([uncond, text_embed])

            latents = noised_latents

            # run the model for the given number of steps
            for i, ts in enumerate(tqdm(scheduler.timesteps[starting_step:], disable=True)):
                # We need to scale the latents to match the variance
                inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)

                # Predicting noise residual using U-Net
                with torch.no_grad(), torch.autocast("cuda"):
                    unconditional, conditional = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

                # cfg
                predicted_sample = unconditional + cfg_strength * (conditional - unconditional)

                latents = scheduler.step(predicted_sample, ts, latents).prev_sample

            final_imgs = latents_to_pil(latents, vae)
            # we are going to save the high-res images
            final_imgs_highres = [final_img.resize((512, 512)) for final_img in final_imgs]
            # we are going to use the low-res images as input for CLIP
            final_imgs = [final_img.resize((224, 224)) for final_img in final_imgs]

            with torch.no_grad(), torch.cuda.amp.autocast():
                imgs = torch.stack([clip_preprocess(final_img) for final_img in final_imgs]).to("cpu")
                image_features = clip_model.encode_image(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # compute the similarities between the generated images and the zero-shot weights
            similarities = (100.0 * image_features.float().cpu() @ zeroshot_weights.T).softmax(dim=-1)

            for _ in range(batch_size):
                predicted_class = similarities[_].argmax().item()

                # either the predicted class is the same as the current class, or we have spent too much time on this class
                if predicted_class == idx or (cycles_spent_this_class > 100):
                    # useful in case of multiple parallel runs
                    random_name = random.randint(0, 1000000000)
                    print(f"Image is most similar to {class_name}")

                    out_path = os.path.join(output_dir, f"{class_}_{generated_this_class}_{random_name}.jpg")
                    final_imgs_highres[_].save(out_path)

                    # in case of multiple parallel runs, we calculate the number of generated images for this class at each step
                    generated_this_class = len(os.listdir(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, default="datasets-by-seed")
    parser.add_argument("--output_dir", type=str, default="sap_generated_images")
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--steps", type=int, default=5, help="Starting step for generation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for Stable Diffusion")
    parser.add_argument("--use_llava", action="store_true")
    parser.add_argument("--llava_model_path", type=str)
    parser.add_argument("--mixup", action="store_true", help="Use mixup for images")
    parser.add_argument("--cfg_strength", type=float, default=8.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--images_per_class", type=int, default=64)

    args = parser.parse_args()

    main(args)

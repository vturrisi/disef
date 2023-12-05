import clip
import torch
import torch.nn as nn
from src.data import templates
from tqdm import tqdm
import math


class ClassificationHead(nn.Linear):
    def __init__(self, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        if weights is not None:
            self.weight = nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = nn.Parameter(biases.clone())
        else:
            self.bias = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


def build_zero_shot_classifier(clip_model, datamodule, template, device="cpu", subsample="all"):
    assert subsample in ["all", "base", "new"]

    class_names = datamodule.classes
    split = math.ceil(len(class_names) / 2)
    if subsample == "base":
        class_names = class_names[:split]
    elif subsample == "new":
        class_names = class_names[split:]

    template = getattr(templates, f"{template}_template")
    logit_scale = clip_model.logit_scale

    clip_model.eval()
    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(class_names, desc="Building zero-shot classifier"):
            texts = []
            for t in template:
                texts.append(t(classname))

            texts = clip.tokenize(texts).to(device)

            # embed with text encoder
            embeddings = clip_model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            # average across multiple prompt templates and re-norm
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights).to(device)
        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(weights=zeroshot_weights)

    return classification_head

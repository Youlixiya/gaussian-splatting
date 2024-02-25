import hashlib
import os
import urllib
import warnings
from collections import OrderedDict
from typing import Union, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .masks import BitMasks
from .utils import expand_box, mask2box, crop_with_mask

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def _preprocess(image: torch.Tensor,
                mask: torch.Tensor,
                mask_fill: str = "mean",
                mask_expand_ratio: float = 1.0,
                # mask_thr: float = 0.5,
                region_resized: bool = True,
                normalize: bool = True):
    """crop, mask and normalize the image

    Args:
        image ([type]): [C,H,W]
        mask ([type]): [K,H,W]
        normalize (bool, optional): [description]. Defaults to True.
    """
    dtype = image.dtype
    pixel_mean = torch.Tensor(PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
    pixel_std = torch.Tensor(PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
    # bin_mask = mask > mask_thr
    # valid = bin_mask.sum(dim=(-1, -2)) > 0
    # bin_mask = bin_mask[valid]
    # mask = mask[valid]
    # bin_mask = BitMasks(mask)
    # bboxes = bin_mask.get_bounding_boxes()
    # bboxes = mask2box
    # crop,mask
    regions = []
    region_masks = []
    if mask_fill == "zero":
            mask_fill = (0.0, 0.0, 0.0)
    elif mask_fill == "mean":
        mask_fill = [255.0 * c for c in PIXEL_MEAN]
    else:
        raise NotImplementedError(
            "Unknown mask_fill method: {}".format(mask_fill)
        )
    
    for single_mask in mask:
        bbox = mask2box(single_mask)
        region, region_mask = crop_with_mask(
            image.type(dtype),
            single_mask.type(dtype),
            bbox,
            fill=mask_fill,
            expand_ratio=mask_expand_ratio,
        )
        regions.append(region.unsqueeze(0))
        region_masks.append(region_mask.unsqueeze(0))
    
    unnorm_regions = regions
    if normalize:
        regions = [(r - pixel_mean ) / pixel_std for r in regions]
    # resize
    if region_resized:
        regions = [
            F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
        ]
        regions = torch.cat(regions)
        region_masks = [
            F.interpolate(r, size=(224, 224), mode="nearest") for r in region_masks
        ]
        region_masks = torch.cat(region_masks)
        unnorm_regions = [
            F.interpolate(r, size=(224, 224), mode="bicubic") for r in unnorm_regions
        ]
        unnorm_regions = torch.cat(unnorm_regions)
    return (regions, unnorm_regions), region_masks
def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(
    name: str,
    mask_prompt_depth: int = 0,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=False,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
        if 'state_dict' in state_dict:
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                if k.startswith('module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
            state_dict = new_state_dict

    if not jit:
        model = build_model(state_dict or model.state_dict(), mask_prompt_depth).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _preprocess

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _preprocess


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
    return_length: bool = False,
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    length = []
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
                length.append(context_length)
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        else:
            length.append(len(tokens))
        result[i, : len(tokens)] = torch.tensor(tokens)
    if return_length:
        return result, length
    return result
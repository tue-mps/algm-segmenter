import click
import einops
import torch
import torchvision

import matplotlib.pyplot as plt
import segm.utils.torch as ptu
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from segm import config
from segm.data.utils import STATS
from segm.model.decoder import MaskTransformer
from segm.model.factory import load_model
from torchvision import transforms
import turboSeg
import numpy as np
from scipy.ndimage import gaussian_filter



@click.command()
@click.argument("model-path", type=str)
@click.argument("image-path", type=str)
@click.argument("output-dir", type=str)
@click.option("--layer-id", default=0, type=int)
@click.option("--x-patch", default=0, type=int)
@click.option("--y-patch", default=0, type=int)
@click.option("--cmap", default="viridis", type=str)
@click.option("--enc/--dec", default=True, is_flag=True)
@click.option("--cls/--patch", default=False, is_flag=True)
@click.option("--patch-type", default="pure", type=str)
@click.option("--merging-window-size", nargs=2, type=int)
@click.option("--selected-layers", nargs=2, type=int)
@click.option("--threshold", default=0.88, type=float)
def visualize(
    model_path,
    image_path,
    output_dir,
    layer_id,
    x_patch,
    y_patch,
    cmap,
    enc,
    cls,
    patch_type,
    selected_layers,
    merging_window_size,
    threshold,
):

    output_dir = Path(output_dir)
    model_dir = Path(model_path).parent

    ptu.set_gpu_mode(True)

    # Build model
    model, variant = load_model(model_path)
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    if patch_type == "pmt":
        print('uploding pmt model....')
        turboSeg.patch.turboSeg_segmenter_patch(model,trace_source=True)
        print(merging_window_size)
        model.encoder.window_size = merging_window_size
        model.encoder.threshold = threshold
        model.encoder.selected_layers = selected_layers
    model.to(ptu.device)

    # Get model config
    patch_size = model.patch_size
    normalization = variant["dataset_kwargs"]["normalization"]
    image_size = variant["dataset_kwargs"]["image_size"]
    n_cls = variant["net_kwargs"]["n_cls"]
    stats = STATS[normalization]


    # open the gt ----
    seg_image_path = image_path.replace("images","annotations").replace("jpg","png")
    image_name = seg_image_path.split('/')[-1].split('.')[0]

    img_seg = Image.open(seg_image_path)
    img_seg = img_seg.resize((512, 512), resample=Image.NEAREST)
    print(img_seg.size)
    img_seg = np.array(img_seg) 
    # get class ids 
    class_list = np.unique(img_seg)
    class_list = class_list[1:]
    #---------------

    # Open image and process it
    try:
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img_vis = img.resize((image_size,image_size))
            img = img.convert("RGB")
    except:
        raise ValueError(f"Provided image path {image_path} is not a valid image file.")

    

    # Normalize and resize
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    img = transform(img)
    

    # Make the image divisible by the patch size
    w, h = (
        image_size - image_size % patch_size,
        image_size - image_size % patch_size,
    )
    
    print("dsdsdsdsdsds",h,w)
    # Crop to image size
    img = img.unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # Sanity checks
    if not enc and not isinstance(model.decoder, MaskTransformer):
        raise ValueError(
            f"Attention maps for decoder are only availabe for MaskTransformer. Provided model with decoder type: {model.decoder}."
        )

    if not cls:
        if x_patch > w_featmap or y_patch > h_featmap:
            raise ValueError(
                f"Provided patch x: {x_patch} y: {y_patch} is not valid. Patch should be in the range x: [0, {w_featmap}), y: [0, {h_featmap})"
            )
        num_patch = w_featmap * y_patch + x_patch

    if layer_id < 0:
        raise ValueError("Provided layer_id should be positive.")

    if enc and model.encoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for encoder with {model.encoder.n_layers}."
        )

    if not enc and model.decoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for decoder with {model.decoder.n_layers}."
        )
    # output_dir = output_dir / image_name
    output_dir= Path (output_dir / image_name / patch_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path.mkdir(output_dir, exist_ok=True)

    # Process input and extract attention maps
    if enc:
        print(f"Generating Attention Mapping for Encoder Layer Id {layer_id}")
        attentions = model.get_attention_map_enc(img.to(ptu.device), layer_id)
        num_extra_tokens = 1 + model.encoder.distilled
        if cls:
            attentions = attentions[0, :, 0, num_extra_tokens:]
        else:
            attentions = attentions[
                0, :, num_patch + num_extra_tokens, num_extra_tokens:
            ]
    else:
        print(f"Generating Attention Mapping for Decoder Layer Id {layer_id}")
        attentions = model.get_attention_map_dec(img.to(ptu.device), layer_id)
        print(attentions.shape)
        if cls:
            attentions = attentions[0, :, -n_cls:, :-n_cls]
        else:
            attentions = attentions[0, :, num_patch, :-n_cls]
    

    # print("ssaaaa",attentions.shape, model.encoder._turbo_info["source"].shape)


    try:
    
        # print("sssss",source.shape)
        idxs = model.encoder._turbo_info["source"][:, 1:, 1:].argmax(dim=1)
        print(idxs.shape, attentions.shape)
        # idxs = source[:, :, :].argmax(dim=1)
        # defining a torch tensor with the output size 
        masks_shape = attentions.shape
        size_x = model.encoder._turbo_info["source"].shape[2]-1
        # size_x = source.shape[2]
        attentions_ =  torch.ones(masks_shape[0],masks_shape[1],size_x,device=attentions.device)
        print(attentions_.shape)
        
        for batch in range(0,masks_shape[0]):
            attentions_[batch,:,:] = attentions[batch,:,idxs[0]]
    except:
        attentions_ = attentions
    

    print("ssaaassa",attentions_.shape)
    
    # Reshape into image shape
    nh = attentions_.shape[0]  # Number of heads
    attentions = attentions_.reshape(nh, -1)

    if cls and not enc:
        attentions = attentions.reshape(nh, n_cls, w_featmap, h_featmap)
    else:
        attentions = attentions.reshape(nh, 1, w_featmap, h_featmap)
    



    # Resize attention maps to match input size
    attentions = (
        F.interpolate(attentions, scale_factor=patch_size, mode="nearest").cpu().numpy()
    )

    # attentions = attentions.sum(0)
    print("ssaaaa",attentions.shape)
    # Save Attention map for each head
    calss_name = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']
    
    for i in range(nh):
        base_name = "enc" if enc else "dec"
        head_name = f"{base_name}_layer{layer_id}_attn-head{i}"
        attention_maps_list = attentions[i]
        print(attention_maps_list.shape)
        for j in class_list:
            attention_map = attention_maps_list[j-1]
            file_name = head_name
            dir_path = output_dir / f"{base_name}_layer{layer_id}"
            Path.mkdir(dir_path, exist_ok=True)
            if cls:
                if enc:
                    file_name = f"{file_name}_cls"
                    dir_path /= "cls"
                else:
                    file_name = f"{file_name}_{calss_name[j-1]}"
                    dir_path /= f"cls_{calss_name[j-1]}"
                Path.mkdir(dir_path, exist_ok=True)
            else:
                dir_path /= f"patch_{x_patch}_{y_patch}"
                Path.mkdir(dir_path, exist_ok=True)

            file_path = dir_path / f"{file_name}.png"
            file_path_atten_overly = dir_path / f"{file_name}_overlay.png"
            attention_map_ = gaussian_filter(attention_map, sigma=10)
          
            plt.imsave(fname=str(file_path), arr=attention_map_, format="png", cmap=cmap)

            #---- overlay image ----#
            attention_weights_normalized = (attention_map - np.min(attention_map)) / \
                               (np.max(attention_map) - np.min(attention_map)) 
            attention_weights_normalized = gaussian_filter(attention_weights_normalized, sigma=10)

            attention_map = (np.array(img_vis) * 0.6 + plt.get_cmap('jet')(attention_weights_normalized)[:, :, :3] * 255 * 0.4).astype(np.uint8)
            plt.imsave(fname=str(file_path_atten_overly), arr=attention_map, format="png", cmap=cmap)
            print(f"{file_path} saved.")

    # Save input image showing selected patch
    if  cls:
        print('sssss')
        
        im_n = torchvision.utils.make_grid(img, normalize=True, scale_each=True)

        # Compute corresponding X and Y px in the original image
        x_px = x_patch * patch_size
        y_px = y_patch * patch_size
        px_v = einops.repeat(
            torch.tensor([1, 0, 0]),
            "c -> 1 c h w",
            h=patch_size,
            w=patch_size,
        )

        # Draw pixels for selected patch
        # im_n[:, y_px : y_px + patch_size, x_px : x_px + patch_size] = px_v
        image_name = image_name+".png"
        
        torchvision.utils.save_image(
            im_n,
            str(output_dir / image_name),
        )
    try: 
        # save the merged image ------#
        vis_out =turboSeg.make_visualization(img_vis, model.encoder._turbo_info["source"], patch_size=16, class_token=True)
        vis_name = image_name+"_vis.png"
        vis_out.save(output_dir / vis_name)
    except:
        pass

if __name__ == "__main__":
    visualize()

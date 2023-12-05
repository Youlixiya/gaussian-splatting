import PIL
from PIL.Image import Image
import numpy as np
from typing import List, Union
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from segment_anything import build_sam

GROUNDING_DINO_CONFIG_PATH = "Grounded-SAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "Grounded-SAM/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "Grounded-SAM/weights/sam_vit_h_4b8939.pth"

class GroundSAM:
    def __init__(self,
                 grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
                 grounfing_dino_ckpt_path=GROUNDING_DINO_CHECKPOINT_PATH,
                 sam_ckpt_path=SAM_CHECKPOINT_PATH,
                 device='cuda'):
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounfing_dino_ckpt_path = grounfing_dino_ckpt_path
        self.grounding_dino_model = Model(model_config_path=grounding_dino_config_path, model_checkpoint_path=grounfing_dino_ckpt_path)
        self.sam_ckpt_path = sam_ckpt_path
        sam = build_sam(checkpoint=sam_ckpt_path)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.device = device


    # def visualize_results(self,
    #                       img: Union[Image, np.ndarray],
    #                       class_list: [List],
    #                       box_threshold: float=0.25,
    #                       text_threshold: float=0.25,
    #                       nms_threshold: float=0.8,
    #                       pil: bool=True):
    #     detections = self.forward(img, class_list, box_threshold, text_threshold)
    #     box_annotator = sv.BoxAnnotator()
    #     nms_idx = torchvision.ops.nms(
    #         torch.from_numpy(detections.xyxy),
    #         torch.from_numpy(detections.confidence),
    #         nms_threshold
    #     ).numpy().tolist()

    #     detections.xyxy = detections.xyxy[nms_idx]
    #     detections.confidence = detections.confidence[nms_idx]
    #     detections.class_id = detections.class_id[nms_idx]
    #     labels = [
    #         f"{class_list[class_id]} {confidence:0.2f}"
    #         for _, _, confidence, class_id, _
    #         in detections]
    #     annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
    #     if pil:
    #         return PIL.Image.fromarray(annotated_frame[:, :, ::-1]), detections
    #     else:
    #         return annotated_frame, detections


    @torch.no_grad()
    def __call__(self,
                img: Union[Image, np.ndarray],
                class_list: [List],
                box_threshold: float=0.25,
                text_threshold: float=0.25,
                nms_threshold: float=0.8,
                pil: bool=True)->sv.Detections:
        if isinstance(img, Image):
            img = np.uint8(img)[:, :, ::-1]
        detections = self.grounding_dino_model.predict_with_classes(
                    image=img,
                    classes=class_list,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
        )
        box_annotator = sv.BoxAnnotator()
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        labels = [
            f"{class_list[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_box = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        self.sam_predictor.set_image(img[:, :, ::-1])
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.asarray(result_masks)
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{class_list[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_mask = mask_annotator.annotate(scene=img.copy(), detections=detections)
        if pil:
            return PIL.Image.fromarray(annotated_box[:, :, ::-1]), PIL.Image.fromarray(annotated_mask[:, :, ::-1]), (np.where(np.sum(detections.mask, axis=0)>0, 1, 0)).astype(bool)
        else:
            return annotated_box, annotated_mask, (np.where(np.sum(detections.mask, axis=0)>0, 1, 0)).astype(bool)

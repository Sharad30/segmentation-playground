import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from loguru import logger

def get_instance_segmentation_model(num_classes, pretrained=True):
    """
    Get a Mask R-CNN model pre-trained on COCO and fine-tune for instance segmentation.
    
    Args:
        num_classes (int): Number of output classes including background
        pretrained (bool): Whether to use a pretrained backbone
        
    Returns:
        nn.Module: Mask R-CNN model
    """
    # Load a pre-trained Mask R-CNN model with ResNet-50 FPN backbone
    # This is a smaller model that should fit in 6GB GPU memory
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained,
        min_size=600,  # Reduce from 800 to 600
        max_size=1000  # Reduce from 1333 to 1000
    )
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    # Reduce memory usage for training
    model.backbone.body.layer4.requires_grad_(True)
    model.backbone.body.layer3.requires_grad_(True)
    model.backbone.body.layer2.requires_grad_(False)
    model.backbone.body.layer1.requires_grad_(False)
    model.backbone.body.conv1.requires_grad_(False)
    
    logger.info(f"Created Mask R-CNN model with {num_classes} classes")
    logger.info(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model 
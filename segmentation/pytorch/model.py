import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from loguru import logger

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, bilinear=True, pretrained=True):
        """
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            bilinear (bool): If True, use bilinear upsampling, otherwise use transposed convolutions
            pretrained (bool): If True, use pretrained ResNet50 weights
        """
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained ResNet50
        self.resnet = resnet50(pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Freeze the backbone if using pretrained weights
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Encoder (ResNet50)
        self.encoder1 = nn.Sequential(
            self.resnet[0],  # conv1
            self.resnet[1],  # bn1
            self.resnet[2],  # relu
            self.resnet[3]   # maxpool
        )  # 64 channels
        
        self.encoder2 = self.resnet[4]   # layer1: 256 channels
        self.encoder3 = self.resnet[5]   # layer2: 512 channels
        self.encoder4 = self.resnet[6]   # layer3: 1024 channels
        self.encoder5 = self.resnet[7]   # layer4: 2048 channels
        
        # Decoder
        self.up1 = Up(2048 + 1024, 1024, bilinear)  # 2048 + 1024 -> 1024
        self.up2 = Up(1024 + 512, 512, bilinear)    # 1024 + 512 -> 512
        self.up3 = Up(512 + 256, 256, bilinear)     # 512 + 256 -> 256
        self.up4 = Up(256 + 64, 64, bilinear)       # 256 + 64 -> 64
        self.outc = OutConv(64, n_classes)
        
        # Final upsampling to match input size
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        logger.info(f"Initialized ResNet50-UNet with {n_channels} input channels and {n_classes} output classes")
        logger.info(f"Using {'bilinear' if bilinear else 'transposed conv'} upsampling")
        logger.info(f"Using {'pretrained' if pretrained else 'randomly initialized'} ResNet50 backbone")

    def forward(self, x):
        # Store input size for final upsampling
        input_size = x.size()[2:]
        
        # Encoder
        x1 = self.encoder1(x)  # 64 channels
        x2 = self.encoder2(x1)  # 256 channels
        x3 = self.encoder3(x2)  # 512 channels
        x4 = self.encoder4(x3)  # 1024 channels
        x5 = self.encoder5(x4)  # 2048 channels
        
        # Decoder
        x = self.up1(x5, x4)  # 2048 + 1024 -> 1024
        x = self.up2(x, x3)   # 1024 + 512 -> 512
        x = self.up3(x, x2)   # 512 + 256 -> 256
        x = self.up4(x, x1)   # 256 + 64 -> 64
        x = self.outc(x)      # 64 -> n_classes
        
        # Final upsampling to match input size
        x = self.final_upsample(x)
        
        return x

    def get_params_count(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_model(config):
    """
    Create a ResNet50-UNet model based on the configuration
    
    Args:
        config: Dictionary containing model configuration:
            - model_type: Type of model ('resnet_unet')
            - n_channels: Number of input channels
            - n_classes: Number of output classes
            - bilinear: Whether to use bilinear upsampling
            - pretrained: Whether to use pretrained ResNet50 weights
    
    Returns:
        model: PyTorch model
    """
    model_type = config.get('model_type', 'resnet_unet')
    
    if model_type == 'resnet_unet':
        model = ResNetUNet(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 21),
            bilinear=config.get('bilinear', True),
            pretrained=config.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Created model: {model_type} with {model.get_params_count():,} parameters")
    
    return model 
"""
Model definitions for mammography classification.
Each model has its own recommended hyperparameters and optimizer settings
based on their original papers and best practices for medical imaging.

Architecture Categories:
1. Baseline Models (Stage 1): ResNet34, ResNet50, VGG16, DenseNet121, EfficientNet-B0, MobileNetV2
2. Advanced Models (Stage 2): ResNet50Stage2, CBAMResNet50, HybridViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from typing import Optional, Dict, Any



# Model Hyperparameters Configuration
# Each model has individually configurable hyperparameters for fine-tuning


MODEL_HYPERPARAMS = {
    'ResNet34': {
        # AdamW is better for transfer learning - adapts per-parameter LR
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,  # Lower LR for Adam-type optimizers
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,  # AdamW uses higher weight decay (decoupled)
        'dropout': 0.0,  # Has batch norm, doesn't need dropout
        'scheduler': 'CosineAnnealingLR',
        'T_max': 60,  # Full training length
        'eta_min': 1e-6,
        'description': 'ResNet34 - 34-layer residual network. Uses AdamW for better fine-tuning.'
    },
    'ResNet50': {
        # AdamW is better for transfer learning - adapts per-parameter LR
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,  # Lower LR for Adam-type optimizers
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,  # AdamW uses higher weight decay (decoupled)
        'dropout': 0.0,  # Has batch norm
        'scheduler': 'CosineAnnealingLR',
        'T_max': 60,  # Full training length
        'eta_min': 1e-6,
        'description': 'ResNet50 - 50-layer residual network with bottleneck. Uses AdamW for better fine-tuning.'
    },
    'VGG16': {
        # VGG has no batch norm, needs Adam for stable training and more regularization
        'optimizer': 'Adam',
        'learning_rate': 1e-4,  # Lower LR for Adam
        'betas': (0.9, 0.999),
        'weight_decay': 5e-4,  # Higher weight decay for regularization
        'dropout': 0.5,  # Heavy dropout needed (no batch norm)
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': 0.1,
        'scheduler_patience': 7,  # More patience since Adam converges slower
        'description': 'VGG16 - Classic deep CNN. Uses Adam optimizer with higher regularization.'
    },
    'DenseNet121': {
        # DenseNet works well with AdamW (decoupled weight decay)
        'optimizer': 'AdamW',
        'learning_rate': 5e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4,
        'dropout': 0.0,  # Dense connections provide regularization
        'scheduler': 'CosineAnnealingWarmRestarts',
        'T_0': 10,  # Restart every 10 epochs
        'T_mult': 2,  # Double the period after each restart
        'description': 'DenseNet121 - Dense connections for feature reuse. Uses AdamW with cosine annealing.'
    },
    'EfficientNet-B0': {
        # Google recommends RMSprop for EfficientNet
        'optimizer': 'RMSprop',
        'learning_rate': 1e-3,
        'alpha': 0.9,  # Smoothing constant
        'eps': 1e-8,
        'momentum': 0.9,
        'weight_decay': 1e-5,  # Lower weight decay
        'dropout': 0.2,  # Moderate dropout
        'scheduler': 'StepLR',
        'step_size': 10,
        'gamma': 0.5,
        'description': 'EfficientNet-B0 - Compound scaling. Uses RMSprop (Google recommended).'
    },
    'MobileNetV2': {
        # MobileNetV2 paper used RMSprop, but AdamW works well for fine-tuning
        'optimizer': 'AdamW',
        'learning_rate': 3e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'scheduler': 'OneCycleLR',
        'max_lr': 1e-3,
        'pct_start': 0.3,  # Warmup for 30% of training
        'description': 'MobileNetV2 - Inverted residuals. Uses AdamW with OneCycleLR scheduler.'
    },
    
    # Stage 2 Advanced Models
    
    'ResNet50Stage2': {
        # Clean ResNet50 backbone - same config as Stage 1 ResNet50
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,
        'dropout': 0.0,  # Matches Stage 1 ResNet50
        'scheduler': 'CosineAnnealingLR',
        'T_max': 60,
        'eta_min': 1e-6,
        'description': 'ResNet50 Stage 2 - Clean backbone for architectural comparison.'
    },
    
    'CBAMResNet50': {
        # CBAM attention - SAME hyperparams as ResNet50Stage2 for fair comparison
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,
        'dropout': 0.0,  # Standardized for fair comparison
        'scheduler': 'CosineAnnealingLR',
        'T_max': 60,
        'eta_min': 1e-6,
        'description': 'CBAM-ResNet50 - Channel + Spatial attention for focused feature learning.'
    },
    
    'HybridViT': {
        # Hybrid ViT - SAME hyperparams as ResNet50Stage2 for fair comparison
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,
        'dropout': 0.0,  # Standardized for fair comparison
        'scheduler': 'CosineAnnealingLR',
        'T_max': 60,
        'eta_min': 1e-6,
        'description': 'Hybrid ViT - CNN backbone + Transformer encoder for global context.'
    }
}


def get_hyperparams(model_name: str) -> dict:
    """Get recommended hyperparameters for a model."""
    return MODEL_HYPERPARAMS.get(model_name, MODEL_HYPERPARAMS['ResNet34'])


# Attention Modules (CBAM)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM paper.
    Learns to focus on important feature channels.
    
    How it works:
    1. Apply global average pooling and max pooling to squeeze spatial dimensions
    2. Pass through shared MLP to learn channel importance
    3. Combine with sigmoid to get attention weights
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        # Squeeze spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP (implemented with 1x1 convolutions for efficiency)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pool branch
        avg_out = self.fc(self.avg_pool(x))
        # Max pool branch
        max_out = self.fc(self.max_pool(x))
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM paper.
    Learns to focus on important spatial regions.
    
    How it works:
    1. Compute channel-wise average and max along spatial dimensions
    2. Concatenate and pass through convolution
    3. Apply sigmoid to get spatial attention map
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        # 7x7 kernel captures wider spatial context
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies channel attention then spatial attention.
    
    Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    
    Benefits for mammography:
    - Channel attention helps focus on texture/density patterns
    - Spatial attention helps localize lesions/masses
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x



# Stage 2 Model Architectures

class ResNet50Stage2(nn.Module):
    """
    Clean ResNet50 backbone with optional dropout-regularized classifier.
    Used as baseline for Stage 2 to compare against attention models.
    
    Note: Default dropout is 0.0 to match Stage 1 ResNet50 for fair comparison.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
        super(ResNet50Stage2, self).__init__()
        
        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # Copy backbone layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CBAMResNet50(nn.Module):
    """
    ResNet50 enhanced with CBAM attention modules after each residual stage.
    
    Architecture:
    - Standard ResNet50 backbone (pretrained on ImageNet)
    - CBAM attention block after each of the 4 residual stages
    - Optional dropout regularization before classifier
    
    Why CBAM for mammography:
    - Channel attention: learns which feature patterns are most relevant
    - Spatial attention: focuses on regions likely to contain lesions
    - Helps ignore irrelevant background while focusing on diagnostically important areas
    
    Note: Default dropout is 0.0 to match Stage 1 ResNet50 for fair comparison.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
        super(CBAMResNet50, self).__init__()
        
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Copy initial layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet stages
        self.layer1 = resnet.layer1  # Output: 256 channels
        self.layer2 = resnet.layer2  # Output: 512 channels
        self.layer3 = resnet.layer3  # Output: 1024 channels
        self.layer4 = resnet.layer4  # Output: 2048 channels
        
        # CBAM attention after each stage
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stage 1 + attention
        x = self.layer1(x)
        x = self.cbam1(x)
        
        # Stage 2 + attention
        x = self.layer2(x)
        x = self.cbam2(x)
        
        # Stage 3 + attention
        x = self.layer3(x)
        x = self.cbam3(x)
        
        # Stage 4 + attention
        x = self.layer4(x)
        x = self.cbam4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer: CNN feature extractor + Transformer encoder.
    
    Architecture:
    - ResNet50 layers 1-3 as CNN backbone (extracts local features)
    - Patch embedding to project CNN features to transformer dimension
    - Transformer encoder to capture global relationships
    - Class token for final classification
    
    Why Hybrid for mammography:
    - CNN captures local texture and edge patterns
    - Transformer captures long-range dependencies (e.g., bilateral asymmetry)
    - Better than pure ViT for medical images (need strong local feature extraction)
    
    Note: Default img_size=224 to match our data loader transforms
    Note: Default dropout is 0.0 to match Stage 1 ResNet50 for fair comparison.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0
    ):
        super(HybridViT, self).__init__()
        
        self.img_size = img_size
        
        # CNN backbone (ResNet50 up to layer3)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,      # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # /4
            resnet.layer1,     # /4, 256 channels
            resnet.layer2,     # /8, 512 channels
            resnet.layer3      # /16, 1024 channels
        )
        
        cnn_out_channels = 1024
        feature_map_size = img_size // 16  # After CNN downsampling (224 -> 14)
        
        # Patch embedding: project CNN features to transformer dimension
        self.patch_embed = nn.Conv2d(
            cnn_out_channels,
            embed_dim,
            kernel_size=1,
            stride=1
        )
        
        num_patches = feature_map_size ** 2  # 14x14 = 196 patches for 224 input
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction
        x = self.cnn_backbone(x)  # [B, 1024, H/16, W/16]
        
        # Project to transformer dimension
        x = self.patch_embed(x)  # [B, embed_dim, H/16, W/16]
        
        # Flatten patches: [B, embed_dim, H, W] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification from class token
        x = self.norm(x[:, 0])
        x = self.dropout(x)
        x = self.head(x)
        
        return x


def create_optimizer(model: nn.Module, model_name: str) -> optim.Optimizer:
    """
    Create the recommended optimizer for a specific model architecture.
    
    Args:
        model: The PyTorch model
        model_name: Name of the model architecture
    
    Returns:
        Configured optimizer
    """
    config = get_hyperparams(model_name)
    optimizer_type = config.get('optimizer', 'SGD')
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config['weight_decay'],
            nesterov=config.get('nesterov', False)
        )
    
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config['weight_decay']
        )
    
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config['weight_decay']
        )
    
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config['learning_rate'],
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            momentum=config.get('momentum', 0),
            weight_decay=config['weight_decay']
        )
    
    else:
        # Default to SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, model_name: str, 
                     num_epochs: int = 60, steps_per_epoch: int = 100):
    """
    Create the recommended learning rate scheduler for a specific model.
    
    Args:
        optimizer: The optimizer to schedule
        model_name: Name of the model architecture
        num_epochs: Total training epochs
        steps_per_epoch: Number of batches per epoch (for OneCycleLR)
    
    Returns:
        Configured scheduler
    """
    config = get_hyperparams(model_name)
    scheduler_type = config.get('scheduler', 'ReduceLROnPlateau')
    
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 5)
        )
    
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2)
        )
    
    elif scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', num_epochs),
            eta_min=config.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.get('pct_start', 0.3)
        )
    
    else:
        # Default to ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5
        )
    
    return scheduler


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True, 
               img_size: int = 224, **kwargs) -> nn.Module:
    """
    Get a model modified for binary classification.
    Each model has its own optimal classifier head configuration.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        img_size: Input image size (used for HybridViT)
        **kwargs: Additional hyperparameter overrides
    
    Returns:
        PyTorch model ready for training
    
    Available models:
        Stage 1 (Baseline): ResNet34, ResNet50, VGG16, DenseNet121, EfficientNet-B0, MobileNetV2
        Stage 2 (Advanced): ResNet50Stage2, CBAMResNet50, HybridViT
    """
    config = get_hyperparams(model_name)
    
    # Allow hyperparameter overrides from kwargs
    dropout = kwargs.get('dropout', config.get('dropout', 0.0))
    
    # Use weights enum for explicit progress display
    if pretrained:
        print(f"Loading pretrained weights for {model_name}...")
    

    # Stage 1: Baseline Models

    
    if model_name == 'ResNet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'ResNet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'VGG16':
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
        
    elif model_name == 'DenseNet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == 'EfficientNet-B0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name == 'MobileNetV2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        num_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    

    # Stage 2: Advanced Models

    
    elif model_name == 'ResNet50Stage2':
        model = ResNet50Stage2(
            num_classes=num_classes, 
            pretrained=pretrained,
            dropout=dropout
        )
        
    elif model_name == 'CBAMResNet50':
        model = CBAMResNet50(
            num_classes=num_classes, 
            pretrained=pretrained,
            dropout=dropout
        )
        
    elif model_name == 'HybridViT':
        model = HybridViT(
            num_classes=num_classes,
            img_size=img_size,
            embed_dim=kwargs.get('embed_dim', 384),
            num_heads=kwargs.get('num_heads', 6),
            num_layers=kwargs.get('num_layers', 4),
            mlp_ratio=kwargs.get('mlp_ratio', 4),
            dropout=dropout
        )
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: ResNet34, ResNet50, VGG16, DenseNet121, EfficientNet-B0, MobileNetV2, "
            f"ResNet50Stage2, CBAMResNet50, HybridViT"
        )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model_name: str) -> str:
    """Get information string about a model."""
    params = get_hyperparams(model_name)
    return params.get('description', model_name)


def print_model_config(model_name: str, config: Dict[str, Any] = None):
    """
    Print detailed configuration for a model.
    
    Args:
        model_name: Name of the model
        config: Optional custom config dict (if None, uses MODEL_HYPERPARAMS)
    """
    if config is None:
        config = get_hyperparams(model_name)
    
    print(f"\n{model_name} Configuration:")
    print(f"  Optimizer: {config.get('optimizer', 'SGD')}")
    print(f"  Learning Rate: {config.get('learning_rate', 1e-3)}")
    print(f"  Weight Decay: {config.get('weight_decay', 1e-4)}")
    print(f"  Dropout: {config.get('dropout', 0.0)}")
    print(f"  Scheduler: {config.get('scheduler', 'ReduceLROnPlateau')}")
    
    # Print optimizer-specific parameters
    opt = config.get('optimizer', 'SGD')
    if opt == 'SGD':
        print(f"  Momentum: {config.get('momentum', 0.9)}")
        print(f"  Nesterov: {config.get('nesterov', False)}")
    elif opt in ['Adam', 'AdamW']:
        print(f"  Betas: {config.get('betas', (0.9, 0.999))}")
    elif opt == 'RMSprop':
        print(f"  Alpha: {config.get('alpha', 0.99)}")
        print(f"  Momentum: {config.get('momentum', 0)}")


# Hyperparameter Experiment Support

def create_optimizer_from_config(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer from a custom configuration dict.
    Useful for hyperparameter tuning experiments.
    
    Args:
        model: The PyTorch model
        config: Dictionary with optimizer settings
    
    Returns:
        Configured optimizer
    
    Example config:
        {
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        }
    """
    optimizer_type = config.get('optimizer', 'AdamW')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', False)
        )
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            momentum=config.get('momentum', 0),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler_from_config(optimizer: optim.Optimizer, config: Dict[str, Any],
                                  num_epochs: int = 60, steps_per_epoch: int = 100):
    """
    Create scheduler from a custom configuration dict.
    Useful for hyperparameter tuning experiments.
    
    Args:
        optimizer: The optimizer to schedule
        config: Dictionary with scheduler settings
        num_epochs: Total training epochs
        steps_per_epoch: Number of batches per epoch (for OneCycleLR)
    
    Returns:
        Configured scheduler
    
    Example config:
        {
            'scheduler': 'CosineAnnealingLR',
            'T_max': 60,
            'eta_min': 1e-6
        }
    """
    scheduler_type = config.get('scheduler', 'CosineAnnealingLR')
    
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 5)
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2)
        )
    elif scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', num_epochs),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', config.get('learning_rate', 1e-3)),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.get('pct_start', 0.3)
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    
    return scheduler


def get_experiment_name(model_name: str, config: Dict[str, Any]) -> str:
    """
    Generate a unique experiment name from model and hyperparameters.
    Useful for tracking multiple runs of the same model with different settings.
    
    Args:
        model_name: Base model name
        config: Hyperparameter configuration
    
    Returns:
        Unique experiment name like "ResNet50_AdamW_lr1e-4_wd0.01"
    """
    opt = config.get('optimizer', 'AdamW')
    lr = config.get('learning_rate', 1e-4)
    wd = config.get('weight_decay', 1e-4)
    
    # Format learning rate nicely
    lr_str = f"{lr:.0e}".replace('-0', '-')
    
    return f"{model_name}_{opt}_lr{lr_str}_wd{wd}"


def list_available_models() -> Dict[str, str]:
    """List all available models with their descriptions."""
    return {name: config.get('description', 'No description') 
            for name, config in MODEL_HYPERPARAMS.items()}

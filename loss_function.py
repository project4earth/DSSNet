import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config
  

class BFCrossEntropyIoULoss(nn.Module):
    def __init__(self, theta0=3, theta=5, boundary_weight=1.0, iou_weight=1.0, smooth=1e-6, n_classes=config.NUM_CLASSES):
        """
        Combined BF mIoU Cross-Entropy Loss and mmIoU Loss.
        
        Args:
            theta0 (int): Kernel size for initial boundary map generation.
            theta (int): Kernel size for extended boundary map generation.
            boundary_weight (float): Weight factor for boundary loss.
            iou_weight (float): Weight factor for mmIoU loss.
            smooth (float): Smoothing factor to avoid division by zero.
            n_classes (int): Number of classes for segmentation.
        """
        super(BFCrossEntropyIoULoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.boundary_weight = boundary_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.n_classes = n_classes
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_boundary_map(self, tensor, kernel_size):
        """
        Computes boundary map for a given tensor using max pooling.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            kernel_size (int): Kernel size for max pooling.
        
        Returns:
            torch.Tensor: Boundary map.
        """
        assert tensor.ndim == 4, "Input tensor must have 4 dimensions (N, C, H, W)"
        assert tensor.min() >= 0 and tensor.max() <= 1, "Tensor values should be in the range [0, 1] for boundary computation"

        #tensor = torch.clamp(tensor, min=0.0, max=1.0)

        boundary_map = (F.max_pool2d(1 - tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2) - (1 - tensor))
        #boundary_map = torch.clamp(boundary_map, min=0.0, max=1.0)

        return boundary_map
    
    def forward(self, outputs, labels):
        """
        Forward pass for the combined loss function.
        
        Args:
            outputs (torch.Tensor): Model outputs with shape (N, C, H, W).
            labels (torch.Tensor): Ground truth labels with shape (N, H, W).
        
        Returns:
            torch.Tensor: Computed loss value.
        """
 
        # Cross-Entropy Loss
        cross_entropy_loss = self.cross_entropy_loss(outputs, labels)
        #if torch.isnan(cross_entropy_loss) or torch.isinf(cross_entropy_loss):
        #    cross_entropy_loss = torch.tensor(0.0, device=outputs.device)

        # Compute Softmax probabilities for outputs
        outputs_softmax = F.softmax(outputs, dim=1)
            
        # Compute Boundary Maps
        boundary_preds = self.compute_boundary_map(outputs_softmax, self.theta0)
        boundary_labels = self.compute_boundary_map(labels, self.theta0)

        # Extended Boundary Maps
        extended_boundary_preds = F.max_pool2d(boundary_preds, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        extended_boundary_labels = F.max_pool2d(boundary_labels, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # Reshape tensors for precision and recall calculation
        n, c, h, w = outputs.shape
        boundary_preds = boundary_preds.view(n, c, -1)
        extended_boundary_preds = extended_boundary_preds.view(n, c, -1)
        boundary_labels = boundary_labels.view(n, c, -1)
        extended_boundary_labels = extended_boundary_labels.view(n, c, -1)

        # Precision and Recall
        P = torch.sum(boundary_preds * extended_boundary_labels, dim=2) / (torch.sum(boundary_preds, dim=2) + self.smooth)
        R = torch.sum(extended_boundary_preds * boundary_labels, dim=2) / (torch.sum(boundary_labels, dim=2) + self.smooth)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + self.smooth)
        
        # Mean BF1 loss
        bf_loss = 1 - BF1.mean()
        #if torch.isnan(bf_loss) or torch.isinf(bf_loss):
        #    bf_loss = torch.tensor(0.0, device=outputs.device)

        # Calculate mmIoU Loss
        # Compute Intersection and Union for IoU
        inter = (outputs_softmax * labels).view(n, self.n_classes, -1).sum(2)
        union = (outputs_softmax + labels - (outputs_softmax * labels)).view(n, self.n_classes, -1).sum(2)
        
        # Compute IoU for each class
        iou = torch.clamp(inter / (union + self.smooth), min=0.0, max=1.0)

        # Minimum IoU for any class
        min_iou = torch.min(iou, dim=1).values

        # mmIoU Loss
        mm_iou_loss = -min_iou.mean() - iou.mean()
        #if torch.isnan(mm_iou_loss) or torch.isinf(mm_iou_loss):
        #    mm_iou_loss = torch.tensor(0.0, device=outputs.device)

        # Combine Losses
        total_loss = cross_entropy_loss + self.boundary_weight * bf_loss + self.iou_weight * mm_iou_loss

        # Check for NaNs or Infs
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)

        return total_loss   
    
  
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, n_classes=config.NUM_CLASSES):
        """
        Initialize the Tversky Loss.

        Args:
            alpha (float): Weight of false positives.
            beta (float): Weight of false negatives.
            smooth (float): Smoothing to avoid division by zero.
            n_classes (int): Number of classes.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.n_classes = n_classes

    def forward(self, outputs, labels):
        """
        Compute the Tversky Loss.

        Args:
            outputs (torch.Tensor): Raw logits from the model (N, C, H, W).
            labels (torch.Tensor): Ground truth labels (N, H, W) or one-hot encoded (N, C, H, W).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        outputs_softmax = F.softmax(outputs, dim=1)

        # Convert labels to one-hot encoding if necessary
        if labels.ndim == 3:
            one_hot_labels = F.one_hot(labels, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        elif labels.ndim == 4:
            one_hot_labels = labels
        else:
            raise ValueError("Labels must be (N, H, W) or (N, C, H, W)")

        # Flatten tensors
        outputs_flat = outputs_softmax.view(outputs.size(0), outputs.size(1), -1)
        labels_flat = one_hot_labels.view(one_hot_labels.size(0), one_hot_labels.size(1), -1)

        # Calculate Tversky components
        TP = (outputs_flat * labels_flat).sum(dim=2)
        FP = (outputs_flat * (1 - labels_flat)).sum(dim=2)
        FN = ((1 - outputs_flat) * labels_flat).sum(dim=2)

        TI = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Final Tversky loss
        loss = (1 - TI).mean()

        # Handle NaN or Inf safely
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            loss = torch.clamp(loss, min=0.0, max=1e6)

        return loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=[0.7, 0.3], reduction='mean', n_classes=config.NUM_CLASSES):
        """
        Initialize Focal Loss.

        Args:
            gamma (float): Focusing parameter.
            alpha (list or float or None): Weighting factor for classes. 
                Can be a list (per-class), float (single scalar for rare class), or None.
            reduction (str): 'mean', 'sum', or 'none'.
            n_classes (int): Number of classes.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_classes = n_classes

        if isinstance(alpha, (list, torch.Tensor)):
            assert n_classes is not None, "n_classes must be provided if alpha is a list."
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha.float()
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([1 - alpha] * n_classes, dtype=torch.float32)
            self.alpha[1] = alpha  # optionally prioritize class 1
        else:
            self.alpha = None

    def forward(self, outputs, labels):
        """
        Compute Focal Loss.

        Args:
            outputs (torch.Tensor): Raw logits from model (N, C, H, W)
            labels (torch.Tensor): One-hot encoded labels (N, C, H, W)

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Apply softmax to logits
        prob = F.softmax(outputs, dim=1)  # (N, C, H, W)

        if labels.ndim != 4 or labels.shape != outputs.shape:
            raise ValueError(f"Expected one-hot labels with shape {outputs.shape}, but got {labels.shape}")

        # Clamp for numerical stability
        prob = prob.clamp(min=1e-6, max=1. - 1e-6)

        # Cross-entropy per class
        ce_loss = -labels * torch.log(prob)

        # Focal scaling
        focal_term = (1 - prob) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(outputs.device).view(1, -1, 1, 1)
            loss = alpha * loss

        # Reduce
        if self.reduction == 'mean':
            return loss.sum() / labels.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for one-hot encoded labels with 2 classes.

        Args:
            smooth (float): Smoothing to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        """
        Args:
            outputs (Tensor): Raw logits from model of shape (N, 2, H, W)
            labels (Tensor): One-hot encoded labels of shape (N, 2, H, W)

        Returns:
            Tensor: Dice loss
        """
        if outputs.shape != labels.shape:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")

        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)

        # Flatten tensors
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        labels_flat = labels.view(labels.size(0), labels.size(1), -1)

        # Compute intersection and union
        intersection = (probs_flat * labels_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + labels_flat.sum(dim=2)

        # Compute dice score per class and batch
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Final dice loss = 1 - mean over batch and classes
        loss = 1 - dice.mean()

        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        BCE loss for one-hot encoded labels with 2 classes.

        Args:
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, labels):
        """
        Args:
            outputs (Tensor): Raw logits from model of shape (N, 2, H, W)
            labels (Tensor): One-hot encoded labels of shape (N, 2, H, W)

        Returns:
            Tensor: BCE loss
        """
        if outputs.shape != labels.shape:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")

        # Apply softmax to logits
        probs = F.softmax(outputs, dim=1)

        # Clamp for numerical stability
        probs = probs.clamp(min=1e-6, max=1 - 1e-6)

        # BCE Loss
        bce = -labels * torch.log(probs)

        if self.reduction == 'mean':
            return bce.sum() / labels.sum()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce
        
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        """
        IoU Loss for multi-class with one-hot encoded targets.

        Args:
            smooth (float): Smoothing to avoid division by zero.
            reduction (str): 'mean' or 'sum'.
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits from model, shape (N, C, H, W)
            targets (Tensor): One-hot encoded labels, shape (N, C, H, W)

        Returns:
            Tensor: IoU loss
        """
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}")

        # Apply softmax across classes
        inputs = F.softmax(inputs, dim=1)

        # Flatten per class
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        # Intersection and Union
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection

        # IoU per class per batch
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Mean over classes and batch
        loss = 1 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: (N, C)
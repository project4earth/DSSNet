import torch
import torch.nn.functional as F

class Metrics:
    def __init__(self, num_classes, threshold=0.5, zero_division=1):
        self.num_classes = num_classes
        self.threshold = threshold
        self.zero_division = zero_division

    def safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-6)

    def f1_score(self, logits, target):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # (N, H, W)

        if target.ndim == 4:
            target_idx = target.argmax(dim=1)
        else:
            target_idx = target

        f1_sum = torch.tensor(0.0, device=logits.device)
        weight_sum = torch.tensor(0.0, device=logits.device)

        for cls in range(self.num_classes):
            pred_cls = (preds == cls).float()
            target_cls = (target_idx == cls).float()

            tp = (pred_cls * target_cls).sum()
            pred_sum = pred_cls.sum()
            target_sum = target_cls.sum()

            if pred_sum > 0 and target_sum > 0:
                precision = self.safe_divide(tp, pred_sum)
                recall = self.safe_divide(tp, target_sum)
                f1 = self.safe_divide(2 * precision * recall, precision + recall)
                weight = target_sum
                f1_sum += f1 * weight
                weight_sum += weight

        score = self.safe_divide(f1_sum, weight_sum)
        return score

    def pixel_accuracy(self, logits, target):
        preds = logits.softmax(dim=1).argmax(dim=1)
        if target.ndim == 4:
            target = target.argmax(dim=1)

        valid_mask = (target != -1)
        correct = ((preds == target) & valid_mask).float().sum()
        total = valid_mask.float().sum()
        return self.safe_divide(correct, total)

    def calculate_iou_scores(self, pred, target):
        if target.ndim == 4:
            target = target.argmax(dim=1)  # one-hot to index
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)  # logits or probs to index

        ious = []
        weights = []
        class_presence = []

        valid_mask = (target != -1)

        for cls in range(self.num_classes):
            pred_cls = (pred == cls) & valid_mask
            target_cls = (target == cls) & valid_mask

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union > 0:
                iou = self.safe_divide(intersection, union)
                weight = target_cls.sum().float()
                ious.append(iou.item())
                weights.append(weight.item())
                class_presence.append(True)
            elif target_cls.sum() > 0:
                ious.append(0.0)
                weights.append(target_cls.sum().item())
                class_presence.append(True)
            else:
                class_presence.append(False)

        iou_sum = sum(i * w for i, w in zip(ious, weights))
        weight_sum = sum(weights)
        weighted_iou = iou_sum / (weight_sum + 1e-6) if weight_sum > 0 else 0.0

        valid_ious = [iou for iou, present in zip(ious, class_presence) if present]
        miou_valid = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        miou_traditional = sum(ious) / self.num_classes if self.num_classes > 0 else 0.0

        return weighted_iou, miou_valid, miou_traditional

    def iou_score(self, pred, target):
        weighted_iou, _, _ = self.calculate_iou_scores(pred, target)
        return weighted_iou

    def miou_score(self, pred, target, mode='valid'):
        """Mode bisa 'valid' (hitung kelas yang ada) atau 'traditional' (semua kelas)"""
        weighted_iou, miou_valid, miou_traditional = self.calculate_iou_scores(pred, target)
        return miou_valid if mode == 'valid' else miou_traditional
"""
Occupancy evaluation metrics
Lightweight mIoU/F-Score calculators that aggregate confusion matrices and
print per-class scores.
"""

import numpy as np


def _format_rows(headers, rows):
    col_widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(row):
        return " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
    lines = [fmt_row(headers), "-+-".join("-" * w for w in col_widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


class Metric_mIoU:
    """Mean Intersection over Union metric for occupancy prediction"""

    def __init__(self, num_classes=None, ignore_index=None, class_names=None, **kwargs):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.cm = None  # confusion matrix accumulator
        self.results = []

    def _update_cm(self, pred, target):
        """Accumulate confusion matrix from raw predictions/targets"""
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        max_class = max(pred.max(initial=0), target.max(initial=0)) + 1
        num_classes = self.num_classes or max_class
        cm = np.bincount(
            (target * num_classes + pred).astype(np.int64),
            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        if self.cm is None:
            self.cm = cm
        else:
            self.cm += cm

    def add_batch(self, pred=None, target=None, scene_id=None, frame_id=None, count_matrix=None):
        """Add a batch of predictions and targets"""
        if count_matrix is not None:
            cm = np.asarray(count_matrix)
            self.num_classes = self.num_classes or cm.shape[0]
            self.cm = cm if self.cm is None else self.cm + cm
        elif pred is not None and target is not None:
            self._update_cm(np.asarray(pred), np.asarray(target))

        self.results.append({
            'scene_id': scene_id,
            'frame_id': frame_id
        })

    def _compute_iou(self):
        if self.cm is None:
            return None, None
        eps = 1e-6
        tp = np.diag(self.cm)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + eps)
        valid = np.ones_like(iou, dtype=bool)
        if self.ignore_index is not None and self.ignore_index < len(iou):
            valid[self.ignore_index] = False
        miou = iou[valid].mean() if valid.any() else 0.0
        return iou, miou

    def print(self, runner=None):
        """Print evaluation results"""
        logger = runner.logger if runner is not None else None
        if self.cm is None:
            msg = "No confusion matrix accumulated; skipping mIoU print."
            (logger.info(msg) if logger else print(msg))
            return
        iou, miou = self._compute_iou()
        headers = ["Class", "IoU"]
        rows = []
        for idx, val in enumerate(iou):
            name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else f"cls_{idx}"
            rows.append([name, f"{val*100:.2f}"])
        rows.append(["mIoU", f"{miou*100:.2f}"])
        msg = "\n" + _format_rows(headers, rows)
        (logger.info(msg) if logger else print(msg))

    def compute_metric(self):
        """Compute the final mIoU"""
        if self.cm is None:
            return {}
        iou, miou = self._compute_iou()
        return {'mIoU': float(miou), 'IoU': iou.tolist()}


class Metric_FScore:
    """F-Score metric for occupancy prediction"""

    def __init__(self, num_classes=None, ignore_index=None, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.cm = None
        self.results = []

    def _update_cm(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        max_class = max(pred.max(initial=0), target.max(initial=0)) + 1
        num_classes = self.num_classes or max_class
        cm = np.bincount(
            (target * num_classes + pred).astype(np.int64),
            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        if self.cm is None:
            self.cm = cm
        else:
            self.cm += cm

    def add_batch(self, pred=None, target=None, scene_id=None, frame_id=None, count_matrix=None):
        if count_matrix is not None:
            cm = np.asarray(count_matrix)
            self.num_classes = self.num_classes or cm.shape[0]
            self.cm = cm if self.cm is None else self.cm + cm
        elif pred is not None and target is not None:
            self._update_cm(np.asarray(pred), np.asarray(target))
        self.results.append({
            'scene_id': scene_id,
            'frame_id': frame_id
        })

    def _compute_fscore(self):
        if self.cm is None:
            return None, None
        eps = 1e-6
        tp = np.diag(self.cm)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fscore = 2 * precision * recall / (precision + recall + eps)
        valid = np.ones_like(fscore, dtype=bool)
        if self.ignore_index is not None and self.ignore_index < len(fscore):
            valid[self.ignore_index] = False
        fscore_mean = fscore[valid].mean() if valid.any() else 0.0
        return fscore, fscore_mean

    def print(self, runner=None):
        if self.cm is None:
            msg = "No confusion matrix accumulated; skipping F-Score print."
            (runner.logger.info(msg) if runner else print(msg))
            return
        fscore, fscore_mean = self._compute_fscore()
        headers = ["Class", "F-Score"]
        rows = []
        for idx, val in enumerate(fscore):
            name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else f"cls_{idx}"
            rows.append([name, f"{val*100:.2f}"])
        rows.append(["Mean", f"{fscore_mean*100:.2f}"])
        msg = "\n" + _format_rows(headers, rows)
        (runner.logger.info(msg) if runner else print(msg))

    def compute_metric(self):
        if self.cm is None:
            return {}
        fscore, fscore_mean = self._compute_fscore()
        return {'FScore': float(fscore_mean), 'per_class_FScore': fscore.tolist()}

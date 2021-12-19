import cv2
import numpy as np
import sklearn.metrics

class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

    # Add your own metrics here, such as mAP, class-weighted accuracy, ...

    def accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self, tp, fp):
        return tp / (tp + fp)

    def recall(self, tp, fn):
        return tp / (tp + fn)

    def precision_recall_curve(self, y_true, pred_scores, thresholds):
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in pred_scores]

            precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label=1)
            recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)

            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls

    def mAP(self):
        return None
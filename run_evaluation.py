import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
import sklearn.metrics
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import matplotlib.pyplot as plt

TP_THRESHOLD = 0.5

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def convert_bb(self, bb):
        return [(bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3])]

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        import detectors.CNN.vgg16_detector as vgg16_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.Detector()

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        y_true = []
        y_pred = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing



            ## Histogram Equalalization
            #img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse

            ## Brightness Correction
            #img = preprocess.brightness_correction(img)

            ## Edge Enhancment
            #img = preprocess.edge_enhancment(img)


            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)



            y_true.append(1)
            if len(prediction_list) == 0:
                fn += 1
                y_pred.append(0)
                iou_arr.append(0.0)
                continue


            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)

            #true_bb = self.convert_bb(annot_list[0])
            #pred_bb = self.convert_bb(prediction_list[0])
            #img = cv2.rectangle(img, true_bb[0], true_bb[1], (0, 255, 0), 2) # green
            #img = cv2.rectangle(img, pred_bb[0], pred_bb[1], (0, 0, 255), 2) # red
            #cv2.imshow('Detection', img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)
            if iou >= TP_THRESHOLD:
                tp += 1
                y_pred.append(1)
            else:
                fp += 1
                y_pred.append(0)

        # Precision recall curve
        #thresholds = np.arange(start=0.2, stop=0.7, step=0.05)
        #precisions, recalls = eval.precision_recall_curve(y_true=y_true, pred_scores=iou_arr, thresholds=thresholds)
        #plt.plot(recalls, precisions, linewidth=4, color="red")
        #plt.xlabel("Recall", fontsize=12, fontweight='bold')
        #plt.ylabel("Precision", fontsize=12, fontweight='bold')
        #plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
        #plt.show()
        #f1s = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))
        #precisions.append(1)
        #recalls.append(0)
        #precisions = np.array(precisions)
        #recalls = np.array(recalls)
        #avg_precision = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        #print("Avg. Precision:", f"{avg_precision:.2%}")

        miou = np.average(iou_arr)
        precision = eval.precision(tp, fp)
        recall = eval.recall(tp, fn)
        accuracy = eval.accuracy(tp, tn, fp, fn)
        confusion_matrix = np.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))
        #f1 = 2 * precision * recall / (precision + recall)


        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("Accuracy:", f"{accuracy:.2%}")
        print("Confusion matrix: (From Left to Right & Top to Bottom: True Positive, False Negative, False Positive, True Negative)\n", confusion_matrix)
        print("Precision:", f"{precision:.2%}")
        print("Recall:", f"{recall:.2%}")
        #print("F1 score:", f"{f1:.2%}")
        #print("mAP:", f"{mAP:.2%}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
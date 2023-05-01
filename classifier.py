import cv2
import numpy as np
import matplotlib.pyplot as plt

from image import Image

# Bidirectional Map of Class Names to Class Numbers
ClassMap = {
    "bedroom": 0,
    "coast": 1,
    "forest": 2,
    0: "bedroom",
    1: "coast",
    2: "forest",
}


class Classifier:
    def __init__(self, train, test):
        # Image Data
        self._train = train
        self._test = test

        # Classifiers
        self._50NN = None
        self._siftNN = None
        self._histNN = None
        self._siftSVM = None

        # Results
        self.pctCorrect = {
            "50NN": None,
            "siftNN": None,
            "histNN": None,
            "siftSVM": None,
        }
        self.pctFalsePos = {
            "50NN": None,
            "siftNN": None,
            "histNN": None,
            "siftSVM": None,
        }
        self.pctFalseNeg = {
            "50NN": None,
            "siftNN": None,
            "histNN": None,
            "siftSVM": None,
        }

    def train(self):
        print("\n--- Training Classifiers ---")
        print("  Training Nearest Neighbor classifier using 50x50 pixels...")
        self._50NN = self.__train50NN()
        print("  Training Nearest Neighbor classifier using SIFT data...")
        self._siftNN = self.__trainSiftNN()
        print("  Training Nearest Neighbor classifier using Histogram data...")
        self._histNN = self.__trainHistNN()
        print("  Training SVM classifier using SIFT data...")
        self._siftSVM = self.__trainSiftSVM()

    def test(self):
        print("\n--- Testing Classifiers ---")
        print("  Testing 50x50 Nearest Neighbor classifier...")
        (
            self.pctCorrect["50NN"],
            self.pctFalsePos["50NN"],
            self.pctFalseNeg["50NN"],
        ) = self.__test50NN()

        print("  Testing SIFT Nearest Neighbor classifier...")
        (
            self.pctCorrect["siftNN"],
            self.pctFalsePos["siftNN"],
            self.pctFalseNeg["siftNN"],
        ) = self.__testSiftNN()

        print("  Testing Histogram Nearest Neighbor classifier...")
        (
            self.pctCorrect["histNN"],
            self.pctFalsePos["histNN"],
            self.pctFalseNeg["histNN"],
        ) = self.__testHistNN()

        print("  Testing SIFT SVM classifier...")
        (
            self.pctCorrect["siftSVM"],
            self.pctFalsePos["siftSVM"],
            self.pctFalseNeg["siftSVM"],
        ) = self.__testSiftSVM()

    def __train50NN(self):
        X = np.array([i.proc["50"].flatten() for i in self._train], dtype=np.float32)
        y = np.array([i.class_ for i in self._train])

        knn = cv2.ml.KNearest_create()
        knn.train(X, cv2.ml.ROW_SAMPLE, y)

        return knn

    def __trainSiftNN(self):
        X = np.array(
            [x for i in self._train for x in (i.f_sift["50"], i.f_sift["200"])]
        )
        y = np.array([x for i in self._train for x in (i.class_, i.class_)])

        knn = cv2.ml.KNearest_create()
        knn.train(X, cv2.ml.ROW_SAMPLE, y)

        return knn

    def __trainHistNN(self):
        X = np.array(
            [x for i in self._train for x in (i.f_hist["50"], i.f_hist["200"])]
        )
        y = np.array([x for i in self._train for x in (i.class_, i.class_)])

        knn = cv2.ml.KNearest_create()
        knn.train(X, cv2.ml.ROW_SAMPLE, y)

        return knn

    def __trainSiftSVM(self):
        X = np.array(
            [x for i in self._train for x in (i.f_sift["50"], i.f_sift["200"])]
        )
        y = np.array([x for i in self._train for x in (i.class_, i.class_)])

        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        svm.train(X, cv2.ml.ROW_SAMPLE, y)

        return svm

    def __test50NN(self):
        X = np.array([i.proc["50"].flatten() for i in self._test], dtype=np.float32)
        y = np.array([i.class_ for i in self._test])

        _, res, _, _ = self._50NN.findNearest(X, k=1)
        return self.__calcMetrics(res, y)

    def __testSiftNN(self):
        X = np.array([x for i in self._test for x in (i.f_sift["50"], i.f_sift["200"])])
        y = np.array([x for i in self._test for x in (i.class_, i.class_)])

        _, res, _, _ = self._siftNN.findNearest(X, k=1)
        return self.__calcMetrics(res, y)

    def __testHistNN(self):
        X = np.array([x for i in self._test for x in (i.f_hist["50"], i.f_hist["200"])])
        y = np.array([x for i in self._train for x in (i.class_, i.class_)])

        _, res, _, _ = self._histNN.findNearest(X, k=1)
        return self.__calcMetrics(res, y)

    def __testSiftSVM(self):
        X = np.array([x for i in self._test for x in (i.f_sift["50"], i.f_sift["200"])])
        y = np.array([x for i in self._test for x in (i.class_, i.class_)])

        res = self._siftSVM.predict(X)[1]
        return self.__calcMetrics(res, y)

    def __calcMetrics(self, res, y):
        c = self.__calcPctCorrect(res, y)
        fp = 1 - c
        fn = 0

        return c, fp, fn

    def __calcPctCorrect(self, res, y):
        count = 0
        for i in range(len(y)):
            if res[i] == y[i]:
                count += 1

        return count / len(y)

    def __calcPctFalsePos(self, res, y):
        pass

    def __calcPctFalseNeg(self, res, y):
        pass

    def generateResults(self):
        print("\n--- Generating Results ---")
        print("  Correct Rates:")
        print("  NN 50x50: {:.2f}%".format(self.pctCorrect["50NN"] * 100))
        print("  NN SIFT:  {:.2f}%".format(self.pctCorrect["siftNN"] * 100))
        print("  NN Hist:  {:.2f}%".format(self.pctCorrect["histNN"] * 100))
        print("  SVM SIFT: {:.2f}%".format(self.pctCorrect["siftSVM"] * 100))
        print("")
        print("  False Positive Rates:")
        print("  NN 50x50: {:.2f}%".format(self.pctFalsePos["50NN"] * 100))
        print("  NN SIFT:  {:.2f}%".format(self.pctFalsePos["siftNN"] * 100))
        print("  NN Hist:  {:.2f}%".format(self.pctFalsePos["histNN"] * 100))
        print("  SVM SIFT: {:.2f}%".format(self.pctFalsePos["siftSVM"] * 100))
        print("")
        print("  False Negative Rates:")
        print("  NN 50x50: {:2.2f}%".format(self.pctFalseNeg["50NN"] * 100))
        print("  NN SIFT:  {:2.2f}%".format(self.pctFalseNeg["siftNN"] * 100))
        print("  NN Hist:  {:2.2f}%".format(self.pctFalseNeg["histNN"] * 100))
        print("  SVM SIFT: {:2.2f}%".format(self.pctFalseNeg["siftSVM"] * 100))
        print("--------------------------")

        # use plt to generate a bar graph of the results
        labels = self.pctCorrect.keys()
        correct = [self.pctCorrect[k] for k in labels]
        falsePos = [self.pctFalsePos[k] for k in labels]
        falseNeg = [self.pctFalseNeg[k] for k in labels]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar(x - width, correct, width, label="Correct")
        ax.bar_label(ax.containers[0], fmt="%.2f")
        ax.bar(x, falsePos, width, label="False Positive")
        ax.bar_label(ax.containers[1], fmt="%.2f")
        ax.bar(x + width, falseNeg, width, label="False Negative")
        ax.bar_label(ax.containers[2], fmt="%.2f")

        ax.set_ylabel("Percent")
        ax.set_title("Classifier Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.legend()

        fig.tight_layout()

        plt.savefig("results.jpg")
        plt.show()

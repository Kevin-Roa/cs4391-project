import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        self.results = {
            "50NN": {
                "bedroom": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "coast": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "forest": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                }
            },
            "siftNN": {
                "bedroom": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "coast": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "forest": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                }
            },
            "histNN": {
                "bedroom": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "coast": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "forest": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                }
            },
            "siftSVM": {
                "bedroom": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "coast": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                },
                "forest": {
                    "TP": None,
                    "FP": None,
                    "FN": None
                }
            }
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
        self.results["50NN"] = self.__test50NN()

        print("  Testing SIFT Nearest Neighbor classifier...")
        self.results["siftNN"] = self.__testSiftNN()

        print("  Testing Histogram Nearest Neighbor classifier...")
        self.results["histNN"] = self.__testHistNN()

        print("  Testing SIFT SVM classifier...")
        self.results["siftSVM"] = self.__testSiftSVM()
#
# TRAINING
#
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
#
# TESTING
#
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
#
# CALCULATIONS
#
    def __calcMetrics(self, res, y):
        results = {
                "bedroom": {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0
                },
                "coast": {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0
                },
                "forest": {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0
                }
            }
        for classifier in range(3):
            total = 0
            tp = 0
            fp = 0
            fn = 0
            for i,val in enumerate(y):
                if val == classifier:
                    total += 1
                    if res[i] == val:
                        tp += 1
                    else:
                        fn += 1
                elif res[i] == classifier:
                    total += 1
                    fp += 1
            tp /= total
            fp /= total
            fn /= total
            results[ClassMap[classifier]]["TP"] = tp
            results[ClassMap[classifier]]["FP"] = fp
            results[ClassMap[classifier]]["FN"] = fn
        
        return results


    def generateResults(self):
        print("\n--- Generating Results ---")
        print("  Correct Rates: bedroom, coast, forest")
        print("    NN 50x50: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["50NN"]['bedroom']['TP'] * 100, self.results["50NN"]['coast']['TP'] * 100, self.results["50NN"]['forest']['TP'] * 100))
        print("    NN SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftNN"]['bedroom']['TP'] * 100, self.results["siftNN"]['coast']['TP'] * 100, self.results["siftNN"]['forest']['TP'] * 100))
        print("    NN Hist: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["histNN"]['bedroom']['TP'] * 100, self.results["histNN"]['coast']['TP'] * 100, self.results["histNN"]['forest']['TP'] * 100))
        print("    SVM SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftSVM"]['bedroom']['TP'] * 100, self.results["siftSVM"]['coast']['TP'] * 100, self.results["siftSVM"]['forest']['TP'] * 100))
        print("")
        print("  False Positive Rates: bedroom, coast, forest")
        print("    NN 50x50: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["50NN"]['bedroom']['FP'] * 100, self.results["50NN"]['coast']['FP'] * 100, self.results["50NN"]['forest']['FP'] * 100))
        print("    NN SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftNN"]['bedroom']['FP'] * 100, self.results["siftNN"]['coast']['FP'] * 100, self.results["siftNN"]['forest']['FP'] * 100))
        print("    NN Hist: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["histNN"]['bedroom']['FP'] * 100, self.results["histNN"]['coast']['FP'] * 100, self.results["histNN"]['forest']['FP'] * 100))
        print("    SVM SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftSVM"]['bedroom']['FP'] * 100, self.results["siftSVM"]['coast']['FP'] * 100, self.results["siftSVM"]['forest']['FP'] * 100))
        print("")
        print("  False Negative Rates: bedroom, coast, forest")
        print("    NN 50x50: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["50NN"]['bedroom']['FN'] * 100, self.results["50NN"]['coast']['FN'] * 100, self.results["50NN"]['forest']['FN'] * 100))
        print("    NN SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftNN"]['bedroom']['FN'] * 100, self.results["siftNN"]['coast']['FN'] * 100, self.results["siftNN"]['forest']['FN'] * 100))
        print("    NN Hist: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["histNN"]['bedroom']['FN'] * 100, self.results["histNN"]['coast']['FN'] * 100, self.results["histNN"]['forest']['FN'] * 100))
        print("    SVM SIFT: {:.2f}%, {:.2f}%, {:.2f}%".format(self.results["siftSVM"]['bedroom']['FN'] * 100, self.results["siftSVM"]['coast']['FN'] * 100, self.results["siftSVM"]['forest']['FN'] * 100))
        print("--------------------------")

        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(10)

        df = pd.DataFrame(self.results['50NN']).transpose()
        df.plot(kind="bar", ax=axes[0,0], title='50NN', rot=0, ylim=(0,1))

        df = pd.DataFrame(self.results['siftNN']).transpose()
        df.plot(kind="bar", ax=axes[0,1], title='siftNN', rot=0, ylim=(0,1))

        df = pd.DataFrame(self.results['histNN']).transpose()
        df.plot(kind="bar", ax=axes[1,0], title='histNN', rot=0, ylim=(0,1))

        df = pd.DataFrame(self.results['siftSVM']).transpose()
        df.plot(kind="bar", ax=axes[1,1], title="siftSVM", rot=0, ylim=(0,1))
        plt.savefig("results.jpg")
        plt.show()

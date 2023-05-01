import cv2
import numpy as np
import pickle
import os


class Image:
    def __init__(self):
        # Image Data
        self.orig = None
        self.class_ = None

        # Feature Vectors
        self.proc = {"50": None, "200": None}  # 50x50 and 200x200
        self.f_sift = {"50": None, "200": None}  # 128 elements each
        self.f_hist = {"50": None, "200": None}  # 256 elements each

    def loadData(self, imagePath: str, class_: int):
        cachePath = imagePath.replace("ProjData", "cache").replace(".jpg", ".pkl")

        if os.path.exists(cachePath):
            try:
                with open(cachePath, "rb") as f:
                    data = pickle.load(f)
                    self.orig = data["orig"]
                    self.class_ = data["class_"]
                    self.proc = data["proc"]
                    self.f_sift = data["f_sift"]
                    self.f_hist = data["f_hist"]
                    return True
            except:
                # If error in pkl file, regenerate image data
                return self.__getImageData(imagePath, cachePath, class_)
        else:
            return self.__getImageData(imagePath, cachePath, class_)

    def __getImageData(self, imagePath: str, cachePath: str, class_: int):
        self.orig = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if self.orig is None:
            return False

        self.class_ = class_
        self.__getFeatures()
        self.__saveData(cachePath)
        return True

    def __saveData(self, cachePath: str):
        # save to pkl file
        data = {
            "orig": self.orig,
            "class_": self.class_,
            "proc": self.proc,
            "f_sift": self.f_sift,
            "f_hist": self.f_hist,
        }
        os.makedirs(os.path.dirname(cachePath), exist_ok=True)
        with open(cachePath, "wb") as f:
            pickle.dump(data, f)

    def __getFeatures(self):
        self.__preprocess()
        self.__getSift()
        self.__getHist()

    def __preprocess(self):
        brightness = np.mean(self.orig) / 255
        img = self.orig
        if brightness < 0.4:
            img = cv2.addWeighted(self.orig, 1.5, 0, 0, 0)
        elif brightness > 0.6:
            img = cv2.addWeighted(self.orig, 0.5, 0, 0, 0)

        proc50 = cv2.resize(img, (50, 50))
        proc200 = cv2.resize(img, (200, 200))

        self.proc = {"50": proc50, "200": proc200}

    def __getSift(self):
        sift = cv2.SIFT_create()
        kp50, des50 = sift.detectAndCompute(self.proc["50"], None)
        kp200, des200 = sift.detectAndCompute(self.proc["200"], None)

        # if no keypoints, set to 0
        if des50 is None:
            des50 = np.zeros((1, 128))
        if des200 is None:
            des200 = np.zeros((1, 128))

        # homogenize shape of descriptors
        if des50.shape[0] < 128:
            des50 = np.vstack((des50, np.zeros((128 - des50.shape[0], 128))))
        if des200.shape[0] < 128:
            des200 = np.vstack((des200, np.zeros((128 - des200.shape[0], 128))))
        if des50.shape[0] > 128:
            des50 = des50[:128, :]
        if des200.shape[0] > 128:
            des200 = des200[:128, :]

        des50 = np.float32(des50)
        des200 = np.float32(des200)

        self.f_sift = self.__formatRet(des50, des200)

    def __getHist(self):
        hist50 = cv2.calcHist([self.proc["50"]], [0], None, [256], [0, 256])
        hist200 = cv2.calcHist([self.proc["200"]], [0], None, [256], [0, 256])

        self.f_hist = self.__formatRet(hist50, hist200)

    def __formatRet(self, r50, r200):
        r50 = np.array(r50.flatten())
        r200 = np.array(r200.flatten())
        return {"50": r50, "200": r200}

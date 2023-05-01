import cv2
import numpy as np
import os
import sys
from image import Image
from classifier import Classifier, ClassMap


def main():
    train, test = loadImages()
    classifier = Classifier(train, test)
    classifier.train()
    classifier.test()
    classifier.generateResults()


def loadImages():
    print("--- Loading Images ---")

    trainPath = ".\\ProjData\\Train"
    testPath = ".\\ProjData\\Test"

    print("  Reading Training Images...")
    train_imgs = readImages(trainPath)
    print("  Reading Testing Images...")
    test_imgs = readImages(testPath)

    return train_imgs, test_imgs


def readImages(path):
    imgs = []
    for cat in os.listdir(path):
        folder = os.path.join(path, cat)
        for file in os.listdir(folder):
            filePath = os.path.join(folder, file)
            img = Image()
            if img.loadData(filePath, ClassMap[cat.lower()]):
                imgs.append(img)
            else:
                print("Error loading image: " + filePath)
    return imgs


if __name__ == "__main__":
    main()

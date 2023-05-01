# CS4391 (Computer Vision) Final Project
The goal for this project is to combine image processing, feature extraction, clustering and classification methods to achieve basic scene understanding. 

There are two main parts to this project. The first part is to extract features from the images. The second part is to classify the images based on the features extracted. Feature extraction is done using grayscale histograms and SIFT. Classification is done using a nearest neighbor classifier and a support vector machine classifier.

### Usage
1. Clone the repository
   ``` bash
    git clone
   ```
2. Navigate to the project directory
   ``` bash
    cd cs4391-project
   ```
3. Run the program
   ``` bash
    python3 main.py
   ```

### Dependencies
- opencv-python
- numpy
- matplotlib

### File Structure
main.py - The main program file. This is where the program is run from. It loads the images and classifies them.

classifier.py - Contains the Classifier class. This class is used to classify images based on the features extracted from them.

image.py - Contains the Image class. This class stores the image data and provides methods to extract features from the image.

./ProjData - Images are available within the ProjData folder and are structured as follows:
- /ProjData
  - /Train
    - /bedroom
      - image1.jpg
    - /coast
      - ...
    - /forest
      - ...
  - /Test
    - /bedroom
      - ...
    - ...

### Notes
- The program will generate a cache folder in the project directory to store and read image data. This will be used to speed up the program on subsequent runs. If you wish to clear the cache, simply delete the folder.
- The program will output runtime logs to the console. At the end of the program, it will generate a report as well as a graph of the runtime data. These will be stored in the project directory.


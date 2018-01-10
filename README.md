# PeopleDetect.py

One of the most popular and successful “person detectors” out there right now is the HOG with SVM approach. HOG stands for Histograms of Oriented Gradients. HOG is a type of “feature descriptor”. The intent of a feature descriptor is to generalize the object in such a way that the same object (in this case a person) produces as close as possible to the same feature descriptor when viewed under different conditions.
The HOG person detector uses a sliding detection window which is moved around the image. At each position of the detector window, a HOG descriptor is computed for the detection window. This descriptor is then shown to the trained SVM, which classifies it as either “person” or “not a person”. To recognize persons at different scales, the image is subsampled to multiple sizes. Each of these subsampled images is searched.

## PeopleDetect.py in python with OpenCV

In OpenCV, there is peopleDetect python library which lets us to detect person by giving input image.

![peopledetect](https://user-images.githubusercontent.com/12676867/34783332-1ed58d42-f64d-11e7-889e-5da492255bf2.png)



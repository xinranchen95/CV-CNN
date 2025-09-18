# CV-CNN
This is a repository for predicting electro-organic reaction performance using cyclic voltammogram (CV) images as learning material with convolutional neural network (CNN). 
Since raw cyclic voltammetry data points are rarely disclosed, the repository also provides a tool for extracting individual CV curves from published figures. 
The curated dataset comprises 150 CVs corresponding to 50 individual reactions, each represented by a mixture CV, a substrate CV, and a catalyst CV. Plural-view strategy is employed for the reactivity calssification. and reactions with yield below 10% were labelled as unreactive.
We also carried out separate regression tasks for different metals, starting from cobalt catalysis, which constitutes the largest proportion of the collected dataset.

<p align="center">
  <img src="READMEpics/pic1.png" alt="image1" width="400"/>
</p>

# Data curation
The data curation was completed with a single CV profile extraction tool. individual_CV_extraction. ipynb file gives an example. With a reported CV figure in hand, the first step employs OCR technique to locate the current and voltage axes, identify the scale values for subsequent normalization, and remove the irrelevant legends. In the second step, the target redox profile is isolated by specifying its HSV (hue–saturation–value) range. To ensure quantitative comparability across all collected images in the final step, each extracted single curve is standardized into unified current (-500 to 500 μA) and voltage (-2.0 to 2.0 V vs. SCE) scales according to the scale values recorded in the first step.

<p align="center">
  <img src="READMEpics/pic2.png" alt="image2" width="400"/>
</p>

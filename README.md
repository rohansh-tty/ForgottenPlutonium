## FORGOTTEN PLUTONIUM

Why this name? Because I think TinyImageNet Dataset is underrated and I wanted to make it cool again and radioactive	:sunglasses:

**Notebook**: ![TinyImageNet Classifier](https://github.com/Gilf641/ForgottenPlutonium/blob/master/classifier.ipynb)

**Model Features:**

1. Used GPU as Device.
2. CNN Type: ResNet18
3. Total Params: 11,271,432
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: 0.01
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 50 Epochs 

        * Highest Validation Accuracy: 51.20%
        
9. GradCam for 25 Misclassified Images.


**Library Documentation:**

1.![AlbTransforms.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/AlbTransforms.py) : Applies required image transformation to both Train & Test dataset using Albumentations library.

2.![DataPrep.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/DataPrep.py): Consists of Custom DataSet Class and some helper functions to apply transformations, extract classID etc.

3.![resNet.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/resNet.py): Consists of main ResNet model

4.![execute.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/DataLoaders.py): Scripts to load the datasets.

6.![displayData.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/displayData.py): Consists of helper functions to plot images from dataset & misclassified images.

7.![Gradcam.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/Gradcam.py): Consists of Gradcam class & other related functions.

8.![LR Finder.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/LR_Finder.py): LR finder using FastAI Approach.

9.![cyclicLR.py](https://github.com/Gilf641/ForgottenPlutonium/blob/master/src/cyclicLR.py): Consists helper functions related to CycliclR.

**Plots & Curves**


![LR Finder](https://github.com/Gilf641/ForgottenPlutonium/blob/master/Images/LR%20finder.png)



**Model Performance**

![Accuracy Plot](https://github.com/Gilf641/ForgottenPlutonium/blob/master/Images/AccPlot.png)


![Loss Plot](https://github.com/Gilf641/ForgottenPlutonium/blob/master/Images/LossPlot.png)


**Misclassified Images**

![Misclassified](https://github.com/Gilf641/ForgottenPlutonium/blob/master/Images/Misclassified.png)

**GradCam for Misclassified Images**

![GradCam](https://github.com/Gilf641/ForgottenPlutonium/blob/master/Images/GradCam.png)




**Model Logs**
* ![Model Logs](https://github.com/Gilf641/ForgottenPlutonium/blob/master/ModelLogs.md)

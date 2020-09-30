# S12 Assignment

Task: 


    Assignment:
    
    Assignment A:
        Download this TINY IMAGENET (Links to an external site.) dataset. 
        Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
        Submit Results. Of course, you are using your own package for everything. You can look at this (Links to an external site.) for reference. 
    Assignment B:
        Download 50 images of dogs. 
        Use this (Links to an external site.) to annotate bounding boxes around the dogs.
        Download JSON file. 
        Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
        Refer to this tutorial (Links to an external site.). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub. 

 

**Assignment A Solution**: ![S12 Assignment A Solution](https://github.com/Gilf641/EVA4/blob/master/S12/S12_AssignmentSolution.ipynb)

**Assignment B Solution**: ![S12 Assignment B Solution](https://github.com/Gilf641/EVA4/blob/master/S12/S12_AssignmentSolution(K-Means).ipynb)



# Assignment A

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

1.![AlbTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/AlbTransforms.py) : Applies required image transformation to both Train & Test dataset using Albumentations library.

2.![DataPrep.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/DataPrep.py): Consists of Custom DataSet Class and some helper functions to apply transformations, extract classID etc.

3.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/resNet.py): Consists of main ResNet model

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/DataLoaders.py): Scripts to load the datasets.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/displayData.py): Consists of helper functions to plot images from dataset & misclassified images.

7.![Gradcam.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.

8.![LR Finder.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/LR_Finder.py): LR finder using FastAI Approach.

9.![cyclicLR.py](https://github.com/Gilf641/EVA4/blob/master/S12/evaLibrary/cyclicLR.py): Consists helper functions related to CycliclR.

**Plots & Curves**


![LR Finder](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/LR%20finder.png)



**Model Performance**

![Accuracy Plot](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/AccPlot.png)


![Loss Plot](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/LossPlot.png)


**Misclassified Images**

![Misclassified](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/Misclassified.png)

**GradCam for Misclassified Images**

![GradCam](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/GradCam.png)




**Model Logs**
* ![Model Logs](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/ModelLogs.md)

# Assignment B 

* ![DogImages Folder](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-B/Dogs)
* ![JSON File](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Dogs/dogsData.json)


**Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work)**

File which helps in storing Data Structures & Objects in JavaScript Object Notation is called JSON File. It consists of key-value pairs similar to Python Dictionaries. 
Now in this JSON file there are around 4 Keys/Attributes i.e Filename, Size, Regions and Attributes

Example: 
> "dog11.jpg5671":{"filename":"dog11.jpg","size":5671,"regions":[{"shape_attributes":{"name":"rect","x":97,"y":29,"width":82,"height":74},"region_attributes":{"Class":"Dog"}}],"file_attributes":{"caption":"","public_domain":"no","image_url":""}}

1. First, the Image name which is same as original image name with size attached at the end. This is the main Key. 
2. For this we have around 1 Value, which a dictionary consisting of 4 Keys i.e Filename, Size, Regions and File Attributes.
3. **Filename** is the Original Image Name.
4. **Size** is the image size.
5. **Region** consists of two attributes i.e **Shape** and **Region**
> **Shape Attribute** consists of 4 elements, which refer to the Bounding Box dimension. It consists of **X, Y, W** & ** 
    **(X,Y)** is the starting left (point/corner)coordinate of the bounding box, while W & H are width and height of the Bounding Box. Adding X to W & Y to H results in Bottom Right Coordinate of the Bounding Box. 
    
![Bounding Box](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/WhatsApp%20Image%202020-06-28%20at%206.24.48%20PM.jpeg)    
    
    
> While **Region Attribute** consists of **Class** as one of its key, which is equal to the assigned class for that particular object. 
6. Lastly, the **File Attributes** consists of 3 different attributes viz *Caption, Public_domain, image_url*. Caption is related any text passed based on the image. Image_url is link to the particular image. 


**K-Means Clustering**

* Data Distribution Scatter Plot
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/BBX-Data%20Distributio.png)




* Using Elbow Method, best K is 2
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/Elbow%20method.png)




* Calculating IOU for K using this 

![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/IOU%20Over%20K.png)




* Clustering Bounding Boxes with K=2
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/K-means%202.png)




* Clustering Bounding Boxes with K=3
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/K-means%203.png)



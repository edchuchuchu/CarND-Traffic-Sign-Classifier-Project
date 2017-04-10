## Build a Traffic Sign Recognition Program

Overview
---
In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. To train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, will then try out the model on images of German traffic signs find on the web.

The goals / steps of this project:
---
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Setup
To do this project, you will need Python 3 along with the following libraries, as well as Jupyter Notebook installed.
```
> pip install jupyter  
> pip install matplotlib 
> pip install numpy  
> pip install pickle  
> pip install sklearn  
> pip install cv2  
> pip install tensorflow
```  
  
### Dataset and Repository

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/edchuchuchu/Traffic-Sign-Classifier-Project
cd Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

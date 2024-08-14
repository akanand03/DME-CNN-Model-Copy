# Automatic Diabetic Macular Edema detection from Color Fundus Images

## Introduction:
Diabetic Macular Edema (DME) represents a major complication of diabetic retinopathy and is a leading cause of vision impairment and blindness among the diabetic population. 
The condition results from fluid accumulation in the macula due to leaking blood vessels in the eye, often exacerbated by poorly controlled blood sugar levels. 
Early detection and treatment are crucial to prevent irreversible damage to the retina and to maintain vision. Traditional methods of DME detection rely heavily on direct ophthalmoscopy and expert interpretation of optical coherence tomography (OCT) and fundus photographs. However, these methods require highly skilled professionals and are not always able to meet the demand for widespread screening in high-risk populations.

Recent advances in digital imaging and artificial intelligence (AI) offer new opportunities for automating the detection process, potentially increasing its accuracy and accessibility. Particularly, convolutional neural networks (CNNs) have shown exceptional capability in image recognition tasks and are increasingly being applied in the field of medical image analysis. 

## Proposed Methodology:
The proposed methodology to detect and classify Diabetic Macular Edema(DME) into its three grades is constructed by first collecting the relevant datasets, in our case the IDRiD dataset and Messidor Dataset. Afterwards applying numerous image pre-processing techniques on the images in the datasets to aid the CNN model that shall be implemented to extract the suitable features on which to train and classify DME with high accuracy.

### Dataset Collection and Preparation
For our model we have decided to utilize two established datasets for diabetic macular edema (DME) detection: Messidor and IDRiD. 
The Messidor Dataset has a total of 1187 images. To support research on computer-assisted diagnosis of diabetic retinopathy, the Messidor database was created. 
There are 964 images in the 0th grade of DME, 73 images in the 1st grade of DME and finally 150 images in the 2nd grade for DME. 
The pixel resolutions of the photos are 1440 x 960, 2240 x 1488, and 2304 x 1536.

In the IDRiD dataset, there are 407 images in total, with 177 in the 0th grade of DME, 41 images in the 1st grade of DME and finally 195 images in the 2nd grade for DME.
A retina specialist at an eye clinic in Nanded, Maharashtra, India, took these pictures. 
Experts have confirmed that every image is of sufficient quality, clinically important, unique, and contains a reasonable combination of disease stratification that is typical of DME. The resolution of the photos is 4288 x 2848 pixels. 
Upon analyzing the datasets the severity stage of DME can be predicted.

### Data Preprocessing
Before model training, preprocessing steps will be undertaken to ensure the data quality and enhance model robustness. The data preprocessing is essential to remove noise from the images, solve variations prevalent in the images due to lighting conditions among other things. This step will also include normalization to standardize pixel values across images. 
The acquired standardized dataset is preprocessed to enhance image contrast, assisting in the visible separation of exudates from the background.

### 3.2 Convolutional Neural Network (CNN) Model Architecture:
1. Model Design
Initial Setup: We began with input images from the IDRiD and Messidor datasets. 
These datasets were chosen because they contain a diverse range of color fundus images, classified based on the severity of diabetic retinopathy and macular edema. Utilizing these images allows our model to learn from a broad spectrum of cases.

Convolutional Layers: We applied multiple convolutional layers, each followed by a ReLU (Rectified Linear Unit) activation function. 

These layers are responsible for feature extraction:
First, we implemented depthwise convolutions to efficiently process the spatial information in the input images. By applying a single filter per input channel, we minimized the computational complexity and the number of parameters, enhancing the model's performance and reducing overfitting.
After each convolutional operation, we applied ReLU, which introduces non-linearity to the process, allowing the model to learn more complex patterns.

Pooling Layers: Pooling layers were strategically placed after certain convolutional layers to reduce the spatial dimensions of the feature maps:
We used pooling operations to down-sample the feature dimensions, which decreases the amount of computational power needed and helps to prevent overfitting by abstracting the features further.

Global Average Pooling (GAP): We incorporated a Global Average Pooling (GAP) layer towards the end of the convolutional layers. This layer reduces each feature map to a single value, summarizing the most important information from the feature maps, which simplifies the network architecture by drastically reducing the number of parameters.

Fully Connected Layers and Output: A Flatten layer was introduced to convert the pooled feature maps into a 1D feature vector, making it suitable for processing in the fully connected layers.We then added fully connected layers, where each neuron is connected to all activations in the previous layer. This dense network architecture uses the features extracted and pooled from previous layers to perform classification. ReLU was also employed in the fully connected layers to maintain non-linearity in deeper parts of the network.

Output with Softmax: The final part of our architecture is the output layer using a softmax function. This layer converts the logits, numerical outputs from the last fully connected layer, into probabilities by classifying the input images into one of three categories: Grade 0 (Absence of DME), Grade 1 (Mild DME), and Grade 2 (Severe DME).

Adaptive Learning Rate: Throughout the training process, we employed an adaptive learning rate to optimize the convergence of the model. By adjusting the learning rate based on the training epoch and observed error rates using techniques like Adam or RMSprop, we ensured that the model training was both efficient and effective.

Skip Connections: Skip connections, also known as shortcut connections or residual connections, are a key feature in modern deep learning architectures, particularly in Convolutional Neural Networks (CNNs).Skip connections allow the output from earlier layers to bypass one or more layers and be fed directly to deeper layers. They create "shortcuts" in the network, allowing information to flow more easily through the entire network. Skip connections allow the model to reuse features from earlier layers. In DME detection, this can help preserve fine-grained details from the original image that might be lost in deeper layers.

Concatenation: Lower-level features (from earlier layers) might capture fine details, while higher-level features (from later layers) might capture more abstract patterns. By concatenating these, we give the model access to a rich set of features for making its final decision.





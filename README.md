# Automatic Diabetic Macular Edema Detection from Color Fundus Images

## Introduction
Diabetic Macular Edema (DME) is a major complication of diabetic retinopathy and a leading cause of vision impairment and blindness among individuals with diabetes. The condition arises from fluid accumulation in the macula due to leaking blood vessels in the retina, often aggravated by poorly controlled blood sugar levels. Early detection and treatment are critical to prevent irreversible retinal damage and preserve vision.

Traditional detection methods, such as direct ophthalmoscopy and expert analysis of optical coherence tomography (OCT) and fundus photographs, are resource-intensive and rely on highly skilled professionals. These limitations make widespread screening in high-risk populations challenging.

Recent advancements in digital imaging and artificial intelligence (AI) offer promising opportunities to automate the detection process. Convolutional Neural Networks (CNNs), known for their exceptional performance in image recognition tasks, have emerged as powerful tools in medical image analysis, enabling improved accuracy and scalability.

## Proposed Methodology
The proposed methodology aims to detect and classify DME into three grades using advanced CNN architectures. The approach involves dataset preparation, image preprocessing, and a robust CNN model designed for high accuracy in DME detection.

### 1. Dataset Collection and Preparation
We utilize two established datasets for DME detection: Messidor and IDRiD.

#### Messidor Dataset:
- Total Images: 1187
- Distribution:
  - Grade 0: 964 images
  - Grade 1: 73 images
  - Grade 2: 150 images
- Resolutions: 1440x960, 2240x1488, and 2304x1536 pixels
- Purpose: To support research on computer-assisted diagnosis of diabetic retinopathy.

#### IDRiD Dataset:
- Total Images: 407
- Distribution:
  - Grade 0: 177 images
  - Grade 1: 41 images
  - Grade 2: 195 images
- Resolution: 4288x2848 pixels
- Captured by a retina specialist at an eye clinic in Nanded, Maharashtra, India.
- Characteristics: High-quality, clinically relevant, and diverse images.

These datasets provide a comprehensive basis for training and evaluating the model's ability to detect DME severity stages.

### 2. Data Preprocessing
Preprocessing is essential to enhance data quality and model robustness. Key steps include:
- **Noise Reduction:** Removing artifacts and noise from images.
- **Normalization:** Standardizing pixel values across images.
- **Contrast Enhancement:** Improving the visibility of exudates against the background.
- **Resizing:** Ensuring uniform input dimensions for the CNN model.

### 3. Convolutional Neural Network (CNN) Model Architecture

#### 3.1 Model Design
- **Input:** Fundus images from the IDRiD and Messidor datasets, representing diverse cases of DME.
- **Convolutional Layers:** Feature extraction using multiple convolutional layers with ReLU activation.
  - **Depthwise Convolution:** Efficiently processes spatial information, reducing computational complexity and mitigating overfitting.
  - **ReLU Activation:** Introduces non-linearity, enabling the model to learn complex patterns.
- **Pooling Layers:** Downsamples feature maps to reduce dimensionality and computational load.
- **Global Average Pooling (GAP):** Summarizes feature maps into single values, reducing parameters and simplifying the architecture.

#### 3.2 Fully Connected Layers and Output
- **Flatten Layer:** Converts feature maps into a 1D feature vector for dense layer processing.
- **Dense Layers:** Uses extracted features for classification, with ReLU activation maintaining non-linearity.
- **Output Layer:** Softmax function outputs probabilities for three categories:
  - Grade 0 (Absence of DME)
  - Grade 1 (Mild DME)
  - Grade 2 (Severe DME)

#### 3.3 Additional Architectural Features
- **Adaptive Learning Rate:** Techniques like Adam or RMSprop adjust the learning rate dynamically, optimizing model convergence.
- **Skip Connections:** Bypasses layers to retain fine-grained details from earlier layers, enhancing feature reuse and preserving spatial information.
- **Feature Concatenation:** Combines low-level and high-level features to improve decision-making.

### 4. Training and Evaluation
- **Training:** The model is trained using a balanced combination of the Messidor and IDRiD datasets, employing data augmentation to enhance generalization.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score are calculated to assess performance.

### Model Architecture Diagram

![Model Architecture](https://github.com/user-attachments/assets/26301314-9854-4772-8709-2303b8556644)

### Conclusion
This project leverages CNN-based architectures to address the critical challenge of DME detection. By automating the process, it enhances accuracy, scalability, and accessibility, contributing to early intervention and improved patient outcomes. The integration of advanced image preprocessing techniques, adaptive learning, and architectural innovations positions this model as a valuable tool in the field of medical imaging.

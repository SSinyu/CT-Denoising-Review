# CT image denoising with deep learning
-----------
#
#
#
## 01. A deep convolutional neural network using directional wavelets for low-dose X-ray CT reconstruction (KAIST-net)
> [`paper`](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.12344)
### **Dataset**
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge (only abdominal CT images)
>>  - 512x512, 10 patients, 5743 slices
>>  - use a 55x55 patches
### **Model**
> - This method works on wavelet coefficients of low-dose CT images
> - Network contains 24 convolution layers  
> ![KAISTNET](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B01%5DKAISTNET.PNG)
--------
#
#
#
## 02. Low-dose CT via Convolutional Neural Network
> [`paper`](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5330597/pdf/679.pdf)
### **Dataset**
> - TCIA(The Cancer Imaging Archive) normal-dose CT images.
>>  - 256x256, 165 patients, 7015 slices.
>>  - impose Poisson noise into normal-dose sinogram.
>>  - use a 33x33 patches.
### **Model**
> - Network use only 3 conoluional layers (Conv - ReLU - Conv - ReLU - Conv).
--------
#
#
#
## 03. Improving Low-Dose CT Image Using Residual Convolutional Network
> [`paper`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8082505)
### **Dataset**
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>>  - 512x512, 10 patients, 5080 slices
>>  - use a 44x44 patches(2D), 44x44x24 patches(3D)
### **Model**
> - 2D residual convolution net
> - 3D residual convolution net (take into account the spatial continuity of tissues)  
![ResCNN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B03%5DResidual_CNN.PNG)
--------
#
#
#
## 04. CT Image Denoising with Perceptive Deep Neural Networks
> [`paper`](https://arxiv.org/pdf/1702.07019.pdf)
### **Dataset**
> - cadaver CT image dataset collected at Massachusetts General Hospital (MGH)
### **Model**
> - Compare the denoised output against the ground truth in another high-dimensional feature space (from VGG)  
![PerDNN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B04%5DPerceptive_DNN.PNG)
--------
#
#
#
## 05. Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
> [`paper`](https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf)
### **Dataset**
> - NBIA(Natioanl Biomedical Imaging Archive) normal-dose CT images
>>  - 256x256, 165 patients, 7015 slices
>>  - adding Poisson noise into the sinogram simulated from the normal-dose images
>- AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>>  - 512x512, 10 patients, 2378 slices
>>  - use a 55x55 patches
### **Model**
> - Incoporated a deconvolution network and shortcut connections into a CNN model  
![REDCNN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B05%5DRED-CNN.PNG)
--------
#
#
#
## 06. Generative adversarial networks for noise reduction in low-dose CT
> [`paper`](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7934380)
### **Dataset**
> - Phantom CT scans
>>  - An anthropomorphic thorax phantom (QRM anthropomorphic thorax phantom)
>>  - voltage of 120 kVp. 50mAs(routine-dose), 10mAs(low-dose)
> - Cardiac CT scan (28 patients)
>>  - voltage of 120 kVp. 50~60mAs(routine-dose), 10~12mAs(low-dose)
### **Model**
> - Generator transforms the low-dose CT image into noise reduced image
> - Discriminator determines whether the input is a real routine-dose image or not  
![GAN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B06%5DGAN.PNG)
--------
#
#
#
## 07. Structure-sensitive Multi-scale Deep Neural Network for Low-Dose CT Denoising
> [`paper`](https://arxiv.org/pdf/1805.00587.pdf)
### **Dataset**
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>>  - 512x512, 10 patients, 2378 slices
>>  - use a 80x80x11 patches
### **Model**
> - (Part 1). Generator consist of eight 3D convolutional (Conv) layers
> - (Part 2). Calculate patch-wise error between the 3D output and the 3D NDCT images   
![SMGAN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B07%5DSMGAN.PNG)
> - (Part 3). Discriminator distinguishes between two images  
![SMGAN_loss](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B07%5Dloss.PNG)
--------
#
#
#
## 08. Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss
> [`paper`](https://arxiv.org/pdf/1708.00961.pdf)
### **Dataset**
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>>  - 512x512, 10 patients, 4000 slices
>>  - use a 64x64 patches
### **Model**
> - GAN with Wasserstein distance
> - (Part 2). Comparing the perceptual feature of a denoised output against those of the ground truth in an established feature space  
![WGAN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B08%5DWGAN.PNG)
--------
#
#
#
## 09. Sharpness-aware Low Dose CT Denoising Using Conditional Generative Adversarial Network
> [`paper`](https://arxiv.org/pdf/1708.06453.pdf)
### **Dataset**
> - NBIA(Natioanl Biomedical Imaging Archive) normal-dose CT images
>>  - 512x512, 239 slices
>>  - adding Poisson + normally Gaussian noise
>>  - use a 256x256 patches (sampled from the 4 corners and center)
> - Deceased piglet CT
>>  - voltage of 100 kVp. 300mAs(full-dose) ~ 15mAs(low-dose)
> - Phantom CT scans
>>  - voltage of 120 kVp. 300mAs(full-dose) ~ 15mAs(low-dose)
> - Data Science Bowl 2017 
>>  - Detect lung cancer from LDCTs
### **Model**
> - Sharpness detection network : generate a similar sharpness map as closs as to real CT  
![SAGAN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B09%5DSAGAN.PNG)
![SAGAN_loss](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B09%5DSAGNA_loss.PNG)
--------
#
#
#
## 10. 3D Convolutional Encoder-Decoder Network for Low-Dose CT via Transfer Learning from a 2D Trained Network
> [`paper`](https://arxiv.org/pdf/1802.05656.pdf)
### **Dataset**
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>>  - 512x512, 10 patients
>>  - use a 64x64 patches
### **Model**
> - Concatenation of feature-maps from the two sides of the conveying-path
> - Learn the 2D model first, and use it to initialize the 3D network. This transfer learning shows much faster convergence and better performance  
![Transfer](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B10%5DTransfer.PNG)
--------
#
#
#
## 11. Cycle Consistent Adversarial Denoising Network for Multiphase Coronary CT Angiography
> [`paper`](https://arxiv.org/abs/1806.09748)
### **Dataset**
> - 50 CT scans of mitral valve prolapse patients, and 50 CT scans of coronary artery disease patients
>>  - use a 56x56 patches
> - AAPM-Mayo Clinic Low-Dose CT Grand Challenge
### **Model**
> - In coronary CTA, the images at the low-dose and routine-dose phases do not match each other exactly due to the cardiac motion
> - Two generator denotes the mapping form low-dose to routine-dose image and from routine-dose to low-dose image, two adversarial discriminators distinguish between input images and synthesized images from the generators
> - Using cycle-consistent adversarial denoising network, learn the mapping between the low and routine dose cardiac phases  
![CYCLEGAN](https://github.com/SSinyu/CT_DENOISING/blob/master/img/%5B11%5DCYCLEGAN.PNG)


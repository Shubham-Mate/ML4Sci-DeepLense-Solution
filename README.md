# DeepLense Project - GSoC 2025 Application Solutions

This repository contains my solutions to the technical tests for the DeepLense project as part of my Google Summer of Code (GSoC) 2025 application. The tasks focus on machine learning approaches for analyzing gravitational lensing images, including classification and super-resolution challenges.

## Project Overview

### Task I: Multi-Class Classification of Gravitational Lensing Images
**Objective**: Build models to classify images into three categories:
1. Strong lensing with no substructure
2. Subhalo substructure
3. Vortex substructure

### Task II: Lens Finding
Build a model identifying lenses using PyTorch or Keras.

### Task III: Image Super-Resolution
- **Task III.A**: Train a deep learning-based super resolution algorithm of your choice to upscale low-resolution strong lensing images using the provided high-resolution samples as ground truths.
- **Task III.B**: Train a deep learning-based super-resolution algorithm of your choice to enhance low-resolution strong lensing images using a limited dataset of real HR/LR pairs collected from HSC and HST telescopes.

### Task IV: Diffusion Models for Gravitational Lensing Simulation
Develop a generative model to simulate realistic strong gravitational lensing images using diffusion models (DDPM). Implement and evaluate various architectural choices within the diffusion framework.


## Solution

### Task I: Multi-Class Classification

#### Approach

- Implemented by fine-tuning of ResNet model
- Compared performance using ROC curves and AUC scores

#### Results
- ROC-AUC Score: 0.9738

### Task II: Lens Finding

#### Approach
- Implemented a custom CNN architecture
- Fine-tuned a pretrained ResNet-18 Model
- Utilized Upsampling and downsampling to deal with class imbalances

#### Metrics
- AUC score
- Plotting of ROC curve (Can be found in the notebook)
- Accuracy

#### Results
This table is for AUC scores

| Model                  | Unmodified Dataset | Downsampling      | Upsampling         |
|------------------------|--------------------|-------------------|--------------------|
| LensCNN                | 0.9762768782660841 | 0.960415291039809 | 0.9725461123302295 |
| ResNet-18 (finetuned)  | 0.9813188884276008 | 0.977040639477031 | **0.9821499976935598** |


#### Future work
- Look into image specific techniques for dealing with class imbalances
- Utilize larger pretrained models

### Task III.A: Simulated Image Super-Resolution

#### Models Implemented
1. **SRCNN** (Small and Large variants)
   - Basic CNN architecture for super-resolution
2. **ESPCN**
   - Efficient Sub-Pixel Convolutional Network
3. **SRGAN**
   - Generative adversarial network approach
4. **SRResNet**
   - Residual network based approach

#### Evaluation Metrics
Compared models using:
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

#### Results

| Model         | PSNR (dB)         | SSIM              | MSE                   |
|---------------|-------------------|-------------------|-----------------------|
| SRCNN-small   | 42.01907361871431 | 0.974398921251297 | 6.332411169569241e-05 |
| SRCNN-large   | 42.10237243971328 | 0.974830070137977 | 6.21188970944786e-05  |
| ESPCN         | **42.28710515495454** | **0.975952082574367** | **5.95304630096507e-05**  |
| SRGAN         | 40.48610935549050 | 0.958761047363281 | 9.002151059030438e-05 |
| SRResNet      | 41.65625927735547 | 0.971796224296093 | 6.882382733238046e-05 |

### Task III.B: Real Telescope Image Super-Resolution

#### Approach
- Fine-tuned models from Task III.A on real HSC/HST data
- Employed transfer learning and data augmentation techniques

#### Evaluation Metrics
Compared models using:
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

#### Results

| Model         | PSNR (dB)         | SSIM               | MSE                   |
|---------------|-------------------|--------------------|-----------------------|
| SRCNN-small   | 34.09434717290395 | 0.8582045892874400 | 0.0008473546695313416 |
| SRCNN-large   | 35.21323692730531 | 0.8576015591621399 | **0.0007645150971560118** |
| ESPCN         | **35.39417348870955** | **0.8734989364941915** | 0.0007788492572823694 |
| SRResNet      | 35.24461601752956 | 0.8417196949323018 | 0.0007827904955775011 |

#### Future Work
- Look into diffusion model based approaches
- Look into training larger model
- Implement a special augmentation technique called CutBlur, which is designed for Image Super-Resolution


### Task 4:

#### Approach
- Used a cosine scheduler to add noise during forward diffusion process
- Used a U-Net architecture with residual connection to predict the added noise during backward diffusion process

#### Evaluation Metrics
-  Fréchet Inception Distance (FID)

#### Results:
- FID = **0.04538939893245697**

#### Saved Weights:
- link: [Model Weights](https://drive.google.com/file/d/1iLRwvWrmS2kKMosVNYsuLSGjakoVph9l/view?usp=sharing)

**Note**: This task has 2 notebooks, 1st notebook has model trained till 15 epochs and then checkpointed, while second notebook continues training that checkpointed model for 8 more epochs. This was done because of switching between different cloud platforms due to compute limitations





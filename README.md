# OCT-Generation-Project
## Overview

This project focuses on generating Optical Coherence Tomography (OCT) images using a UNet based architecture with Fourier Transform. The repository contains various scripts and modules that facilitate the training and generation of OCT images.

## Files

### `unet_fft_generator.py`
This code processes image sequences in the Fourier domain to predict and reconstruct intermediate sequence frames using Fourier-transformed data. This code:
 - Prepares the dataset, converts them to Fourier magnitude and phase. Splits sequences into odd (input) and even (target) frames for model training.
 - A U-Net variant tailored for Fourier data, integrating temporal embedding. Generates even frames from odd frames and conditional inputs.
 - Converts Fourier components back to spatial images using inverse FFT.
 - Uses the model to generate even frames, compares them with ground truth, and saves results.

### `data_prep.py`
This script renames .jpeg files in subdirectories by standardizing their numeric components. Standardizes 

### `train_test_split.py`
This script splits subfolders from a given root directory into "test" and "train" directories while preserving their structure.

## Dataset

The data collection part of the project is done by an outside collaboration. Medical school student Hilal Hacıo˘glu from Bezmialem University gathered OCT data from 141 different dibetic retinopathy patients. There are 19 slices from each eye. To augment the data, horizontally flipped versions of the images are included in the dataset as well. In total, there are 5358 different diseased OCT images in the dataset. The images are identificated with regard to which patient they belong to.
This is necessary for the context completeness. During the experiment phase of the project, the dataset might be expanded in case of a need occurs. This expansion might include both diseased and healthy people’s OCT images.


## Conclusion

This project aims to develop a model for generating intermediate OCT slice images, addressing the limitations of hardware, acquisition time, and patient comfort in capturing fine-grained sequential slices. By filling the gaps between real slices, the model enhances the completeness of visual data for medical diagnosis in ophthalmology.
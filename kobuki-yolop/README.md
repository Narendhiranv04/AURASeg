# Free Space Segmentation Model for COTS Perception Pipeline (Manuscript In-draft)


**Overview**

This repository contains the implementation of a Free Space Segmentation Model designed for COTS (Commercial Off-The-Shelf) Perception Pipelines, specifically tailored for autonomous navigation tasks using the Kobuki robot. The project focuses on optimizing ground plane segmentation in real-time using a modified streamlined version of the YOLOP (You Only Look Once Panoptic) architecture with the CSPDarknet53 backbone, enabling efficient and accurate segmentation of drivable areas in indoor environments.



**Key Features**
**Modified YOLOP Architecture:** Adapted the YOLOP model to focus solely on ground plane segmentation, enhancing real-time performance.    
ASPP Module: Utilized a lightweight Atrous Spatial Pyramid Pooling (ASPP) module for capturing multi-scale context, ensuring the model adapts to varying environments.

**Global Feature Upsample (GFU):** The GFU module integrates high-level features with the attention-guided feature maps in the decoder to generate a final refined segmentation output. It uses bilinear upsampling and global pooling to capture context and combines these features with the attention-guided outputs for efficient feature integration

**Attention Pyramid Fusion (APF) with Scale-aware Strip Attention Module:** The APF module is designed to bridge the semantic gap between different feature levels. It uses the Scale-aware Strip Attention Module (SSAM) to aggregate multi-scale contextual information by capturing long-range dependencies through vertical striping operations. This helps to relate pixels with similar semantic labels, making the model more effective in handling multi-scale objects in the scene​.

**Border Refinement Module (LABRM):** The LABRM is a specialized module introduced to enhance the segmentation of object boundaries. It combines Laplace convolution and spatial attention mechanisms to eliminate noise from low-level features and refine the edges, ensuring that the model captures sharper and more accurate object boundaries

**Model Architecture:**

![modified_architecture](https://github.com/user-attachments/assets/ee18a556-f15d-4d11-bd32-58b4c04bbe92)

**Dataset and Training**

Dataset: Custom Gazebo images with the simulation of kobuki turtlebot. 

Preprocessing: Applied noise filtering, data augmentation, and normalization to improve the model’s robustness.

Transfer Learning: Fine-tuned the model using pre-trained weights on the BDD100k dataset and stremlined the model to only Drivable Area Segmentation, focusing on adapting to the COTS environment.

**Qualitative Results**

![Screenshot from 2024-09-10 03-38-13](https://github.com/user-attachments/assets/91c399eb-f6bf-4f2b-beb2-019bd0ad8b63)
![Screenshot from 2024-09-10 03-16-15(1)](https://github.com/user-attachments/assets/d50b2fd6-a04d-4156-aa13-7f04cd18c31c)
![Screenshot from 2024-09-10 03-17-08](https://github.com/user-attachments/assets/cb1877f3-f5aa-420a-8e33-61ef30d8da08)


**Acknowledgments**

This work was carried out during my research internship at IIT Bombay, with guidance from Dr. Vivek Yogi and Professor Leena Vachhani. Special thanks to them for their support and resources.

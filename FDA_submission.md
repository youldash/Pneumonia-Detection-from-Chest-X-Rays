# FDA Submission

[flowchart]: misc/Flowchart.png "Algorithm Flowchart"

**Your Name:** Mustafa Youldash

**Name of your Device:** ChestXrayAnalyzer

## Algorithm Description 

### 1. General Information

#### Intended Use Statement

An algorithm intended for use by radiologists, assisting them with Pneumonia detection and identification, using two-dimensional (2D) chest X-ray scans as the data source.

#### Indications for Use

The algorithm may be deployed in a clinical setting that can analyze patient data to assist them in detecting the existence of Pneumonia by relying on their medical X-ray scans as input to the algorithm (or model if you may will). Patient’s medical history may well indicate Pneumonia or not. Patents are expected to be aged between their first year of age till 95 yrs. A patient can be either male or female. The X-ray scans (or images) are expected to be captured (scanned) in either a PA-, or an AP-viewing position (for clarity, PA stands for Posterior-Anterior, and AP stands for Anterior-Posterior).

#### Device Limitations

For the algorithm to run effectively, it must be installed and configured on a workstation (or computer) equipped with a capable **[NVIDIA Graphics Processing Unit (GPU)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/)** that is **[CUDA-capable](https://developer.nvidia.com/cuda-gpus/)**. More importantly, the GPU (or GPUs if the computer is equipped with more than one) must have enough VRAM (i.e., GPU memory) to handle the X-ray data properly. The algorithm, once trained and validated, can then be deployed on an application setting that can be either a desktop application or Web-based. There are no specific requirements for running the algorithm on the Operating System of choice – if the system can support CUDA. 

#### Clinical Impact of Performance

The goal here is to assist radiologists with their review of medical X-rays scans for indications of Pneumonia in them. A radiologist is expected to review all X-ray data and validate the validate the outcomes (using the device). In a situation where the device * inaccurately* claims the presents of Pneumonia in an image (in other words, it results in predicting a False Positive (FP) case from an image), the patient can then be directed for a sputum culture (or test) at a medica facility, and then the diagnoses/results from such a test can be further used to verify and validate the prediction. In the case of a False Negative (FN) however, a patient is most likely to be impacted adversely. To prevent this, a radiologist is advised to report to the clinician (in charge) to revise the outcomes of the prediction (including the original X-ray scan(s)) and proceed to performing a sputum culture. Hence, the radiologist’s review of the results is crucial.

### 2. Algorithm Design and Function

<div align="center">
	<img src="misc/Flowchart.png" width="100%" />
</div>

#### DICOM Checking Steps

The implementation checks the **DICOM** file *headers* looking for certain parameters like the following:
```
BodyPartExamined == 'CHEST'
Modality == 'DX'
PatientPosition is in a 'PA' or 'AP' viewing position
```
In other words, the checks involve:

- Analyzing the DICOM file, ensuring that the body part examined is a valid chest X-ray.
- Analyzing the DICOM file again to ensuring that the Imaging Modality is “DX.”
- Analyzing the DICOM file again to ensure that the Viewing Positions are valid AP and PA positions.

If *any* of the three categories did not match the requirements, a warning message will then be presented telling the user that the image did not meet the expected criteria.

#### Preprocessing Steps

If a DICOM file passes the initial DICOM file header checks, the DICOM pixel array will then be edited. A copy of the DICOM pixel array data will then be **normalized**, and **resized** to fit a `224 x 224` pixel ratio (or resolution).

In short, DICOM pixel array modifications will include:

- Image standardization (a.k.a. normalization), and
- Image resizing.

#### CNN Architecture


### 3. Algorithm Training

#### Parameters:

* Types of augmentation used during training
* Batch size
* Optimizer learning rate
* Layers of pre-existing architecture that were frozen
* Layers of pre-existing architecture that were fine-tuned
* Layers added to pre-existing architecture

<< Insert algorithm training performance visualization >> 

<< Insert P-R curve >>

**Final Threshold and Explanation:**

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:** 


**Description of Validation Dataset:** 


### 5. Ground Truth



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**

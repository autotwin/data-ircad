# ircad data

## Advisory

> 2022-08-01: Historical document included below.  Links and directories may or may not work.  No effort is undertaken to restore broken links, as this is document is now only for context.

## Goal

Create a 3D solid model of the human ~~head~~ chest from medical images without human intervention.

## Data

The 3D-IRCADb-01 database contains 3D CT scans of 10 males and 10 females with
hepatic tumors in 75 percent of the cases.  Patient-specific data follow, which
is an extraction from the source 
[table](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/).

> 2022-08-01: Currently only `patient_1` and `patient_2` have been uploaded to the G Drive to save hard drive space.  We are not certain if we will make use of the ircad data set because it is torso data.  We want to focus on head data for application to brain injury, emphasized by ONR.

| Patient No. | Sex | DOB  | Voxel Size (mm)  | Image Size (pixels) | G drive link                                                                                          |
| ----------- | --- | ---- | ---------------- | ------------------- | ----------------------------------------------------------------------------------------------------- |
| 1           | F   | 1944 | 0.57, 0.57, 1.6  | 512, 512, 129       | [patient_1](https://drive.google.com/drive/folders/11NsNjMkWVIVS9l6xd44x282TyW7PRxjD?usp=**sharing**) |
| 2           | F   | 1987 | 0.78, 0.78, 1.6  | 512, 512, 172       | [patient_2](https://drive.google.com/drive/folders/1TYpklL2Se09Y2LhzhGs9Ckuuap4iYlUQ?usp=sharing)     |
| 3           | M   | 1956 | 0.62, 0.62, 1.25 | 512, 512, 200       | to come                                                                                               |
| 4           | M   | 1942 | 0.74, 0.74, 2.0  | 512, 512, 91        | to come                                                                                               |
| 5           | M   | 1957 | 0.78, 0.78, 1.6  | 512, 512, 139       | to come                                                                                               |
| 6           | M   | 1929 | 0.78, 0.78, 1.6  | 512, 512, 135       | to come                                                                                               |
| 7           | M   | 1946 | 0.78, 0.78, 1.6  | 512, 512, 151       | to come                                                                                               |
| 8           | F   | 1970 | 0.56, 0.56, 1.6  | 512, 512, 124       | to come                                                                                               |
| 9           | M   | 1949 | 0.87, 0.87, 2.0  | 512, 512, 111       | to come                                                                                               |
| 10          | F   | 1953 | 0.73, 0.73, 1.6  | 512, 512, 122       | to come                                                                                               |
| 11          | M   | 1966 | 0.72, 0.72, 1.6  | 512, 512, 132       | to come                                                                                               |
| 12          | F   | 1973 | 0.68, 0.68, 1.0  | 512, 512, 260       | to come                                                                                               |
| 13          | M   | 1951 | 0.67, 0.67, 1.6  | 512, 512, 122       | to come                                                                                               |
| 14          | F   | 1970 | 0.72, 0.72, 1.6  | 512, 512, 113       | to come                                                                                               |
| 15          | F   | 1946 | 0.78, 0.78, 1.6  | 512, 512, 125       | to come                                                                                               |
| 16          | M   | 1950 | 0.70, 0.70, 1.6  | 512, 512, 155       | to come                                                                                               |
| 17          | M   | 1942 | 0.74, 0.74, 1.6  | 512, 512, 119       | to come                                                                                               |
| 18          | F   | 1958 | 0.74, 0.74, 2.5  | 512, 512, 74        | to come                                                                                               |
| 19          | F   | 1970 | 0.70, 0.70, 4    | 512, 512, 124       | to come                                                                                               |
| 20          | F   | 1949 | 0.81, 0.81, 2    | 512, 512, 225       | to come                                                                                               |

### Potential New Data Source

[CSI 2014](http://csi-workshop.weebly.com/challenges.html)

## HPC SYN hx

### Accessing the p2m directory

Our directory is located at `data/wg-p2m`. You can get to it from your home directory by running `cd ../../data/wg-p2m`.

### Using the same python environment

From `data/wg-p2m`, run `source miniconda/bin/activate` followed by `conda activate p2m`.  

**IMPORTANT: Don't mess with the version of tensorflow, keras, or cuda. That took a while to set up, and while I [Anirudh] do have notes, recreating it might be a bit of a pain.**

### Data and Code

* The data are located in `geometry/doc/pixel2mesh/unet/data/`
* The code is located in `geometry/doc/pixel2mesh/unet/unet_keras/`

## Code description

### `data_gen_np.py`

This module defines the `PatientData` object.  

* `PatientData` looks for files in the 3DIRCAD data folder. The database
    contains DICOM files located in different folders, PATIENT_DICOM which
    contains the original CT scans, and MASKS_DICOM, which contains subfolders
    of masks of the CT scan. For example, the directory MASKS_DICOM/bone will
    contain the corresponding bone masks for images in PATIENT_DICOM
* You can then call `PatientData.get_data`, which resizes and normalises the
    data before splitting it into train and test and saving them out as .npy files

### `make_data_np.py`

This module creates a PatientData object and calls get_data to save the scans and
masks as numpy files.

### `model.py`

This module defines the function `get_unet` which returns a compiled U-Net model.
A description of the model can be found in the file itself or in the [U-Net paper](https://arxiv.org/abs/1505.04597).

### `losses.py`

This module defines three functions.

1. [`dice_coef`](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), which is defined as DSC = 2|X ∩ Y| / (|X| + |Y|) 
2. `dice_coef_loss`, which defines the loss as `1-dice_coef`. `-dice_coef` is equally acceptable, but less readable as we generally expect the loss to have converged at 0 and not -1
3. `dice_coef_no_tf` again defines the dice coefficient, but this time without using tensorflow functions. This is useful for visualizing results.

### `callbacks.py`

This module implements the callbacks used during the training of our U-Net model. We use the standard ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping callbacks, while defining an additional StopTraining callback, which stops the training if the dice coefficient reaches 1.0.

### `train_np.py`

This module loads the numpy files and trains.

### `eval_model.py`

This module evaluates our model on the test data.

### `make_predictions.py`

This module predicts what the mask for a given scan will look like and saves the result to a .npy file

### `visualize_pred.py`

This module shows a plot comparing our prediction with the true result.

### `patient_film.py`

This module plots a nice animation scrolling through a patient's scans.


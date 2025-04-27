# Volume Slicing, ADS-net Training, and Prediction for Adsorption Maps

This repository contains a pipeline for processing microstructure volumes (in Tiff format) and masks (adsorption maps), using them for training an ADS-net architecture and predicting adsorption maps for slices of the volumes.

The workflow is divided into three main stages:

1. **Slicing**: Slices 3D microstructure volumes and masks into 2D slices for training.
2. **Training**: Trains an ADS-net model on the sliced data using a SSIM loss function.
3. **Prediction**: Predicts adsorption maps for slices and reconstructs the 3D volume.

---

## Slicing

### Purpose
The `slice_volumes.py` script slices the 3D volumes and corresponding masks into 2D slices. It processes volumes stored in Tiff format and masks in FDA format, creating a set of 2D slices for both the images and masks.

### Input
- **Microstructure Volumes**: Tiff files stored in the `train_data/{TARGET}/imgDT/` directory.
- **Adsorption Maps (Masks)**: FDA files stored in the `train_data/{TARGET}/mask/` directory.

### Output
- **Sliced Images and Masks**: The script saves the 2D slices into the `slices/{TARGET}/img/` and `slices/{TARGET}/mask/` directories.

### Usage
Ensure that the `TARGET` is set to either `"train"` or `"test"`. Run the following command:
```bash
python slice_volumes.py 
```

## Training

### Purpose
The `train.py` script allows to train the ADS-net model on the sliced data. The goal is to train the model on 2D slices of the microstructure volumes and adsorption maps (masks) and save the models weights.

### Input
- **Sliced Images**: 2D images of microstructures stored in `slices/train/img/`.
- **Sliced Masks**: 2D images of adsorption maps stored in `slices/train/mask/`.

### Output
- **Model Weights**: Saved in the `checkpoints/` directory after every epoch.
- **Training and Validation Plots**: Accuracy and loss graphs for both training and validation sets, saved as visualizations during the training process.

### Functionality
- **Model Architecture**: The model uses a U-Net++ style architecture.
- **Custom SSIM Loss**: The training utilizes a custom SSIM (Structural Similarity Index) loss function.
- **Checkpointing**: The model’s weights are saved at each epoch.
- **Visualization**: The script generates and displays training/validation accuracy and loss curves to monitor the model’s performance over time.

### Usage
Run the following command to begin the training process:
```bash
python train.py 
```

## Prediction

### Purpose
The `predict.py` script is designed to apply the trained ADS-net model to predict adsorption maps for slices of a target volume. It reconstructs the full 3D volume from 2D slice predictions and saves the predicted volume in Tiff format. Additionally, the script generates a PDF visualization of the predicted results.

### Input
- **Trained Model Weights**: The trained ADS-net model weights (e.g., `ads-net-dt.h5`) from the `checkpoints/` directory.
- **Target Volume**: A Tiff file representing the target microstructure volume to predict adsorption maps for, stored in the `predict/img/` directory.

### Output
- **Predicted Volume**: The full 3D predicted volume of adsorption maps, saved as a Tiff file in the `predict/result/` directory.
- **Prediction Visualization**: A PDF file containing a visualization of the predicted adsorption maps and slices, saved in the `predict/result/` directory.

### Functionality
- **Slice-wise Prediction**: The script processes each slice of the target volume along the X axis. The model predicts the adsorption maps for each slice independently.
- **Volume Reconstruction**: After processing individual slices, the script reconstructs the predicted 3D volume from the predicted 2D slices.
- **Normalization and Scaling**: The input target volume is normalized before making predictions, and the predicted output is scaled back to the original dimensions.
- **Visualization**: The predicted volume and selected slices are visualized and saved as a PDF.

### Usage
To run the prediction, use the following command:
```bash
python predict.py
```
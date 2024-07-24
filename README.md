### [Development of Fully Automated Models for Staging Liver Fibrosis Using Non-Contrast MRI and Artificial Intelligence: A Retrospective Multicenter Study]

## Installation
```
git clone https://github.com/CLL-CMU/StagingLiverFibrosis
cd StagingLiverFibrosis
pip install -r requirements.txt

```

## Training
To train the model, use the Trainer.py script. You need to specify the directory containing the MRI images, the mode, and the fold number.
--data_dir: Path to the directory containing MRI image datasets.
--mode: Type of MRI image (e.g., T1, T2).
--fold: Fold number to use for cross-validation (e.g., 1).
Example command to start the training process:
```
python /StagingLiverFibrosis/Trainer.py \
--data_dir /imagepath/ \
--mode T1 --fold 1 

```
## Evaluation
After training the model, you can evaluate its performance using the Infer.py script. The script requires the same parameters as the training script to locate and appropriately use the trained model.
--data_dir: Path to the directory containing MRI image datasets.
--mode: Type of MRI image (e.g., T1, T2).
--fold: Fold number to use for cross-validation (e.g., 1).
Example command to start the evaluation process:


```
python /StagingLiverFibrosis/Infer.py \
--data_dir /imagepath/ \
--mode T1 --fold 1
```

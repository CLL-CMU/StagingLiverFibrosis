### [Development of Fully Automated Models for Staging Liver Fibrosis Using Non-Contrast MRI and Artificial Intelligence: A Retrospective Multicenter Study]

## Installation
git clone https://github.com/CLL-CMU/StagingLiverFibrosis
cd StagingLiverFibrosis
pip install -r requirements.txt 

## Training

```
python /StagingLiverFibrosis/Trainer.py \
--data_dir /imagepath/ \
--mode T1 --fold 1 

```
## Evaluation
```
python /StagingLiverFibrosis/Infer.py \
--data_dir /imagepath/ \
--mode T1 --fold 1
```

# Complete Project Plan

## Title
Plant Health Monitoring Using Deep Learning: Whole-Plant Segmentation and Stem Disease Detection

## Objective
Develop a deep learning system that analyzes whole-plant images and segments healthy leaves, stems, leaf disease spots, stem lesions, and wilting/dead tissue.

## Architecture
Input Image → Preprocessing → COCO Annotation → Mask R-CNN → Segmentation Output → Health Summary → Streamlit Web App

## Main Model
Mask R-CNN with ResNet-50 FPN backbone pretrained on COCO.

## Baseline Model
Mini U-Net trained from scratch for semantic segmentation comparison.

## Dataset
Collect 100–200 plant images using smartphone cameras.

Recommended split:
- 70% training
- 15% validation
- 15% testing

## Annotation
Use CVAT/Roboflow to create COCO instance segmentation masks.

Classes:
1. healthy_leaf
2. stem_branch
3. leaf_disease_spot
4. stem_disease_lesion
5. wilting_dead_region

## Metrics
- Mask IoU
- Dice score
- mAP
- Precision
- Recall
- F1-score
- Inference time
- Model size

## Expected Results
The pretrained Mask R-CNN should perform better on segmentation quality. The Mini U-Net should be faster and smaller but less accurate for instance-level detection.

## Presentation Flow
1. Title
2. Problem statement
3. Motivation
4. Dataset
5. Annotation classes
6. Sample annotations
7. Mask R-CNN architecture
8. Training process
9. Evaluation metrics
10. Mini U-Net comparison
11. Results
12. Web app demo
13. Challenges
14. Conclusion

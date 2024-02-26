Introduction
This code is the open source implementation of the paper "Evaluation and Analysis of Feature Point Detection Methods Based on vSLAM Systems". We have categorized the S-PTAM project into four types: based on traditional detection methods, R2D2, D2-Net, and SuperPoint, and have completed the part for the TartanAir dataset. Here we provide a runnable usage process, with a detailed introduction to follow.

Requirements
- Python 3.7
- NumPy
- OpenCV (cv2)
- g2o
- Pangolin

Usage Process

We have refined the code. You only need to modify the `path` and `dataset` parameters in `sptam.py` to run it successfully.
You can also use:
- `python sptam.py --dataset kitti --path /path/to/your/KITTI_odometry_dataset/sequences/00`
- `python sptam.py --dataset euroc --path /path/to/your/EuRoC_MAV_dataset/MH_01_easy`
- `python sptam.py --dataset tartanair --path /path/to/your/TartanAir_dataset/your_sequence_name`

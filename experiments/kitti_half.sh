cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py tracking --exp_id kitti_01 --dataset kitti_tracking --dataset_version train_half --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 8 --load_model ../models/nuScenes_3Ddetection_e140.pth
# test
python test.py tracking --exp_id kitti_01 --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --resume
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --load_model ../models/kitti_half.pth
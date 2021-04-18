# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN True
# test
python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

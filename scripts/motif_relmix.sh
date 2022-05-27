CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --master_port 10225 --nproc_per_node=4 \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CumixMotifPredictor \
    SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" \
    SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /data_local/chenqianyu/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT /data_local/chenqianyu/pysgg_checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR /data_local/chenqianyu/SGG_output/motif-relmix-sgcls > motif-relmix-sgcls-cumix-w-obj.out

CUDA_VISIBLE_DEVICES=8,9 nohup python -m torch.distributed.launch --master_port 10235 --nproc_per_node=2 \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CumixMotifPredictor \
    SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /data_local/chenqianyu/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT /data_local/chenqianyu/pysgg_checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR /data_local/chenqianyu/SGG_output/motif-relmix-sgcls-wo-obj > motif-relmix-sgcls-cumix-wo-obj.out

CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 \
    tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CumixMotifPredictor   \
    TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /data_local/chenqianyu/glove  \
    MODEL.PRETRAINED_DETECTOR_CKPT  /data_local/chenqianyu/SGG_output/motif-relmix-sgcls \
    OUTPUT_DIR  /data_local/chenqianyu/SGG_output/motif-relmix-sgcls

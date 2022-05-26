CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port 10025\
 --nproc_per_node=6 tools/relation_train_net.py\
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR CumixMotifPredictor \
  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 6 DTYPE "float16" \
  SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/chenqianyu/glove \
  MODEL.PRETRAINED_DETECTOR_CKPT /data_local/chenqianyu/SGG_checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR /data_local/chenqianyu/SGG_checkpoints/cumix-motif-predcls-tmp \
  MODEL.ROI_BOX_HEAD.NUM_CLASSES 70099 \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 1808 \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.CUMIX True MODEL.HUBNESS_SCALE 1.0 \
  MODEL.AUG_PERCENT 30 MODEL.RANDOM_LAMBDA True \
  MODEL.MIXUP False SOLVER.PRE_VAL True


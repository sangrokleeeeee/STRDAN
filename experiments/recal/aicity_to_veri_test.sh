python train_xent_tri.py -s veri -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 3 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-aicity-to-veri \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--random-erase \
--evaluate
# --load-weights log/resnet50-baseline-aicity-to-veri/model.pth.tar-60 \
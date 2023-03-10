python train_xent_tri.py -s veri -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-sim-veri \
--gpu-devices 1 \
--train-sampler sim \
--include-sim \
--random-erase \
--include-ori --include-color --include-type
name="PCT"
data_dir="datasets/University-Release/train"
test_dir="datasets/University-Release/test"
pretrain_path="/home/ps/Code/wxm_project/Polar_Contrastive_Transfomer/main_project/pretrain_model/vit_small_p16_224-15ec54c9.pth"
gpu_ids=0
num_worker=16
lr=0.22
# It seems that lr=0.01 is suitable for batchsize = 8 (On NVIDIA RTX 4070 Laptop GPU)
# When you change the batchsize, you may need to adjust the learning rate accordingly.
# Apply Linear Scaling Rule: Scale learning rate linearly with batch size.
# For example, when batchsize=16, lr=0.02 may be better.
sample_num=1
block=3
batchsize=192
triplet_loss=0.3
num_epochs=150
pad=0
views=2
warmup_epochs=10
save_epoch_freq=10
save_start_epoch=120

# training
python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views $views --lr $lr \
--sample_num $sample_num --block $block --batchsize $batchsize --triplet_loss $triplet_loss --num_epochs $num_epochs --pretrain_path $pretrain_path\
--warmup_epochs $warmup_epochs --save_epoch_freq $save_epoch_freq --save_start_epoch $save_start_epoch\

# testing
cd checkpoints/$name
for((i = $save_start_epoch; i <= $num_epochs; i += $save_epoch_freq)); 
    # testing the model every $save_epoch_freq epochs
do
  for((p = 0; p <= $pad; p+=10));
  do
    for ((j = 1; j < 3; j++));
    do
        python test_server.py --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker --pad $pad
    done
  done
done



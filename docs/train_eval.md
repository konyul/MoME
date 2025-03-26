## Train & Inference
#### Train

Train 1st Stage

```shell
tools/dist_train.sh ./projects/configs/moad_voxel0075_vov_1600x640_cbgs.py 8
```

Train 2nd Stage
```shell
tools/dist_train.sh ./projects/configs/mome/mome.py 4
```

#### Inference
```shell
tools/dist_test.sh ./projects/configs/mome/mome.py $path_to_weight$ $num_gpus --eval bbox
```

## Train & Inference
### 1st stage
#### Train
```shell
tools/dist_train.sh $path_to_config$ 8
```
#### Inference
```shell
tools/dist_test.sh $path_to_config$ $path_to_weight$ 8 --eval bbox
```

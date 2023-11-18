# Deit-Pytorch

This is the DeiT model code warehouse, which mainly includes models: ___DeiT___, ___DeiTV2___, ___ResMLP___, ___CaiT___, ___patchconvnet-models___. The code is mainly derived from the official source code of facebookresearch, and has been modified based on it. Now it can be applied to your own image classification datasets.

## Precautions
<1>Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to remove the comments of ___Plot_ROC___ and ___Predictor___ at the end of the code. The comment of the function is enough, and the third parameter can be changed to the path of your own model weights file(.pth).

<2>If you want to use another model, import it in the ___train_gpu.py___ file, then find the following code and replace the name of model function.
```
model = deit_tiny_patch16_224(pretrained=False,
                              num_classes=args.nb_classes,
                              drop_rate=args.drop,
                              drop_path_rate=args.drop_path,
                              img_size=args.input_size
                              )
```

## Train this model
### train model with single-machine single-card：
```
python train_gpu.py
```

### train model with single-machine multi-card：
```
torchrun --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-card: 
(using a specified part of the cards: for example, I want to use the second and fourth cards)
```
CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-card:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain card, just add CUDA_VISIBLE_DEVICES= to specify the index number of the card before each command. The principle is the same as single-machine multi-card training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Paper
```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```

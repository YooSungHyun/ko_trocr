capability=`CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"`
capability=${capability: 1:1}.${capability: -2:1}
echo $capability
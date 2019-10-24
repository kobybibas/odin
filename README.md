# ODIN: Out-of-Distribution Detector for Neural Networks


This is an unofficial [PyTorch](http://pytorch.org) implementation for detecting out-of-distribution examples in neural networks. The method is described in the paper [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) by S. Liang, [Yixuan Li](www.yixuanli.net) and [R. Srikant](https://sites.google.com/a/illinois.edu/srikant/). The method reduces the false positive rate from the baseline 34.7% to 4.3% on the DenseNet (applied to CIFAR-10) when the true positive rate is 95%.
<p align="center">
<img src="./figures/original_optimal_shade.png" width="500">
</p>




## Experimental Results

We used two neural network architectures, [DenseNet-BC](https://arxiv.org/abs/1608.06993) and [Wide ResNet](https://arxiv.org/abs/1605.07146).
The PyTorch implementation of [DenseNet-BC](https://arxiv.org/abs/1608.06993) is provided by [Andreas Veit](https://github.com/andreasveit/densenet-pytorch) and [Brandon Amos](https://github.com/bamos/densenet.pytorch). The PyTorch implementation of [Wide ResNet](https://arxiv.org/abs/1605.07146) is provided  by [Sergey Zagoruyko](https://github.com/szagoruyko/wide-residual-networks).
The experimental results are shown as follows. The definition of each metric can be found in the [paper]().
![performance](./figures/performance.png)




## Running the code

### Dependencies

Install requirements: 
```bash
while read requirement; conda install --yes $requirement;or pip install $requirement; end < requirements.txt
```

Download datasets:
```bash
cd data
chmod 777 ./download_data.sh
./download_data.sh
```



### Running

Here is an example code reproducing the results of DenseNet-BC trained on CIFAR-10 where TinyImageNet (crop) is the out-of-distribution dataset. The temperature is set as 1000, and perturbation magnitude is set as 0.0014. In the **root** directory, run

```
cd src
python main.py --nn densenet10 --dataset Uniform --magnitude 0.0014 --temperature 1000
```

or using tmux
```
cd bash_scripts
tmuxp load produce_score.yaml
```

### Analyze
```
cd notebooks
jupyter lab
```


### License
Please refer to the [LICENSE](https://github.com/facebookresearch/odin/blob/master/LICENSE).

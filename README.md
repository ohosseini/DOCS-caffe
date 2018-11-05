# Deep Object Co-Segmentation (DOCS) - Caffe

This is the [caffe](https://github.com/BVLC/caffe) implementation of our paper **Deep Object Co-Segmentation** published at ACCV18. For more information, you can visit the [project page](https://ohosseini.github.io/projects/DOCS/).

![DOCS Network](_assets/network.png)

## Requirements

- Python 2.7
- OpenCV 2.4: If you are using anaconda you can install it by running

  ```console
  conda install -c conda-forge opencv=2.4
  ```

## Compiling Caffe

1. Follow the installation instructions in [caffe official website](http://caffe.berkeleyvision.org/installation.html) for installing the dependencies.
2. Make a ```Makefile.config``` file using the ```Makefile.config.example```.
3. Set ```WITH_PYTHON_LAYER := 1``` in ```Makefile.config```.
4. Then compile it by running

    ```console
    make all
    make pycaffe
    ```

## Demo

First download the model from [here](https://drive.google.com/file/d/1dXlHNW4Zf7JGLXOIowJXUI1IC3Cxth0T/view) and put it in the ```models/DOCS/```.

Then you can run the demo with

```console
sh demo.sh
```

![demo](_assets/demo.png)

For more information on how to apply the demo on other images you can check

```console
python models/DOCS/demo.py --help
```

## Training

You can train your own model by running

```console
cd model/DOCS
python train.py 0 --steps 100000 --weights VGG_ILSVRC_16_layers.caffemodel
```

You can download the initial weights ```VGG_ILSVRC_16_layers.caffemodel``` from [here](http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), or start the training from scratch by running 

```console
python train.py 0 --steps 100000
```

For more information about training you can check

```console
python train.py --help
```

A small training set is provided in ```data/train_examples/``` as an example.

## Citation

If you use this code, please cite our publication:

**Deep Object Co-Segmentation**, Weihao Li*, Omid Hosseini Jafari*, Carsten Rother, *ACCV 2018*.

```
@InProceedings{DOCS_ACCV18,
  title={Deep Object Co-Segmentation},
  author={Li, Weihao and Hosseini Jafari, Omid and Rother, Carsten},
  booktitle={ACCV},
  year={2018}
}
```
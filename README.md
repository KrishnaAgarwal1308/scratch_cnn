# scratch cnn
This is an from scratch implementation of cnn.

```cnn_manual.py ```contains scratch implementation of a cnn layer 
and ```cnn_part2.py``` contains scratch implementation of cnn net.

I don't recommend running both since it would take eternity to run since they are not optimized. But are good for learning purpose.

You may run ```cnn_using_modules.py``` it shows how to train a cnn network and how to save it. The model is trained on cifar-10 dataset.

## How to run this code 
```
git clone https://github.com/KrishnaAgarwal1308/scratch_cnn.git
cd scratch_cnn
pip install torch, torchvision, numpy
python cnn_using_modules.py
```

# scratch cnn
This is an from scratch implementation of cnn.

```cnn_manual.py ```contains scratch implementation of a cnn layer 
and ```cnn_part2.py``` contains scratch implementation of cnn net.

I don't recommend running both since it would take eternity to run since they are not optimized. But are good for learning purpose.

You may run ```cnn_using_modules.py``` it shows how to train a cnn network and how to save it. The model is trained on cifar-10 dataset.

you can now run alex-net and ViT from this repo and look at it's implemenation from scratch. Both of which could be selected through ```run_models.py``` and could be run individually.

## How to run this code 
```
git clone https://github.com/KrishnaAgarwal1308/scratch_cnn.git
cd scratch_cnn
pip install torch, torchvision, numpy
python cnn_using_modules.py
```

## To run visual transformers
```
python run_models.py
```
Note: while running ViT make sure to shrink down the size of the image to something like 64 or 128 or else it won't fit in most commercial gpu's vram.


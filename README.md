PCNN-ATT Model for Relation Extraction
=========================

This repo contains the *pytorch* code for paper [Neural Relation Extraction with Selective Attention over Instances.](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/acl2016_nre.pdf).

## Requirements

- Python 2 (tested on 2.7)
- PyTorch (tested on 0.4.1)


## Dataset
Use data from [Lin et. al. (2016)](https://github.com/thunlp/NRE/blob/master/data.zip), and put all training, test, vocab and relation files under the same directory.


## Training

Train a PCNN-ATT model with:
```
python train.py --data_dir data/ --lr 0.001 --num_epoch 15 --save_dir saved_models/
```

Model checkpoints and logs will be saved to `./saved_models/`.

## Evaluation

Run held-out evaluation on the test set with:
```
python eval.py --model_dir saved_models/ --model best_model.tar --data_dir data/
```

Use `--model checkpoint_epoch_10.tar` to specify a model checkpoint file of 10th training epoch. Add `--out saved_models/pr.dump` to write model precision/recall output to a file.

## Reference
[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.
[Original C++ code for PCNN-ATT](https://github.com/thunlp/NRE)























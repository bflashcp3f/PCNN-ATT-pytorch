PCNN-ATT Model for Relation Extraction
=========================

This repo contains the *pytorch* code for paper [Neural Relation Extraction with Selective Attention over Instances.](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/acl2016_nre.pdf).

## Requirements

- Python 2 (tested on 2.7)
- PyTorch (tested on 0.4.1)


Train an PCNN-ATT model with:
```
python train.py --data_dir data/ --vocab_dir data/ --rel_dir data/ --lr 0.001 --num_epoch 15 --save_dir saved_models
```

Model checkpoints and logs will be saved to `./saved_models/`.

## Evaluation

Run held-out evaluation on the test set with:
```
python eval.py saved_models/ --data_dir data/
```

This will use the `best_model.tar` by default. Use `--model checkpoint_epoch_10.tar` to specify a model checkpoint file of 10th training epoch. Add `--out saved_models/pr.dump` to write model precision/recall output to a file.

## Reference
[Original C++ code for PCNN-ATT](https://github.com/thunlp/NRE)























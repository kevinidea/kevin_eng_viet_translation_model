#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda --batch-size=64

elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es outputs/test_outputs.txt --cuda

elif [ "$1" = "train_local" ]; then
  	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --batch-size=2

elif [ "$1" = "test_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    python run.py decode model.bin ./en_es_data/grader.es outputs/test_outputs.txt

elif [ "$1" = "train_local_q1" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder
elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q1.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt \
        --no-char-decoder
elif [ "$1" = "train_local_q2" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100
elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2.txt 
elif [ "$1" = "vocab" ]; then
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --size=200 --freq-cutoff=1 vocab_tiny_q1.json
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        vocab_tiny_q2.json
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
elif [ "$1" = "vocab_en_vi" ]; then
    python vocab.py --train-src=./en_vi_data/train.en --train-tgt=./en_vi_data/train.vi \
        --size=80000 --freq-cutoff=1 vocab_en_vi.json
elif [ "$1" = "train_en_vi" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py train \
        --train-src=./en_vi_data/train.en \
        --train-tgt=./en_vi_data/train.vi \
        --dev-src=./en_vi_data/dev.en \
        --dev-tgt=./en_vi_data/dev.vi \
        --vocab=vocab_en_vi.json \
        --batch-size=8 \
        --save-to=model_en_vi.bin --cuda
elif [ "$1" = "test_en_vi" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_en_vi.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode \
        model_en_vi.bin \
        ./en_vi_data/test.en \
        ./en_vi_data/test.vi \
        outputs/test_outputs_en_vi.txt --cuda
elif [ "$1" = "train_en_vi_local" ]; then
    python run.py train \
        --train-src=./en_vi_data/train.en \
        --train-tgt=./en_vi_data/train.vi \
        --dev-src=./en_vi_data/dev.en \
        --dev-tgt=./en_vi_data/dev.vi \
        --vocab=vocab_en_vi.json \
        --batch-size=8 \
        --save-to=model_en_vi.bin
elif [ "$1" = "test_en_vi_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_en_vi.txt
    python run.py decode \
        model_en_vi.bin \
        ./en_vi_data/test.en \
        ./en_vi_data/test.vi \
        outputs/test_outputs_en_vi.txt
elif [ "$1" = "retrain_en_vi_local" ]; then
    python run.py train \
        --train-src=./en_vi_data/train.en \
        --train-tgt=./en_vi_data/train.vi \
        --dev-src=./en_vi_data/dev.en \
        --dev-tgt=./en_vi_data/dev.vi \
        --vocab=vocab_en_vi.json \
        --batch-size=8 \
        --save-to=model_en_vi.bin \
        --retrain \
        --retrain-model=model_en_vi_1.bin \
        --retrain-optimizer=model_en_vi_1.bin.optim
elif [ "$1" = "retrain_en_vi" ]; then
    python run.py train \
        --train-src=./en_vi_data/train.en \
        --train-tgt=./en_vi_data/train.vi \
        --dev-src=./en_vi_data/dev.en \
        --dev-tgt=./en_vi_data/dev.vi \
        --vocab=vocab_en_vi.json \
        --batch-size=16 \
        --save-to=model_en_vi.bin \
        --retrain \
        --retrain-model=model_en_vi.bin \
        --retrain-optimizer=model_en_vi.bin.optim \
        --cuda \
        --max-num-trial=100 \
        --max-epoch=50
else
	echo "Invalid Option Selected"
fi

# SoundStream: An End-to-End Neural Audio Codec

Implementation of the [SoundStream](https://arxiv.org/abs/2107.03312) neural audio codec
trained on [LibriSpeech](https://www.openslr.org/12) `train-clean-100` partition.

Built on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
with Hydra configuration, Comet ML logging,
and automatic mixed precision.

## Setup

```bash
git clone https://github.com/Alekseyka20x/SoundStream
cd SoundStream
pip install -r requirements.txt
pre-commit install   # optional
```

## Training

The training can be performed after setup by running
```bash
export COMET_API_KEY=<your key> && python train.py
```

Without any additional arguments, program will use `src/configs/train.yaml` default config and download `train-clean-100` and `test-clean` partitions from [LibriSpeech](https://www.openslr.org/12) automatically. This behavior can be changed from command line (see [Hydra](https://hydra.cc/docs/intro/)).


### Training on Kaggle (example)

For training on Kaggle we use [kaggle version](https://www.kaggle.com/datasets/a24998667/librispeech) of Librispeech dataset as input and this modifiers:
- `datasets.validation.save_index=False` &mdash; do not save dataset index
- `datasets.train.save_index=False` &mdash; same for train partition
- `datasets.validation.dataset_dir=<path>` &mdash; set path to validation dataset
- `datasets.train.dataset_dir=<path>` &mdash; same for train dataset.

### Some key training parameters

| Parameter | Value |
|---|---|
| Steps | 45 000 |
| Batch size | 12 |
| Crop length | 0.5 s |
| Sample rate | 16 kHz |
| Learning rate | 1e-4 |
| Optimizer | Adam ($\beta_1=0.5, \beta_2=0.9$) |

### Using pretrained checkpoint

Our best checkpoint is saved on [google drive](https://drive.google.com/file/d/1xTPJxA5CwqCU2Ei9J4quxIFPqib3pUzz/view?usp=sharing) and can be downloaded with `gdown`:
```python
import gdown

gdown.download(id="1xTPJxA5CwqCU2Ei9J4quxIFPqib3pUzz", output="checkpoint_best.pth")
```

Training can be continued by running
```bash
python train.py trainer.resume_from=<path to checkpoint>
```

## Inference

Checkpoint from training contains useless for inference data: discriminator weights, parameters of optimizer/scheduler/grad scaler and hydra configs. So before using this checkpoint we convert it to more compact format, which stores only generator and its config. This transform can be performed using
```bash
python3 src/utils/convert.py
```
and then entering in the terminal:
- path to the checkpoint
- path to where to store converted checkpoint for inference

A converted version of our best checkpoint is availible [here](https://drive.google.com/file/d/1VdAG_X5E2tTEqI7jjmpvM_p7iZas7WET/view?usp=drive_link) and can be downloaded using `gdown` in the same way as full checkpoint (see [demo](demo.ipynb) for an example of inference)

### Using pretrained model
After obtaining checkpoint for inference, model can be simply loaded by running
```python
from src.models.generator import SoundStream

model = SoundStream.from_pretrained(path)
```

Then you can use it for encoding and decoding audio
```python
# encoding audio
encoded_audio = model.encode(audio, sample_rate=sample_rate)

# decoding audio
decoded_audio = model.decode(encoded_audio, sample_rate=sample_rate)
```
where `audio` is a torch tensor of shape `(1, T)`.

Also it is possible to evaluate on a dataset for calculating metrics by running:

```bash
python3 inference.py inferencer.from_pretrained=model_best.pth
```
where `model_best.pth` &mdash; result of converting a checkpoint (compressed version)

## Results

Evaluated on the full LibriSpeech `test-clean` partition:

| Metric | Value |
|---|---|
| STOI | 0.81 |
| NISQA | 2.17 |

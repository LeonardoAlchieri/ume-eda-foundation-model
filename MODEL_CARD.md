# Model card: UME (base)

## Model details

- **What it is:** an encoder that turns 60-second windows of wrist electrodermal activity (EDA) into 64-dimensional embeddings.
- **Architecture:** EfficientNet-style 1D CNN (stem convolution, 16 MBConv1D blocks with squeeze-and-excitation, head convolution, global average pooling, linear projection). See `edamame/model/efficientnet_contrastive.py`.
- **Training objective:** sample-wise contrastive learning (InfoNCE, temperature 0.1) between two augmented views of the same window, using the EDA augmentations of [Matton et al. (2024)](https://arxiv.org/abs/2404.08783).
- **Files:** `artifacts/ume_weights.pt` (weights and config) and `artifacts/ume.ckpt` (full training checkpoint). Both hold the same weights.
- **Developed by:** Leonardo Alchieri, Università della Svizzera italiana (USI).

## Input

- Shape `(batch, 240, 3)`: 240 samples = 60 s at 4 Hz.
- Channels, in order: mixed EDA, phasic component, tonic component.
- Signals at other sampling rates or window lengths should be resampled and cut to match.

## Intended use

- Extracting features from wearable EDA for research on downstream tasks such as stress, cognitive load, or sleep/wake classification.
- Fine-tuning on a labelled research dataset with `EdamamePipeline.fit`.

## Out-of-scope use

- Clinical diagnosis, or any medical or safety-critical decision about an individual.
- Inferring sensitive traits or emotional states of people without their informed consent, including workplace or surveillance monitoring.
- Signals that are not wrist EDA (e.g. PPG, ECG) or that are sampled very differently, without first validating performance.

## Training data

Pretrained without labels on 17 public and research EDA datasets from the EDAMAME collection (listed in the README, section "Pretraining data"). The data is not distributed with the model. Recordings come mostly from research studies with wrist-worn devices such as the Empatica E4, which shapes the populations, devices and settings the model has seen.

## Evaluation

Downstream evaluation (frozen features and a classical classifier, validated leave-one-person-out) is run by `train.py` through the optional `edamame_downstream` package. Results will be reported in the accompanying paper.

## Limitations and risks

- **Population and device bias.** Performance can drop on populations, devices, body locations or recording conditions that are underrepresented in the pretraining data.
- **Artifacts.** EDA is sensitive to motion, temperature and sensor contact. The model does not detect or flag poor-quality signal.
- **Individual differences.** EDA varies a lot between people. Validate with person-independent splits (e.g. leave-one-person-out) rather than random splits, which overestimate performance.
- **Privacy.** Physiological signals are personal data. Handle any data you use with the model according to its consent terms and applicable law.

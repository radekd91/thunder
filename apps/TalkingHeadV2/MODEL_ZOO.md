# THUNDER Model Zoo

This file records model bundles, auxiliary assets, and configured download targets for the V1 / unversioned and TFHP / 3DV demos. Download availability has not yet been verified for this release. The paths below describe the expected layout in the INFERNO checkout.

## Target Layout

Talking-head diffusion models:

```text
<inferno>/assets/DiffusionTalkingHead/models/<model_name>/
```

Mesh-to-speech and silent-video-to-speech models:

```text
<inferno>/assets/MeshToSpeech/models/<model_name>/
```

Shared THUNDER assets:

```text
<inferno>/assets/FLAME/geometry/
<inferno>/assets/HuBERT/
<inferno>/assets/ImageBind/
<inferno>/assets/Lip2SpeechUnit/lip2speech/
<inferno>/assets/TFHP/processed/
<inferno>/data/THUNDER_test_example_data/
```

External repositories and their model files:

```text
<inferno>/external/DiffPoseTalk/
<inferno>/external/ImageBind/
```

## Paper Models

The paper-relevant demo set is the V1 / unversioned THUNDER models, the TFHP / 3DV models, and the mesh-to-speech or silent-video-to-speech support models referenced by them.

### Version Families

The initial release includes these model families:

```text
V1 / unversioned / ICCV 2025
  THUNDER-F_m2s-0
  THUNDER-F_m2s-1
  THUNDER-T_m2s-0
  THUNDER-T_m2s-1
  M2F-THUNDER-F_m2s-0
  M2F-THUNDER-F_m2s-1
  M2F-THUNDER-T_m2s-0
  M2F-THUNDER-T_m2s-1

TFHP / 3DV 2026
  TFHP-THUNDER-*
  TFHP-pose-THUNDER-*
  TFHP-pose-style-THUNDER-*
  TFHP-pose-style-THUNDER-*_perc_1
```

### V1 / Unversioned Models

These models use the unversioned THUNDER naming:

```text
THUNDER-F_m2s-1
THUNDER-T_m2s-1
M2F-THUNDER-F_m2s-1
M2F-THUNDER-T_m2s-1
```

The `m2s-1` variants need:

```text
assets/MeshToSpeech/models/MouthToSpeech/
```

### TFHP / 3DV Models

These models use the TFHP naming:

```text
TFHP-THUNDER-*
TFHP-pose-THUNDER-*
TFHP-pose-style-THUNDER-*
```

The TFHP models need:

```text
assets/MeshToSpeech/models/TFHP-MouthToSpeech/
assets/TFHP/processed/stats_train.npz
```

The styled TFHP models also need the original DiffPoseTalk style encoder. This is a third-party/checkpoint dependency from DiffPoseTalk, not part of the THUNDER talking-head model bundle:

```text
external/DiffPoseTalk/experiments/1-with_head/SE/head-L4H4-T0.1-BS32/checkpoints/iter_0026000.pt
```

Without that file, models such as `TFHP-pose-style-THUNDER-F_m2s-1` cannot be constructed because their config instantiates `OrigDiffPoseTalkStylePreprocessor`.

### Mesh-To-Speech And Choi Et Al. Models

These support models should be distributed as separate model-zoo entries because several talking-head models reference them, and the Choi et al. models are a different silent-video-to-speech path:

```text
MouthToSpeech
TFHP-MouthToSpeech
ChoiEtAlOrig
ChoiEtAlFinetuned
```

They need the shared assets listed below.

## Shared Assets

Depending on the chosen model, the demo may require the following assets:

```text
assets/FLAME/geometry/generic_model.pkl
assets/FLAME/geometry/landmark_embedding.npy
assets/FLAME/geometry/mouth_selection_3.txt
assets/HuBERT/km.bin
assets/ImageBind/imagebind_huge.pth
assets/Lip2SpeechUnit/lip2speech/frontend.pt
assets/Lip2SpeechUnit/lip2speech/conformer.pt
assets/Lip2SpeechUnit/lip2speech/unit_decoder.pt
assets/Lip2SpeechUnit/lip2speech/mel_decoder.pt
assets/TFHP/processed/stats_train.npz
data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_audio.wav
data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_embeddings.npy
data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_rec_CVT-EMICA2020_rec.hdf5
```

Third-party assets have their own license terms; consult their original projects before use or redistribution.

## Download Scripts

The download scripts use these URL patterns by default:

```text
https://download.is.tue.mpg.de/thunder/models/talking_head/<model_name>.zip
https://download.is.tue.mpg.de/thunder/models/mesh_to_speech/<model_name>.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/<asset_bundle>.zip
https://download.is.tue.mpg.de/thunder/assets/mesh_to_speech/<asset_bundle>.zip
```

Use:

```text
demos/download_thunder_assets.sh
demos/download_thunder_v1_assets.sh
demos/download_thunder_talking_head_models.sh
../Mesh2Speech/demos/download_thunder_mesh_to_speech_models.sh
../Mesh2Speech/demos/download_thunder_v1_mesh_to_speech_models.sh
```

The scripts reuse existing model directories or valid downloaded zip files and extract without overwriting existing files. `download_thunder_assets.sh` and `download_thunder_v1_assets.sh` select the baseline assets. The download helpers accept `INFERNO_ROOT` for the checkout location and `THUNDER_BASE_URL` for the download base URL.

## Configured Talking-Head URLs

The default talking-head download script fetches the V1 THUNDER/M2F models and the TFHP family:

```text
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-F_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-F_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-0.1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-0.5.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-5.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/THUNDER-T_m2s-10.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-0.1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-0.5.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-5.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-F_m2s-10.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-T_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/M2F-THUNDER-T_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-THUNDER-F_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-THUNDER-F_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-THUNDER-T_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-THUNDER-T_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-THUNDER-F_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-THUNDER-F_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-THUNDER-T_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-THUNDER-T_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-F_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-F_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-T_m2s-0.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-T_m2s-1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-F_m2s-0_perc_1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-F_m2s-1_perc_1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-T_m2s-0_perc_1.zip
https://download.is.tue.mpg.de/thunder/models/talking_head/TFHP-pose-style-THUNDER-T_m2s-1_perc_1.zip
```

## Configured Asset URLs

```text
https://download.is.tue.mpg.de/thunder/assets/talking_head/FLAME.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/HuBERT.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/TFHP.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/ReferenceSpeakerEmbeddings.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/DiffusionReferenceImages.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/THUNDER_test_example_data.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/ImageBind.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/DiffPoseTalkStyleEncoders.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/Perceptual3DTalkingHead.zip
https://download.is.tue.mpg.de/thunder/assets/mesh_to_speech/Lip2SpeechUnit.zip
```

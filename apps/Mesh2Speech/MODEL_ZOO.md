# THUNDER Mesh-To-Speech Model Zoo

This file tracks mesh-to-speech and silent-video-to-speech model bundles used by the initial THUNDER demos. Download targets are taken from the existing scripts; availability has not yet been verified for this release.

## Target Layout

```text
<inferno>/assets/MeshToSpeech/models/<model_name>/
```

Shared assets are expected under:

```text
<inferno>/assets/FLAME/geometry/
<inferno>/assets/HuBERT/
<inferno>/assets/Lip2SpeechUnit/
<inferno>/data/THUNDER_test_example_data/
```

## Default Models

The default download script fetches:

```text
ChoiEtAlOrig
ChoiEtAlFinetuned
MouthToSpeech
TFHP-MouthToSpeech
```

Configured model URLs:

```text
https://download.is.tue.mpg.de/thunder/models/mesh_to_speech/ChoiEtAlOrig.zip
https://download.is.tue.mpg.de/thunder/models/mesh_to_speech/ChoiEtAlFinetuned.zip
https://download.is.tue.mpg.de/thunder/models/mesh_to_speech/MouthToSpeech.zip
https://download.is.tue.mpg.de/thunder/models/mesh_to_speech/TFHP-MouthToSpeech.zip
```

## Configured Asset URLs

```text
https://download.is.tue.mpg.de/thunder/assets/talking_head/FLAME.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/HuBERT.zip
https://download.is.tue.mpg.de/thunder/assets/talking_head/THUNDER_test_example_data.zip
https://download.is.tue.mpg.de/thunder/assets/mesh_to_speech/Lip2SpeechUnit.zip
```

## Download Scripts

- [Default mesh-to-speech models](demos/download_thunder_mesh_to_speech_models.sh)
- [V1 mesh-to-speech models](demos/download_thunder_v1_mesh_to_speech_models.sh)
- [Shared THUNDER assets](../TalkingHeadV2/demos/download_thunder_assets.sh)

The helpers accept `INFERNO_ROOT` for the checkout location and `THUNDER_BASE_URL` for the download base URL. Third-party assets and model bundles retain their separate license requirements.

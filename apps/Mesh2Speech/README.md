# Mesh-To-Speech

This INFERNO app generates speech from reconstructed facial motion. Its architecture is inspired by [lip2speech-unit](https://github.com/choijeongsoo/lip2speech-unit/) and borrows code from that project.

The initial release covers inference and demos. Training workflows are planned for a later release. Model bundles, including the silent-video-to-speech support models, are listed in the [model zoo](MODEL_ZOO.md).

## Setup and assets

See the repository [setup instructions](../../README.md#setup) for environment prerequisites. The [Docker guide](../../docker/inference/README.md#choose-an-image) supports either downloading and loading the prebuilt image or building it locally. Models and assets are downloaded separately.

The [model download script](demos/download_thunder_mesh_to_speech_models.sh) selects the model bundles documented in the model zoo. Shared assets are handled by the [THUNDER asset script](../TalkingHeadV2/demos/download_thunder_assets.sh). Asset and model license terms apply separately from the source code.

## Demo

After preparing the environment, models, and assets, run the [demo script](demos/mesh_to_speech_demo.py) from this app directory:

```bash
python demos/mesh_to_speech_demo.py --model_name MouthToSpeech
```

The defaults select an example reconstruction, reference audio, and speaker embedding from `data/THUNDER_test_example_data` in the INFERNO checkout. To use your own compatible inputs:

```bash
python demos/mesh_to_speech_demo.py --model_name MouthToSpeech --reconstruction_path <reconstruction.hdf5> --audio_path <reference_audio.wav> --speaker_embedding_path <speaker_embeddings.npy>
```

See [demo arguments](demos/mesh_to_speech_demo_args.py) for options. The main architecture is implemented in [MeshToSpeech.py](../../models/MeshToSpeech.py).

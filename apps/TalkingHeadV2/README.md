# Diffusion-based 3D Speech-Driven Avatars

This app adapts the INFERNO TalkingHead project for THUNDER. The initial release covers V1 / unversioned and TFHP / 3DV inference and demos. Training workflows are planned for a later release.

## Setup and assets

See the repository [setup instructions](../../README.md#setup) for environment prerequisites. The [Docker guide](../../docker/inference/README.md#choose-an-image) supports either downloading and loading the prebuilt image or building it locally. Models and assets are downloaded separately.

The [model zoo](MODEL_ZOO.md) lists the required model bundles, shared assets, and expected paths. The existing download entry points are:

- [Shared assets](demos/download_thunder_assets.sh)
- [Talking-head models](demos/download_thunder_talking_head_models.sh)
- [Mesh-to-speech support models](../Mesh2Speech/demos/download_thunder_mesh_to_speech_models.sh)

Asset and model license terms apply separately from the source code.

Styled TFHP models, including `TFHP-pose-style-THUNDER-F_m2s-1`, also require the original third-party DiffPoseTalk style encoder checkpoint described in the model zoo.

## Demo

Once the environment, models, and assets are available, run the [demo script](demos/thunder_demo.py) from this app directory:

```bash
python demos/thunder_demo.py --model_name THUNDER-F_m2s-1 --audio_path <path_to_your_audio> --num_variations 6
```

The classifier-free guidance options control the balance between output diversity and audio conditioning:

```bash
python demos/thunder_demo.py --model_name THUNDER-F_m2s-1 --audio_path <path_to_your_audio> --num_variations 6 --cf_guidance_weight_uncond 0.2 --cf_guidance_weight_audio 1.0
```

See [demo arguments](demos/thunder_demo_args.py) for the available options.

## Citation 

If you use this work in your publication, please cite the following:
```
@misc{danecek2025supervising3dtalkinghead,
      title={Supervising 3D Talking Head Avatars with Analysis-by-Audio-Synthesis}, 
      author={Radek Daněček and Carolin Schmitt and Senya Polikovsky and Michael J. Black},
      year={2025},
      eprint={2504.13386},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2504.13386}, 
}
```

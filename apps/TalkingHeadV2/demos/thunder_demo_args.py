import argparse
from inferno.utils.other import get_path_to_assets 

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Thunder Demo Arguments")

    parser.add_argument(
        "--model_folder",
        type=str,
        default=str(get_path_to_assets() / "DiffusionTalkingHead/models"),
        # required=True,
        help="Path to the folder containing the model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="THUNDER-F_m2s-1",
        # default="THUNDER-T_m2s-1",
        # default="M2F-THUNDER-F_m2s-1", 
        # default="TFHP-pose-style-THUNDER-F_m2s-0",
        default="TFHP-pose-style-THUNDER-F_m2s-1",
        # default="TFHP-pose-style-THUNDER-F_m2s-0_perc_1"
        # default="TFHP-pose-style-THUNDER-F_m2s-1_perc_1",
        # default="TFHP-pose-style-THUNDER-T_m2s-0",
        # default="TFHP-pose-style-THUNDER-T_m2s-1",
        help="Name of the model to use."
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=6,
        help="Number of generated variations."
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        default=str(get_path_to_assets().parent / "data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_audio.wav"),
        help="Path to the audio file to be used for generation."
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length."
    )

    parser.add_argument(
        "--cf_guidance_weight_audio",
        type=float,
        default=1.0,
        help="Classifier-free guidance weight for audio. Increase to trade lip sync for diversity."
    )
    parser.add_argument(
        "--cf_guidance_weight_uncond",
        type=float,
        default=0.0,
        help="Classifier-free guidance weight for unconditioned case. Increase to trade lip sync for diversity."
    )

    parser.add_argument(
        "--clip_image_conditions",
        type=str,
        default=None,
        help="Comma-separated list of image conditions to use. Only possible with models that have been trained with image conditions."
    )

    parser.add_argument(
        "--cf_guidance_weight_clip",
        type=float,
        default=None,
        help="Classifier-free guidance weight for unconditioned case. Increase to trade lip sync for diversity."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--save_meshes",
        action="store_true",
        help="If set, saves the generated meshes to disk."
    )

    parser.add_argument(
        "--dont_save_videos",
        action="store_true",
        help="If set, does not save the generated videos."
    )

    parser.add_argument(
        "--save_predicted_flame", 
        action="store_true",
        help="If set, saves the predicted FLAME parameters."
    )

    parser.add_argument(
        "--diff_steps_to_visualize",
        type=str,
        default="final",
        help="Comma-separated list of diffusion steps to visualize or 'final'. If the model has 1000 denoising steps, than the final inference is 999. The first denoising step is 0 and so on."
    )

    parser.add_argument(
        "--output_path", 
        type=str,
        default="videos", 
        help="Path to the folder where the output videos and meshes will be saved. If relative, it is relative to the folder of the loaded model."
    )

    # Add more arguments as needed in the future
    return parser


def get_args():
    parser = get_arg_parser()
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

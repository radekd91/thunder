import argparse
from inferno.utils.other import get_path_to_assets 

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Thunder Demo Arguments")

    parser.add_argument(
        "--model_folder",
        type=str,
        default=str(get_path_to_assets() / "MeshToSpeech/models"),
        help="Path to the folder containing the model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MouthToSpeech",
        # default="FaceToSpeech",
        # default="FlameToSpeech",
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
        help="Path to the GT audio file."
    )

    parser.add_argument(
        "--speaker_embedding_path",
        type=str,
        default=str(get_path_to_assets().parent / "data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_embeddings.npy"),
        help="Path to the reconstruction file to be used for generation."
    )

    parser.add_argument(
        "--reconstruction_path",
        type=str,
        default=str(get_path_to_assets().parent / "data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_rec_CVT-EMICA2020_rec.hdf5"),
        help="Path to the reconstruction file to be used for generation."
    )

    # parser.add_argument(
    #     "--video_path",
    #     type=str,
    #     default=str(get_path_to_assets().parent / "data/THUNDER_test_example_data/ravdess/Actor_24_01-01-01-01-01-01-24_audio.wav"),
    #     help="Path to the GT video file to be used for generation."
    # )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Optional maximum number of frames at 25 fps; by default, process the entire input audio."
    )

    parser.add_argument(
        "--save_meshes",
        action="store_true",
        help="If set, saves the generated meshes to disk."
    )

    parser.add_argument(
        "--output_dir",
        type =str,
        default="test_results",
        help="Path to the folder where the output will be saved. If relative, it will be relative to the model folder."
    )

    # Add more arguments as needed in the future
    return parser


def get_args():
    parser = get_arg_parser()
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

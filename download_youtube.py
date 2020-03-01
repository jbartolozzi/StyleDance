import argparse
import os
import subprocess
from pytube import Playlist, YouTube


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--playlist", help="YouTube playlist url.", type=str)
    parser.add_argument("-o", "--output_directory", help="Path to output directory.", type=str, required=True)
    parser.add_argument("-gpu", "--gpu", type=int, help="Gpu Number.", default=2)
    parser.add_argument("-co", "--convert_only",
                        help="Run conversion on downloaded WEBM", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    file_counter = 0

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    if not args.convert_only:
        playlist = Playlist(args.playlist)
        videos_to_convert = []
        for video in playlist:
            filename = "video_%s" % file_counter
            file_counter += 1
            v = YouTube(video).streams.order_by('resolution')[-1]
            downloaded_video = v.download(args.output_directory, filename=filename)
            rel_path = downloaded_video.replace(os.getcwd() + "/", "")
            videos_to_convert.append(rel_path)

    else:
        videos_to_convert = list(os.path.join(args.output_directory, file)
                                 for file in os.listdir(args.output_directory) if "webm" in file)

    for rel_path in videos_to_convert:
        convert_command = "docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%s --volume %s:/workspace willprice/nvidia-ffmpeg -y -hwaccel cuvid -i %s -vcodec h264_nvenc -preset slow %s" % (
            args.gpu, os.getcwd(), rel_path, rel_path.replace("webm", "mp4"))
        print(subprocess.check_call(convert_command.split()))


if __name__ == "__main__":
    main()

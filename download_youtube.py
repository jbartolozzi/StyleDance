import subprocess
import os
from pytube import Playlist, YouTube
playlist = Playlist(
    "https://www.youtube.com/playlist?list=PL1P8jsym3NDE13a2rSXu5-vytv7bGm5Ik")

file_counter = 0
root_downloads = "downloads"
output_directory = "dance_videos"

download_directory = os.path.join(root_downloads, output_directory)

if not os.path.exists(download_directory):
    os.mkdir(download_directory)

for video in playlist:
    filename = "video_%s" % file_counter
    file_counter += 1
    v = YouTube(video).streams.order_by('resolution')[-1]
    downloaded_video = v.download(download_directory, filename=filename)
    rel_path = downloaded_video.replace(os.getcwd()+"/", "")
    convert_command = "docker run --rm -it --runtime=nvidia --volume %s:/workspace willprice/nvidia-ffmpeg -y -hwaccel cuvid -i %s -vcodec h264_nvenc %s" % (
        os.getcwd(), rel_path, rel_path.replace("webm", "mp4"))
    print(subprocess.check_call(convert_command.split()))

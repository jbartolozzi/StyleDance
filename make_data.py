import os
import cv2
import tqdm
import argparse
import multiprocessing as mp
import numpy as np
from detectron2.config import get_cfg
import predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs='+', help="Path(s) to input files.")
    parser.add_argument("output_directory", help="Path to output directory.")
    parser.add_argument("-gpu", "--gpu", type=int,
                        help="Gpu Number.", default=2)

    return parser.parse_args()


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()

    config_file = "./detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"
    cfg.merge_from_file(config_file)

    opts = ["MODEL.WEIGHTS",
            "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"]
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    confidence_threshold = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def main():
    args = parse_args()
    cfg = setup_cfg()

    mp.set_start_method("spawn", force=True)

    inputs = args.inputs
    output_directory = args.output_directory

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            os.mkdir(os.path.join(output_directory, "images"))
        except:
            print("Unable to create output directory %s" % output_directory)
            return

    inferencer = predictor.VisualizationDemo(cfg, parallel=True)
    item_counter = 0
    for input in inputs:
        # Handle Video
        video_file = cv2.VideoWriter(
            filename=os.path.join(output_directory, "video.mkv"),
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(24),
            frameSize=(512, 512),
            isColor=True,
        )

        if input.endswith("mp4"):
            video = cv2.VideoCapture(input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # basename = os.path.basename(input)

            for vis_frame in tqdm.tqdm(inferencer.run_on_video(video, width, height, 512), total=num_frames):
                if vis_frame is not None:
                    vis_frame.save(
                        os.path.join(output_directory, "images", "%s.png" % item_counter))
                    video_file.write(np.asarray(vis_frame)[:, :, ::-1])
                    item_counter += 1

        video_file.release()
        # # Handle single images
        # elif input.endswith("jpg") or input.endswith("png"):
        #     pass
        # else:
        #     print("Unhandled file type %s." % input)
        #     continue


if __name__ == "__main__":
    main()

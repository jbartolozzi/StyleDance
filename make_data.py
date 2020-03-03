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
    parser.add_argument("inputs", help="Path(s) to input files.")
    parser.add_argument("output_directory", help="Path to output directory.")
    parser.add_argument("-resize", type=int, default=512, help="Resolution to resize to.")
    parser.add_argument("-padding", type=int, default=16, help="Image padding around bbox.")
    parser.add_argument("-gpu", "--gpu", type=int,
                        help="Gpu Number.", default=2)

    return parser.parse_args()


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()

    config_file = "./detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"
    cfg.merge_from_file(config_file)

    # opts = ["MODEL.WEIGHTS",
    #         "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"]
    
    opts = ["MODEL.WEIGHTS", "detectron_model.pkl"]
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

    resolution = args.resize

    video_file = cv2.VideoWriter(
        filename=os.path.join(output_directory, "video.mkv"),
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*"x264"),
        fps=float(60),
        frameSize=(resolution, resolution),
        isColor=True,
    )

    inputs = []
    inputs = list(os.path.join(args.inputs, file) for file in os.listdir(
        args.inputs) if os.path.exists(os.path.join(args.inputs, file) and "mp4" in os.path.join(args.inputs, file)))

    for input_video in inputs:
        print("Processing:", input_video)
        video = cv2.VideoCapture(input_video)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for vis_frame in tqdm.tqdm(inferencer.run_on_video(video, width, height, resolution, args.padding), total=num_frames):
            if vis_frame is not None:
                vis_frame.save(
                    os.path.join(output_directory, "images", "%s.png" % item_counter))
                video_file.write(np.asarray(vis_frame)[:, :, ::-1])
                item_counter += 1

    video_file.release()


if __name__ == "__main__":
    main()

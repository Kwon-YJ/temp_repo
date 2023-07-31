import os

import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default=None, help='Path to output directory')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'], 
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
        # '--score-thr', type=float, default=0.001, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    print(model.CLASSES)

    # get all image files in the directory
    img_files = []
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img_files.append(os.path.join(root, file))

    # inference and save the results for each image
    for img_file in img_files:
        result = inference_detector(model, img_file)

        save_txt = ''

        for i, arr in enumerate(result):
            temp = list(arr)
            if len(temp) != 0:
                temp = list(temp[0])
                temp.append(i)
                save_txt += f"{temp[-1]} {round(temp[-2],2)} {int(temp[0])} {int(temp[1])} {int(temp[2])} {int(temp[3])}\n"
        # [[99.31321, 75.46702, 232.6153, 139.26889, 0.916291, 1]]


        file_name = img_file.split("/")[-1]
        # file_name = file_name.split(".")[0]
        file_name = file_name[:-4]
        f = open(f'result/{file_name}.txt', 'w')
        f.write(save_txt)
        f.close()

        # save the result with the same filename in the output directory
        if args.out_dir is not None:
            basename = os.path.basename(img_file)
            out_file = os.path.join(args.out_dir, os.path.splitext(basename)[0] + '.png')
        else:
            out_file = None

        
        show_result_pyplot(
            model,
            img_file,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=out_file)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
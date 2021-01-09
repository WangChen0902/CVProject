import sys
import argparse
from my_yolo import YOLO, detect_video
from PIL import Image
import os
import time

PATH = '../Pic'
OUT_PATH = '../Result'
TEXT_PATH = '../Result'
coco_class  = []
coco_anno = open('model_data/coco_classes.txt')
coco_line = coco_anno.readline()
while coco_line:
    coco_class.append(coco_line.strip())
    coco_line = coco_anno.readline()
coco_anno.close()


def detect_img(yolo):
    st = time.time()
    for root,dirs,files in os.walk(PATH):
        for file in files:
            img = os.path.join(root, file)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, r_boxes, r_scores, r_classes = yolo.detect_image(image)
                # r_image.show()
                r_image.save(os.path.join(OUT_PATH, file), quality=95)
                # print(r_boxes, r_scores, r_classes)
                txt_name = file.split('.')[0]+'.txt'
                f_name = os.path.join(TEXT_PATH, txt_name)
                f = open(f_name, 'w')
                # f.write(str(r_classes)+' '+str(r_scores)+' '+str(r_boxes))
                for i in range(len(r_classes)):
                    line_str = coco_class[int(r_classes[i])]+' '+str(round(r_scores[i],6))+' '+str(int(r_boxes[i][1]))+\
                               ' '+str(int(r_boxes[i][0]))+' '+str(int(r_boxes[i][3]))+' '+str(int(r_boxes[i][2]))+'\n'
                    f.write(line_str)
                f.close()
    yolo.close_session()
    et = time.time()
    t = et - st
    print('total time: ', t)

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

import os
import xml.etree.ElementTree as ET

PATH ='D:\\VOC_Test\\VOCdevkit\\VOC2007\\KindOutText'
ANNO_PATH = 'D:\\VOC_Test\\VOCdevkit\\VOC2007\\Annotations\\%s.xml'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
coco_class  = []
coco_anno = open('model_data/coco_classes.txt')
coco_line = coco_anno.readline()
while coco_line:
    coco_class.append(coco_line.strip())
    coco_line = coco_anno.readline()

print(coco_class)
recalls = []
i = 0
for root, dirs, files in os.walk(PATH):
    for file in files:
        i += 1
        print(i)
        f_name = os.path.join(root, file)
        img_id = file.split('.')[0]
        f = open(f_name)
        line = f.readline()
        total_line = ''
        while line:
            total_line += line
            line = f.readline()
        test_cls = []
        cls_idx = total_line.split('[')[-1].split(']')[0].strip().replace('  ', ' ').split(' ')
        # print(cls_idx)
        if cls_idx == ['']:
            test_cls = []
        else:
            test_cls = [coco_class[int(idx)] for idx in cls_idx]
        # print(test_cls)
        f.close()
        in_file = open(ANNO_PATH % (img_id))
        tree = ET.parse(in_file)
        tree_root = tree.getroot()
        gt_cls = []
        gt_box = []
        for obj in tree_root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            box = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            gt_cls.append(cls)
            gt_box.append(box)
        # print(gt_cls)
        recall = 0.0
        correct = 0
        total = len(gt_cls)
        for c in gt_cls:
            if c in test_cls:
                correct += 1
                test_cls.remove(c)
        recall = correct/total
        recalls.append(recall)
avg_recall = sum(recalls)/len(recalls)
print(avg_recall)
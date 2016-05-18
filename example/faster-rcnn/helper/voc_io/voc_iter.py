import mxnet as mx
import numpy as np
import random
from voc import PascalVOC
import image_processing
import cv2


class PascalIter(mx.io.DataIter):
    def __init__(self, image_set, year, root_path, devkit_path,
                 image_shape, max_edge, rand_crop=True, rand_flip=True, shuffle=True,
                 normalization=False, **kwargs):
        super(PascalIter, self).__init__()
        self.dataset = PascalVOC(image_set, year, root_path, devkit_path)
        self.image_shape = image_shape
        self.rand_crop = rand_crop
        self.min_edge = min(image_shape)
        self.max_edge = max_edge
        self.shuffle = shuffle
        self.rand_flip = rand_flip
        self.max_gt_boxes = 0
        self.idxs = self.dataset.load_image_set_index()
        self._get_max_gt_boxes()
        self.num_instance = len(self.idxs)
        self.reset()

    def _get_max_gt_boxes(self):
        for i in range(len(self.idxs)):
            gt_boxes = self.dataset.load_pascal_annotation(self.idxs[i])
            self.max_gt_boxes = max(gt_boxes.shape[0], self.max_gt_boxes)

    def _flip(self, im, boxes):
        h = boxes[:, 2] - boxes[:, 0]
        boxes[:, 0] = im.shape[1] - (boxes[:, 0] + h)
        boxes[:, 2] = im.shape[1] - (boxes[:, 2] - h)
        return cv2.flip(im, 1), boxes

    def _random_crop(self, im, boxes):
        dim0 = im.shape[0] - self.image_shape[0]
        dim1 = im.shape[1] - self.image_shape[1]
        assert(dim0 >= 0)
        assert(dim1 >= 0)
        if self.rand_crop:
            try:
                dim0 = np.random.randint(0, dim0)
            except:
                # ignore 0,0
                pass
            try:
                dim1 = np.random.randint(0, dim1)
            except:
                # ignore 0,0
                pass
        else:
            dim0 /= 2
            dim1 /= 2
        return image_processing.crop(im, boxes,
                                     int(dim0),
                                     int(dim1),
                                     self.image_shape[0],
                                     self.image_shape[1])

    def _random_resize(self, im, boxes):
        ss = np.random.randint(self.min_edge, self.max_edge)
        return _resize(im, boxes, ss)

    def _resize(self, im, boxes, short_edge):
        im_new, scale = image_processing.resize(im, short_edge, self.max_edge)
        return im_new, boxes * scale

    @property
    def provide_data(self):
        return [("image", (1, 3, self.image_shape[0], self.image_shape[1])),
                ("gt_boxes", (self.max_gt_boxes, 5)),
                ("im_info", (1, 3)),
                ("gt_pad", (1,))]

    def hard_reset(self):
        self.cursor = 0

    def reset(self):
        if self.shuffle == True:
            random.shuffle(self.idxs)
        self.cursor = 0

    def iter_next(self):
        self.cursor += 1
        if self.cursor < self.num_instance:
            return True
        else:
            return False

    def next(self):
        find_next = True
        while find_next:
            if self.iter_next():
                tmp = self.getdata()
                if (len(tmp[1]) >= 1):
                    return mx.io.DataBatch(data=tmp, label=None, \
                            pad = 0, index=None)
            else:
                raise StopIteration


    def getdata(self):
        gt_boxes = self.dataset.load_pascal_annotation(self.idxs[self.cursor])
        path = self.dataset.image_path_from_index(self.idxs[self.cursor])
        print path
        im = cv2.imread(path)
        if self.rand_flip:
            if random.uniform(0, 1) > 0.5:
                im, gt_boxes = self._flip(im, gt_boxes)
        im, gt_boxes = self._resize(im, gt_boxes, self.min_edge)
        im, gt_boxes = self._random_crop(im, gt_boxes)
        im_info = np.asarray((self.image_shape[0], self.image_shape[1], 1.0))
        pad = np.asarray([self.max_gt_boxes - len(gt_boxes)])
        return [im, gt_boxes, im_info, pad]


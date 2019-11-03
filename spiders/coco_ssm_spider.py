'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@author Harm Matthias Harms
@date 11/03/19
'''

from config import *
import numpy as np
from alchemy.datasets.coco import COCO_DS

from spiders.base_coco_ssm_spider import BaseCOCOSSMSpiderAttentionBox, BaseCOCOSSMSpiderAttSizeTest, NoLabelException


class COCOSSMSpiderAttentionBox(BaseCOCOSSMSpiderAttentionBox):

    attr = ['image', 'objAttBox_8', 'objAttBox_16', 'objAttBox_24', 'objAttBox_32', 'objAttBox_48', 'objAttBox_64', 'objAttBox_96', 'objAttBox_128',
            'objAttBox_8_org', 'objAttBox_16_org', 'objAttBox_24_org', 'objAttBox_32_org', 'objAttBox_48_org', 'objAttBox_64_org', 'objAttBox_96_org', 'objAttBox_128_org']

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(
                ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            self.__class__.cats_to_labels = dict(
                [(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpiderAttentionBox, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE


class COCOSSMDemoSpider(BaseCOCOSSMSpiderAttSizeTest):

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(
                ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, False)
        super(COCOSSMDemoSpider, self).__init__(*args, **kwargs)
        try:
            self.RFs = RFs
        except Exception:
            pass
        try:
            self.SCALE = TEST_SCALE
        except Exception:
            pass

    def fetch(self):
        idx = self.get_idx()
        item = self.dataset[idx]
        self.image_path = item.image_path
        self.anns = item.imgToAnns
        self.max_edge = self.SCALE
        self.fetch_image()
        return {"image": self.img_blob}

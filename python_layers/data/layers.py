'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@author Harm Matthias Harms
@date 11/03/19
'''

from spiders.coco_ssm_spider import COCOSSMSpiderAttentionBox
from alchemy.engines.caffe_python_layers import AlchemyDataLayer


class COCOSSMSpiderAttentionBox(AlchemyDataLayer):

    spider = COCOSSMSpiderAttentionBox

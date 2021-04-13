from data.triplet_input import SketchyDatabase
from data.triplet_input import QUMLDataV1
from data.triplet_input import QUMLDataV2
from data.image_input import ImageDataset


class DatasetCreator(object):
    @staticmethod
    def getDataset(name):
        if name == 'sketchydb':
            return SketchyDatabase
        elif name == 'shoev1' or name == 'chair':
            return QUMLDataV1
        elif name == 'shoev2':
            return QUMLDataV2
        elif name == 'test':
            return ImageDataset

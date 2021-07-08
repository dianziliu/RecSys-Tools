from DataManager.encoderManager import EncoderManager,EncoderManagerPlus
from featureConfig.feature1 import *


class ItemManager(EncoderManager):
    name="ItemManager"
    pass

class ItemManagerPlus(EncoderManagerPlus):
    name="ItemManager"
    pass


if __name__=="__main__":
    a=ItemManager([1,2,3])
    print(a.defaultPath)
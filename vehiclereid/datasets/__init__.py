from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .veri import VeRi
from .vehicleid import VehicleID
from .aicity import Aicity
from .sim import Sim
from .veri_wild import VeriWild


__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
    'aicity': Aicity,
    'sim': Sim,
    'veri_wild': VeriWild
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


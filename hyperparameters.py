
import json
import os
from copy import deepcopy


class HyperParameters(dict):

    def __init__(self, loadpath=None, **kwargs):
        super().__init__(**kwargs)

        # update user params
        self.load(loadpath)

    def load(self, loadpath=None):
        """
        Load and overwrite parameters from json file.
        """

        if loadpath:
            with open(loadpath, mode='r') as f:
                self.update(json.load(f))

    def save(self, savedir='.', savename='savehyperparams.json'):
        """
        Dump Hyperparams to json file
        """

        with open(os.path.join(savedir, savename), mode='w') as f:
            json.dump(self, f, indent=1, sort_keys=True)

    def overwirte(self, srcdic, keys):
        srcdic = deepcopy(srcdic)

        for key in keys:
            if key in srcdic:
                val = srcdic.pop(key)
                if val is not None:    
                    self[key] = val

        return srcdic
    


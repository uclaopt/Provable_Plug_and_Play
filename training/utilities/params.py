import json

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self):
        pass

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    @staticmethod
    def from_json(json_path):
        """ Create a parameter manager from a json file on the disk. """
        params = Params()
        with open(json_path) as f:
            loaded = json.load(f)
            params.__dict__.update(loaded)

    @staticmethod
    def from_dict(dictionary):
        """ Create a parameter manager from a dictionary. """
        params = Params()
        params.__dict__.update(dictionary)

    @staticmethod
    def from_parser(parser):
        """ Create a parameter manager from a parser instance from argparse. """
        params = Params()
        params.__dict__.update(vars(parser))


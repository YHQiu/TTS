import json

class TrainConfig:

    def __init__(self, batch_size=32, learning_rate=1e5, num_epochs=10, outputs=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.outputs = outputs

    @classmethod
    def load_from_json(cls, path):
        with open(path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
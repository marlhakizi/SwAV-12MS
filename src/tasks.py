# This file holds the mean and standard deviation for each task
class Task:
    def __init__(self, name):
        if name == "RGB":
            self.mean = [0.507, 0.513, 0.462]
            self.std = [0.172, 0.133, 0.114]
        elif name == "12_channels":
            self.mean = [
                0.17784427,
                0.20382293,
                0.21353255,
                0.24842924,
                0.31372614,
                0.34154329,
                0.35239194,
                0.35575098,
                0.31944108,
                0.26813008,
                -0.0010185,
                -0.00173495,
            ]
            self.std = [
                0.0621233,
                0.06649029,
                0.08300084,
                0.07792479,
                0.07930023,
                0.09247079,
                0.09787765,
                0.09431137,
                0.09293508,
                0.09194112,
                0.00045928,
                0.00044843,
            ]
        else:
            raise NotImplementedError("Channels not specified!")

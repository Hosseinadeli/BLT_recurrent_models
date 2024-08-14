from torch import nn

class BLTLoss(nn.Module):
    def __init__(self, args, return_dict=True):
        super().__init__()
        
        if not isinstance(args, dict):
            args = vars(args)

        self.return_dict = return_dict
        if return_dict:
            self.weight_dict = {'loss_labels': 1}
        self.loss_choice = args["loss_choice"]
        self.loss_gamma = args["loss_gamma"]
        self.loss_func = nn.CrossEntropyLoss()

        if self.loss_choice not in ["weighted", "decay"]:
            raise ValueError(f"Unknown loss choice: {self.loss_choice}")

    def forward(self, outputs, targets):
        loss = 0
        weight_sum = 1
        loss = self.loss_func(outputs[-1], targets)

        if self.loss_gamma > 0:
            for s in range(1, len(outputs)):
                # here `s` represents the number of steps back in time
                # e.g., s=1 corresponds to the second-to-last output, outputs[-2]
                if self.loss_choice == "decay":
                    weight = self.loss_gamma ** s
                elif self.loss_choice == "weighted":
                    weight = self.loss_gamma
                weight_sum += weight
                loss += weight * self.loss_func(outputs[-(s+1)], targets)

        loss = loss / weight_sum

        if self.return_dict:
            return {'loss_labels': loss}
        else:
            return loss

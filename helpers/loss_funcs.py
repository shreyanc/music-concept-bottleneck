import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VarCriterion:
    def __init__(self, loss_functions, loss_aggregation='sum'):
        self.loss_functions = loss_functions
        self.log_vars = []
        self.num_tasks = len(loss_functions)
        self.loss_aggregation = loss_aggregation
        for i in range(self.num_tasks):
            self.log_vars.append(torch.zeros((1,), requires_grad=True, device=device))

    def __call__(self, y_pred, y_true):
        total_loss = 0
        all_losses = []
        all_scaled_losses = []
        for i in range(len(self.loss_functions)):
            precision = torch.exp(-self.log_vars[i])
            loss = self.loss_functions[i](y_pred[i].float(), y_true[i].float())
            scaled_regularized_loss = torch.sum(precision * loss + self.log_vars[i], -1)
            total_loss += scaled_regularized_loss
            all_losses.append(loss)
            all_scaled_losses.append(precision * loss)
        if self.loss_aggregation == 'mean':
            return torch.mean(torch.stack(all_losses)), all_losses
        else:
            return total_loss, all_losses, all_scaled_losses


class ScaledCriterion:
    def __init__(self, loss_functions, loss_weights, loss_aggregation='sum'):
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.num_tasks = len(loss_functions)
        self.loss_aggregation = loss_aggregation
        # for i in range(self.num_tasks):
        #     self.loss_weights.append(torch.zeros((1,), requires_grad=True, device=device))

    def __call__(self, y_pred, y_true):
        total_loss = 0
        all_losses = []
        all_scaled_losses = []
        for i in range(len(self.loss_functions)):
            weight = self.loss_weights[i]
            loss = self.loss_functions[i](y_pred[i].float(), y_true[i].float())
            scaled_loss = weight * loss
            total_loss += scaled_loss
            all_losses.append(loss)
            all_scaled_losses.append(scaled_loss)
        if self.loss_aggregation == 'mean':
            return torch.mean(torch.stack(all_losses)), all_losses
        else:
            return total_loss, all_losses, all_scaled_losses


# class TaskCriterion:
#     def __init__(self, tasks):

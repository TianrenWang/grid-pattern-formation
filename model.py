# -*- coding: utf-8 -*-
import torch


class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(
            input_size=2,
            hidden_size=self.Ng,
            nonlinearity=options.activation,
            bias=False,
        )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.coordinateDecoder = torch.nn.Linear(self.Ng, 2, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        """
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        """
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)
        return g

    def predict(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        """
        grid_code = self.g(inputs)
        place_preds = self.decoder(grid_code)
        grid_code = grid_code.detach()
        coordinate_preds = self.coordinateDecoder(grid_code)

        return place_preds, coordinate_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        """
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        """
        y: torch.Tensor = pc_outputs
        preds, coordinatePreds = self.predict(inputs)
        loss = torch.nn.functional.cross_entropy(preds.flatten(0, 1), y.flatten(0, 1))
        loss += torch.nn.functional.mse_loss(
            pos.flatten(0, 1), coordinatePreds.flatten(0, 1)
        )

        # Weight regularization
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()
        abs_coord_err = torch.sqrt(((pos - coordinatePreds) ** 2).sum(-1)).mean()

        return loss, err, abs_coord_err

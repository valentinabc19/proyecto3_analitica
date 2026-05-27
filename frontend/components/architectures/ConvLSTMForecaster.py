import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. DEFINICIÓN DEL CONVLSTM CELL ──────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=self.padding,
            padding_mode='replicate'
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, gates.size(1) // 4, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_next = forgetgate * c + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)
        return h_next, c_next


# ── 2. RED CONVLSTM BIDIRECCIONAL DE 2 CAPAS ──────
class ConvLSTM(nn.Module):
    """
    Red ConvLSTM BIDIRECCIONAL de 2 capas (hidden_dim=128, kernel_size=3).
    Procesa secuencias espaciales en dirección Forward y Backward.
    """
    def __init__(self, in_channels=260, hidden_dim=128, kernel_size=3, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Celdas para la dirección Forward
        self.forward_cells = nn.ModuleList([
            ConvLSTMCell(in_channels, hidden_dim, kernel_size),
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        ])

        # Celdas para la dirección Backward
        self.backward_cells = nn.ModuleList([
            ConvLSTMCell(in_channels, hidden_dim, kernel_size),
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        ])

        # Convolución final 1x1: Recibe los canales concatenados (128 fwd + 128 bwd = 256)
        # y mapea a las 9 predicciones simultáneas [3 horizontes x 3 gases]
        self.conv_output = nn.Conv2d(hidden_dim * 2, 9, kernel_size=1)

    def forward(self, x):
        b, t, _, h, w = x.size()

        # Inicializar estados de memoria Forward
        h_fwd = [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)]

        for seq_t in range(t):
            h_fwd[0], c_fwd[0] = self.forward_cells[0](x[:, seq_t, :, :, :], h_fwd[0], c_fwd[0])
            h_fwd[1], c_fwd[1] = self.forward_cells[1](h_fwd[0], h_fwd[1], c_fwd[1])

        # Inicializar estados de memoria Backward
        h_bwd = [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)]

        for seq_t in reversed(range(t)):
            h_bwd[0], c_bwd[0] = self.backward_cells[0](x[:, seq_t, :, :, :], h_bwd[0], c_bwd[0])
            h_bwd[1], c_bwd[1] = self.backward_cells[1](h_bwd[0], h_bwd[1], c_bwd[1])

        # Concatenación de características bidireccionales en la dimensión de canales: [B, 256, H, W]
        out_combined = torch.cat([h_fwd[1], h_bwd[1]], dim=1)

        # Proyectar a las 9 salidas simultáneas
        predictions_raw = self.conv_output(out_combined)

        predictions_raw = F.softplus(predictions_raw) # Corregido: Usar F.softplus

        # Reestructurar a [B, 3, 3, H, W]
        predictions = predictions_raw.view(b, 3, 3, h, w)
        return predictions
"""The network is entirely a practice exercise: no hidden implementation."""

from torch import Tensor, nn

from .exercises import ExerciseNotImplemented


class ToyMLP(nn.Module):
    """A direct-output network, independent of x/eps/v interpretation.

    Contract: forward(z [B,D], t [B,1]) -> raw prediction [B,D].
    It must not know P, intrinsic points, or the prediction-to-velocity formula.
    Any future nn.Module with this contract can replace this baseline.
    """

    def __init__(self, observed_dim: int, hidden_dim: int = 256, linear_layers: int = 5) -> None:
        super().__init__()
        # M01: Register a time-conditioned ReLU MLP.
        # Practice nn.Linear, nn.ReLU, nn.Sequential or nn.ModuleList and
        # module registration. Plain Python lists won't register parameters.
        # References: model_jit.py::TimestepEmbedder.mlp / JiT.blocks show
        # registration; FinalLayer.linear shows an unconstrained output head.
        # JiT uses sinusoidal time embedding and adaLN; scalar concatenation
        # is our explicit toy simplification, not a copy of that architecture.
        # Chosen depth convention: five Linear layers total = four hidden
        # layers and one output layer. Concatenate a scalar t with z, giving
        # D+1 input features. First map to H, repeat H->H (L-2) times, then
        # H->D. Put ReLU after hidden layers only: all targets can be negative.
        # Use identical architecture for all three prediction spaces.
        # Do not add an input-output residual, analytic noise passthrough,
        # P-based bottleneck, or x/eps/v conversion inside this network.
        # Such shortcuts would change the capacity comparison in the paper.
        # Acceptance: inspect named_parameters/state_dict; all layers are
        # registered, all parameters move with .to(), five Linear layers at L=5.
        # Keep nn.Linear defaults initially; JiT.initialize_weights contains
        # Transformer-specific zero initialization that is not required here.
        raise ExerciseNotImplemented("M01", "register the time-conditioned MLP layers")

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """M02: Implement the complete forward pass yourself.

        Practice torch.cat(dim=...), batch vs feature axes, nn.Module calls,
        dtype/device consistency and autograd. Validate [B,D] and [B,1]; do not
        use an unqualified squeeze(), which removes the batch axis for B=1.
        Time is already a per-example column; no .item() or Python loop over B.
        Concatenate along features, run hidden layers, return unconstrained
        [B,D]. Avoid detach(), torch.tensor(existing_tensor), or no_grad here.
        Acceptance: B=1 and short batches work; backward reaches every layer.
        Optional later extension: replace scalar conditioning with sinusoidal
        time features, recording that as a separate architecture experiment.
        """
        raise ExerciseNotImplemented("M02", "implement the MLP forward pass")

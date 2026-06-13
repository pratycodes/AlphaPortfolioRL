import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    """
    Linear layer with a fixed binary connectivity mask.

    The mask is applied in the forward pass, so masked connections contribute
    neither activations nor gradients. This gives the actor/critic MLP heads a
    sparse inductive bias without changing their external interface.

    Inspired by the local reference paper:
    networkSparsity.pdf, "Network Sparsity Unlocks the Scaling Potential of
    Deep Reinforcement Learning".
    """

    def __init__(self, in_features, out_features, density=0.35, bias=True, seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.density = float(density)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        mask = torch.bernoulli(
            torch.full((out_features, in_features), self.density),
            generator=generator,
        )
        mask = self._ensure_connectivity(mask)
        self.register_buffer("mask", mask)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)

    def _ensure_connectivity(self, mask):
        for row in range(mask.size(0)):
            if mask[row].sum() == 0:
                mask[row, row % mask.size(1)] = 1.0
        for column in range(mask.size(1)):
            if mask[:, column].sum() == 0:
                mask[column % mask.size(0), column] = 1.0
        return mask


def build_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    use_sparse=False,
    density=0.35,
    seed=None,
    topology="erdos_renyi",
    dropout=0.0,
):
    layers = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    shapes = [(dims[index], dims[index + 1]) for index in range(len(dims) - 1)]
    densities = _layer_densities(shapes, density, topology) if use_sparse else [density] * len(shapes)

    for layer_index, hidden_dim in enumerate(hidden_dims):
        layers.append(_linear(dims[layer_index], hidden_dim, use_sparse, densities[layer_index], seed, layer_index))
        layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    output_index = len(shapes) - 1
    layers.append(_linear(dims[-2], output_dim, use_sparse, densities[output_index], seed, output_index))
    return nn.Sequential(*layers)


def _linear(in_features, out_features, use_sparse, density, seed, layer_index):
    if not use_sparse:
        return nn.Linear(in_features, out_features)

    layer_seed = None if seed is None else int(seed) + layer_index
    return SparseLinear(in_features, out_features, density=density, seed=layer_seed)


def _layer_densities(shapes, target_density, topology):
    target_density = max(0.0, min(float(target_density), 1.0))
    if _canonical_topology(topology) != "erdos_renyi":
        return [target_density for _ in shapes]

    layer_sizes = [in_features * out_features for in_features, out_features in shapes]
    total_params = sum(layer_sizes)
    target_nonzeros = target_density * total_params
    remaining_target = target_nonzeros
    remaining = set(range(len(shapes)))
    densities = [0.0 for _ in shapes]

    while remaining:
        epsilon_denominator = sum(shapes[index][0] + shapes[index][1] for index in remaining)
        epsilon = remaining_target / max(epsilon_denominator, 1.0)
        capped = []

        for index in remaining:
            in_features, out_features = shapes[index]
            probability = epsilon * (in_features + out_features) / max(layer_sizes[index], 1)
            if probability >= 1.0:
                capped.append(index)

        if not capped:
            for index in remaining:
                in_features, out_features = shapes[index]
                probability = epsilon * (in_features + out_features) / max(layer_sizes[index], 1)
                densities[index] = float(min(1.0, max(probability, 0.0)))
            break

        for index in capped:
            densities[index] = 1.0
            remaining_target -= layer_sizes[index]
            remaining.remove(index)

        if remaining_target <= 0:
            break

    return densities


def _canonical_topology(topology):
    value = str(topology or "erdos_renyi").lower().replace("-", "_").replace(" ", "_")
    if value in {"er", "erdos", "erdosrenyi", "erdos_renyi"}:
        return "erdos_renyi"
    return value

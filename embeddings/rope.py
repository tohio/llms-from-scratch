import torch


def get_rope_frequencies(dim, seq_len, base=10000):
    # Compute the rotation angles for each position and dimension
    # dim must be even since we rotate in pairs
    # base=10000 is the standard value from the original transformer paper

    # Compute theta for each dimension pair: θ = 1 / (10000 ^ (2i / dim))
    # torch.arange(0, dim, 2) gives [0, 2, 4, ...] — one per pair
    thetas = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Positions in the sequence [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len).float()

    # Outer product — every position multiplied by every theta
    # shape: (seq_len, dim/2)
    freqs = torch.outer(positions, thetas)

    return freqs


def apply_rope(x, freqs):
    # x shape: (batch_size, seq_len, dim)
    # freqs shape: (seq_len, dim/2)

    # Split x into pairs along the last dimension
    # x1 and x2 are both shape (batch_size, seq_len, dim/2)
    x1 = x[..., 0::2]  # even indices — first of each pair
    x2 = x[..., 1::2]  # odd indices  — second of each pair

    # Compute sin and cos of the rotation angles
    # unsqueeze(0) adds the batch dimension so shapes align with x1 and x2
    cos = freqs.cos().unsqueeze(0)  # (1, seq_len, dim/2)
    sin = freqs.sin().unsqueeze(0)  # (1, seq_len, dim/2)

    # Apply the rotation to each pair
    # [x1, x2] → [x1·cos(θ) - x2·sin(θ), x1·sin(θ) + x2·cos(θ)]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave rotated pairs back into the original shape
    # stack combines them, flatten reassembles into (batch_size, seq_len, dim)
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    out = out.flatten(-2)

    return out


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 1
    seq_len = 4
    dim = 8

    # Simulate a Query or Key vector
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, dim)
    print("Input shape:", x.shape)
    print("Input:\n", x)

    # Compute frequencies and apply RoPE
    freqs = get_rope_frequencies(dim, seq_len)
    x_rotated = apply_rope(x, freqs)

    print("\nOutput shape:", x_rotated.shape)
    print("Output:\n", x_rotated)
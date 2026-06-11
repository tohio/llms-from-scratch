import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Expert ───────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        # Each expert is a standard FFN with SwiGLU activation
        # identical architecture to the FeedForward class in llama
        self.w_gate  = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out   = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: (Swish(gate) * value) → output
        gate  = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return self.w_out(gate * value)


# ─── Router ───────────────────────────────────────────────────────────────────

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()

        # Linear layer that scores each token against each expert
        # output shape: (batch * seq_len, num_experts)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k):
        # Compute raw scores for each expert
        # shape: (batch * seq_len, num_experts)
        logits = self.gate(x)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Select top_k experts per token
        # weights: the probability assigned to each selected expert
        # indices: which expert was selected
        weights, indices = torch.topk(probs, k=top_k, dim=-1)

        # Renormalise weights so they sum to 1 across selected experts
        # ensures the weighted sum of expert outputs is properly scaled
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices


# ─── Mixture of Experts ───────────────────────────────────────────────────────

class MoE(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, top_k):
        super().__init__()

        assert top_k <= num_experts, "top_k cannot exceed num_experts"

        self.d_model     = d_model
        self.num_experts = num_experts
        self.top_k       = top_k

        # Pool of expert FFNs — each is an independent learnable transformation
        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim)
            for _ in range(num_experts)
        ])

        # Router — decides which experts handle each token
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        b, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for routing
        # shape: (batch * seq_len, d_model)
        x_flat = x.view(-1, d_model)

        # Get expert weights and indices for each token
        # weights shape: (batch * seq_len, top_k)
        # indices shape: (batch * seq_len, top_k)
        weights, indices = self.router(x_flat, self.top_k)

        # Accumulate weighted expert outputs
        # start with zeros — each selected expert adds its weighted contribution
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            # Get the expert index for this slot across all tokens
            expert_idx = indices[:, k]         # (batch * seq_len,)
            weight     = weights[:, k]         # (batch * seq_len,)

            # Route each token to its assigned expert
            # process tokens per expert to avoid running all experts on all tokens
            for i in range(self.num_experts):
                # Find which tokens are assigned to expert i for this slot
                token_mask = (expert_idx == i)

                if token_mask.any():
                    # Select the tokens assigned to this expert
                    expert_input  = x_flat[token_mask]

                    # Run the expert
                    expert_output = self.experts[i](expert_input)

                    # Scale by the router weight and accumulate
                    output[token_mask] += weight[token_mask].unsqueeze(-1) * expert_output

        # Restore original shape
        return output.view(b, seq_len, d_model)


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(123)

    # Hyperparameters
    batch_size  = 2
    seq_len     = 4
    d_model     = 8
    hidden_dim  = 32
    num_experts = 8    # total number of expert FFNs
    top_k       = 2   # each token is routed to 2 experts

    # Random input — simulates token embeddings after attention
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape:       {x.shape}")

    # Build MoE layer
    moe = MoE(
        d_model     = d_model,
        hidden_dim  = hidden_dim,
        num_experts = num_experts,
        top_k       = top_k
    )

    # Forward pass
    output = moe(x)
    print(f"Output shape:      {output.shape}")
    print(f"Output:\n{output}")

    # Show which experts were selected for the first token
    x_flat          = x.view(-1, d_model)
    weights, indices = moe.router(x_flat, top_k)
    print(f"\nRouter weights (first 4 tokens):\n{weights[:4]}")
    print(f"Router indices (first 4 tokens):\n{indices[:4]}")

    # Parameter count
    total_params  = sum(p.numel() for p in moe.parameters())
    active_params = sum(p.numel() for p in moe.experts[0].parameters()) * top_k
    print(f"\nTotal parameters:  {total_params:,}")
    print(f"Active per token:  {active_params:,}  ({100 * active_params / total_params:.1f}% of total)")
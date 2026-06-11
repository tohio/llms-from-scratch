Good catch to push back. Let me be thorough:

**torch.nn:**
- `nn.Embedding`
- `nn.MultiheadAttention`
- `nn.TransformerDecoderLayer`
- `nn.TransformerDecoder`
- `nn.TransformerEncoderLayer`
- `nn.TransformerEncoder`
- `nn.Transformer`
- `nn.LayerNorm`
- `nn.RMSNorm`
- `nn.Linear`
- `nn.Dropout`
- `nn.Sequential`
- `nn.ModuleList`
- `nn.GELU`
- `nn.SiLU`

**torch.nn.functional:**
- `F.scaled_dot_product_attention`
- `F.cross_entropy`
- `F.softmax`
- `F.silu`
- `F.gelu`

**torch.optim:**
- `torch.optim.AdamW`
- `torch.optim.Adam`
- `torch.optim.lr_scheduler.CosineAnnealingLR`
- `torch.optim.lr_scheduler.StepLR`

**torch.utils.data:**
- `Dataset`
- `DataLoader`
- `random_split`

**torch:**
- `torch.compile`
- `torch.arange`
- `torch.no_grad`
- `torch.manual_seed`
- `torch.inference_mode`
- `torch.amp.autocast`
- `torch.cuda.is_available`
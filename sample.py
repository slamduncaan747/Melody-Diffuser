import torch

def sample(model, cond, T, vocab_size, temperature=1, p=.95, w=.9):
    x = torch.randint(0, vocab_size, (1, 64), device=cond.device)
    for t in reversed(range(T)):
        conditioned_logits = model(x, t, cond)
        unconditioned_logits = model(x, t)
        logits = unconditioned_logits + w * (conditioned_logits - unconditioned_logits)
        probs = torch.softmax(logits / temperature, dim=-1)

        sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(sorted_probs.view(-1, sorted_probs.size(-1)), 1)
        x = idx.view(-1, sorted_probs.size(-1)).gather(-1, next_token).view(probs.shape[0], probs.shape[1])
    return x

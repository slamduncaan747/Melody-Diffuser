import torch
from model import MelodyDiffusor
import pickle
from diffusion_utils import add_noise, get_loss, get_betas, Config

config = Config(
    vocab_size=90,
    T=16,
    dim=512,
    epochs=1000
)


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

with open("melodies_test.pkl", "rb") as f:
    data = pickle.load(f)
data_tensor = torch.tensor(data[:1000], dtype=torch.long).squeeze(1).to(device)

dataset = torch.utils.data.TensorDataset(data_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)



model = MelodyDiffusor(vocab_size=90, seq_len=1024, dim=512, n_layers=6, n_heads=8, ffn_inner_dim=2048, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
betas = get_betas(.02, .1, 16).to(device)
alphas = 1 - betas
alpha_cum = torch.cumprod(alphas, dim=0)

print("Starting training...")
for epoch in range(config.epochs):
    total_loss = 0
    for batch in dataloader:
        x0 = batch[0].to(device)
        t = torch.randint(0, 16, (x0.shape[0],)).long().to(device)
        noise = 1-alpha_cum[t]
        noisy_inputs = add_noise(x0, noise, config.vocab_size)
        optimizer.zero_grad()
        loss = get_loss(model, noisy_inputs, x0, betas, config.vocab_size, t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "melody_diffusor_model.pth")
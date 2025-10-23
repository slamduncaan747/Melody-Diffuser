import torch
from model import MelodyDiffusor
import pickle
from diffusion_utils import add_noise, get_loss, get_betas, Config

config = Config(
    vocab_size=130,
    T=16,
    dim=512,
    epochs=1000,
    val_interval=5,
    checkpoint_interval=50
)


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

with open("/content/drive/MyDrive/MelodyDiffusor/melodies_test.pkl", "rb") as f:
    data = pickle.load(f)
data_tensor = torch.tensor(data[:len(data)-1025], dtype=torch.long).squeeze(1).to(device)
val_tensor = torch.tensor(data[len(data)-1025:], dtype=torch.long).squeeze(1).to(device)

dataset = torch.utils.data.TensorDataset(data_tensor)
val_set = torch.utils.data.TensorDataset(val_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)



model = MelodyDiffusor(vocab_size=130, seq_len=1024, dim=512, n_layers=6, n_heads=8, ffn_inner_dim=2048, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
betas = get_betas(.02, .25, 16).to(device)
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

    if (epoch + 1) % config.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"/content/drive/MyDrive/BachNet/checkpoints/melody_diffusor_model_epoch{epoch+1}.pth")
    
    if (epoch + 1) % config.val_interval == 0:
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch[0].to(device)
                t = torch.randint(0, 16, (val_batch.shape[0],)).long().to(device)
                noise = 1 - alpha_cum[t]
                noisy_inputs = add_noise(val_batch, noise, config.vocab_size)
                val_loss = get_loss(model, noisy_inputs, val_batch, betas, config.vocab_size, t)
                val_loss_total += val_loss.item()
            val_loss = val_loss_total / len(val_loader)
            print(f"Validation Loss at epoch {epoch+1}: {val_loss_total:.4f}")
        model.train()
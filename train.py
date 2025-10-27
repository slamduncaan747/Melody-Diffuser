import torch
from model import MelodyDiffusor
import pickle
from diffusion_utils import add_noise, get_loss, get_betas, Config, add_cond_noise
import torch.multiprocessing as mp
import numpy as np
import os
from datetime import datetime
config = Config(
    vocab_size=130,
    T=64,
    dim=512,
    epochs=100,
    val_interval=3,
    checkpoint_interval=10
)
onColab = True
test_amount = 16
batch_size = 256
workers = 12
data_string = "melodies_test.pkl" if not onColab else "Melody-Diffuser/testandval_data.pkl"
cond_string = "gesture_conditions.npy" if not onColab else "Melody-Diffuser/gesture_conditions.pkl"

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f"/content/drive/MyDrive/MelodyDiffusor/checkpoints/{run_timestamp}"
os.makedirs(checkpoint_dir, exist_ok=True)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    with open(data_string, "rb") as f:
        data = pickle.load(f)
    with open(cond_string, "rb") as c:
        conds = pickle.load(c)
    if onColab:
        conds_tensor_t = torch.tensor(conds[:len(data)-5000], dtype=torch.long).squeeze(1)
        conds_tensor_v = torch.tensor(conds[len(data)-5000:], dtype=torch.long).squeeze(1)
        data_tensor = torch.tensor(data[:len(data)-5000], dtype=torch.long).squeeze(1)
        val_tensor = torch.tensor(data[len(data)-5000:], dtype=torch.long).squeeze(1)
    else:
        conds_tensor_t = torch.tensor(conds[:test_amount], dtype=torch.long).squeeze(1)
        conds_tensor_v = torch.tensor(conds[-test_amount:], dtype=torch.long).squeeze(1)
        data_tensor = torch.tensor(data[:test_amount], dtype=torch.long).squeeze(1)
        val_tensor = torch.tensor(data[-test_amount:], dtype=torch.long).squeeze(1)
    print(len(data))
    dataset = torch.utils.data.TensorDataset(data_tensor, conds_tensor_t)
    val_set = torch.utils.data.TensorDataset(val_tensor, conds_tensor_v)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))



    model = MelodyDiffusor(vocab_size=130, seq_len=64, dim=512, n_layers=6, n_heads=8, ffn_inner_dim=2048, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    betas = get_betas(.05, .5, config.T).to(device)
    alphas = 1 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)

    print("Starting training...")
    print("Dataset of length: " ,len(dataloader))
    for epoch in range(config.epochs):
        total_loss = 0
        for batch, cond in dataloader:
            x0 = batch.to(device)
            condition = cond.to(device)
            condition = add_cond_noise(condition, 8, .1)
            t = torch.randint(0, config.T, (x0.shape[0],)).long().to(device)
            noise = 1-alpha_cum[t]
            noisy_inputs = add_noise(x0, noise, config.vocab_size)
            optimizer.zero_grad()

            loss = get_loss(model, noisy_inputs, x0, betas, config.vocab_size, t, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % config.checkpoint_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"melody_diffusor_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for val_batch, val_cond in val_loader:
                    val_batch = val_batch.to(device)
                    val_cond = val_cond.to(device)
                    val_cond = add_cond_noise(val_cond, 8, .1)
                    t = torch.randint(0, config.T, (val_batch.shape[0],)).long().to(device)
                    noise = 1 - alpha_cum[t]
                    noisy_inputs = add_noise(val_batch, noise, config.vocab_size)
                    val_loss = get_loss(model, noisy_inputs, val_batch, betas, config.vocab_size, t, val_cond)
                    val_loss_total += val_loss.item()
                val_loss = val_loss_total / len(val_loader)
                print(f"Validation Loss at epoch {epoch+1}: {val_loss_total:.4f}")
            model.train()
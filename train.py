import torch
from model import MelodyDiffusor
import pickle
from diffusion_utils import add_noise, get_loss, get_betas, Config, add_cond_noise, get_scheduler
import torch.multiprocessing as mp
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import wandb


config = Config(
    vocab_size=130,
    T=64,
    dim=512,
    epochs=10,
    val_interval=2000,
    checkpoint_interval=1
)
onColab = True
test_amount = 16
batch_size = 2048
workers = 12
val_size = 100000
data_string = "melodies_test.pkl" if not onColab else "Melody-Diffuser/testandval_data.pkl"
cond_string = "gesture_conditions.npy" if not onColab else "Melody-Diffuser/big_gestures_V2.pkl"



if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = f"/content/drive/MyDrive/MelodyDiffusor/checkpoints/{run_timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        project="melody-diffuser",  
        name=run_timestamp,         
        config={
            "learning_rate": 2e-4,
            "epochs": config.epochs,
            "batch_size": batch_size,
            "dim": config.dim,
            "n_layers": 6,
            "n_heads": 8,
            "dropout": 0.1,
            "T_steps": config.T,
            "val_size": val_size,
            "dataset_size": "10M" 
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    with open(data_string, "rb") as f:
        data = pickle.load(f)
    with open(cond_string, "rb") as c:
        conds = pickle.load(c)
    print("Data Loaded")
    indices = np.random.permutation(len(data))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    data_np = np.stack(data)
    conds_np = np.stack(conds)

    train_data = data_np[train_indices]
    val_data = data_np[val_indices]
    train_conds= conds_np[train_indices]
    val_conds = conds_np[val_indices]

    train_data_tensor = torch.tensor(train_data, dtype=torch.long).squeeze(1)
    val_data_tensor = torch.tensor(val_data, dtype=torch.long).squeeze(1)
    train_conds_tensor = torch.tensor(train_conds, dtype=torch.long).squeeze(1)
    val_conds_tensor = torch.tensor(val_conds, dtype=torch.long).squeeze(1)

    dataset = torch.utils.data.TensorDataset(train_data_tensor, train_conds_tensor)
    val_set = torch.utils.data.TensorDataset(val_data_tensor, val_conds_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))

    model = MelodyDiffusor(vocab_size=130, seq_len=64, dim=512, n_layers=6, n_heads=8, ffn_inner_dim=2048, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    total_training_steps = len(dataloader) * config.epochs
    warmup_steps = len(dataloader) * 2

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_scheduler(step, warmup_steps=warmup_steps, total_training_steps=total_training_steps))

    betas = get_betas(1e-4, .05, config.T).to(device)
    alphas = 1 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)

    print("Starting training...")
    print("Dataset of length: " ,len(dataloader))

    training_step = 0
    for epoch in range(config.epochs):
        total_loss = 0
        batch_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for batch, cond in batch_loop:
            x0 = batch.to(device)
            condition = cond.to(device)
            t = torch.randint(0, config.T, (x0.shape[0],)).long().to(device)
            noise = 1-alpha_cum[t]
            noisy_inputs = add_noise(x0, noise, config.vocab_size)
            optimizer.zero_grad()

            loss = get_loss(model, noisy_inputs, x0, betas, config.vocab_size, t, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_loop.set_postfix(loss=loss.item())
            training_step += 1

            if training_step % 100 == 0: 
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": training_step
                })

            if training_step % config.val_interval == 0:
                model.eval()
                val_loss_total = 0
                with torch.no_grad():
                    val_batch_loop = tqdm(val_loader, desc="validation", leave=False)
                    for val_batch, val_cond in val_batch_loop:
                        val_batch = val_batch.to(device)
                        val_cond = val_cond.to(device)
                        
                        t_val = torch.randint(0, config.T, (val_batch.shape[0],)).long().to(device)
                        noise_val = 1 - alpha_cum[t_val]
                        noisy_inputs_val = add_noise(val_batch, noise_val, config.vocab_size)

                        val_loss = get_loss(model, noisy_inputs_val, val_batch, betas, config.vocab_size, t_val, val_cond)
                        val_loss_total += val_loss.item()
                    
                val_loss = val_loss_total / len(val_loader)
                print(f"\nStep {training_step}, Validation Loss: {val_loss:.4f}\n")

                wandb.log({
                    "val_loss": val_loss,
                    "step": training_step
                })
                
                save_path = os.path.join(checkpoint_dir, f"melody_diffusor_model_step{training_step}.pth")
                torch.save(model.state_dict(), save_path)
                
                model.train()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

        wandb.log({
            "avg_epoch_loss": avg_loss,
            "epoch": epoch + 1
        })

    wandb.finish()


        
        
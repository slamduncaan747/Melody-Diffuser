
from model import MelodyDiffusor
from diffusion_utils import get_betas, add_noise
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import sys
import pickle
import pretty_midi
import pygame


def play_and_show_melody(pitch_seq, filename="output_melody.mid"):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    current_time = 0.0
    step_duration = 0.25 
    current_note_obj = None

    for pitch in pitch_seq:
        pitch = int(pitch)
        
        if pitch == 129: 
            if current_note_obj:
                instrument.notes.append(current_note_obj)
                current_note_obj = None
            current_time += step_duration
            
        elif pitch == 128:
            if current_note_obj:
                current_note_obj.end += step_duration
            current_time += step_duration
            
        else:
            if current_note_obj:
                instrument.notes.append(current_note_obj)
            
            current_note_obj = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=current_time,
                end=current_time + step_duration
            )
            current_time += step_duration
            
    if current_note_obj:
        instrument.notes.append(current_note_obj)

    midi_data.instruments.append(instrument)
    
    midi_data.write(filename)
    print(f"MIDI file saved to {filename}")
    
    print("Playing melody...")
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"error")
    finally:
        pygame.mixer.quit()


def get_condition_from_video():
    print("Starting webcam... Press 'q' to quit.")
    beat_interval = 0.5
    min_x_interval = 25
    seq_len = 63
    small_jump, medium_jump = 25, 300

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.75, min_tracking_confidence=0.75,
        model_complexity=1
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    last_beat_time = time.time()
    flash_on = False
    graph = []
    encoding = []

    while len(encoding) < seq_len:
        success, img = cap.read()
        if not success: 
            break
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, c = img.shape
        x, y = 0, 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    lm = hand_landmarks.landmark[8]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (x, y), 8, (0, 255, 0), cv2.FILLED)

        current_time = time.time()
        if current_time - last_beat_time >= beat_interval:
            is_hold = False
            if len(graph) == 0:
                graph.append((current_time, y))
            elif abs(x - graph[-1][0]) < min_x_interval:
                is_hold = True
                graph.append((current_time, graph[-1][1]))
            else:
                graph.append((current_time, y))
            
            last_beat_time = current_time
            flash_on = not flash_on

            if len(graph) > 1:
                jump = graph[-1][1] - graph[-2][1]
                
                if y > (h * 0.9): 
                    encoding.append(3) 
                elif is_hold:
                    encoding.append(4)
                elif jump == 0:
                    encoding.append(4)  
                elif 0 < jump <= small_jump:
                    encoding.append(5)  
                elif small_jump < jump <= medium_jump:
                    encoding.append(6)  
                elif jump > medium_jump:
                    encoding.append(7)  
                elif -small_jump <= jump < 0:
                    encoding.append(2)  
                elif -medium_jump <= jump <= -small_jump:
                    encoding.append(1)  
                else: 
                    encoding.append(0)  
                
                print(f"Beat {len(encoding)}/{seq_len} -> Token: {encoding[-1]}")

        color = (255, 255, 255) if flash_on else (0, 0, 0)
        cv2.circle(img, (w - 30, 30), 25, color, cv2.FILLED)
        cv2.putText(img, f"Gestures: {len(encoding)}/{seq_len}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Live Gesture Input", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    if len(encoding) < seq_len: 
        return None
    return encoding


def get_condition_from_file(filepath):
    try:
        with open(filepath, "rb") as f:
            all_gestures = pickle.load(f)

        idx = random.randint(0, len(all_gestures) - 1)
        cond_list = all_gestures[idx]

        if len(cond_list) != 63:
            cond_list = list(cond_list[:63])
            while len(cond_list) < 63:
                cond_list.append(0)
        return list(cond_list)

    except FileNotFoundError:
        print("Condition file not found.")
        return None
    except Exception as e:
        print(f"Error loading condition from file: {e}")
        return None

def sample(model, cond, T, alpha_cum, vocab_size, temperature=1.0, p=0.95, w=3.0, device='cpu'):
    B, L = 1, 64 
    x = torch.randint(0, vocab_size, (B, L), device=device)
    for t in tqdm(reversed(range(T)), desc="Sampling", total=T):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            conditioned_logits = model(x, t_tensor, cond)
            unconditioned_logits = model(x, t_tensor, None) 
        
        logits = unconditioned_logits + w * (conditioned_logits - unconditioned_logits)
        probs = torch.softmax(logits / temperature, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        orig_order_probs = torch.zeros_like(sorted_probs)
        orig_order_probs.scatter_(-1, sorted_indices, sorted_probs)

        x_0_pred_flat = torch.multinomial(orig_order_probs.view(-1, vocab_size), 1)
        x_0_pred = x_0_pred_flat.view(B, L)
        
        if t == 0:
            x = x_0_pred
            break
            
        noise_prob_tm1 = (1 - alpha_cum[t-1]).view(-1, 1)
        x = add_noise(x_0_pred, noise_prob_tm1, vocab_size)
    return x

if __name__ == "__main__":
    
    CHECKPOINT_PATH = "checkpoints/Melody-Diffuser.pth" 
    
    GESTURE_FILE_PATH = "best_gestures.pkl" 
    T = 64
    VOCAB_SIZE = 130
    
    TEMPERATURE = 1
    TOP_P = .95
    CFG_SCALE = 3


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MelodyDiffusor(
        vocab_size=VOCAB_SIZE, seq_len=64, dim=512, 
        n_layers=6, n_heads=8, ffn_inner_dim=2048, dropout=0.1
    ).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"--- ERROR: Checkpoint file not found at: {CHECKPOINT_PATH} ---")
        sys.exit(1)
    
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Successfully loaded checkpoint from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.eval()

    betas = get_betas(1e-4, .05, T).to(device)
    alphas = 1 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)
    
    cond_list = None
    print("\n--- Select Conditioning Mode ---")
    print("(1) Live Video (via webcam)")
    print("(2) Load Random from File")
    
    while True:
        mode = input("Enter 1 or 2: ")
        if mode == '1':
            cond_list = get_condition_from_video()
            break
        elif mode == '2':
            cond_list = get_condition_from_file(GESTURE_FILE_PATH)
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    if cond_list:
        if len(cond_list) != 63:
            print(f"Error: Gesture list length {len(cond_list)}, expected 63.")
        else:
            cond_tensor = torch.tensor(cond_list,dtype=torch.long).to(device).unsqueeze(0)
            print(f"Sampling with condition: {cond_tensor}")
            
            sampled_sequence = sample(
                model=model, cond=cond_tensor,T=T, alpha_cum=alpha_cum, 
                vocab_size=VOCAB_SIZE, temperature=TEMPERATURE,
                p=TOP_P, w=CFG_SCALE, device=device
            )
            
            sampled_np = sampled_sequence.squeeze().cpu().numpy()
            print("\n--- Sampled Sequence (Raw) ---")
            print(np.array2string(sampled_np, separator=', '))
            
            play_and_show_melody(sampled_np, "output_melody.mid")

    else:
        print("No condition was generated. Exiting.")
from music21 import *
import kagglehub
import os
from magenta.music import NoteSequence
import pickle
from tqdm import tqdm

# Download MIDI files
path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
midi_files = []
note_sequences = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".mid"):
            midi_files.append(os.path.join(root, file))

# Process MIDI file into notes
def process_midi(file_path):
    try:
        midi_data = converter.parse(file_path)
        notes = []
        for element in midi_data.flat.notes:
            if isinstance(element, note.Note):
                notes.append((element.pitch, float(element.offset), float(element.quarterLength+element.offset)))
            elif isinstance(element, chord.Chord):
                pitch = element.pitches[0]
                notes.append((pitch, float(element.offset), float(element.quarterLength+element.offset)))
        return notes
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# Process notes into Note Sequence Object
def process_notes(notes):
    note_sequence = NoteSequence()
    for pitch, start, end in notes:
        note = note_sequence.notes.add()
        note.pitch = pitch.midi
        note.start_time = start
        note.end_time = end
    note_sequence.total_time = max(end_time for _, _, end_time in notes) if notes else 0
    note_sequence.tempos.add(qpm=120)
    return note_sequence
    
# Process all midi files into Note Sequence objects
for file in tqdm(midi_files):
    notes = process_midi(file)
    if len(notes) == 0:
        continue

    note_sequence = process_notes(notes)
    if note_sequence.total_time < 5.0:
        continue

    note_sequences.append(note_sequence)

# Save Note Sequences
for i, ns in enumerate(note_sequences):
    with open(f"note_sequences/ns_{i:05d}.pkl", "wb") as f:
        pickle.dump(ns, f)
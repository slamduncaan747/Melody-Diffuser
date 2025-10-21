import os
import glob
import pickle
import tensorflow as tf

context_features = {
    "toussaint": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "note_density": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "pitch_range": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "contour": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "note_change_ratio": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "dynamic_range": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "len_longest_rep_section": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "repetitive_section_ratio": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "ratio_hold_note_steps": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "ratio_note_off_steps": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "unique_notes_ratio": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "unique_bigrams_ratio": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "unique_trigrams_ratio": tf.io.FixedLenFeature([], tf.float32, default_value=0),
}
sequence_features = {
    "pitch_seq": tf.io.VarLenFeature(tf.int64),
}

def parse_sequence_example(serialized_example):
    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    pitch_sequence = tf.sparse.to_dense(sequence_parsed["pitch_seq"]).numpy()
    return pitch_sequence

def process_directory_to_pickle(tfrecord_dir, pickle_path):
    tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    parsed_data = []
    for tf_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tf_file)
        for raw_record in dataset:
            pitch_seq = parse_sequence_example(raw_record)
            parsed_data.append(pitch_seq)

    with open(pickle_path, "wb") as f:
        pickle.dump(parsed_data, f)


process_directory_to_pickle("train/4_bars_melodies_pitchseq_validation", "melodies_test.pkl")

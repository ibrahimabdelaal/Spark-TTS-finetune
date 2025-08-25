import os
import csv
import json
from tqdm import tqdm
import argparse
import torch
import numpy as np
from pathlib import Path
import traceback # For detailed error logging
import torch.multiprocessing as mp # For GPU parallel processing

# Assuming these imports work because BiCodecTokenizer and its dependencies are in the Python path
# or the sparktts package is installed.
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.file import load_config
from sparktts.utils.audio import load_audio
# from sparktts.models.bicodec import BiCodec # BiCodec is used internally by BiCodecTokenizer

# Helper function to format the tokenized output string (remains the same)
def format_prompt_text(text, global_token_ids, semantic_token_ids):
    if isinstance(global_token_ids, torch.Tensor):
        global_token_ids = global_token_ids.squeeze().cpu().tolist()
    if isinstance(semantic_token_ids, torch.Tensor):
        semantic_token_ids = semantic_token_ids.squeeze().cpu().tolist()

    global_tokens_str = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids]
    )
    semantic_tokens_str = "".join(
        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids]
    )
    inputs_list = [
        "<|task_tts|>", "<|start_content|>", text, "<|end_content|>",
        "<|start_global_token|>", global_tokens_str, "<|end_global_token|>",
        "<|start_semantic_token|>", semantic_tokens_str, "<|end_semantic_token|>",
        "<|im_end|>"
    ]
    return "".join(inputs_list)

class AudioPromptDataset:
    def __init__(self, model_name_or_path, device_str):
        self.current_device_str = device_str # Store for tqdm description
        self.device = torch.device(device_str)
        self.audio_tokenizer = BiCodecTokenizer(
            model_dir=Path(model_name_or_path),
            device=self.device
        )

    def _process_batch_data(self, audio_paths_batch, texts_batch):
        # This is the corrected version from our previous discussion
        processed_prompts = []
        loaded_raw_wavs_list = []
        ref_wav_tensors_list = []

        if not audio_paths_batch:
            return []
        try:
            for audio_path_str in audio_paths_batch:
                audio_path = Path(audio_path_str)
                wav_np = load_audio(
                    audio_path,
                    sampling_rate=self.audio_tokenizer.config["sample_rate"],
                    volume_normalize=self.audio_tokenizer.config["volume_normalize"],
                )
                if wav_np.ndim > 1: # Ensure wav_np is 1D (mono)
                    wav_np_squeezed = wav_np.squeeze()
                    if wav_np_squeezed.ndim == 1:
                        wav_np = wav_np_squeezed
                    elif wav_np.ndim > 1 and wav_np.shape[0] == 1:
                         wav_np = wav_np[0]
                    elif wav_np.ndim > 1 and wav_np.shape[1] == 1:
                         wav_np = wav_np[:,0]
                    else: # Fallback if still not 1D, take first channel
                        print(f"Warning: Audio {audio_path_str} is not mono after squeeze, shape: {wav_np.shape}. Using first channel.")
                        if wav_np.shape[0] < wav_np.shape[1] and wav_np.shape[0] != 0 :
                             wav_np = wav_np[0,:]
                        elif wav_np.shape[1] < wav_np.shape[0] and wav_np.shape[1] != 0 :
                             wav_np = wav_np[:,0]
                        elif wav_np.shape[0] == 0 or wav_np.shape[1] == 0:
                            raise ValueError(f"Audio {audio_path_str} resulted in empty array after trying to get mono.")
                        else: # If shape[0] == shape[1] > 1, it's ambiguous, take first.
                            wav_np = wav_np[0,:]


                loaded_raw_wavs_list.append(wav_np)
                ref_clip_np = self.audio_tokenizer.get_ref_clip(wav_np)
                ref_wav_tensor_single = torch.from_numpy(ref_clip_np).unsqueeze(0).float()
                ref_wav_tensors_list.append(ref_wav_tensor_single)

            batched_ref_wavs = torch.cat(ref_wav_tensors_list, dim=0).to(self.device)
            batch_payload = {
                "wav": loaded_raw_wavs_list,
                "ref_wav": batched_ref_wavs
            }
            batched_global_tokens, batched_semantic_tokens = self.audio_tokenizer.tokenize_batch(batch_payload)

            for i in range(batched_global_tokens.size(0)):
                inputs_str = format_prompt_text(texts_batch[i], batched_global_tokens[i], batched_semantic_tokens[i])
                processed_prompts.append({"text": inputs_str})
        except Exception as e:
            print(f"Error processing batch on {self.current_device_str}: {e}. This batch will be skipped.")
            traceback.print_exc()
        return processed_prompts

    def load_all_metadata(self, data_dir, csv_delimiter='|'):
        """Loads all metadata and returns a list of valid entries."""
        all_entries = []
        metadata_path = os.path.join(data_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Error: metadata.csv not found at {metadata_path}")
            return []

        with open(metadata_path, mode="r", encoding="utf-8") as file_in:
            reader = csv.reader(file_in, delimiter=csv_delimiter)
            print("Reading metadata and checking audio files (main process)...")
            for row in tqdm(reader, desc="Scanning metadata"):
                try:
                    audio_path_or_name, text = row[0], row[1]
                except IndexError:
                    continue
                if os.path.isabs(audio_path_or_name):
                    audio_path = audio_path_or_name
                else:
                    audio_path = os.path.join(data_dir, "wavs", audio_path_or_name + ".wav")
                if not os.path.exists(audio_path):
                    continue
                all_entries.append({"path": audio_path, "text": text})
        
        if not all_entries:
            print("No valid entries found to process after checking paths.")
        else:
            print(f"Found {len(all_entries)} valid audio entries to process.")
        return all_entries

    def tokenize_chunk_and_save(self, entries_chunk, output_filepath, batch_size=16):
        """Tokenizes a given chunk of entries and saves to a specific file."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True) # Ensure output dir for temp file exists
        
        with open(output_filepath, "w", encoding="utf-8") as f_out:
            for i in tqdm(range(0, len(entries_chunk), batch_size), desc=f"Processing on {self.current_device_str}", position=int(self.current_device_str.split(':')[-1])): # Use rank for position
                batch_entries = entries_chunk[i:i+batch_size]
                audio_paths_current_batch = [item['path'] for item in batch_entries]
                texts_current_batch = [item['text'] for item in batch_entries]
                
                if not audio_paths_current_batch:
                    continue
                processed_prompts = self._process_batch_data(audio_paths_current_batch, texts_current_batch)
                for prompt_data in processed_prompts:
                    if prompt_data and not prompt_data.get("error"):
                        f_out.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        print(f"Chunk processing complete on {self.current_device_str}. Output: {output_filepath}")


def worker_fn(rank, world_size, args_dict):
    """Worker function for each process."""
    gpu_id = rank
    torch.cuda.set_device(gpu_id)
    device_str = f"cuda:{gpu_id}"
    print(f"Worker {rank}/{world_size} started, using {device_str}")

    entries_chunk = args_dict['data_chunks'][gpu_id]
    temp_output_file = args_dict['temp_files'][gpu_id]
    
    processor = AudioPromptDataset(
        model_name_or_path=args_dict['model_name_or_path'],
        device_str=device_str
    )
    processor.tokenize_chunk_and_save(
        entries_chunk,
        temp_output_file,
        batch_size=args_dict['batch_size']
    )
    print(f"Worker {rank}/{world_size} finished.")

def merge_files(temp_files, final_output_path):
    print(f"Merging {len(temp_files)} temporary files into {final_output_path}...")
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as f_out:
        for temp_file in tqdm(temp_files, desc="Merging files"):
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        f_out.write(line)
                try:
                    os.remove(temp_file) # Clean up temp file
                except OSError as e:
                    print(f"Warning: Could not remove temp file {temp_file}: {e}")
            else:
                print(f"Warning: Temp file {temp_file} not found for merging.")
    print("Merging complete.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Recommended for CUDA with multiprocessing

    parser = argparse.ArgumentParser(description="Tokenize audio dataset for TTS in batches using BiCodecTokenizer and multiple GPUs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory (must contain metadata.csv).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the final tokenized output directory.")
    parser.add_argument("--model_name_or_path", type=str, default="pretrained_models/Spark-TTS-0.5B", help="Model name or path for BiCodecTokenizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--csv_delimiter", type=str, default='|', help="Delimiter used in the metadata.csv file.")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use. 0 means use all available. Default is 0.")


    args = parser.parse_args()

    if args.num_gpus == 0:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.num_gpus, torch.cuda.device_count())

    if world_size == 0:
        print("No GPUs available or specified for use. Please check your CUDA setup or --num_gpus argument.")
        exit()
    
    print(f"Using {world_size} GPU(s).")

    # Load metadata once in the main process
    # Use a temporary processor instance on CPU for this to avoid unnecessary GPU init here
    # Note: BiCodecTokenizer init loads models, so this is not ideal.
    # Better to just have a static method or function for loading metadata if possible.
    # For now, we'll instantiate it, but it's a bit heavy for just metadata.
    # A lighter way would be to move the CSV reading logic out of the class.
    # Let's create a light version of metadata loading here:
    
    all_entries = []
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        exit()
    with open(metadata_path, mode="r", encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=args.csv_delimiter)
        print("Reading metadata and checking audio files (main process)...")
        for row in tqdm(reader, desc="Scanning metadata"):
            try:
                audio_path_or_name, text = row[0], row[1]
            except IndexError: continue
            if os.path.isabs(audio_path_or_name):
                audio_path = audio_path_or_name
            else:
                audio_path = os.path.join(args.data_dir, "wavs", audio_path_or_name + ".wav")
            if not os.path.exists(audio_path): continue
            all_entries.append({"path": audio_path, "text": text})
    
    if not all_entries:
        print("No valid entries found to process.")
        exit()
    print(f"Found {len(all_entries)} valid audio entries to distribute among {world_size} GPUs.")


    # Split all_entries for each GPU
    entries_per_gpu_chunk = [[] for _ in range(world_size)]
    for i, entry in enumerate(all_entries):
        entries_per_gpu_chunk[i % world_size].append(entry)

    # Prepare temporary output file paths
    # Ensure output_dir exists for temp files
    os.makedirs(args.output_dir, exist_ok=True)
    temp_output_files = [os.path.join(args.output_dir, f"temp_output_gpu_{i}.jsonl") for i in range(world_size)]

    spawn_args_dict = {
        'model_name_or_path': args.model_name_or_path,
        'data_chunks': entries_per_gpu_chunk,
        'temp_files': temp_output_files,
        'batch_size': args.batch_size,
        # csv_delimiter is not needed by worker_fn as data is already loaded
    }

    mp.spawn(worker_fn,
             args=(world_size, spawn_args_dict),
             nprocs=world_size,
             join=True)

    # Merge temporary files
    final_output_filename = os.path.basename(args.data_dir) + ".jsonl" # Consistent with single GPU version
    final_output_path = os.path.join(args.output_dir, final_output_filename)
    merge_files(temp_output_files, final_output_path)

    print("All processing complete.")
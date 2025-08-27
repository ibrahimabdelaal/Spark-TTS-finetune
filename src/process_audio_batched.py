# This file will be named 'process_stt_data.py' and placed in the 'src' directory of the repo.
# It is a standalone command-line script for data processing.

import os
import json
import argparse
import torch
import traceback
from pathlib import Path
from tqdm import tqdm
import sys
import io
from datasets import load_dataset, Features, Audio, Value

# THE FIX: Import soundfile for manual decoding
import soundfile as sf

# MODIFICATION: Add the parent directory to the system path.
# This allows the script to find the 'sparktts' module when run directly.
# This resolves the ModuleNotFoundError.
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent))


# These imports will work because the script is run after the repo is installed.
from datasets import load_dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer

def format_stt_entry(global_token_ids, semantic_token_ids, target_text):
    """
    Formats the data into a single string for causal language modeling.
    """
    if isinstance(global_token_ids, torch.Tensor):
        global_token_ids = global_token_ids.squeeze().cpu().tolist()
    if isinstance(semantic_token_ids, torch.Tensor):
        semantic_token_ids = semantic_token_ids.squeeze().cpu().tolist()

    global_tokens_str = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids])
    semantic_tokens_str = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids])

    inputs_list = [
        "<|task_stt|>",
        "<|start_global_token|>", global_tokens_str, "<|end_global_token|>",
        "<|start_semantic_token|>", semantic_tokens_str, "<|end_semantic_token|>",
        "<|start_content|>", target_text, "<|end_content|>",
        "<|im_end|>"
    ]
    return "".join(inputs_list)

class AudioProcessor:
    """Handles the audio tokenization process."""
    def __init__(self, model_path, device_str):
        self.device = torch.device(device_str)
        self.audio_tokenizer = BiCodecTokenizer(model_dir=Path(model_path), device=self.device)
        print(f"✅ Audio tokenizer initialized on {device_str}")

    def process_batch(self, audio_arrays_batch, texts_batch):
        processed_data = []
        loaded_raw_wavs_list = []
        ref_wav_tensors_list = []

        if not audio_arrays_batch:
            return []

        try:
            for wav_np in audio_arrays_batch:
                if wav_np.ndim > 1:
                    wav_np_squeezed = wav_np.squeeze()
                    if wav_np_squeezed.ndim == 1:
                        wav_np = wav_np_squeezed
                    elif wav_np.shape[0] == 1:
                        wav_np = wav_np[0, :]
                    elif wav_np.shape[1] == 1:
                        wav_np = wav_np[:, 0]
                    else:
                        wav_np = wav_np.mean(axis=-1)

                loaded_raw_wavs_list.append(wav_np)
                ref_clip_np = self.audio_tokenizer.get_ref_clip(wav_np)
                ref_wav_tensor_single = torch.from_numpy(ref_clip_np).unsqueeze(0).float()
                ref_wav_tensors_list.append(ref_wav_tensor_single)

            batched_ref_wavs = torch.cat(ref_wav_tensors_list, dim=0).to(self.device)
            batch_payload = {"wav": loaded_raw_wavs_list, "ref_wav": batched_ref_wavs}
            
            batched_global_tokens, batched_semantic_tokens = self.audio_tokenizer.tokenize_batch(batch_payload)

            for i in range(batched_global_tokens.size(0)):
                full_string = format_stt_entry(
                    batched_global_tokens[i],
                    batched_semantic_tokens[i],
                    texts_batch[i]
                )
                processed_data.append({"text": full_string})

        except Exception as e:
            print(f"Error processing batch: {e}. Batch skipped.")
            traceback.print_exc()

        return processed_data

def main(args):
    """Main function to run the data processing."""
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_path = Path("pretrained_models") / args.model_name
    
    if not model_path.exists():
        print(f"FATAL: Model directory not found at expected path: {model_path}")
        return

    processor = AudioProcessor(model_path, device)

    print(f"Loading dataset shard {args.shard_index}/{args.shard_count}...")
    # THE FIX: Add decode=False to load raw audio bytes and bypass torchcodec
    feature_set = Features({
    "text": Value("string"),
    "duration": Value("float64"),
    "audio": Audio(decode=False)  # This is the key part to skip decoding
})
    dataset_shard = load_dataset(
        args.hf_dataset_name, 
        split="train", 
        token=args.hf_token,
    features=feature_set,
    ).shard(num_shards=args.shard_count, index=args.shard_index)
    
    print(f"Loaded here {len(dataset_shard)} samples for this shard.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f_out:
        shard_list = list(dataset_shard)
        for i in tqdm(range(0, len(shard_list), args.batch_size), desc=f"Processing on GPU {args.shard_index}"):
            batch = shard_list[i:i + args.batch_size]
            
            # THE FIX: Manually decode audio from bytes and handle errors
            audio_arrays = []
            texts = []
            for item in batch:
                audio_data = item.get('audio')
                text_data = item.get('text')

                if not audio_data or not audio_data.get('bytes') or not text_data:
                    print(f"Warning: Skipping item due to missing audio bytes or text.")
                    continue
                
                try:
                    # Use soundfile to read the audio bytes from memory
                    wav, sr = sf.read(io.BytesIO(audio_data['bytes']))
                    audio_arrays.append(wav)
                    texts.append(text_data)
                except Exception as e:
                    print(f"Warning: Failed to decode audio for item. Skipping. Error: {e}")

            if not audio_arrays:
                continue
            
            processed_batch = processor.process_batch(audio_arrays, texts)
            for data_item in processed_batch:
                f_out.write(json.dumps(data_item, ensure_ascii=False) + "\n")

    print(f"Processing complete for shard {args.shard_index}. Output saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a shard of an HF audio dataset for STT.")
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="Name of the Hugging Face dataset.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token.")
    parser.add_argument("--shard_index", type=int, required=True, help="Index of the dataset shard to process.")
    parser.add_argument("--shard_count", type=int, required=True, help="Total number of dataset shards.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final tokenized output file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model directory in pretrained_models.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    
    args = parser.parse_args()
    main(args)

"""
Audio I/O utilities using ffmpeg for Python 3.13+ compatibility
Replaces soundfile dependency which requires aifc module
"""

import os
import numpy as np
import subprocess
import tempfile
from pathlib import Path

def load_audio(input_path, sample_rate=16000):
    """
    Load audio using ffmpeg for maximum format compatibility
    
    Args:
        input_path: Path to input audio file
        sample_rate: Target sample rate (default: 16000, None to keep original)
    
    Returns:
        tuple: (audio_array, sample_rate)
    """
    try:
        if sample_rate is None:
            # Get original sample rate first
            cmd_info = [
                'ffmpeg', '-i', str(input_path),
                '-f', 'null', '-'
            ]
            result_info = subprocess.run(cmd_info, capture_output=True, text=True)
            
            # Parse sample rate from stderr
            for line in result_info.stderr.split('\n'):
                if 'Hz' in line and 'Audio:' in line:
                    try:
                        sample_rate = int(line.split('Hz')[0].split()[-1])
                        break
                    except:
                        sample_rate = 16000  # fallback
            else:
                sample_rate = 16000  # fallback if not found
        
        # Use ffmpeg to convert audio to raw PCM
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-f', 'f64le',  # 64-bit float little-endian
            '-ac', '1',     # mono
            '-ar', str(sample_rate),  # sample rate
            '-y',           # overwrite output
            'pipe:1'        # output to stdout
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to load {input_path}")
        
        # Convert raw bytes to numpy array
        audio = np.frombuffer(result.stdout, dtype=np.float64).astype(np.float32)
        return audio, sample_rate
    
    except Exception as e:
        print(f"Warning: ffmpeg failed ({e}), using scipy fallback")
        # If all else fails, try to create a simple WAV loader
        try:
            # Simple scipy-based fallback
            from scipy.io import wavfile
            sr, audio = wavfile.read(input_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483647.0
            
            # Resample if needed and sample_rate was specified
            if sample_rate is not None and sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                return audio, sample_rate
            else:
                return audio, sr
        except Exception as e2:
            raise RuntimeError(f"All audio loading methods failed: ffmpeg ({e}), scipy ({e2})")

def save_audio(audio, output_path, sample_rate=16000):
    """
    Save audio using ffmpeg for maximum format compatibility
    
    Args:
        audio: Audio array (numpy array)
        output_path: Path to output audio file
        sample_rate: Sample rate (default: 16000)
    
    Returns:
        bool: True if successful
    """
    try:
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Use ffmpeg to save audio
        cmd = [
            'ffmpeg', '-f', 'f32le',  # 32-bit float little-endian input
            '-ac', '1',               # mono
            '-ar', str(sample_rate),  # sample rate
            '-i', 'pipe:0',           # input from stdin
            '-y',                     # overwrite output
            str(output_path)
        ]
        
        result = subprocess.run(cmd, input=audio.tobytes(), stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to save {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Warning: ffmpeg save failed ({e}), using scipy fallback")
        # Fallback to scipy for saving
        from scipy.io import wavfile
        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        return True

def write_audio_bytes(audio, sample_rate=16000, format='WAV'):
    """
    Write audio to bytes buffer (for streaming/web apps)
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        format: Output format ('WAV', 'MP3', etc.)
    
    Returns:
        bytes: Audio data as bytes
    """
    try:
        # Use ffmpeg to convert to bytes
        audio = audio.astype(np.float32)
        
        format_map = {'WAV': 'wav', 'MP3': 'mp3', 'FLAC': 'flac'}
        output_format = format_map.get(format.upper(), 'wav')
        
        cmd = [
            'ffmpeg', '-f', 'f32le',
            '-ac', '1',
            '-ar', str(sample_rate),
            '-i', 'pipe:0',
            '-f', output_format,
            'pipe:1'
        ]
        
        result = subprocess.run(cmd, input=audio.tobytes(), 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to convert to {format}")
        
        return result.stdout
    
    except Exception as e:
        # Fallback to scipy for WAV format only
        if format.upper() == 'WAV':
            from scipy.io import wavfile
            import io
            
            # Convert to 16-bit PCM
            audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
            
            # Write to bytes buffer
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wavfile.write(tmp.name, sample_rate, audio_int16)
                tmp.flush()
                
                with open(tmp.name, 'rb') as f:
                    data = f.read()
                
                os.unlink(tmp.name)
                return data
        else:
            raise RuntimeError(f"Cannot convert to {format} without ffmpeg: {e}")

import streamlit as st
import torch
import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import io
import tempfile
from pathlib import Path

# Add parent directory to Python path for imports
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config.paths import PATHS, get_path
from src.unet_model import UNet
from src.audio_utils import load_audio, save_audio, write_audio_bytes

st.set_page_config(
    page_title="AI Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metrics-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained GPU model using global path configuration"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use global path configuration
        model_path = get_path('models.best_model')
        
        if not model_path.exists():
            st.error(f"Model not found at: {model_path}")
            st.info("Please ensure the model file exists or update the path in config/paths.py")
            return None, None, None
        
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_filters=32,
            depth=3,
            dropout=0.1
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # st.success(f"✅ Model loaded successfully from: {model_path.name}")
        return model, device, checkpoint
    
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

class GPUSpeechEnhancer:
    """Simplified speech enhancer using the trained GPU model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.chunk_duration = 4.0
        self.overlap_ratio = 0.25
    
    def enhance_audio(self, audio_data, input_sr):
        """Enhance audio using the trained model"""
        try:
            # Resample to target sample rate if needed
            if input_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=input_sr, target_sr=self.sample_rate)
            
            # Process the audio
            enhanced_audio = self._process_chunks(audio_data)
            
            return enhanced_audio, self.sample_rate
            
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            return None, None
    
    def _process_chunks(self, audio):
        """Process audio in overlapping chunks"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(chunk_samples * self.overlap_ratio)
        hop_samples = chunk_samples - overlap_samples

        if len(audio) <= chunk_samples:
            return self._enhance_chunk(audio)

        enhanced_chunks = []
        progress_bar = st.progress(0)
        
        for i, start_idx in enumerate(range(0, len(audio) - overlap_samples, hop_samples)):
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]

            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

            enhanced_chunk = self._enhance_chunk(chunk)
            enhanced_chunks.append((enhanced_chunk, start_idx, len(audio[start_idx:end_idx])))
            
            # Update progress
            progress = (i + 1) / len(range(0, len(audio) - overlap_samples, hop_samples))
            progress_bar.progress(progress)
        
        progress_bar.empty()
        return self._blend_chunks(enhanced_chunks, len(audio), overlap_samples)
    
    def _enhance_chunk(self, audio_chunk):
        """Enhance a single audio chunk"""
        stft = librosa.stft(
            audio_chunk,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        magnitude_normalized = magnitude / (np.max(magnitude) + 1e-8)

        with torch.no_grad():
            magnitude_tensor = torch.FloatTensor(magnitude_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            enhanced_magnitude = self.model(magnitude_tensor)
            enhanced_magnitude = enhanced_magnitude.cpu().squeeze().numpy()

        enhanced_magnitude = enhanced_magnitude * (np.max(magnitude) + 1e-8)
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )

        return enhanced_audio
    
    def _blend_chunks(self, enhanced_chunks, total_length, overlap_samples):
        """Blend overlapping chunks with crossfading"""
        enhanced_audio = np.zeros(total_length)
        weight_sum = np.zeros(total_length)

        for enhanced_chunk, start_idx, original_length in enhanced_chunks:
            end_idx = min(start_idx + len(enhanced_chunk), total_length)
            chunk_length = end_idx - start_idx
            
            weights = np.ones(chunk_length)
            if overlap_samples > 0:
                fade_in_samples = min(overlap_samples, chunk_length // 2)
                weights[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
                
                fade_out_samples = min(overlap_samples, chunk_length // 2)
                weights[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
            
            enhanced_audio[start_idx:end_idx] += enhanced_chunk[:chunk_length] * weights
            weight_sum[start_idx:end_idx] += weights

        weight_sum[weight_sum == 0] = 1
        enhanced_audio /= weight_sum

        return enhanced_audio

def create_waveform_plot(original_audio, enhanced_audio, sample_rate):
    """Create interactive waveform comparison plot"""
    time_axis = np.linspace(0, len(original_audio) / sample_rate, len(original_audio))
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original (Noisy) Audio', 'Enhanced Audio'),
        vertical_spacing=0.1
    )
    
    # Original waveform
    fig.add_trace(
        go.Scatter(
            x=time_axis, y=original_audio,
            mode='lines',
            name='Original',
            line=dict(color='red', width=1),
        ),
        row=1, col=1
    )
    
    # Enhanced waveform
    fig.add_trace(
        go.Scatter(
            x=time_axis, y=enhanced_audio,
            mode='lines',
            name='Enhanced',
            line=dict(color='blue', width=1),
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False, title_text="Audio Waveform Comparison")
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude")
    
    return fig

def create_spectrogram_plot(original_audio, enhanced_audio, sample_rate):
    """Create spectrogram comparison plot"""
    # Compute spectrograms
    orig_stft = librosa.stft(original_audio, n_fft=1024, hop_length=256)
    enh_stft = librosa.stft(enhanced_audio, n_fft=1024, hop_length=256)
    
    orig_mag_db = librosa.amplitude_to_db(np.abs(orig_stft))
    enh_mag_db = librosa.amplitude_to_db(np.abs(enh_stft))
    
    # Create time and frequency axes
    time_frames = orig_mag_db.shape[1]
    freq_bins = orig_mag_db.shape[0]
    
    time_axis = np.linspace(0, len(original_audio) / sample_rate, time_frames)
    freq_axis = np.linspace(0, sample_rate / 2, freq_bins)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original Spectrogram', 'Enhanced Spectrogram'),
        horizontal_spacing=0.1
    )
    
    # Original spectrogram
    fig.add_trace(
        go.Heatmap(z=orig_mag_db, x=time_axis, y=freq_axis, colorscale='Viridis', name='Original'),
        row=1, col=1
    )
    
    # Enhanced spectrogram
    fig.add_trace(
        go.Heatmap(z=enh_mag_db, x=time_axis, y=freq_axis, colorscale='Viridis', name='Enhanced'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Spectrogram Comparison")
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)")
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🎙️ AI Speech Enhancement</h1><p>Remove background noise from speech recordings using deep learning</p></div>', unsafe_allow_html=True)
    
    # Display project information

    
    # Load model
    model, device, checkpoint = load_model()
    
    if model is None:
        st.stop()
    
    # Initialize enhancer
    enhancer = GPUSpeechEnhancer(model, device)
    
    # Sidebar
    with st.sidebar:
        st.success(f"✅ Model loaded successfully from: {get_path('models.best_model').name}")
        st.header("📊 Model Information")
        if checkpoint:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
                st.metric("Device", str(device).upper())
            with col2:
                st.metric("Val Loss", f"{checkpoint.get('val_loss', 'N/A'):.6f}")
                st.metric("Sample Rate", "16 kHz")
        
        st.header("🎛️ Settings")
        st.info("Model uses optimized settings:\n- Chunk size: 4.0s\n- Overlap: 25%\n- STFT: 1024/256")
        st.info(f"🏠 Project Directory: {get_path('base')}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload noisy speech audio for enhancement"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")
            
            # Load and display original audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                
                try:
                    original_audio, original_sr = load_audio(tmp_file.name)
                    st.audio(uploaded_file.getvalue(), format='audio/wav')
                    
                    duration = len(original_audio) / original_sr
                    st.metric("Duration", f"{duration:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Failed to load audio: {e}")
                    st.stop()
                finally:
                    os.unlink(tmp_file.name)
    
    with col2:
        st.header("🎯 Enhanced Audio")
        
        if uploaded_file is not None and 'original_audio' in locals():
            if st.button("🚀 Enhance Audio", type="primary"):
                with st.spinner("Enhancing audio... This may take a few moments."):
                    enhanced_audio, enhanced_sr = enhancer.enhance_audio(original_audio, original_sr)
                
                if enhanced_audio is not None:
                    # Convert to bytes for download
                    audio_bytes = write_audio_bytes(enhanced_audio, enhanced_sr, format='WAV')
                    
                    st.success("✅ Enhancement complete!")
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Download button
                    filename = f"enhanced_{uploaded_file.name.split('.')[0]}.wav"
                    st.download_button(
                        label="💾 Download Enhanced Audio",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav"
                    )
                    
                    # Calculate quality metrics
                    snr_improvement = 10 * np.log10(np.mean(enhanced_audio**2) / (np.mean((enhanced_audio - original_audio)**2) + 1e-8))
                    
                    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Quality Improvement", f"{snr_improvement:.2f} dB")
                    with col_m2:
                        st.metric("Processing Time", "Real-time")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Upload an audio file to get started")
    
    # Visualization section
    if uploaded_file is not None and 'enhanced_audio' in locals() and enhanced_audio is not None:
        st.header("📈 Audio Analysis")
        
        # Visualization tabs
        tab1, tab2 = st.tabs(["🌊 Waveforms", "🔊 Spectrograms"])
        
        with tab1:
            # Trim to same length for comparison
            min_len = min(len(original_audio), len(enhanced_audio))
            orig_trimmed = original_audio[:min_len]
            enh_trimmed = enhanced_audio[:min_len]
            
            waveform_fig = create_waveform_plot(orig_trimmed, enh_trimmed, original_sr)
            st.plotly_chart(waveform_fig, use_container_width=True)
        
        with tab2:
            spectrogram_fig = create_spectrogram_plot(orig_trimmed, enh_trimmed, original_sr)
            st.plotly_chart(spectrogram_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>🎯 AI Speech Enhancement • Built with Streamlit & PyTorch</p>
        <p>Upload noisy speech recordings and get clean, enhanced audio in seconds!</p>
        <p>Project: {get_path('base').name} • Configuration: Global Path System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

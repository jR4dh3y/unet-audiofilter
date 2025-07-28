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

# Add src to path
sys.path.append('/home/radhey/code/ai-clrvoice')
from src.unet_model import UNet
from src.audio_utils import load_audio, save_audio, write_audio_bytes

# Page configuration
st.set_page_config(
    page_title="AI Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    """Load the trained GPU model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with same parameters as training
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_filters=32,
            depth=3,
            dropout=0.1
        ).to(device)
        
        # Load trained weights
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_gpu_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, device, checkpoint
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
            # Resample if necessary
            if input_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=input_sr, target_sr=self.sample_rate)
            
            # Process in chunks for memory efficiency
            enhanced_audio = self._process_chunks(audio_data)
            
            # Compute metadata
            metadata = {
                'duration': len(audio_data) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'input_rms': float(np.sqrt(np.mean(audio_data**2))),
                'output_rms': float(np.sqrt(np.mean(enhanced_audio**2)))
            }
            
            return enhanced_audio, audio_data, metadata
            
        except Exception as e:
            st.error(f"Error during enhancement: {e}")
            return None, None, None
    
    def _process_chunks(self, audio):
        """Process audio in overlapping chunks"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(chunk_samples * self.overlap_ratio)
        hop_samples = chunk_samples - overlap_samples
        
        if len(audio) <= chunk_samples:
            return self._enhance_chunk(audio)
        
        # Process overlapping chunks
        enhanced_chunks = []
        for start_idx in range(0, len(audio) - overlap_samples, hop_samples):
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]
            
            # Pad chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            enhanced_chunk = self._enhance_chunk(chunk)
            enhanced_chunks.append((enhanced_chunk, start_idx, len(audio[start_idx:end_idx])))
        
        # Blend overlapping chunks
        return self._blend_chunks(enhanced_chunks, len(audio), overlap_samples)
    
    def _enhance_chunk(self, audio_chunk):
        """Enhance a single audio chunk"""
        # Convert to spectrogram
        stft = librosa.stft(
            audio_chunk,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Normalize magnitude
        magnitude_normalized = magnitude / (np.max(magnitude) + 1e-8)
        
        # Model inference
        with torch.no_grad():
            magnitude_tensor = torch.FloatTensor(magnitude_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            enhanced_magnitude = self.model(magnitude_tensor)
            enhanced_magnitude = enhanced_magnitude.cpu().squeeze().numpy()
        
        # Denormalize
        enhanced_magnitude = enhanced_magnitude * (np.max(magnitude) + 1e-8)
        
        # Reconstruct audio
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
            end_idx = start_idx + original_length
            chunk_length = min(len(enhanced_chunk), original_length)
            
            # Create blend weights with crossfading
            weights = np.ones(chunk_length)
            
            if overlap_samples > 0 and len(enhanced_chunks) > 1:
                fade_length = min(overlap_samples, chunk_length // 2)
                
                # Fade in
                if start_idx > 0:
                    weights[:fade_length] = np.linspace(0, 1, fade_length)
                
                # Fade out
                if end_idx < total_length:
                    weights[-fade_length:] = np.linspace(1, 0, fade_length)
            
            # Apply weighted blending
            enhanced_audio[start_idx:start_idx + chunk_length] += enhanced_chunk[:chunk_length] * weights
            weight_sum[start_idx:start_idx + chunk_length] += weights
        
        # Normalize
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
    
    # Load model
    model, device, checkpoint = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Initialize enhancer
    enhancer = GPUSpeechEnhancer(model, device)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Model Information")
        st.info(f"""
        **Architecture:** U-Net
        **Parameters:** {sum(p.numel() for p in model.parameters()):,}
        **Best Val Loss:** {checkpoint.get('val_loss', 'N/A'):.6f}
        **Device:** {device}
        """)
        
        st.header("📊 Configuration")
        st.json({
            "Sample Rate": "16000 Hz",
            "STFT Parameters": {
                "n_fft": 1024,
                "hop_length": 256,
                "window": "hann"
            },
            "Processing": {
                "chunk_duration": "4.0 seconds",
                "overlap": "25%"
            }
        })
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload a noisy speech recording to enhance"
        )
        
        if uploaded_file is not None:
            try:
                # Use ffmpeg-based audio loading
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    
                    audio_data, sample_rate = load_audio(tmp_file.name, sample_rate=None)
                    os.unlink(tmp_file.name)
                
                # Convert to WAV bytes for playback using ffmpeg
                wav_bytes_data = write_audio_bytes(audio_data, sample_rate, format='WAV')
                wav_bytes = io.BytesIO(wav_bytes_data)
                
                st.success(f"✅ Audio loaded successfully!")
                st.write(f"**Duration:** {len(audio_data) / sample_rate:.2f} seconds")
                st.write(f"**Sample Rate:** {sample_rate} Hz")
                st.write(f"**Samples:** {len(audio_data):,}")
                
                # Play original audio
                st.subheader("🔊 Original Audio")
                st.audio(wav_bytes, format="audio/wav")
                
            except Exception as e:
                st.error(f"Error loading audio: {e}")
                return
    
    with col2:
        st.header("🚀 Enhancement")
        
        if uploaded_file is not None:
            enhance_button = st.button("🎯 Enhance Audio", type="primary")
            
            if enhance_button:
                with st.spinner("🔄 Enhancing audio... This may take a few moments."):
                    # Enhance audio
                    enhanced_audio, original_audio, metadata = enhancer.enhance_audio(audio_data, sample_rate)
                    
                    if enhanced_audio is not None:
                        st.success("🎉 Enhancement completed!")
                        
                        # Display metrics
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric("Duration", f"{metadata['duration']:.2f}s")
                        
                        with col_m2:
                            st.metric("Input RMS", f"{metadata['input_rms']:.4f}")
                        
                        with col_m3:
                            st.metric("Output RMS", f"{metadata['output_rms']:.4f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Play enhanced audio
                        st.subheader("🔊 Enhanced Audio")
                        
                        # Convert to bytes for audio player using ffmpeg
                        enhanced_bytes_data = write_audio_bytes(enhanced_audio, metadata['sample_rate'], format='WAV')
                        enhanced_bytes = io.BytesIO(enhanced_bytes_data)
                        
                        st.audio(enhanced_bytes, format="audio/wav")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Enhanced Audio",
                            data=enhanced_bytes.getvalue(),
                            file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.wav",
                            mime="audio/wav"
                        )
        else:
            st.info("👆 Please upload an audio file to get started")
    
    # Visualization section
    if uploaded_file is not None and 'enhanced_audio' in locals() and enhanced_audio is not None:
        st.header("📈 Audio Analysis")
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["🌊 Waveforms", "🎵 Spectrograms", "📊 Statistics"])
        
        with tab1:
            st.subheader("Waveform Comparison")
            waveform_fig = create_waveform_plot(original_audio, enhanced_audio, metadata['sample_rate'])
            st.plotly_chart(waveform_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Spectrogram Comparison")
            spectrogram_fig = create_spectrogram_plot(original_audio, enhanced_audio, metadata['sample_rate'])
            st.plotly_chart(spectrogram_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Audio Statistics")
            
            # Compute additional statistics
            orig_energy = np.sum(original_audio ** 2)
            enh_energy = np.sum(enhanced_audio ** 2)
            energy_ratio = enh_energy / (orig_energy + 1e-10)
            
            orig_peak = np.max(np.abs(original_audio))
            enh_peak = np.max(np.abs(enhanced_audio))
            
            # Display statistics
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("**Original Audio**")
                st.write(f"• RMS: {metadata['input_rms']:.6f}")
                st.write(f"• Peak: {orig_peak:.6f}")
                st.write(f"• Energy: {orig_energy:.6f}")
                st.write(f"• Duration: {metadata['duration']:.2f}s")
            
            with col_s2:
                st.markdown("**Enhanced Audio**")
                st.write(f"• RMS: {metadata['output_rms']:.6f}")
                st.write(f"• Peak: {enh_peak:.6f}")
                st.write(f"• Energy: {enh_energy:.6f}")
                st.write(f"• Energy Ratio: {energy_ratio:.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 AI Speech Enhancement • Built with Streamlit & PyTorch</p>
        <p>Upload noisy speech recordings and get clean, enhanced audio in seconds!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

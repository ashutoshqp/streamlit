# Save the Streamlit app
import streamlit as st
import torch
import torchaudio
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import IPython

def setup_environment():
    """
    Installs necessary packages and sets up the environment.
    This would typically be done in Colab setup cells.
    """
    try:
        import tortoise
    except ImportError:
        # These commands would be run in Colab's terminal
        print("Setting up environment...")
        os.system('pip3 install -U scipy')
        os.system('git clone https://github.com/jnordberg/tortoise-tts.git')
        os.chdir('tortoise-tts')
        os.system('pip3 install -r requirements.txt')
        os.system('pip3 install transformers==4.19.0 einops==0.5.0 rotary_embedding_torch==0.1.5 unidecode==1.3.5')
        os.system('python3 setup.py install')

def clone_voice(uploaded_files, text, preset='high_quality'):
    """
    Clone voice using uploaded audio files and generate speech

    Args:
    - uploaded_files: List of uploaded audio files
    - text: Text to be spoken
    - preset: Quality preset for voice generation

    Returns:
    - Path to generated audio file
    """
    # Create a temporary directory for custom voice
    CUSTOM_VOICE_NAME = "custom"
    custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
    os.makedirs(custom_voice_folder, exist_ok=True)

    # Save uploaded files
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(custom_voice_folder, f'{i}.wav')
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

    # Initialize Text-to-Speech
    tts = TextToSpeech()

    # Load voice
    voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)

    # Generate speech
    output_path = f'generated-{CUSTOM_VOICE_NAME}.wav'
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset
    )

    # Save generated audio
    torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)

    return output_path

def main():
    st.title('🎤 Tortoise Voice Cloning App')

    # Setup environment (comment out if already done in Colab)
    setup_environment()

    # Text input
    text = st.text_area(
        "Enter the text you want to generate speech for:",
        "Let's strive to make the world a better place, one code block at a time."
    )

    # Preset selection
    preset = st.selectbox(
        "Select voice generation quality:",
        ["ultra_fast", "fast", "standard", "high_quality"],
        index=3
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload voice samples (2-5 WAV files, 6-10 seconds each)",
        type=['wav'],
        accept_multiple_files=True
    )

    # Validation
    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 voice sample audio files.")
        return

    # Generate button
    if st.button('Clone Voice'):
        with st.spinner('Generating voice...'):
            try:
                output_path = clone_voice(uploaded_files, text, preset)

                # Display audio
                st.audio(output_path)

                # Provide download
                with open(output_path, 'rb') as audio_file:
                    st.download_button(
                        label="Download Generated Audio",
                        data=audio_file,
                        file_name='cloned_voice.wav',
                        mime='audio/wav'
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
import gradio as gr
from models.megatts import Megatts
from modules.tokenizer import HIFIGAN_SR

megatts = Megatts(
    g_ckpt='generator.ckpt',
    g_config='configs/config_gan.yaml',
    plm_ckpt='plm.ckpt',
    plm_config='configs/config_plm.yaml',
    adm_ckpt='adm.ckpt',
    adm_config='configs/config_adm.yaml',
    symbol_table='unique_text_tokens.k2symbols'
)
megatts.eval()

def generate_audio(
    audio_files, 
    text
):
    audio_paths = [audio_file.name for audio_file in audio_files]
    audio_tensor = megatts.forward(audio_paths, text)
    audio_numpy = audio_tensor.cpu().numpy()
    return audio_numpy, HIFIGAN_SR

iface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.inputs.File(
            type="file", 
            label="Upload Audio Files", 
            multiple=True, 
            filetype="audio/wav"
        ),
        gr.inputs.Textbox(lines=2, label="Input Text")
    ],
    outputs=[
        gr.outputs.Audio(type="numpy", label="Generated Audio")
    ],
    title="MegaTTS2 Speech Synthesis",
    description="Upload your audio files (only .wav format) and enter text to generate speech."
)

if __name__ == "__main__":
    iface.launch()

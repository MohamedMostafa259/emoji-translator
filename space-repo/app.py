import gradio as gr
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from peft import PeftModel

# Model configuration
MODEL_NAME = "facebook/bart-large"
PEFT_MODEL_ID = "mohamedmostafa259/bart-emoji-translator"
MAX_LENGTH = 32

CREATIVITY_SETTINGS = {
    0: {"top_p": 0.2, "temperature": 0.5},  # Strict
    1: {"top_p": 0.4, "temperature": 1.0},  # Balanced
    2: {"top_p": 0.6, "temperature": 1.5},  # Creative
}

# Load model and tokenizer at startup (automatically cached)
print("Loading model...")
base_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
tokenizer = BartTokenizer.from_pretrained(PEFT_MODEL_ID)

# Set to eval mode
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model loaded on {device}")

# Translation function
def translate_to_emoji(text, creativity_level=0):
    if not text.strip():
        return "⚠️ Please enter some text!"
    
    try:
        tokens = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_length=MAX_LENGTH,
                num_beams=1,
                do_sample=True,
                temperature=CREATIVITY_SETTINGS[creativity_level]["temperature"],
                top_p=CREATIVITY_SETTINGS[creativity_level]["top_p"],
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "") 
        return result if result else "🤷 (No translation generated)"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="🎭 Emoji Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 BART Emoji Translator
        
        Transform your text into emojis! This model uses curriculum learning and LoRA fine-tuning 
        to translate English sentences into appropriate emoji sequences.
        
        **Dataset:** Synthetic dataset generated using **Gemini 3 Pro** to ensure high quality and diversity.
        
        **Examples:** Try "I am happy", "I eat dinner with my family", or "I feel misunderstood"
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Enter your text",
                placeholder="Type something like: I am happy today...",
                lines=3
            )
            
            # --- Creativity Slider (0, 1, 2) ---
            creativity = gr.Slider(
                minimum=0,
                maximum=2,
                value=1,
                step=1,
                label="🎨 Creativity",
                info="0 = strict, 1 = balanced, 2 = creative (more randomness; maybe not accurate)"
            )

            translate_btn = gr.Button("✨ Translate to Emojis", variant="primary")
        
        with gr.Column(scale=1):
            emoji_output = gr.Textbox(
                label="🎉 Emoji Translation",
                lines=3,
                scale=2
            )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["I am happy", 0],
            ["I feel sad", 2],
            ["I eat dinner with my family", 1],
            ["I tweeted the news to my followers", 1],
            ["My parents want to have a new baby", 2],
        ],
        inputs=[text_input, creativity],
        outputs=emoji_output,
        fn=translate_to_emoji,
        cache_examples=False,
        run_on_click=True
    )
    
    # Button click event
    translate_btn.click(
        fn=translate_to_emoji,
        inputs=[text_input, creativity],
        outputs=emoji_output
    )
    
    # Also trigger on Enter key
    text_input.submit(
        fn=translate_to_emoji,
        inputs=[text_input, creativity],
        outputs=emoji_output
    )
    
    gr.Markdown(
        """
        ---
        ### 📚 Model Information
        
        - **Model:** BART-Large with LoRA fine-tuning
        - **Training:** 6-phase curriculum learning with strategic data retention
        - **Parameters:** 128 LoRA rank, 256 alpha
        - **Dataset:** Progressive difficulty from single emojis to complex sequences
        
        ### 🔗 Links
        
        - [🤗 Model on HuggingFace](https://huggingface.co/mohamedmostafa259/bart-emoji-translator)
        - [💻 Training Code](https://www.kaggle.com/code/mohamedmostafa259/emoji-translator-curriculum-learning)
        - [📊 Dataset](https://www.kaggle.com/datasets/mohamedmostafa259/english-to-emoji)

        ### ⚠️ Limitations
        
        - Works best with English text under 32 tokens
        - May not recognize very rare or newly created emojis
        - Performance varies with text complexity
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

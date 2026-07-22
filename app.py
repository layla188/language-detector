import gradio as gr
import spaces
from inference import predict

# We put the GPU decorator on YOUR actual function
@spaces.GPU
def detect_language(text: str) -> tuple[str, str, str]:
    """
    Use the inference module to detect the language
    and prepare readable results for the Gradio interface.
    """

    try:
        result = predict(text)

        language = result["language"]
        language_code = result["language_code"]
        confidence = result["confidence"]

        return (
            language,
            language_code,
            f"{confidence:.2%}",
        )

    except (TypeError, ValueError) as error:
        return (
            str(error),
            "",
            "",
        )

# This is your original, correct interface configuration
demo = gr.Interface(
    fn=detect_language,

    inputs=gr.Textbox(
        lines=5,
        label="Enter Text",
        placeholder="Type or paste a sentence here...",
    ),

    outputs=[
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Language Code"),
        gr.Textbox(label="Confidence"),
    ],

    title="Multilingual Language Detector",

    description=(
    "Enter a sentence and the application will detect "
    "its language using the pretrained Papluca "
    "XLM-RoBERTa language-detection model."
    ),

    examples=[
        ["I am learning artificial intelligence."],
        ["أنا أتعلم الذكاء الاصطناعي."],
        ["Bonjour, comment allez-vous aujourd'hui ?"],
        ["La reunión comienza mañana por la mañana."],
        ["Das Wetter ist heute sehr angenehm."],
    ],
)

if __name__ == "__main__":
    demo.launch()
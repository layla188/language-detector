from transformers import pipeline


# Hugging Face model used for language detection.
MODEL_ID = "papluca/xlm-roberta-base-language-detection"


# Languages supported by the Papluca model.
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


# Load the model and tokenizer once when the program starts.
# device=-1 means that the model runs on the CPU.
classifier = pipeline(
    task="text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,
)


def normalize_label(raw_label: str) -> str:
    """
    Clean and normalize the language label returned by the model.

    The Papluca model normally returns language codes directly,
    such as en, ar, fr, es, or de.
    """

    return raw_label.strip().lower()


def predict(text: str) -> dict:
    """
    Detect the language of one text using the Papluca model.

    Args:
        text:
            The sentence entered by the user.

    Returns:
        A dictionary containing:
        - readable language name
        - language code
        - confidence score
        - raw model label
    """

    # Make sure the input is text.
    if not isinstance(text, str):
        raise TypeError("The input must be a string.")

    # Remove extra spaces from the beginning and end.
    cleaned_text = text.strip()

    # Prevent inference on an empty input.
    if not cleaned_text:
        raise ValueError("Please enter some text.")

    # Run the pretrained language-detection model.
    # The pipeline returns a list, so [0] selects the first result.
    output = classifier(
        cleaned_text,
        truncation=True,
    )[0]

    # Example:
    # {
    #     "label": "en",
    #     "score": 0.998
    # }

    raw_label = output["label"]

    # Standardize the language code.
    language_code = normalize_label(raw_label)

    # Convert the language code into a readable language name.
    # If the code is not found, display the code in uppercase.
    language_name = LANGUAGE_NAMES.get(
        language_code,
        language_code.upper(),
    )

    return {
        "language": language_name,
        "language_code": language_code,
        "confidence": float(output["score"]),
        "raw_label": raw_label,
    }


# This test section runs only when inference.py
# is executed directly from the terminal.
#
# It will not run when predict() is imported inside app.py.
if __name__ == "__main__":
    test_sentences = [
        "I am learning artificial intelligence.",
        "أنا أتعلم الذكاء الاصطناعي.",
        "Bonjour, comment allez-vous aujourd'hui ?",
        "La reunión comienza mañana por la mañana.",
        "Das Wetter ist heute sehr angenehm.",
        "Sto imparando come funzionano i modelli linguistici.",
    ]

    for sentence in test_sentences:
        result = predict(sentence)

        print("-" * 60)
        print("Text:", sentence)
        print("Language:", result["language"])
        print("Language code:", result["language_code"])
        print(f"Confidence: {result['confidence']:.2%}")
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def generate_summary(text):

    max_chunk = 1024
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    final_summary = ""

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=180,
            min_length=80,
            do_sample=False,
            num_beams=6,
            length_penalty=2.0
        )
        final_summary += summary[0]['summary_text'] + " "

    return final_summary

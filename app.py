import gradio as gr
from transformers import pipeline


chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def get_response(user_message, mood):
    if not user_message.strip():
        return "Please share what you're feeling so I can support you."

    prompt = (
        f"You are a supportive AI mental health assistant.\n"
        f"Someone is feeling '{mood}' and said: \"{user_message}\"\n"
        f"Respond in a kind and empathetic way, providing emotional support.\n\n"
        f"AI Response:"
    )

    try:
        result = chatbot(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )[0]['generated_text']

      
        response_start = result.rfind("AI Response:")
        response = result[response_start + len("AI Response:"):].strip()

        
        for junk in ["Twitter", "Facebook", "Instagram", "Mood:", "You:" ]:
            response = response.split(junk)[0].strip()

        return response

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


description = "Chat with a compassionate AI for emotional support and coping strategies."

demo = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Textbox(label="Your Message", placeholder="e.g. I'm feeling overwhelmed."),
        gr.Dropdown(
            choices=["Good", "Okay", "Stressed", "Sad", "Angry", "Confused"],
            label="Mood"
        )
    ],
    outputs=gr.Textbox(label="Bot Response"),
    title="🧠 AI Mental Health Chatbot",
    description=description,
    theme="soft"
)

demo.launch()

import json
import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from prompts import stronger_prompt
from tooling import handle_tool_calls, tools

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

model_openai = "gpt-5.4-mini"
model_deepseek = "deepseek-chat"
model_transcribe = "whisper-1"
model_transcribe = "gpt-4o-mini-transcribe"
model_tts = "gpt-4o-mini-tts"

st.title("📊 Invest-AI")
st.caption("💰 Inversiones simplificadas.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        message_block = st.chat_message(msg["role"])
        message_block.write(msg["content"])
        audio_payload = msg.get("audio")
        if audio_payload:
            message_block.audio(audio_payload, format="audio/mp3")

with st.sidebar:
    st.subheader("Entrada de audio")
    audio_value = st.audio_input("Graba un mensaje de voz (opcional)")
    send_audio = st.button("Enviar audio", key="send_audio_button", use_container_width=True)

user_prompt = None
user_display_content = None

if text_prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
    user_prompt = text_prompt
    user_display_content = text_prompt
elif send_audio:
    raw_audio = None
    file = None
    source = None

    if audio_value is not None:
        raw_audio = audio_value.getvalue()
        filename = audio_value.name or "voz_usuario.wav"
        source = "Audio grabado"

    if raw_audio:
        audio_file = BytesIO(raw_audio)
        audio_file.name = filename or "voz_usuario.wav"
        with st.spinner("Trancribiendo audio..."):
            transcription = client_openai.audio.transcriptions.create(
                model=model_transcribe,
                file=audio_file,
            )
        user_prompt = transcription.text.strip()
        if user_prompt:
            user_display_content = f"({source}) {user_prompt}" if source else user_prompt
        else:
            st.info("La transcripción no contiene texto interpretable. Intenta nuevamente.")
    else:
        st.warning("Graba un mensaje de voz antes de enviarlo.")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_display_content or user_prompt)
    conversation = [{"role": "assistant", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content":m["content"]} for m in st.session_state.messages)

    done = False
    while not done:
        with st.chat_message("assistant"):
            completion = client_openai.chat.completions.create(
                model = model_openai,
                messages = conversation,
                tools = tools,
            )
            choice = completion.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls" and message.tool_calls:
                tool_calls = message.tool_calls
                tool_calls_serialized = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "type": tc.type,
                    }
                    for tc in tool_calls
                ]
                results = handle_tool_calls(tool_calls)
                safe_content = message.content or ""
                if safe_content:
                    st.session_state.messages.append({"role": message.role, "content": safe_content})
                conversation.append(
                    {
                        "role": message.role,
                        "content": safe_content,
                        "tool_calls": tool_calls_serialized,
                    }
                )
                conversation.extend(results)
                continue

            done = True
            response = message.content or ""
            st.write(response)

#    with st.chat_message("assistant"):
#        stream = client_openai.chat.completions.create(model=model_openai, messages=conversation, stream=True)
#        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

    audio_bytes = None
    with st.spinner("Generando respuesta en audio..."):
        try:
            speech = client_openai.audio.speech.create(model=model_tts, voice="ash", input=response)
            audio_bytes = speech.read()
            if not audio_bytes:
                st.info("No se pudo obtener audio para esta respuesta.")
        except Exception as exc:
            st.error(f"No se pudo generar la voz sintética: {exc}")
    
    st.audio(audio_bytes, format="audio/mp3", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=True, width="stretch")

    if audio_bytes:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant":
            last_message["audio"] = audio_bytes
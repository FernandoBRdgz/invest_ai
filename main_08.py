import os
import re
import urllib.request
import streamlit as st
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from prompts import stronger_prompt
from tooling import handle_tool_calls, tools
from visualization import (
    viz_income_engineering,
    viz_fcf_engineering,
    viz_roic_engineering,
    viz_price_history,
)

FALLBACK_SP500_COMPANIES = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corp."),
    ("NVDA", "NVIDIA Corp."),
    ("GOOGL", "Alphabet Inc. (Class A)"),
    ("AMZN", "Amazon.com Inc."),
    ("META", "Meta Platforms Inc."),
    ("TSLA", "Tesla Inc."),
    ("BRK.B", "Berkshire Hathaway Inc. (Class B)"),
    ("JPM", "JPMorgan Chase & Co."),
    ("UNH", "UnitedHealth Group Inc."),
    ("V", "Visa Inc."),
    ("MA", "Mastercard Inc."),
    ("HD", "Home Depot Inc."),
    ("PG", "Procter & Gamble Co."),
    ("XOM", "Exxon Mobil Corp."),
    ("BAC", "Bank of America Corp."),
    ("PFE", "Pfizer Inc."),
    ("ADBE", "Adobe Inc."),
    ("ABBV", "AbbVie Inc."),
    ("COST", "Costco Wholesale Corp."),
    ("AVGO", "Broadcom Inc."),
    ("PEP", "PepsiCo Inc."),
    ("NFLX", "Netflix Inc."),
    ("CRM", "Salesforce Inc."),
    ("MRK", "Merck & Co. Inc."),
    ("INTC", "Intel Corp."),
    ("CSCO", "Cisco Systems Inc."),
    ("ORCL", "Oracle Corp."),
    ("WMT", "Walmart Inc."),
    ("KO", "Coca-Cola Co."),
    ("NKE", "Nike Inc."),
    ("MCD", "McDonald's Corp."),
    ("VZ", "Verizon Communications Inc."),
    ("T", "AT&T Inc."),
    ("CVX", "Chevron Corp."),
    ("UPS", "United Parcel Service Inc."),
    ("LLY", "Eli Lilly & Co."),
    ("BMY", "Bristol Myers Squibb Co."),
    ("TMO", "Thermo Fisher Scientific Inc."),
    ("ACN", "Accenture plc"),
    ("MDT", "Medtronic plc"),
    ("ABT", "Abbott Laboratories"),
    ("TXN", "Texas Instruments Inc."),
    ("DHR", "Danaher Corp."),
    ("QCOM", "Qualcomm Inc."),
    ("SBUX", "Starbucks Corp."),
    ("AMGN", "Amgen Inc."),
    ("UNP", "Union Pacific Corp."),
    ("RTX", "Raytheon Technologies Corp."),
]

NO_SELECTION_LABEL = "Selecciona una compañía..."

SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"

VIZ_TYPES = [
    ("income", "Ingresos"),
    ("fcf", "Flujo de Caja Libre"),
    ("roic", "ROIC e inversiones"),
    ("price", "Precio histórico"),
]
VIZ_LABEL_TO_KEY = {label: key for key, label in VIZ_TYPES}
VIZ_KEY_TO_LABEL = {key: label for key, label in VIZ_TYPES}
VIZ_REGISTRY = {
    "income": viz_income_engineering,
    "fcf": viz_fcf_engineering,
    "roic": viz_roic_engineering,
    "price": viz_price_history,
}

def _clean_text(html_fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_fragment or "")
    return text.strip()


@st.cache_data(show_spinner=False)
def _fetch_sp500_companies(limit=250):
    try:
        request = urllib.request.Request(
            SLICKCHARTS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InvestAI/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    companies = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    for row in rows:
        if "<th" in row.lower():
            continue
        cols = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)
        if len(cols) < 3:
            continue
        name = _clean_text(cols[1])
        ticker = _clean_text(cols[2])
        if not ticker or not name:
            continue
        companies.append((ticker, name))
        if len(companies) >= limit:
            break
    return companies


def _get_company_list(limit=250):
    try:
        companies = _fetch_sp500_companies(limit)
    except Exception:
        companies = []
    if len(companies) < limit:
        seen = {ticker for ticker, _ in companies}
        for ticker, name in FALLBACK_SP500_COMPANIES:
            if ticker not in seen:
                companies.append((ticker, name))
                seen.add(ticker)
                if len(companies) >= limit:
                    break
    return companies[:limit]


COMPANY_LIST = _get_company_list(limit=250)
COMPANY_DISPLAY_OPTIONS = [f"{ticker} — {name}" for ticker, name in COMPANY_LIST]
COMPANY_LABELS = {ticker: name for ticker, name in COMPANY_LIST}


load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

model_openai = "gpt-5.4-mini"
model_transcribe = "whisper-1"
model_tts = "gpt-4o-mini-tts"

# funcion auxiliar para streaming
def stream_assistant_answer(client, model, conversation):
    """
    Llama al modelo con stream=True y pinta la respuesta progresivamente.
    Devuelve el texto completo generado.
    """
    full_response = ""
    placeholder = st.empty()

    stream = client.chat.completions.create(
        model=model,
        messages=conversation,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full_response += delta.content
            placeholder.markdown(full_response)

    return full_response


@st.cache_data(show_spinner=False)
def _cached_dashboard_visualization(ticker: str, viz_key: str):
    """
    Cache wrapper around the visualization helper selected by type.
    """
    func = VIZ_REGISTRY[viz_key]
    viz_result = func(ticker.strip().upper())
    return viz_result if isinstance(viz_result, list) else [viz_result]


def _refresh_dashboard(ticker: str, viz_key: str):
    ticker_clean = (ticker or "").strip().upper()
    if not ticker_clean:
        st.session_state.dashboard_figures = []
        st.session_state.dashboard_error = "Inserta un ticker válido antes de actualizar el panel."
        st.session_state.dashboard_loaded = False
        return
    try:
        st.session_state.dashboard_figures = _cached_dashboard_visualization(ticker_clean, viz_key)
        st.session_state.dashboard_error = ""
        st.session_state.dashboard_loaded = True
    except Exception as exc:
        st.session_state.dashboard_figures = []
        st.session_state.dashboard_error = str(exc)
        st.session_state.dashboard_loaded = False
    st.session_state.dashboard_ticker = ticker_clean
    st.session_state.dashboard_viz_key = viz_key


def _refresh_comparison(tickers: list[str], viz_key: str):
    cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
    if not cleaned:
        st.session_state.comparison_data = []
        st.session_state.comparison_error = "Selecciona al menos una compañía para comparar."
        st.session_state.comparison_loaded = False
        st.session_state.comparison_tickers = []
        st.session_state.comparison_viz_key = viz_key
        return

    results = []
    errors = []
    for ticker in cleaned[:3]:
        try:
            figures = _cached_dashboard_visualization(ticker, viz_key)
            results.append({"ticker": ticker, "figures": figures})
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    st.session_state.comparison_data = results
    st.session_state.comparison_error = "; ".join(errors) if errors else ""
    st.session_state.comparison_loaded = bool(results)
    st.session_state.comparison_tickers = cleaned[:3]
    st.session_state.comparison_viz_key = viz_key


st.title("📊 Invest-AI")
st.caption("💰 Inversiones simplificadas.")

st.markdown(
    """
    Usa el panel lateral para alternar entre el chat, el dashboard de una sola empresa y la comparativa.
    En cada sección puedes elegir el tipo de visualización y el ticker que quieras analizar.
    """
)

if "dashboard_ticker" not in st.session_state:
    st.session_state["dashboard_ticker"] = ""
if "dashboard_figures" not in st.session_state:
    st.session_state["dashboard_figures"] = []
if "dashboard_error" not in st.session_state:
    st.session_state["dashboard_error"] = ""
if "dashboard_loaded" not in st.session_state:
    st.session_state["dashboard_loaded"] = False
if "dashboard_company_selectbox" not in st.session_state:
    st.session_state["dashboard_company_selectbox"] = NO_SELECTION_LABEL
if "dashboard_viz_key" not in st.session_state:
    st.session_state["dashboard_viz_key"] = VIZ_TYPES[0][0]
if "dashboard_viz_selectbox" not in st.session_state:
    st.session_state["dashboard_viz_selectbox"] = VIZ_TYPES[0][1]
if "comparison_data" not in st.session_state:
    st.session_state["comparison_data"] = []
if "comparison_error" not in st.session_state:
    st.session_state["comparison_error"] = ""
if "comparison_loaded" not in st.session_state:
    st.session_state["comparison_loaded"] = False
if "comparison_tickers" not in st.session_state:
    st.session_state["comparison_tickers"] = []
if "comparison_viz_key" not in st.session_state:
    st.session_state["comparison_viz_key"] = VIZ_TYPES[0][0]
if "comparison_viz_selectbox" not in st.session_state:
    st.session_state["comparison_viz_selectbox"] = VIZ_KEY_TO_LABEL[VIZ_TYPES[0][0]]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

audio_value = None
send_audio = False
section_choice = "Chat"

with st.sidebar:
    section_choice = st.radio("Secciones", ["Chat", "Dashboard", "Comparación"], index=0)
    if section_choice == "Chat":
        st.subheader("Entrada de audio")
        audio_value = st.audio_input("Graba un mensaje de voz (opcional)")
        send_audio = st.button("Enviar audio", key="send_audio_button", use_container_width=True)
    elif section_choice == "Dashboard":
        st.subheader("Dashboard financiero")
        viz_labels = [label for _, label in VIZ_TYPES]
        default_viz_index = (
            viz_labels.index(st.session_state.dashboard_viz_selectbox)
            if st.session_state.dashboard_viz_selectbox in viz_labels
            else 0
        )
        selected_viz_label = st.selectbox(
            "Tipo de visualización",
            viz_labels,
            index=default_viz_index,
            key="dashboard_viz_selectbox",
        )
        selected_viz_key = VIZ_LABEL_TO_KEY.get(selected_viz_label, VIZ_TYPES[0][0])
        st.session_state.dashboard_viz_key = selected_viz_key

        select_options = [NO_SELECTION_LABEL] + COMPANY_DISPLAY_OPTIONS
        default_index = (
            select_options.index(st.session_state.dashboard_company_selectbox)
            if st.session_state.dashboard_company_selectbox in select_options
            else 0
        )
        st.selectbox(
            "Selecciona una empresa del S&P 500",
            select_options,
            index=default_index,
            key="dashboard_company_selectbox",
        )
        if st.button("Actualizar dashboard", key="dashboard_refresh_button_sidebar"):
            selected_entry = st.session_state.dashboard_company_selectbox
            if selected_entry == NO_SELECTION_LABEL:
                st.session_state.dashboard_figures = []
                st.session_state.dashboard_error = "Selecciona una compañía antes de actualizar."
                st.session_state.dashboard_loaded = False
                st.session_state.dashboard_ticker = ""
            else:
                selected_ticker = selected_entry.split(" — ")[0]
                _refresh_dashboard(selected_ticker, st.session_state.dashboard_viz_key)
    else:
        st.subheader("Comparativa de compañías")
        viz_labels = [label for _, label in VIZ_TYPES]
        default_viz_index = (
            viz_labels.index(st.session_state.comparison_viz_selectbox)
            if st.session_state.comparison_viz_selectbox in viz_labels
            else 0
        )
        selected_viz_label = st.selectbox(
            "Tipo de visualización",
            viz_labels,
            index=default_viz_index,
            key="comparison_viz_selectbox",
        )
        selected_viz_key = VIZ_LABEL_TO_KEY.get(selected_viz_label, VIZ_TYPES[0][0])
        st.session_state.comparison_viz_key = selected_viz_key
        comparison_options = COMPANY_DISPLAY_OPTIONS
        default_selections = [
            f"{ticker} — {COMPANY_LABELS.get(ticker, ticker)}"
            for ticker in st.session_state.comparison_tickers
        ]
        comparison_multiselect = st.multiselect(
            "Selecciona hasta 3 compañías del S&P 500",
            comparison_options,
            default=default_selections,
            key="comparison_company_multiselect",
        )
        if len(comparison_multiselect) > 3:
            st.warning("Solo se mostrarán las primeras 3 compañías seleccionadas.")
            comparison_multiselect = comparison_multiselect[:3]
        if st.button("Actualizar comparativa", key="comparison_refresh_button_sidebar"):
            if not comparison_multiselect:
                st.session_state.comparison_data = []
                st.session_state.comparison_error = "Selecciona al menos una compañía para comparar."
                st.session_state.comparison_loaded = False
                st.session_state.comparison_tickers = []
            else:
                selected_tickers = [entry.split(" — ")[0] for entry in comparison_multiselect]
                _refresh_comparison(selected_tickers, selected_viz_key)

if section_choice == "Chat":
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            message_block = st.chat_message(msg["role"])
            message_block.write(msg["content"])
            audio_payload = msg.get("audio")
            if audio_payload:
                message_block.audio(audio_payload, format="audio/mp3")

    user_prompt = None
    user_display_content = None

    if text_prompt := st.chat_input(placeholder="Escribe tu mensaje aquí..."):
        user_prompt = text_prompt
        user_display_content = text_prompt
    elif send_audio:
        raw_audio = None
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
        conversation.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.messages)

        done = False

        while not done:
            completion = client_openai.chat.completions.create(
                model=model_openai,
                messages=conversation,
                tools=tools,
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

            last_non_stream_response = message.content or ""
            done = True

        with st.chat_message("assistant"):
            response = stream_assistant_answer(
                client=client_openai,
                model=model_openai,
                conversation=conversation,
            )

        st.session_state.messages.append({"role": "assistant", "content": response})
        # La siguiente sección estaba generando voz sintética; se comenta mientras se revisa.
        # audio_bytes = None
        # with st.spinner("Generando respuesta en audio..."):
        #     try:
        #         speech = client_openai.audio.speech.create(model=model_tts, voice="ash", input=response)
        #         audio_bytes = speech.read()
        #         if not audio_bytes:
        #             st.info("No se pudo obtener audio para esta respuesta.")
        #     except Exception as exc:
        #         st.error(f"No se pudo generar la voz sintética: {exc}")
        #
        # st.audio(audio_bytes, format="audio/mp3", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=True, width="stretch")
        #
        # if audio_bytes:
        #     last_message = st.session_state.messages[-1]
        #     if last_message["role"] == "assistant":
        #         last_message["audio"] = audio_bytes

elif section_choice == "Dashboard":
    st.divider()
    with st.expander("Dashboard financiero", expanded=True):
        viz_label_display = VIZ_KEY_TO_LABEL.get(st.session_state.dashboard_viz_key, "")
        st.caption(f"Visualiza la evolución de indicadores clave por año ({viz_label_display}).")
        if st.session_state.dashboard_error:
            st.error(f"No se pudo generar el panel: {st.session_state.dashboard_error}")
        elif not st.session_state.dashboard_loaded:
            st.info("Selecciona una compañía y presiona actualizar para cargar el panel.")
        else:
            for idx, fig in enumerate(st.session_state.dashboard_figures):
                st.plotly_chart(fig, use_container_width=True, key=f"dashboard_plot_{idx}")
elif section_choice == "Comparación":
    st.divider()
    with st.expander("Comparativa de compañías", expanded=True):
        viz_label_display = VIZ_KEY_TO_LABEL.get(st.session_state.comparison_viz_key, "")
        st.caption(f"Comparativa de {len(st.session_state.comparison_tickers)} compañía(s) ({viz_label_display}).")
        if st.session_state.comparison_error:
            st.error(f"No se pudo generar la comparativa: {st.session_state.comparison_error}")
        elif not st.session_state.comparison_loaded:
            st.info("Selecciona hasta tres compañías y presiona actualizar para ver la comparativa.")
        else:
            columns = st.columns(len(st.session_state.comparison_data))
            for col, panel in zip(columns, st.session_state.comparison_data):
                with col:
                    st.subheader(panel["ticker"])
                    for idx, fig in enumerate(panel["figures"]):
                        st.plotly_chart(fig, use_container_width=True, key=f"comparison_{panel['ticker']}_{idx}")

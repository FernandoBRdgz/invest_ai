import json
from utils import get_income_statement, get_balance_sheet, get_cashflow_statement, get_earnings
from utils import get_income_engineering, get_fcf_engineering, get_roic_engineering
from utils import get_call_transcripts, get_intrinsic_value

get_intrinsic_value_json = {
    "name": "get_intrinsic_value",
    "description": "Usa esta herramienta (tool) para obtener el valor intrínseco de la compañía solicitado por el usuario. \
                    Evita cálculos ajenos a la herramienta proporcionada como WACC, DFC, entre otros.\
                    Contrasta siempre con el precio final que también provee la herramienta para dar insights sin ser recomendación de compra o venta.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
        },
        "required": ["ticker"],
        "additionalProperties": False
    }
}


get_income_statement_json = {
    "name": "get_income_statement",
    "description": "Usa esta herramienta para obtener cualquier dato contenido en el estado de resultados de la companía que solicite el usuario.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
            "period": {
                "type": "string",
                "enum": ["anual", "trimestral"],
                "description": "Periodo del reporte a consultar, anual o trimestral."
            }
        },
        "required": ["ticker", "period"],
        "additionalProperties": False
    }
}

get_balance_sheet_json = {
    "name": "get_balance_sheet",
    "description": "Usa esta herramienta para obtener cualquier dato contenido en el balance general de la companía que solicite el usuario.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
            "period": {
                "type": "string",
                "enum": ["anual", "trimestral"],
                "description": "Periodo del reporte a consultar, anual o trimestral."
            }
        },
        "required": ["ticker", "period"],
        "additionalProperties": False
    }
}

get_cashflow_statement_json = {
    "name": "get_cashflow_statement",
    "description": "Usa esta herramienta para obtener cualquier dato contenido en el flujo de efectivo de la companía que solicite el usuario.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
            "period": {
                "type": "string",
                "enum": ["anual", "trimestral"],
                "description": "Periodo del reporte a consultar, anual o trimestral."
            }
        },
        "required": ["ticker", "period"],
        "additionalProperties": False
    }
}

get_earnings_json = {
    "name": "get_earnings",
    "description": "Usa esta herramienta para obtener cualquier dato contenido en los resultados de ganancias por acción de la companía que solicite el usuario.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
            "period": {
                "type": "string",
                "enum": ["anual", "trimestral"],
                "description": "Periodo del reporte a consultar, anual o trimestral."
            }
        },
        "required": ["ticker", "period"],
        "additionalProperties": False
    }
}

get_call_transcripts_json = {
    "name": "get_call_transcripts",
    "description": "Usa esta herramienta (tool) para obtener insights clave de una transcripción de earnings call",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "El TICKER de la compañía a analizar."
            },
            "quarter": {
                "type": "string",
                "description": "El trimestre de transcripciones de la llamada con Inversionistas"
            }
        },
        "required": ["ticker", "quarter"],
        "additionalProperties": False
    }
}

tools = [
         {"type": "function", "function": get_intrinsic_value_json},
         {"type": "function", "function": get_income_statement_json},
         {"type": "function", "function": get_balance_sheet_json},
         {"type": "function", "function": get_cashflow_statement_json},
         {"type": "function", "function": get_earnings_json},
         {"type": "function", "function": get_call_transcripts_json}
         ]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results
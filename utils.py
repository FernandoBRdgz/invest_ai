import os
import json
import time
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel, Field


load_dotenv(override=True)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def _convert_columns_to_numeric(df, columns_to_exclude=None):
    """
    Converts all columns in a DataFrame to numeric, except for those specified
    in the exclusion list.

    Args:
        df: pandas DataFrame.
        columns_to_exclude: A list of column names to exclude from conversion.
                            Defaults to None (convert all columns).

    Returns:
        pandas DataFrame with specified columns converted to numeric.
    """
    if columns_to_exclude is None:
        columns_to_exclude = []

    # Identify columns to convert to numeric
    columns_to_convert = df.columns.difference(columns_to_exclude)

    # Convert relevant columns to numeric, coercing errors
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def _calculate_financial_growth(df, columns):
    """
    Calculates the Year-over-Year growth for specified columns in a DataFrame
    using a custom logic for negative previous values.

    Args:
        df: pandas DataFrame with financial statement data.
        columns: A list of column names for which to calculate YoY growth.

    Returns:
        pandas DataFrame with YoY growth columns added for the specified columns.
    """
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df_sorted = df.sort_values(by='fiscalDateEnding').reset_index(drop=True)

    # Convert specified columns to numeric, coercing errors
    for col in columns:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

    # Define the custom Year-over-Year growth calculation function
    def calculate_yoy_growth_custom(current, previous):
        if pd.isna(previous) or previous == 0:
            return None
        if previous < 0:
            return ((current - previous) / abs(previous)) * 100
        else:
            return ((current - previous) / previous) * 100

    # Calculate Year-over-Year growth for each specified column using the custom function
    for col in columns:
        df_sorted[f'previous_{col}'] = df_sorted[col].shift(1)
        df_sorted[f'{col}_YoY_growth%'] = df_sorted.apply(
            lambda row: calculate_yoy_growth_custom(row[col], row[f'previous_{col}']),
            axis=1
        )
        df_sorted = df_sorted.drop(columns=[f'previous_{col}']) # Drop the temporary column

    return df_sorted[['fiscalDateEnding'] + [f'{col}_YoY_growth%' for col in columns]]

def _impute_yoy_median(df):
    """
    Imputes the median for columns containing "YoY" in a DataFrame.

    Args:
        df: pandas DataFrame.

    Returns:
        pandas DataFrame with NaN values in YoY columns imputed with the median.
    """
    yoy_columns = [col for col in df.columns if 'YoY' in col]
    for col in yoy_columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def _convert_columns_to_millions(df, columns_to_convert):
    """
    Converts specified numeric columns in a DataFrame to millions.

    Args:
        df: pandas DataFrame.
        columns_to_convert: A list of column names to convert to millions.

    Returns:
        pandas DataFrame with specified columns converted to millions.
    """
    # Convert specified columns to numeric, coercing errors and then fillna
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Divide the specified columns by 1,000,000
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col] / 1_000_000

    return df

def _fetch_alpha_vantage_data(url, description, ticker, max_retries=3, backoff_seconds=5):
    """Retrieve data from AlphaVantage with retries that respect burst limits."""
    for attempt in range(max_retries):
        response = requests.get(url)
        payload = response.json()
        note = payload.get('Note') or payload.get('Information')
        if note and 'Burst pattern detected' in note:
            if attempt == max_retries - 1:
                raise ValueError(
                    f"{description} ({ticker}) alcanzó el límite de peticiones. {note}"
                )
            time.sleep(backoff_seconds * (attempt + 1))
            continue
        return payload
    return payload


def get_income_statement(ticker, period = "anual"):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={alphavantage_api}'
    income_statement = _fetch_alpha_vantage_data(
        url_income,
        description="Estado de Resultados",
        ticker=ticker,
    )
    report_key = 'quarterlyReports' if period == "trimestral" else 'annualReports'
    income_df = _extract_reports_from_response(
        income_statement,
        report_key,
        description="Estado de Resultados",
        ticker=ticker,
        period=period,
    )
    income_df = _convert_columns_to_numeric(income_df, columns_to_exclude=['fiscalDateEnding', 'reportedCurrency'])
    return income_df.to_dict()

def get_balance_sheet(ticker, period = "anual"):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_balance = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={alphavantage_api}'
    balance_sheet = _fetch_alpha_vantage_data(
        url_balance,
        description="Balance General",
        ticker=ticker,
    )
    report_key = 'quarterlyReports' if period == "trimestral" else 'annualReports'
    balance_df = _extract_reports_from_response(
        balance_sheet,
        report_key,
        description="Balance General",
        ticker=ticker,
        period=period,
    )
    balance_df = _convert_columns_to_numeric(balance_df, columns_to_exclude=['fiscalDateEnding', 'reportedCurrency'])
    return balance_df.to_dict()

def get_cashflow_statement(ticker, period = "anual"):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_cashflow = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={alphavantage_api}'
    cashflow_statement = _fetch_alpha_vantage_data(
        url_cashflow,
        description="Flujo de Caja",
        ticker=ticker,
    )
    report_key = 'quarterlyReports' if period == "trimestral" else 'annualReports'
    cashflow_df = _extract_reports_from_response(
        cashflow_statement,
        report_key,
        description="Flujo de Caja",
        ticker=ticker,
        period=period,
    )
    cashflow_df = _convert_columns_to_numeric(cashflow_df, columns_to_exclude=['fiscalDateEnding', 'reportedCurrency'])
    return cashflow_df.to_dict()

def get_earnings(ticker, period = "anual"):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_earnings = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={alphavantage_api}'
    earnings_history = _fetch_alpha_vantage_data(
        url_earnings,
        description="Historial de ganancias",
        ticker=ticker,
    )
    report_key = 'quarterlyEarnings' if period == "trimestral" else 'annualEarnings'
    earnings_df = _extract_reports_from_response(
        earnings_history,
        report_key,
        description="Historial de ganancias",
        ticker=ticker,
        period=period,
    )
    earnings_df['reportedEPS'] = pd.to_numeric(earnings_df['reportedEPS'], errors='coerce')
    return earnings_df.to_dict()

keep_income_cols = ['fiscalDateEnding','totalRevenue','totalRevenue_YoY_growth%',
                    'ebitda','ebitdaMargin%','ebitda_YoY_growth%','depreciationDepletionAndAmortization',
                    'operatingIncome','ebitMargin%','operatingIncome_YoY_growth%',
                    'interestExpense','interestIncome','netInterest',
                    'incomeBeforeTax','incomeTaxExpense','taxRate%',
                    'netIncome','netIncomeMargin%','netIncome_YoY_growth%',
                    'reportedEPS','reportedEPS_YoY_growth%',
                    'commonStockSharesOutstanding','commonStockSharesOutstanding_YoY_growth%']

keep_fcf_cols = ['fiscalDateEnding','ebitda','capitalExpenditures',
       'maintenanceCapex', 'growthCapex','interestExpense',
       'incomeTaxExpense','inventory','currentNetReceivables', 'currentAccountsPayable',
       'deferredRevenue','workingCapital','changeInWC', 'freeCashFlow',
       'freeCashFlow_YoY_growth%','freeCashFlowPerShare',
       'freeCashFlowPerShare_YoY_growth%','maintenanceCapexMargin%',
       'workingCapitalMargin%','freeCashFlowMargin%','cashConversion%']

keep_roic_cols = ['fiscalDateEnding','cashAndShortTermInvestments','shortTermInvestments',
                  'shortTermDebt','longTermDebt','capitalLeaseObligations',
                  'totalShareholderEquity','investedCapital','ROA%','ROE%','ROIC%','reinvestmentRate%']


def _extract_reports_from_response(response: dict, report_key: str, description: str, ticker: str, period: str):
    """Safely extract a list of reports from the AlphaVantage response."""
    reports = response.get(report_key)
    if reports is None:
        detail = response.get('Note') or response.get('Error Message') or response.get('Information')
        if detail and not isinstance(detail, str):
            try:
                detail = json.dumps(detail)
            except TypeError:
                detail = str(detail)
        detail_text = detail or 'Sin detalles adicionales.'
        raise ValueError(
            f"{description} ({ticker}, periodo='{period}') no devolvió '{report_key}'. {detail_text}"
        )
    if not isinstance(reports, list):
        raise ValueError(
            f"{description} ({ticker}, periodo='{period}') devolvió un tipo inesperado para '{report_key}' ({type(reports).__name__})."
        )
    return pd.DataFrame.from_dict(reports)


def get_income_engineering(ticker):
    income_df = pd.DataFrame(get_income_statement(ticker=ticker, period="anual"))
    earnings_df = pd.DataFrame(get_earnings(ticker=ticker, period="anual"))
    balance_df = pd.DataFrame(get_balance_sheet(ticker=ticker, period="anual"))
    cashflow_df = pd.DataFrame(get_cashflow_statement(ticker=ticker, period="anual"))

    # Ensure fiscalDateEnding is in datetime format for both dataframes before merging
    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])
    earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])

    # Merge the dataframes
    income_df = pd.merge(income_df, earnings_df, on='fiscalDateEnding', how='left')

    # Ensure fiscalDateEnding is in datetime format for both dataframes before merging
    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])
    balance_df['fiscalDateEnding'] = pd.to_datetime(balance_df['fiscalDateEnding'])

    # Merge the dataframes
    income_df = pd.merge(income_df, balance_df[['fiscalDateEnding','commonStockSharesOutstanding']], on='fiscalDateEnding', how='left')

    # Ensure fiscalDateEnding is in datetime format for both dataframes before merging
    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])
    cashflow_df['fiscalDateEnding'] = pd.to_datetime(cashflow_df['fiscalDateEnding'])

    # Merge the dataframes
    income_df = pd.merge(income_df, cashflow_df[['fiscalDateEnding','depreciationDepletionAndAmortization']], on='fiscalDateEnding', how='left')

    # EBITDA Margin
    income_df['ebitdaMargin%'] = income_df['ebitda']*100 / income_df['totalRevenue']
    # EBIT Margin
    income_df['ebitMargin%'] = income_df['operatingIncome']*100 / income_df['totalRevenue']
    # Net income Margin
    income_df['netIncomeMargin%'] = income_df['netIncome']*100 / income_df['totalRevenue']
    # Net Interest
    income_df['netInterest'] = income_df['interestIncome'] - income_df['interestExpense']
    # Tax Rate
    income_df['taxRate%'] = income_df['incomeTaxExpense']*100 / income_df['incomeBeforeTax']

    income_df['reportedEPS'] = income_df['reportedEPS'].fillna(0)
    income_df['interestIncome'] = income_df['interestIncome'].fillna(0)
    income_df['netInterest'] = income_df['interestIncome'] - income_df['interestExpense']

    income_columns_to_grow = ['totalRevenue','grossProfit','operatingIncome','ebitda',
                              'netIncome','reportedEPS','commonStockSharesOutstanding']
    income_df_with_growth = _calculate_financial_growth(income_df.copy(), income_columns_to_grow)

    # Merge the two dataframes on 'fiscalDateEnding'
    income_df = pd.merge(income_df, income_df_with_growth, on='fiscalDateEnding', how='left')

    income_df = income_df[keep_income_cols].copy()
    income_df['reportedEPS_YoY_growth%'] = income_df['reportedEPS_YoY_growth%'].fillna(0)
    income_df = _impute_yoy_median(income_df)

    return income_df.to_dict()

def get_fcf_engineering(ticker):
    cashflow_df = pd.DataFrame(get_cashflow_statement(ticker=ticker, period="anual"))
    income_df = pd.DataFrame(get_income_statement(ticker=ticker, period="anual"))
    balance_df = pd.DataFrame(get_balance_sheet(ticker=ticker, period="anual"))

    fcf_df = cashflow_df[['fiscalDateEnding','operatingCashflow','depreciationDepletionAndAmortization','capitalExpenditures']].copy()
    fcf_df['maintenanceCapex'] = fcf_df.apply(
        lambda row: row['capitalExpenditures'] if row['capitalExpenditures'] < row['depreciationDepletionAndAmortization'] else row['depreciationDepletionAndAmortization'],
        axis=1
    )
    fcf_df['growthCapex'] = fcf_df['capitalExpenditures'] - fcf_df['maintenanceCapex']
    # Ensure fiscalDateEnding is in datetime format for both dataframes before merging
    fcf_df['fiscalDateEnding'] = pd.to_datetime(fcf_df['fiscalDateEnding'])
    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])

    # Merge the dataframes
    fcf_df = pd.merge(fcf_df, income_df[['fiscalDateEnding','totalRevenue','operatingIncome','interestExpense','incomeTaxExpense']], on='fiscalDateEnding', how='left')
    
    # Ensure fiscalDateEnding is in datetime format for both dataframes before merging
    fcf_df['fiscalDateEnding'] = pd.to_datetime(fcf_df['fiscalDateEnding'])
    balance_df['fiscalDateEnding'] = pd.to_datetime(balance_df['fiscalDateEnding'])

    # Merge the dataframes
    fcf_df = pd.merge(fcf_df, balance_df[['fiscalDateEnding','inventory','currentNetReceivables',
                                        'currentAccountsPayable','deferredRevenue',
                                        'otherNonCurrentLiabilities','commonStockSharesOutstanding']], on='fiscalDateEnding', how='left')
    fcf_df = fcf_df.fillna(0)

    # Convert float columns to integers
    float_cols = fcf_df.select_dtypes(include='float64').columns
    for col in float_cols:
        fcf_df[col] = fcf_df[col].astype(int)

    # Replace 'deferredRevenue' with 'otherNonCurrentLiabilities' if 'deferredRevenue' is not positive
    fcf_df['deferredRevenue'] = fcf_df.apply(
        lambda row: row['otherNonCurrentLiabilities'] if row['deferredRevenue'] <= 0 else row['deferredRevenue'],
        axis=1
    )

    # Drop the 'otherNonCurrentLiabilities' column
    fcf_df = fcf_df.drop(columns=['otherNonCurrentLiabilities'])

    # Impute the median for 'deferredRevenue' if necessary
    if fcf_df['deferredRevenue'].isnull().any():
        median_deferredRevenue = fcf_df['deferredRevenue'].median()
        fcf_df['deferredRevenue'] = fcf_df['deferredRevenue'].fillna(median_deferredRevenue)
    
    fcf_df['ebitda'] = fcf_df['operatingIncome'] + fcf_df['depreciationDepletionAndAmortization']
    fcf_df['workingCapital'] = fcf_df['inventory'] + fcf_df['currentNetReceivables'] - fcf_df['currentAccountsPayable'] - fcf_df['deferredRevenue']

    fcf_df['previous_currentAccountsPayable'] = fcf_df['currentAccountsPayable'].shift(-1)
    fcf_df['previous_workingCapital'] = fcf_df['workingCapital'].shift(-1)

    fcf_df['changeInWC'] = fcf_df.apply(lambda row: (row['workingCapital'] - row['previous_workingCapital']) if row['previous_currentAccountsPayable'] > 0 else 0, axis=1)

    # Drop the temporary columns
    fcf_df = fcf_df.drop(columns=['previous_currentAccountsPayable', 'previous_workingCapital'])

    fcf_df['freeCashFlow'] = fcf_df['ebitda'] - fcf_df['interestExpense'] - fcf_df['incomeTaxExpense'] - fcf_df['maintenanceCapex'] - fcf_df['changeInWC']
    fcf_df['freeCashFlowPerShare'] = fcf_df['freeCashFlow'] / fcf_df['commonStockSharesOutstanding']
    fcf_df['freeCashFlow2'] = fcf_df['operatingCashflow'] - fcf_df['capitalExpenditures']
    fcf_df_with_growth = _calculate_financial_growth(fcf_df.copy(), ['freeCashFlow','freeCashFlowPerShare'])
    
    # Merge the two dataframes on 'fiscalDateEnding'
    fcf_df = pd.merge(fcf_df, fcf_df_with_growth, on='fiscalDateEnding', how='left')

    # Calculate the requested ratios and add them to the fcf_df DataFrame
    fcf_df['maintenanceCapexMargin%'] = fcf_df['maintenanceCapex']*100 / fcf_df['totalRevenue']
    fcf_df['workingCapitalMargin%'] = fcf_df['workingCapital']*100 / fcf_df['totalRevenue']
    fcf_df['freeCashFlowMargin%'] = fcf_df['freeCashFlow']*100 / fcf_df['totalRevenue']
    fcf_df['cashConversion%'] = fcf_df['freeCashFlow']*100 / fcf_df['ebitda']

    fcf_df = fcf_df[keep_fcf_cols]
    fcf_df = _impute_yoy_median(fcf_df)
    return fcf_df.to_dict()

def get_roic_engineering(ticker):
    balance_df = pd.DataFrame(get_balance_sheet(ticker=ticker, period="anual"))
    income_df = pd.DataFrame(get_income_engineering(ticker=ticker))
    fcf_df = pd.DataFrame(get_fcf_engineering(ticker))

    roic_df = balance_df[['fiscalDateEnding', 'cashAndShortTermInvestments','shortTermInvestments','shortTermDebt',
                      'longTermDebt','capitalLeaseObligations','totalAssets','totalShareholderEquity']].copy()

    # Convert fiscalDateEnding to datetime for merging
    roic_df['fiscalDateEnding'] = pd.to_datetime(roic_df['fiscalDateEnding'], errors='coerce')
    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'], errors='coerce')

    # Merge roic_df with income_df on fiscalDateEnding
    roic_df = pd.merge(roic_df, income_df[['fiscalDateEnding','operatingIncome','taxRate%','netIncome']], on='fiscalDateEnding', how='left')

    # Convert fiscalDateEnding to datetime for merging
    roic_df['fiscalDateEnding'] = pd.to_datetime(roic_df['fiscalDateEnding'], errors='coerce')
    fcf_df['fiscalDateEnding'] = pd.to_datetime(fcf_df['fiscalDateEnding'], errors='coerce')

    # Merge roic_df with fcf_df on fiscalDateEnding
    roic_df = pd.merge(roic_df, fcf_df[['fiscalDateEnding','growthCapex','freeCashFlow']], on='fiscalDateEnding', how='left')

    # Identify columns to convert to numeric (all except 'fiscalDateEnding')
    roic_df = _convert_columns_to_numeric(roic_df, columns_to_exclude=['fiscalDateEnding'])

    # Tax Rate is 21% in USA
    roic_df['taxRate%'] = roic_df['taxRate%'].fillna(21.0)

    # Fill NaN values with 0 before performing the calculation
    roic_df = roic_df.fillna(0)

    # Create the 'investedCapital' feature
    roic_df['investedCapital'] = roic_df['totalShareholderEquity'] + roic_df['shortTermDebt'] + roic_df['longTermDebt'] + roic_df['capitalLeaseObligations'] - roic_df['shortTermInvestments']

    # Calculate ROA, ROE, NOPAT, and ROIC
    roic_df['ROA%'] = roic_df['netIncome']*100 / roic_df['totalAssets']
    roic_df['ROE%'] = roic_df['netIncome']*100 / roic_df['totalShareholderEquity']
    roic_df['NOPAT'] = roic_df['operatingIncome'] * (1 - roic_df['taxRate%'] / 100)
    roic_df['ROIC%'] = roic_df['NOPAT']*100 / roic_df['investedCapital']

    # Replace infinite values with 0 in the specified columns
    for col in ['ROA%', 'ROE%', 'ROIC%']:
        roic_df[col] = roic_df[col].replace([np.inf, -np.inf], 0)

    # Calculate reinvestmentRate based on the condition
    roic_df['reinvestmentRate%'] = roic_df.apply(lambda row: row['growthCapex']*100/row['freeCashFlow'] if row['freeCashFlow'] > 0 else 0, axis=1)

    roic_df = roic_df[keep_roic_cols].copy()
    return roic_df.to_dict()

class EarningsCallInsights(BaseModel):
    """
    Output estructurado con insights clave en español extraídos de una earnings call.
    """
    symbol: Optional[str] = Field(description="Ticker de la compañía, ej. AAPL, AMZN, MSFT, etc")
    sentiment: Optional[str] = Field(description="Sentimiento general: Muy bajista / Bajista / Alcista / Muy alcista")
    summary: str = Field(description="Resumen ejecutivo de la llamada en español (párrafo de 3 líneas)")
    key_topics: List[str] = Field(default_factory=list, description="Temas estratégicos principales, cado uno con un máximo de 4 palabras")
    guidance: List[str] = Field(default_factory=list, description="Guías o proyecciones futuras mencionadas, en español")
    numeric_highlights: List[str] = Field(default_factory=list, description="Cifras o métricas clave reportadas, en español")
    risks: List[str] = Field(default_factory=list, description="Riesgos explícitos o implícitos discutidos, en español")
    catalysts: List[str] = Field(default_factory=list, description="Catalizadores futuros o eventos relevantes, en español")
    analyst_questions: List[str] = Field(default_factory=list, description="Top 3 preguntas destacadas de analistas, en español")
    unanswered_topics: List[str] = Field(default_factory=list, description="Temas abiertos o sin respuesta clara, en español")
    bullish_points: List[str] = Field(default_factory=list, description="Tesis alcistas derivadas de la llamada, en español")
    bearish_points: List[str] = Field(default_factory=list, description="Tesis bajistas derivadas de la llamada, en español")
    red_flags: List[str] = Field(default_factory=list, description="Alertas o señales negativas detectadas, en español")
    emotion: Optional[str] = Field(description="Emoción general: Optimismo / Incertidumbre / Preocupación / Entusiasmo / Frustración")

def _render_transcript(d: dict) -> str:
    """
    Renders the transcript data from a dictionary into a formatted string.

    Args:
        d: A dictionary containing the transcript data. Expected keys are 'symbol',
           'quarter', and 'transcript'. The 'transcript' key should contain a list
           of dictionaries, each with 'speaker', 'title' (optional), and 'content'.

    Returns:
        A formatted string representation of the transcript.
    """
    header_info = f"Symbol: {d.get('symbol','')} | Quarter: {d.get('quarter','')}"
    transcript_lines = [header_info, "TRANSCRIPT:"]
    for index, transcript_entry in enumerate(d.get("transcript", [])):
        speaker_name = transcript_entry.get("speaker", "Unknown")
        speaker_title = transcript_entry.get("title")
        utterance_text = (transcript_entry.get("content") or "").strip()
        speaker_tag = f"{speaker_name} ({speaker_title})" if speaker_title else speaker_name
        transcript_lines.append(f"[{index:04d}] {speaker_tag}: {utterance_text}")
    return "\n".join(transcript_lines)

def _get_earnings_call_insights(call_transcripts: dict) -> EarningsCallInsights:
    """
    Obtiene insights clave de una transcripción de earnings call utilizando un modelo de OpenAI.

    Args:
        transcript_text: El texto de la transcripción de la llamada.

    Returns:
        Un objeto diccionario derivado de EarningsCallInsights con los insights extraídos.
    """
    client = OpenAI()
    transcript_text = _render_transcript(call_transcripts)
    response = client.chat.completions.parse(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Eres un experto Analista Financiero. Devuelve SOLO un JSON válido que siga exactamente el esquema de EarningsCallInsights. Salidas en español."},
            {"role": "user", "content": transcript_text},
        ],
        response_format=EarningsCallInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()

def get_call_transcripts(ticker: str, quarter: str) -> dict:
    client = OpenAI()
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_transcripts = f'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={quarter}&apikey={alphavantage_api}'
    response_transcripts = requests.get(url_transcripts)
    call_transcripts = response_transcripts.json()
    insights = _get_earnings_call_insights(call_transcripts=call_transcripts)
    return insights

def _fetch_and_process_price_data(ticker):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_price = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={alphavantage_api}'
    response_price = requests.get(url_price)
    price_monthly = response_price.json()

    if 'Monthly Adjusted Time Series' not in price_monthly:
        raise ValueError(f"Could not retrieve price data for {ticker}. Response: {price_monthly}")

    price_df = pd.DataFrame.from_dict(price_monthly['Monthly Adjusted Time Series'], orient='index')
    price_df.index = pd.to_datetime(price_df.index)
    price_df.sort_index(inplace=True)
    price_df['adjusted_close'] = pd.to_numeric(price_df['5. adjusted close'])
    price_df = price_df.resample('YE')['adjusted_close'].last().to_frame()
    return int(price_df['adjusted_close'].iloc[-1])


def _get_monthly_price_df(ticker):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    url_price = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={alphavantage_api}'
    response_price = requests.get(url_price)
    price_monthly = response_price.json()
    if 'Monthly Adjusted Time Series' not in price_monthly:
        raise ValueError(f"Could not retrieve price history for {ticker}. Response: {price_monthly}")
    price_df = pd.DataFrame.from_dict(price_monthly['Monthly Adjusted Time Series'], orient='index')
    price_df.index = pd.to_datetime(price_df.index)
    price_df.sort_index(inplace=True)
    price_df['adjusted_close'] = pd.to_numeric(price_df['5. adjusted close'])
    return price_df[['adjusted_close']]


def get_price_history(ticker):
    """
    Devuelve el historial mensual ajustado del precio para ayudar a las visualizaciones.
    """
    price_df = _get_monthly_price_df(ticker)
    return price_df.copy()

def _load_and_merge_financial_data(ticker):
    income_df = pd.DataFrame(get_income_engineering(ticker))
    fcf_df = pd.DataFrame(get_fcf_engineering(ticker))
    roic_df = pd.DataFrame(get_roic_engineering(ticker))

    income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'], errors='coerce')
    fcf_df['fiscalDateEnding'] = pd.to_datetime(fcf_df['fiscalDateEnding'], errors='coerce')
    roic_df['fiscalDateEnding'] = pd.to_datetime(roic_df['fiscalDateEnding'], errors='coerce')

    financial_df = pd.merge(income_df, fcf_df, on='fiscalDateEnding', how='left')
    financial_df = pd.merge(financial_df, roic_df, on='fiscalDateEnding', how='left')
    return financial_df

def _calculate_financial_ratios(financial_df, last_price):
    financial_df['netDebt'] = financial_df['shortTermDebt'] + financial_df['longTermDebt'] - financial_df['cashAndShortTermInvestments'] - financial_df['shortTermInvestments']
    financial_df['netDebt/EBITDA'] = financial_df['netDebt'] / financial_df['ebitda_y']
    financial_df['marketCapitalization'] = financial_df['commonStockSharesOutstanding'] * last_price
    financial_df['enterpriseValue'] = financial_df['marketCapitalization'] + financial_df['netDebt']
    financial_df['PER'] = financial_df['marketCapitalization'] / financial_df['netIncome']
    financial_df['EV/FCF'] = financial_df['enterpriseValue'] / financial_df['freeCashFlow']
    financial_df['EV/EBITDA'] = financial_df['enterpriseValue'] / financial_df['ebitda_y']
    financial_df['EV/EBIT'] = financial_df['enterpriseValue'] / financial_df['operatingIncome']
    return financial_df

def _prepare_forecast_data(financial_df):
    forecast_df = financial_df[['fiscalDateEnding', 'commonStockSharesOutstanding', 'netDebt', 'freeCashFlow', 'operatingIncome', 'netIncome', 'ebitda_y']].copy()
    forecast_df['fiscalDateEnding'] = pd.to_datetime(forecast_df['fiscalDateEnding'], errors='coerce')
    forecast_df = forecast_df.sort_values(by='fiscalDateEnding').reset_index(drop=True)

    numerical_cols = ['commonStockSharesOutstanding', 'netDebt', 'freeCashFlow', 'operatingIncome', 'netIncome', 'ebitda_y']
    last_historical_values = forecast_df[numerical_cols].iloc[-1]

    last_year = forecast_df['fiscalDateEnding'].dt.year.max()
    future_years = pd.to_datetime([f"{last_year + i}-12-31" for i in range(1, 6)])

    future_df = pd.DataFrame({'fiscalDateEnding': future_years})

    for col in numerical_cols:
        future_df[col] = last_historical_values[col]

    extended_forecast_df = pd.concat([forecast_df, future_df], ignore_index=True)
    extended_forecast_df = extended_forecast_df.sort_values(by='fiscalDateEnding').reset_index(drop=True)
    return extended_forecast_df

def _calculate_valuation_metrics(extended_forecast_df, per_multiple, ev_fcf_multiple, ev_ebitda_multiple, ev_ebit_multiple):
    numerical_cols_extended = extended_forecast_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    valuation_df = pd.DataFrame({'fiscalDateEnding': extended_forecast_df['fiscalDateEnding']})

    for col in numerical_cols_extended:
        valuation_df[f'EMA_5Y_{col}'] = extended_forecast_df[col].ewm(span=5, adjust=False).mean()

    valuation_df['priceEV/FCF'] = ((valuation_df['EMA_5Y_freeCashFlow'] * ev_fcf_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_commonStockSharesOutstanding']
    valuation_df['priceEV/EBITDA'] = ((valuation_df['EMA_5Y_ebitda_y'] * ev_ebitda_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_commonStockSharesOutstanding']
    valuation_df['priceEV/EBIT'] = ((valuation_df['EMA_5Y_operatingIncome'] * ev_ebit_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_commonStockSharesOutstanding']
    valuation_df['pricePERexCash'] = np.where(
        valuation_df['EMA_5Y_netDebt'] < 0,
        ((valuation_df['EMA_5Y_netIncome'] * per_multiple) - valuation_df['EMA_5Y_netDebt']) / valuation_df['EMA_5Y_commonStockSharesOutstanding'],
        (valuation_df['EMA_5Y_netIncome'] * per_multiple) / valuation_df['EMA_5Y_commonStockSharesOutstanding'])

    intrinsic_value_df = valuation_df[['fiscalDateEnding', 'priceEV/FCF', 'priceEV/EBITDA', 'priceEV/EBIT', 'pricePERexCash']].tail()
    intrinsic_value_df.set_index('fiscalDateEnding', inplace=True)
    return intrinsic_value_df.iloc[-1].median()

def get_intrinsic_value(ticker):
    alphavantage_api = ALPHAVANTAGE_API_KEY
    last_price = _fetch_and_process_price_data(ticker)
    financial_df = _load_and_merge_financial_data(ticker)
    financial_df = _calculate_financial_ratios(financial_df, last_price)

    per_multiple = int(financial_df['PER'].iloc[0])
    ev_fcf_multiple = int(financial_df['EV/FCF'].iloc[0])
    ev_ebitda_multiple = int(financial_df['EV/EBITDA'].iloc[0])
    ev_ebit_multiple = int(financial_df['EV/EBIT'].iloc[0])

    extended_forecast_df = _prepare_forecast_data(financial_df)
    median_intrinsic_value = _calculate_valuation_metrics(extended_forecast_df, per_multiple, ev_fcf_multiple, ev_ebitda_multiple, ev_ebit_multiple)

    return {'intrinsic_value': median_intrinsic_value, 'last_price': last_price}

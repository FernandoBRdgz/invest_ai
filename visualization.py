import random
import pandas as pd
import plotly.graph_objects as go
from utils import _convert_columns_to_millions
from utils import get_income_engineering, get_fcf_engineering, get_roic_engineering, get_price_history

# List of columns to display in millions
columns_in_millions_income = ['totalRevenue', 'ebitda', 'depreciationDepletionAndAmortization',
                       'operatingIncome', 'interestExpense', 'interestIncome', 'netInterest',
                       'incomeBeforeTax', 'incomeTaxExpense', 'netIncome',
                       'commonStockSharesOutstanding']

columns_in_millions_fcf = ['ebitda','capitalExpenditures',
       'maintenanceCapex', 'growthCapex','interestExpense',
       'incomeTaxExpense','inventory','currentNetReceivables', 'currentAccountsPayable',
       'deferredRevenue','workingCapital','changeInWC', 'freeCashFlow']

columns_in_millions_roic = ['cashAndShortTermInvestments','shortTermInvestments','shortTermDebt','longTermDebt',
                            'capitalLeaseObligations','totalShareholderEquity','investedCapital']


def generate_financial_bar_charts(financial_data_transposed: pd.DataFrame):
    """
    Generates Plotly bar charts for financial metrics over time from a transposed DataFrame.

    Args:
        financial_data_transposed: A pandas DataFrame where the index are the metrics
                                   and columns are the years.

    Returns:
        A list of Plotly Figure objects.
    """
    figs = []
    # Convert the index to string type for Plotly compatibility
    financial_data_transposed.index = financial_data_transposed.index.astype(str)

    for index, row in financial_data_transposed.iterrows():
        # Determine the y-axis title based on the index name
        if index == 'EPS':
            yaxis_title = ''
        elif index.endswith('%'):
            yaxis_title = 'Porcentaje'
        else:
            yaxis_title = 'En millones'

        fig = go.Figure()
        # Generate a random color for the bars
        random_color = f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'

        # Convert row values to numeric before plotting
        row_values_numeric = pd.to_numeric(row.values, errors='coerce')

        fig.add_trace(go.Bar(x=financial_data_transposed.columns, y=row_values_numeric, name=index, marker_color=random_color))
        fig.update_layout(
            title=index,  # Use the index name as the graph title
            xaxis_title='Year',
            yaxis_title=yaxis_title,
            xaxis=dict(
                tickmode='array',
                tickvals=financial_data_transposed.columns,
                ticktext=financial_data_transposed.columns
            )
        )
        figs.append(fig)
    return figs


def viz_income_engineering(ticker):
    income_df = pd.DataFrame(get_income_engineering(ticker))
    income_table = _convert_columns_to_millions(income_df.copy(), columns_in_millions_income)
    income_table['fiscalDateEnding'] = income_table['fiscalDateEnding'].dt.year

    income_mapping = {'fiscalDateEnding':f'{ticker}',
                   'totalRevenue':'Ventas',
                   'totalRevenue_YoY_growth%':'Crecimiento ventas interanual%',
                   'ebitda':'EBITDA',
                   'ebitdaMargin%':'Márgen EBITDA%',
                   'ebitda_YoY_growth%':'Crecimiento EBITDA interanual%',
                   'depreciationDepletionAndAmortization':'Depreciación y Amortización',
                   'operatingIncome':'EBIT',
                   'ebitMargin%':'Márgen EBIT%',
                   'operatingIncome_YoY_growth%':'Crecimiento EBIT interanual%',
                   'interestExpense':'Gastos por intereses',
                   'interestIncome':'Ingresos por intereses',
                   'netInterest':'Interés neto',
                   'incomeBeforeTax':'Ingreso antes de impuestos',
                   'incomeTaxExpense':'Gasto en impuestos',
                   'taxRate%':'Tasa impositiva%',
                   'netIncome':'Ingresos netos',
                   'netIncomeMargin%':'Márgen neto%',
                   'netIncome_YoY_growth%':'Crecimiento neto interanual%',
                   'reportedEPS':'EPS',
                   'reportedEPS_YoY_growth%':'Crecimiento EPS interanual%',
                   'commonStockSharesOutstanding':'Acciones diluidas en circulación',
                   'commonStockSharesOutstanding_YoY_growth%':'Crecimiento acciones interanual%'
                  }
    
    income_table = income_table.rename(columns=income_mapping)
    income_viz = income_table.sort_values(by=ticker).set_index(ticker).T
    
    generated_figs = generate_financial_bar_charts(income_viz)
    return generated_figs

def viz_fcf_engineering(ticker):
    fcf_df = pd.DataFrame(get_fcf_engineering(ticker))
    fcf_table = _convert_columns_to_millions(fcf_df.copy(), columns_in_millions_fcf)
    fcf_table['fiscalDateEnding'] = fcf_table['fiscalDateEnding'].dt.year

    fcf_mapping = {'fiscalDateEnding':f'{ticker}',
               'ebitda':'EBITDA',
               'capitalExpenditures':'Gastos de Capital',
               'maintenanceCapex':'Gastos de Mantenimieto',
               'growthCapex':'Gastos de Expansión',
               'interestExpense':'Gastos por Intereses',
               'incomeTaxExpense':'Gasto en Impuestos',
               'inventory':'Inventarios',
               'currentNetReceivables':'Cuentas por Cobrar',
               'currentAccountsPayable':'Cuentas por Pagar',
               'deferredRevenue':'Ingresos Diferidos',
               'workingCapital':'Capital de Trabajo',
               'changeInWC':'Cambio en Capital de Trabajo',
               'freeCashFlow':'Flujo de Caja',
               'freeCashFlow_YoY_growth%':'Crecimiento Flujo Caja interanual%',
               'freeCashFlowPerShare':'Flujo de Caja por acción',
               'freeCashFlowPerShare_YoY_growth%':'Crecimiento Flujo Caja por acción interanual%',
               'maintenanceCapexMargin%':'Costo de Mantenimieto%',
               'workingCapitalMargin%':'Costo de Capital de Trabajo%',
               'freeCashFlowMargin%':'Márgen de Flujo de Caja%',
               'cashConversion%':'Conversión a Efectivo%'
    }
    fcf_table = fcf_table.rename(columns=fcf_mapping)
    fcf_viz = fcf_table.sort_values(by=ticker).set_index(ticker).T
    generated_figs = generate_financial_bar_charts(fcf_viz)
    return generated_figs

def viz_roic_engineering(ticker):
    roic_df = pd.DataFrame(get_roic_engineering(ticker))
    roic_table = _convert_columns_to_millions(roic_df.copy(), columns_in_millions_roic)
    roic_table['fiscalDateEnding'] = roic_table['fiscalDateEnding'].dt.year

    roic_mapping = {'fiscalDateEnding':f'{ticker}',
                  'cashAndShortTermInvestments':'Efectivo',
                  'shortTermInvestments':'Inversiones a Corto Plazo',
                  'shortTermDebt':'Deuda a Corto Plazo',
                  'longTermDebt':'Deuda a Largo Plazo',
                  'capitalLeaseObligations':'Obligaciones de Arrendamiento',
                  'totalShareholderEquity':'Capital de los Accionistas',
                  'investedCapital':'Capital invertido',
                  'ROA%':'Retorno sobre los Activos%',
                  'ROE%':'Retorno sobre el Capital%',
                  'ROIC%':'Retorno sobre el Capital Invertido%',
                  'reinvestmentRate%':'Tasa de Reinversión%'
    }
    roic_table = roic_table.rename(columns=roic_mapping)
    roic_viz = roic_table.sort_values(by=ticker).set_index(ticker).T
    generated_figs = generate_financial_bar_charts(roic_viz)
    return generated_figs


def viz_price_history(ticker):
    price_df = get_price_history(ticker)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df['adjusted_close'],
            mode='lines',
            name='Precio ajustado',
            line=dict(color='#1f77b4'),
        )
    )
    fig.update_layout(
        title=f'{ticker} — Precio ajustado mensual',
        xaxis_title='Fecha',
        yaxis_title='Precio ajustado (USD)',
        hovermode='x unified',
    )
    return fig

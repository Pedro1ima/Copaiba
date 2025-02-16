from flask import Flask, request, jsonify, render_template
import pandas as pd
import urllib.request
import requests
import json
import re
from textwrap import shorten
from bs4 import BeautifulSoup
from io import BytesIO
import gzip
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/correlacao')
def correlacao():
    return render_template('correlacao.html') 

@app.route('/indicadores')
def indicadores():
    return render_template('indicadores.html') 

# Função que busca os dados do fundo usando o CNPJ
def get_data(cnpj):
    url = f"https://www.okanebox.com.br/api/fundoinvestimento/hist/{cnpj}/19000101/21000101/"
    req = urllib.request.Request(url)
    req.add_header('Accept-Encoding', 'gzip')
    req.add_header('Authorization', 'Bearer caianfrancodecamargo@gmail.com')
    
    try:
        response = urllib.request.urlopen(req)
    
        if response.info().get('Content-Encoding') == 'gzip':
            buf = BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            dados_historicos = json.loads(f.read().decode("utf-8"))
        else:
            content = response.read().decode("utf-8")
            dados_historicos = json.loads(content)

        df = pd.DataFrame(dados_historicos)
        df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
                
        return df

    except Exception as e:
        print(f"Erro ao buscar dados para CNPJ {cnpj}: {e}")
        return None

# Função que busca os dados do Ibovespa
def get_ibovespa_data(df):
    start_date = df.iloc[0]['DT_COMPTC']
    end_date = df.iloc[-1]['DT_COMPTC'] + pd.DateOffset(days=1)
    
    df_ibovespa = yf.download('^BVSP', start=start_date, end=end_date, progress=False)
    df_ibovespa = df_ibovespa.reset_index()
    df_ibovespa = df_ibovespa.rename(columns={'Date': 'DT_COMPTC'})

    return df_ibovespa

#####################################################################################################
# PÁGINA DA CORRELAÇÃO 

# Função para obter o nome do fundo de investimento a partir do CNPJ
def obter_nome_fundo(cnpj):
    cnpj = re.sub(r'\D', '', cnpj)
    url = f"https://www.okanebox.com.br/w/fundo-investimento/{cnpj}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        nome_fundo = soup.find('h1').get_text(strip=True)
        return nome_fundo
    else:
        return "Erro ao acessar a página."

# Função para criar URLs a partir dos CNPJs fornecidos
def criar_urls(cnpjs):
    fundos = {}
    base_url = "https://www.okanebox.com.br/api/fundoinvestimento/hist/"
    for cnpj in cnpjs:
        cnpj = re.sub(r'\D', '', cnpj)
        nome_fundo = obter_nome_fundo(cnpj)
        nome_fundo_abrev = shorten(nome_fundo, width=20, placeholder="")
        url = f"{base_url}{cnpj}/19000101/21000101/"
        fundos[nome_fundo_abrev] = url
    return fundos

# Função para coletar dados dos fundos
def coletar_dados_fundos(fundos):
    dados_fundos = {}
    for nome, url in fundos.items():
        response = requests.get(url)
        try:
            dados_historicos = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            print(f"Erro ao ler o conteúdo JSON: {e}")
            continue

        if dados_historicos:
            df = pd.DataFrame(dados_historicos)
            df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
            df.set_index('DT_COMPTC', inplace=True)
            dados_fundos[nome] = df['VL_QUOTA'].pct_change().dropna()  # Calcular retornos diários
        else:
            print(f"Nenhum dado retornado para o URL: {url}")

    return dados_fundos

# Função para coletar os dados históricos do CDI
def coletar_dados_cdi(start_date, end_date):
    df_cdi = yf.download('^IRX', start=start_date, end=end_date, progress=False)  # CDI usa o ticker ^IRX
    df_cdi = df_cdi.reset_index()
    df_cdi['DT_COMPTC'] = pd.to_datetime(df_cdi['Date'])
    df_cdi.set_index('DT_COMPTC', inplace=True)
    return df_cdi['Close'].pct_change().dropna()  # Retorno diário do CDI
    

def coletar_dados_completos(fundos, start_date, end_date):
    dados_fundos_com_CDI = coletar_dados_fundos(fundos)
    # Obter a data mínima e máxima dos dados dos fundos
    datas = []
    for df in dados_fundos_com_CDI.values():
        datas.append(df.index)
    
    # A data mínima será a menor data de todos os fundos
    start_date = min([min(d) for d in datas])
    
    # A data máxima será a maior data de todos os fundos
    end_date = max([max(d) for d in datas])
    
    # Coletar dados do CDI usando as datas obtidas
    dados_cdi = coletar_dados_cdi(start_date, end_date)
    
    # Adicionar os dados do CDI ao dataframe dos fundos
    for nome, df in dados_fundos_com_CDI.items():
        dados_fundos_com_CDI[nome] = df.loc[start_date:end_date]  # Garantir que estamos utilizando o mesmo intervalo de datas
        
    # Adicionar a coluna do CDI
    dados_fundos_com_CDI['CDI'] = dados_cdi
    
    return dados_fundos_com_CDI

# Função para calcular a matriz de correlação com CDI incluído
def calcular_correlacao_completa(dados_fundos_com_CDI):
    df_completo = pd.concat(dados_fundos_com_CDI, axis=1, join='inner')
    matriz_correlacao = df_completo.corr()
    
    # Garantir que a diagonal seja 1 (auto-correlação)
    np.fill_diagonal(matriz_correlacao.values, 1)
    
    return matriz_correlacao

# Função para calcular a matriz de correlação
def calcular_correlacao_sem_cdi(dados_fundos):
    if not dados_fundos:
        return pd.DataFrame()

    df_completo = pd.concat(dados_fundos, axis=1, join='inner')
    matriz_correlacao_sem_cdi = df_completo.corr()

    np.fill_diagonal(matriz_correlacao_sem_cdi.values, 1)

    return matriz_correlacao_sem_cdi

# Função para calcular os pesos com base nos valores investidos
def calcular_pesos(valores_investidos):
    total_investido = sum(valores_investidos)
    if total_investido == 0:
        return [0] * len(valores_investidos)
    return [valor / total_investido for valor in valores_investidos]

# Função para calcular a correlação média com pesos
def calcular_correlacao_media(matriz_correlacao_sem_cdi, valores_investidos=None):
    if matriz_correlacao_sem_cdi.empty:
        return None

    if valores_investidos is not None:
        pesos = calcular_pesos(valores_investidos)
    else:
        pesos = np.ones(len(matriz_correlacao_sem_cdi.columns)) / len(matriz_correlacao_sem_cdi.columns)  # Pesos iguais

    pesos = np.array(pesos)
    
    # Calculando o numerador da correlação média
    numerator = 0
    for i in range(len(pesos)):
        for j in range(len(pesos)):
            numerator += pesos[i] * pesos[j] * matriz_correlacao_sem_cdi.iloc[i, j]

    # Calculando o denominador (soma dos produtos dos pesos)
    denominator = np.sum(pesos ** 2)
    for i in range(len(pesos)):
        for j in range(len(pesos)):
            if i != j:
                denominator += pesos[i] * pesos[j]

    # Calculando a correlação média
    correlacao_media = numerator / denominator
    correlacao_media = round(correlacao_media, 2)

    return correlacao_media

# Função para plotar a matriz de correlação sem bordas e com fundo transparente
def plotar_matriz_correlacao(matriz_correlacao, save_path=None):
    plt.figure(figsize=(12, 10))

    mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool), k=1)
    cmap = sns.light_palette("seagreen", as_cmap=True)

    ax = sns.heatmap(matriz_correlacao, annot=True, cmap=cmap, fmt='.2f',
                     linewidths=0.5, mask=mask, square=True, cbar_kws={"shrink": .8})

    num_labels = [f'{i+1}: {name}' for i, name in enumerate(matriz_correlacao.index)]
    ax.set_yticklabels(num_labels, rotation=0, fontsize=12, weight='bold')
    ax.set_xticks(np.arange(len(matriz_correlacao.columns)) + 0.5)
    ax.set_xticklabels(np.arange(1, len(matriz_correlacao.columns) + 1), fontsize=14, weight='bold', rotation=0)

    plt.gcf().set_facecolor("none")
    ax.tick_params(axis='both', which='both', length=0)

    plt.title("Matriz de Correlação entre Fundos de Investimento", fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')


@app.route('/calcular_correlacao', methods=['POST'])
def calcular_correlacao_route():
    # Obter CNPJs enviados pelo usuário
    cnpjs = [request.form.get(f'cnpj{i}') for i in range(10)]
    cnpjs = [cnpj for cnpj in cnpjs if cnpj]

    valores_investidos = [
        float(request.form.get(f'valor{i}', 0)) for i in range(10) if request.form.get(f'valor{i}')
    ] if request.form.get('pesos') == 'sim' else None
    
    
    # Coletar dados e calcular a matriz de correlação
    fundos = criar_urls(cnpjs)
    dados_fundos = coletar_dados_fundos(fundos)
    

    dados_completos = coletar_dados_completos(fundos, start_date="2018-01-01", end_date="2024-01-01")
    matriz_correlacao = calcular_correlacao_completa(dados_completos)

    matriz_correlacao_sem_cdi = calcular_correlacao_sem_cdi(dados_fundos)
    # Calcular correlação média
    correlacao_media = calcular_correlacao_media(matriz_correlacao_sem_cdi, valores_investidos)

    # Gerar o gráfico e salvar como imagem
    grafico_path = os.path.join('static', 'correlacao.png')
    plotar_matriz_correlacao(matriz_correlacao, save_path=grafico_path)
    
    # Salvar o gráfico como imagem (no diretório static do Flask)
    grafico_path = 'static/correlacao.png'


    return render_template('correlacao.html', correlacao_media=correlacao_media, grafico_path=grafico_path)






# Função que junta os 2 Dataframes (fundo e ibovespa)
def unindo_tabelas(df, df_ibovespa):
    df_merged = df.merge(df_ibovespa, on='DT_COMPTC', how='left')
    
    df_merged = df_merged.fillna(method='ffill')
    return df_merged
    

@app.route('/buscar_indicadores', methods=['POST', 'GET'])
def buscar_indicadores():
    cnpj = request.form.get('cnpj')
    graficos = request.form.getlist('graficos[]')
    print(f"CNPJ recebido: {cnpj}")  
    
    df = get_data(cnpj)

    if df is None:
        return jsonify({"error": "CNPJ inválido ou erro ao acessar os dados."})
    
    df_ibovespa = get_ibovespa_data(df)
    df_merged = unindo_tabelas(df, df_ibovespa)
    
    # Processamento dos gráficos solicitados
    chart_data = {}

    # Gráfico de Rentabilidade
    if 'rentabilidade' in graficos:
        chart_data['rentabilidade'] = {
            'dates': df['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'values': df['VL_QUOTA_NORM'].tolist()
        }

    # Gráfico de Drawdown
    if 'drawdown' in graficos:
        df['Max_VL_QUOTA'] = df['VL_QUOTA'].cummax()
        df['Drawdown'] = round((df['VL_QUOTA'] - df['Max_VL_QUOTA']) / df['Max_VL_QUOTA'], 4)
        
        # Filtrando para não ter valores NaN
        df = df.dropna(subset=['Drawdown'])

        chart_data['drawdown'] = {
            'dates': df['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'drawdown': (df['Drawdown'] * 100).tolist()  # Convertendo para porcentagem
        }

    # Gráfico de Volatilidade
    if 'volatilidade' in graficos:
        # Cálculo da variação percentual diária
        df['Variacao_Perc'] = df['VL_QUOTA'].pct_change() * 100

        # Cálculo da volatilidade de 21 dias móveis anualizada
        df['Volatilidade'] = round((df['Variacao_Perc'].rolling(window=21).std()) * (252 ** 0.5), 2)  # 252 dias úteis em um ano
        
        # Filtrando para não ter valores NaN
        df = df.dropna(subset=['Volatilidade'])

        # Volatilidade Histórica (a.a)
        retornos_diarios = df['Variacao_Perc'].dropna()  # Remover valores NaN
        daily_volatility = np.std(retornos_diarios)
        annual_volatility = round(daily_volatility * np.sqrt(252), 2)

        chart_data['volatilidade'] = {
            'dates': df['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'volatilidade': df['Volatilidade'].tolist(),
            'annual_volatility': [annual_volatility] * len(df)  # Cria uma lista com o valor anual para o gráfico
        }
     
    # Gráfico de Patrimônio e Captação Líquida
    if 'patrimonio' in graficos:
        df['Captacao_Liquida'] = df['CAPTC_DIA'] - df['RESG_DIA']
        df['Soma_Acumulada'] = df['Captacao_Liquida'].cumsum()
        
        # Converter valores para milhões
        df['Soma_Acumulada_milhoes'] = df['Soma_Acumulada'] / 1_000_000
        df['VL_PATRIM_LIQ_milhoes'] = df['VL_PATRIM_LIQ'] / 1_000_000
        
        # Filtrar valores menores ou iguais a zero e NaN
        df_filtrado = df[df['Soma_Acumulada_milhoes'].notna() & df['VL_PATRIM_LIQ_milhoes'].notna()]


        # Adicionar os dados ao chart_data
        chart_data['patrimonio'] = {
            'dates': df_filtrado['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'captacao_liquida': df_filtrado['Soma_Acumulada_milhoes'].tolist(),
            'patrimonio_liquido': df_filtrado['VL_PATRIM_LIQ_milhoes'].tolist()
        }

    # Captação Líquida Mensal
    if 'captacao_mensal' in graficos:
        df['Captacao_Liquida'] = df['CAPTC_DIA'] - df['RESG_DIA']
        df_monthly = df.groupby(pd.Grouper(key='DT_COMPTC', freq='M')).sum()  # Agrupa por mês
        df_monthly['Captacao_Liquida_milhoes'] = df_monthly['Captacao_Liquida'] / 1_000_000  # Converte para milhões

        # Filtra para garantir que não tenha valores NaN
        df_monthly = df_monthly.dropna(subset=['Captacao_Liquida_milhoes'])

        chart_data['captacao_mensal'] = {
            'dates': df_monthly.index.strftime('%Y-%m-%d').tolist(),
            'values': df_monthly['Captacao_Liquida_milhoes'].tolist()
        }

    # Patrimônio Médio e Nº de Cotistas
    if 'cotistas' in graficos:
        df['Patrimonio_Liq_Medio'] = df['VL_PATRIM_LIQ'] / df['NR_COTST']
        # Filtrando para não ter valores NaN
        df = df.dropna(subset=['Patrimonio_Liq_Medio', 'NR_COTST'])

        chart_data['cotistas'] = {
            'dates': df['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'patrimonio_medio': df['Patrimonio_Liq_Medio'].tolist(),
            'num_cotistas': df['NR_COTST'].tolist()
        }

    # Value at Risk (VaR) e Expected Shortfall (ES)
    if 'var' in graficos:
        df['Variacao_Percentual'] = df['VL_QUOTA'].pct_change(periods=21) * 100

        # Remover NaN da coluna Variacao_Percentual
        df = df.dropna(subset=['Variacao_Percentual'])

        if not df.empty:  # Verifica se o DataFrame não está vazio após remover NaN
            var_5 = np.percentile(df['Variacao_Percentual'], 5)
            mean_below_var_5 = df.loc[df['Variacao_Percentual'] < var_5, 'Variacao_Percentual'].mean()
            var_1 = np.percentile(df['Variacao_Percentual'], 1)
            mean_below_var_1 = df.loc[df['Variacao_Percentual'] < var_1, 'Variacao_Percentual'].mean()

            chart_data['var'] = {
                'dates': df['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
                'variacao_percentual': df['Variacao_Percentual'].tolist(),
                'var_5': [var_5] * len(df),
                'var_1': [var_1] * len(df),
                'es_5': [mean_below_var_5] * len(df),
                'es_1': [mean_below_var_1] * len(df),
            }

    # CAGR e Excesso de Retorno
    if 'cagr' in graficos:
        #Cópia de df
        df_2 = df.copy()

        #Criação de uma coluna com os valores do index invertidos
        df_2['Index_Invertido'] = range(len(df_2)-1, -1, -1)

        # Restringindo o DataFrame ao último ano (252 dias úteis)
        df_2 = df_2[df_2['Index_Invertido'] <= 252]

        # Cálculo do CAGR diário
        end_value_fundo = df_2['VL_QUOTA'].iloc[-1]
        cagr_diario = ((end_value_fundo / df_2['VL_QUOTA']) ** (252 / (df_2['Index_Invertido'] + 1)) - 1) * 100
        cagr_diario_fundo = cagr_diario
        
        # Média do CAGR final
        mean_cagr = round(cagr_diario.mean(), 2)

        # Preparação dos dados para o gráfico
        chart_data['cagr'] = {
            'dates': df_2['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'cagr': cagr_diario.tolist(),
            'mean_cagr': [mean_cagr] * len(df_2)  # Média do CAGR
        }
        
        ### CAGR DO IBOVESPA E EXCESSO DE RETORNO GERAL #######################################################
        end_value_ibovespa = df_ibovespa['Close'].iloc[-1]
        cagr_ibovespa = ((end_value_ibovespa / df_ibovespa['Close'].iloc[0]) ** (252 / len(df_ibovespa)) - 1) * 100

        excesso_retorno_geral = cagr_diario_fundo - cagr_ibovespa

        chart_data['excesso_retorno'] = {
            'dates': df_2['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
            'excesso_retorno': excesso_retorno_geral.tolist(),
            'mean_excesso_retorno': [excesso_retorno_geral.mean()] * len(df_2)  # Média do Excesso de Retorno
        }

        ### EXCESSO DE RETORNO EM JANELAS MÓVEIS ##############################################################
        janelas = [252 * i for i in range(1, 6)]  # 1 a 5 anos em dias úteis

        # Inicializar um dicionário para armazenar os dados do gráfico
        chart_data_excesso_movel = {}

        for janela in janelas:
            # Calcular o CAGR móvel para o fundo
            df_2[f'CAGR_Movel_{janela//252}'] = ((df_2['VL_QUOTA'] / df_2['VL_QUOTA'].shift(janela)) ** (252 / janela) - 1) * 100
            
            # Calcular o CAGR móvel para o Ibovespa
            df_ibovespa[f'CAGR_Movel_{janela//252}'] = ((df_ibovespa['Close'] / df_ibovespa['Close'].shift(janela)) ** (252 / janela) - 1) * 100
            
            # Remover valores NaN
            df_2.dropna(subset=[f'CAGR_Movel_{janela//252}'], inplace=True)
            df_ibovespa.dropna(subset=[f'CAGR_Movel_{janela//252}'], inplace=True)
            
            # Verificar se há dados suficientes em ambos os dataframes após o cálculo da janela móvel
            if not df_2.empty and not df_ibovespa.empty:
                # Calcular o excesso de retorno móvel
                df_2[f'Excesso_Retorno_Movel_{janela//252}'] = df_2[f'CAGR_Movel_{janela//252}'] - df_ibovespa[f'CAGR_Movel_{janela//252}'].values
                
                # Calcular a média do excesso de retorno móvel
                mean_excesso_retorno_movel = df_2[f'Excesso_Retorno_Movel_{janela//252}'].mean()
                
                # Preparação dos dados para o gráfico de janela móvel
                chart_data[f'excesso_retorno_movel_{janela//252}'] = {
                    'dates': df_2['DT_COMPTC'].dt.strftime('%Y-%m-%d').tolist(),
                    'excesso_retorno_movel': df_2[f'Excesso_Retorno_Movel_{janela//252}'].tolist(),
                    'mean_excesso_retorno_movel': [mean_excesso_retorno_movel] * len(df_2)
                }
        chart_data.update(chart_data)
        
        



    return jsonify({'chart_data': chart_data})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

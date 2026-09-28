import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração de reprodutibilidade
np.random.seed(42)

# =============================================================================
# PHASE 0: GERAÇÃO DOS ARQUIVOS DE ENTRADA (MOCK DATA)
# =============================================================================

# --- 1. GERAÇÃO DO ARQUIVO: transacoes.csv ---
datas_transacoes = pd.date_range(start='2026-09-01', periods=1000, freq='h')

df_transacoes_mock = pd.DataFrame({
    'id_cliente': np.random.choice(['C100', 'C101', 'C102', 'C103', 'C104'], size=1000),
    'data_transacao': datas_transacoes,
    'valor': np.random.choice(
        [np.nan, 150.0, 3000.0, 7500.0, 12000.0, 50.0], 
        size=1000, 
        p=[0.05, 0.40, 0.30, 0.15, 0.05, 0.05]
    ),
    'estado_cliente': np.random.choice(['SP', 'RJ', 'MG', 'RS'], size=1000),
    'origem': np.random.choice(['web', 'mobile_app', 'atm'], size=1000)
})

# Forçando duplicados para teste do pipeline
df_transacoes_mock = pd.concat([df_transacoes_mock, df_transacoes_mock.iloc[:15]], ignore_index=True)
df_transacoes_mock.to_csv('transacoes.csv', index=False, encoding='latin1')

# --- 2. GERAÇÃO DO ARQUIVO: cotacoes.csv ---
datas_cotacoes = pd.date_range(start='2026-09-01', end='2026-10-15', freq='D')

variacoes = np.random.normal(loc=0.001, scale=0.015, size=len(datas_cotacoes))
preco_inicial = 5.20
precos = preco_inicial * np.exp(np.cumsum(variacoes))

df_cotacoes_mock = pd.DataFrame({
    'data': datas_cotacoes,
    'cotacao_usd': np.round(precos, 4),
    'volume_negociado': np.random.randint(10000, 500000, size=len(datas_cotacoes))
})

# Inserindo NaNs propositais para tratamento posterior
df_cotacoes_mock.loc[5, 'cotacao_usd'] = np.nan
df_cotacoes_mock.loc[18, 'cotacao_usd'] = np.nan
df_cotacoes_mock.to_csv('cotacoes.csv', index=False, encoding='utf-8')

print("=== [ETAPA 0] Arquivos 'transacoes.csv' e 'cotacoes.csv' gerados com sucesso! ===\n")


# =============================================================================
# PHASE 1: INGESTÃO, PERFORMANCE E LIMPEZA
# =============================================================================

# Leitura com o encoding correto (latin1) evitando quebras de caracteres remotos
df_transacoes = pd.read_csv('transacoes.csv', encoding='latin1')
df_cotacoes = pd.read_csv('cotacoes.csv', encoding='utf-8')

# Imputação vetorizada de nulos no valor da transação pela mediana do estado correspondente
mediana_por_estado = df_transacoes.groupby('estado_cliente')['valor'].transform('median')
df_transacoes['valor'] = df_transacoes['valor'].fillna(mediana_por_estado)

# Adição da coluna estática para fins de rastreabilidade
df_transacoes['plataforma'] = 'Mobile'


# =============================================================================
# PHASE 2: ENGENHARIA DE DADOS & ALINHAMENTO TEMPORAL
# =============================================================================

# Conversão tipada explícita e aplicação de fuso horário localizado
df_transacoes['data_transacao'] = pd.to_datetime(df_transacoes['data_transacao'])
df_transacoes['data_transacao'] = df_transacoes['data_transacao'].dt.tz_localize('America/Sao_Paulo')

# Atributos baseados em tempo utilizando o acessor vetorizado `.dt`
df_transacoes['dia_semana'] = df_transacoes['data_transacao'].dt.day_name()
df_transacoes['mes'] = df_transacoes['data_transacao'].dt.month_name()

# Remoção de registros 100% duplicados mantendo o primeiro evento
df_transacoes = df_transacoes.drop_duplicates(keep='first')


# =============================================================================
# PHASE 3: OPERAÇÕES VETORIZADAS E FILTROS BITWISE
# =============================================================================

# Aplicação de filtros lógicos rápidos na CPU via operadores bitwise (&, |)
filtro_setembro_alta_renda = (
    (df_transacoes['data_transacao'].dt.month == 9) & 
    ((df_transacoes['estado_cliente'] == 'SP') | (df_transacoes['estado_cliente'] == 'RJ')) & 
    (df_transacoes['valor'] > 5000.00)
)
df_setembro_filtrado = df_transacoes[filtro_setembro_alta_renda]


# =============================================================================
# PHASE 4: CRUZAMENTO DE DADOS, MAPEAMENTO & CONSOLIDACÃO
# =============================================================================

# Dicionário de classificação de risco dos clientes da fintech
risco_dict = {
    'C100': 'Baixo', 'C101': 'Alto', 'C102': 'Medio', 'C103': 'Baixo', 'C104': 'Alto'
}
df_transacoes['nivel_risco'] = df_transacoes['id_cliente'].map(risco_dict)

# Tratamento da tabela secundária de cotações (Preenchendo nulos com a última cotação válida)
df_cotacoes['data'] = pd.to_datetime(df_cotacoes['data']).dt.tz_localize('America/Sao_Paulo')
df_cotacoes['cotacao_usd'] = df_cotacoes['cotacao_usd'].ffill().bfill()

# Criação de chave temporária normalizada (por dia) para efetuar o merge de tabelas
df_transacoes['data_dia'] = df_transacoes['data_transacao'].dt.normalize()
df_cotacoes['data_dia'] = df_cotacoes['data'].dt.normalize()

# Enriquecendo a base principal de transações com dados diários do Dólar Comercial
df_consolidado = pd.merge(
    df_transacoes, 
    df_cotacoes[['data_dia', 'cotacao_usd']], 
    on='data_dia', 
    how='left'
)

# Adicionando métrica derivada: Valor convertido para USD
df_consolidado['valor_usd'] = df_consolidado['valor'] / df_consolidado['cotacao_usd']

# Tabela dinâmica estruturada com totais de controle lineares e cruzados
pivot_risco_mensal = df_consolidado.pivot_table(
    index='mes',
    columns='nivel_risco',
    values='valor',
    aggfunc='sum',
    margins=True,
    margins_name='Total Geral'
)


# =============================================================================
# PHASE 5: DETECÇÃO ESTATÍSTICA DE OUTLIERS (Z-SCORE)
# =============================================================================

# Cálculo puramente vetorizado usando agrupamento por estado para evitar assimetria regional
means = df_consolidado.groupby('estado_cliente')['valor'].transform('mean')
stds = df_consolidado.groupby('estado_cliente')['valor'].transform('std')

df_consolidado['z_score'] = (df_consolidado['valor'] - means) / stds

# Dataframe de anomalias estatísticas isoladas (Z > 2.5)
df_anomalias = df_consolidado[df_consolidado['z_score'] > 2.5]


# =============================================================================
# PHASE 6: VISUALIZAÇÃO GRÁFICA ORIENTADA A OBJETOS (MATPLOTLIB)
# =============================================================================

# Agrupamento diário do faturamento bruto processado na plataforma
dados_grafico = df_consolidado.groupby(df_consolidado['data_dia'].dt.date)['valor'].sum().to_frame(name='total_diario')

# Geração de indicador de tendência com janela deslizante de 7 dias
dados_grafico['media_movel_7d'] = dados_grafico['total_diario'].rolling(7).mean()

# Construção da interface visual via API orientada a objetos do Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))

# Plotagem das curvas temporais requisitadas
ax.plot(dados_grafico.index, dados_grafico['total_diario'], label='Valor Total Diário', color='#1f77b4', alpha=0.5, linestyle='--')
ax.plot(dados_grafico.index, dados_grafico['media_movel_7d'], label='Média Móvel (7 Dias)', color='#ff7f0e', linewidth=2.5)

# Elementos textuais, de design e formatação de eixos
ax.set_title('Métricas de Desempenho Diário e Suavização de Tendência', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Data da Operação', fontsize=11, labelpad=10)
ax.set_ylabel('Volume Financeiro (R\$)', fontsize=11, labelpad=10)

# Correção mandatória do ponto de origem do gráfico
ax.set_ylim(bottom=0)

# Detalhes de infraestrutura de leitura do gráfico
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='none')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print informativo final das tabelas criadas no processo de consolidação
print("=== PIPELINE EXECUTADO COM SUCESSO ===")
print(f"\nRegistros originais com duplicados: 1015")
print(f"Registros limpos e unificados: {len(df_consolidado)}")
print(f"Potenciais anomalias detectadas (Z-score > 2.5): {len(df_anomalias)}")
print("\nTabela Dinâmica de Risco (R\$):")
print(pivot_risco_mensal.round(2))


# Databricks notebook source
# MAGIC %md
# MAGIC # Libs

# COMMAND ----------

import requests
import tarfile
import pandas as pd
import gzip
import io
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from scipy.stats import ttest_ind_from_stats


import pandas as pd
from scipy.stats import ttest_ind

def ttest(df, segmentacao, metricas, col_teste):
  """
  Aplica o t-test para grupo teste e controle e retorna tabela pandas com resultados

  Parâmetros:
  - df: base de dados.
  - segmentacao: nome da segmentação em questão
  - metricas: lista com as métricas a serem testadas
  - col_teste: string que indica qual coluna marca controle e teste

  Retorna:
  - dataframe pandas com resultados do t-test
  """
  resultados = []

  # Itera sobre cada segmentação
  for segmento in df[segmentacao].unique():

    # Filtra os dados para o segmento atual
    df_segmento = df[df[segmentacao] == segmento]

    # Separa os grupos de teste e controle
    grupo_teste_data = df_segmento[df_segmento[col_teste] == 'target']
    grupo_controle_data = df_segmento[df_segmento[col_teste] == 'control']

    # Aplica o teste t para cada métrica
    for metrica in metricas:
      # Extrai os valores da métrica para teste e controle
      valores_teste = grupo_teste_data[metrica].dropna()
      valores_controle = grupo_controle_data[metrica].dropna()

      # Aplica o teste t usando ttest_ind
      t_stat, p_value = ttest_ind(valores_teste, valores_controle)

      # Calcula as médias dos grupos
      media_teste = valores_teste.mean()
      media_controle = valores_controle.mean()

      # Armazena os resultados
      resultados.append({
          'customer_categorization': segmento,
          'metrica': metrica,
          'T-Statistic': t_stat,
          'P-Value': p_value,
          'Media Teste': media_teste,
          'Media Controle': media_controle
      })

  # Cria um DataFrame com os resultados
  return pd.DataFrame(resultados)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extraindo e preparando dados

# COMMAND ----------

# Urls com dados
urls = [
'https://data-architect-test-source.s3-sa-east-1.amazonaws.com/order.json.gz',
'https://data-architect-test-source.s3-sa-east-1.amazonaws.com/consumer.csv.gz',
'https://data-architect-test-source.s3-sa-east-1.amazonaws.com/restaurant.csv.gz',
'https://data-architect-test-source.s3-sa-east-1.amazonaws.com/ab_test_ref.tar.gz'
]

# Ajustando formato para leitura
urls = [i.replace('.s3-sa-east-1.amazonaws.com','').replace('https','s3a') for i in urls]

# Dict para armazanar dados
dbs = {}

for u in urls:
  var_name = u.split('/')[-1].split('.')[0]

  # Se for um arquivo json
  if '.json' in u:
    df = (spark
          .read
          .json(u))
    dbs[var_name] = df

  # Se for um arquivo csv
  elif '.csv' in u:
    df = (spark
          .read
          .csv(u, header = True)
          .distinct())
    dbs[var_name] = df

  # Se for um arquivo tar
  elif 'tar.gz' in u:
    
    # Requisição dos dados
    file_content = requests.get('https://data-architect-test-source.s3-sa-east-1.amazonaws.com/' + var_name + '.tar.gz', stream=True).content

    # Lista para armazenas dados
    lista = []

    # Abrindo arquivo zip
    with gzip.GzipFile(fileobj=io.BytesIO(file_content)) as gzipped_file:
      # Abrindo arquivo tar
      with tarfile.open(fileobj=gzipped_file, mode='r') as tar:

        # Iterando entre membros
        for member in tar.getmembers():
          file = tar.extractfile(member)
          
          # Extraindo dados e armazenando em listas
          if file:
            content = file.read().decode('latin1')
            temp_df = pd.read_csv(io.StringIO(content))
            if temp_df.shape[0] > 0:
              lista.append(temp_df)

    # Materializando dados
    df_pd = pd.concat(lista)

    df = (spark.createDataFrame(df_pd)
          .distinct())
    dbs[var_name] = df


# Materializando dados para o SQL
for key in dbs.keys():
  if key != 'order':
    dbs[key].createOrReplaceTempView(key)
  else:
    (dbs[key]
     .filter(
      F.col('customer_id')
      .isNotNull())
      .withColumn('ranking',F.row_number().over(Window.partitionBy(F.col('order_id')).orderBy(F.desc(F.col('order_created_at')))))
      .filter(F.col('ranking') == 1)
      # Selecionando apenas colunas que serão usadas na análise
      .select(['customer_id','items','merchant_id','order_created_at','order_id','order_total_amount'])
      .distinct()
     .createOrReplaceTempView(key))

# COMMAND ----------

# MAGIC %md
# MAGIC # Defininfo Segmentações e preparando base

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definindo frequência semanal
# MAGIC Classificar usuário seguindo a premissa:
# MAGIC - `active engaged`: >= 3 pdd ultima semana
# MAGIC - `active`: = 1 pdd ultima semana
# MAGIC - `early churn`: < 1 pdd ultima semana
# MAGIC - `middle churn`: < 1 pdd ultimas 2 semanas
# MAGIC - `late/lost churn`: < 1 pdd ultimas 3 ou mais semanas
# MAGIC - `new`: primeiro pdd

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   date_trunc('week',order_created_at)::DATE as week,
# MAGIC   round(count(distinct order_id)/count(distinct customer_id),1) as freq
# MAGIC from
# MAGIC   order
# MAGIC group by all

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorização de customer_id

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view db_customer_cat as 
# MAGIC
# MAGIC with db_week as ( 
# MAGIC   select
# MAGIC     date_trunc('week',order_created_at)::DATE as week,
# MAGIC     customer_id,
# MAGIC     count(distinct order_id) as orders_by_week
# MAGIC   from
# MAGIC     order
# MAGIC
# MAGIC   group by all
# MAGIC   order by 1 asc
# MAGIC
# MAGIC )
# MAGIC
# MAGIC ,db_week_lag as (
# MAGIC   select
# MAGIC     week,
# MAGIC     customer_id,
# MAGIC     orders_by_week,
# MAGIC     LAG(orders_by_week) OVER (PARTITION BY customer_id ORDER BY week ASC) as orders_last_week,
# MAGIC     LAG(week) OVER (PARTITION BY customer_id ORDER BY week ASC) as previous_order,
# MAGIC     DATEDIFF(week, previous_order) / 7 as week_diff
# MAGIC   from
# MAGIC     db_week
# MAGIC )
# MAGIC
# MAGIC select
# MAGIC   week,
# MAGIC   customer_id,
# MAGIC   orders_by_week,
# MAGIC   orders_last_week,
# MAGIC   previous_order,
# MAGIC   week_diff,
# MAGIC   (case
# MAGIC     when orders_last_week is null then 'new'
# MAGIC     when week_diff = 1 and orders_last_week >= 3 then 'active engaged'
# MAGIC     when week_diff = 1 and orders_last_week in (1,2) then 'active'
# MAGIC     when week_diff = 2 then 'early churn'
# MAGIC     when week_diff = 3 then 'middle churn'
# MAGIC     when week_diff >= 4 then 'late/lost churn'
# MAGIC   else 'others'
# MAGIC   end) as customer_categorization
# MAGIC from
# MAGIC   db_week_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join de Order + Customer Categorization + flag de teste

# COMMAND ----------

# MAGIC %sql 
# MAGIC create or replace temp view db_orders_cat as
# MAGIC with db_items_per_order as (
# MAGIC   select 
# MAGIC     order_id,
# MAGIC     SUM(CAST(get_json_object(item, '$.quantity') as double)) AS items_tt
# MAGIC   from (
# MAGIC     select 
# MAGIC       order_id,
# MAGIC       explode(from_json(items, 'array<string>')) AS item
# MAGIC     from 
# MAGIC       order
# MAGIC   )
# MAGIC
# MAGIC   group by all
# MAGIC )
# MAGIC   select
# MAGIC     ORD.*,
# MAGIC     CAT.customer_categorization,
# MAGIC     is_target,
# MAGIC     IT.items_tt as items_tt,
# MAGIC     REST.price_range
# MAGIC
# MAGIC   from
# MAGIC     order ORD
# MAGIC   
# MAGIC   left join db_customer_cat CAT on CAT.customer_id = ORD.customer_id and CAT.week = date_trunc('week',ORD.order_created_at)
# MAGIC   left join ab_test_ref AB on AB.customer_id = ORD.customer_id
# MAGIC   left join db_items_per_order IT on IT.order_id = ORD.order_id
# MAGIC   left join restaurant REST on REST.id = ORD.merchant_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Análise de Teste A/B

# COMMAND ----------

# MAGIC %md
# MAGIC ## Levantando métricas T-Test para Amostras Independentes
# MAGIC Indicadores selecionados:
# MAGIC - `AOV`: Aumentamos o tamanho médio do pdd com o teste?
# MAGIC - `Freq`: Aumentamos a frequência de pdd com o teste?
# MAGIC - `GMV per user`: Melhoramos o gmv por usuário?
# MAGIC - `Items per order`: Aumentamos a quantidade de itens no pdd com o teste?
# MAGIC - `Freq High prince range`: Aumentamos a frequência de pdds em restaurantes caros?
# MAGIC - `Freq Low prince range`: Aumentamos a frequência de pdds em restaurantes baratos?

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view db_t_test as
# MAGIC   select
# MAGIC     ORD.customer_categorization,
# MAGIC     ORD.is_target,
# MAGIC     ORD.customer_id,
# MAGIC     round(sum(ORD.order_total_amount) / count(distinct ORD.customer_id),5) as gmv_per_user,
# MAGIC     round(sum(ORD.order_total_amount) / count(distinct ORD.order_id),5) as aov,
# MAGIC     count(distinct ORD.order_id) as freq,
# MAGIC     round(sum(ORD.items_tt) / count(distinct ORD.order_id),5) as items_per_order,
# MAGIC     count ( distinct case when ORD.price_range in (1,2,3) then ORD.order_id end) as freq_low_price_range,
# MAGIC     count ( distinct case when ORD.price_range in (4,5) then ORD.order_id end) as freq_high_price_range
# MAGIC     
# MAGIC   from
# MAGIC     db_orders_cat ORD
# MAGIC   -- where
# MAGIC   --   customer_id = '1d88117c99ccf34f58c395661eaffee49e67ab8b69b72943f4e1eab4695f598e'
# MAGIC   group by all

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aplicando T-Test

# COMMAND ----------

# Carregando para pyspark a temp table
df_ttest = (spark
          .table('db_t_test')
          .toPandas()
          )

# Define colunas para executar o teste
segmentacao = 'customer_categorization'
col_teste = 'is_target'
metricas = ['gmv_per_user', 'aov', 'freq', 'items_per_order','freq_low_price_range','freq_high_price_range']

# Aplica a func ttest
resultados = ttest(df = df_ttest, segmentacao = segmentacao, metricas = metricas, col_teste = col_teste)

# Exibe os resultados
df_resultados = spark.createDataFrame(resultados)
# df_resultados.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analisando Resultados

# COMMAND ----------

df_resultados = (df_resultados
                .withColumn('P-Value check', F.when(F.col('P-Value') <= 0.05, True).otherwise(False))
                .withColumn('T-Statistic check', F.when(F.col('T-Statistic') > 0, True).otherwise(False))
                .withColumn('Test Lift', F.when((F.col('P-Value check') == True) & (F.col('T-Statistic check') == True), True).otherwise(False))
                )

df_resultados.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Anotações para construção de storytelling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quais foram os sucessos do teste?
# MAGIC `New` 
# MAGIC   - Apresentou melhora significativa em	gmv_per_user, freq, items_per_order, freq_low_price_range e freq_high_price_range
# MAGIC   - Desta forma tivemos uma boa melhoria na conversão de usuários new junto de uma melhora no valor que trazem neste primeiro pdd, quantidade de itens que adicionam e frequencia nas duas faixas de preço de restaurantes
# MAGIC
# MAGIC ### Que melhorias podemos implementar para os próximos testes ?
# MAGIC `New`
# MAGIC   - Visto que o teste apresentou melhoria neste público, a recomendação é realizar outro teste segmentando clientes new onde é dado um cupom promocional inferior
# MAGIC   - Desta forma podemos entender até o menor valor promocional que ainda assim se mantém atrativo para o usuário
# MAGIC
# MAGIC `Early, middle e last/lost churn`
# MAGIC   - O teste se mostrou ineficaz para todas as outras segmentações, sendo necessário mudar de abordagem para estes públicos
# MAGIC   - O teste não deu certeza estatística de como realmente o cupom promocional performou nestas segmentações
# MAGIC   - O melhor cenário seria subir outro Teste A/B apenas com esta base segmentada e avalair novamente a performance 
# MAGIC
# MAGIC `Active e active engaged`
# MAGIC - Devido estes clientes possuirem alto engajamento com a plataforma, a melhor estratégia para este público é testar alguma alavanca de fidelização qeu auxilie aumentar ainda mais a quantidade de pdd semanais, como:
# MAGIC   - A cada x pdd ganha um desconto em sobremesa em restaurantes parceiros
# MAGIC   - Descontos leves em restaurantes favoritos do cliente
# MAGIC   - Descontos em momentos onde o cliente está mais propenso a converter como almoço e jantar
# MAGIC
# MAGIC `Próximos passos`
# MAGIC   - Criar acompanhamentos de quais usuários estão se aproximando a se tornarem churn e implementar alavancas que disparem para que mantenhamos baixo o % de ativos que viram churn
# MAGIC   - Realizar mais testes A/B mas separando teste e controle dentro das mesmas segmentações (new, early churn e etc)
# MAGIC   - Realizar testes A/B focados na navegabilidade do marketplace, assim avaliando se há lift's com mudanças de componentes sem a necessidade de investimento promocional

# COMMAND ----------

# MAGIC %md
# MAGIC # Visões para apresentação

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exemplo de New

# COMMAND ----------

# Média de gmv per user do grupo controle da segmentação de New
control_media_gmv_new = (df_resultados
                        .filter((F.col('customer_categorization') == 'new') & (F.col('metrica') == 'gmv_per_user'))
                        .collect()[0]["Media Controle"]
                        )

# Média de gmv per user do grupo target da segmentação de New
target_media_gmv_new = (df_resultados
                      .filter((F.col('customer_categorization') == 'new') & (F.col('metrica') == 'gmv_per_user'))
                      .collect()[0]["Media Teste"]
                      )

# T-Statistic do gmv per user da segmentação de New
t_gmv_new = (df_resultados
            .filter((F.col('customer_categorization') == 'new') & (F.col('metrica') == 'gmv_per_user'))
            .collect()[0]['T-Statistic']
            )

# P-Value do gmv per user da segmentação de New
p_gmv_new = (df_resultados
            .filter((F.col('customer_categorization') == 'new') & (F.col('metrica') == 'gmv_per_user'))
            .collect()[0]['P-Value']
            )

# Valor de gmv per user do grupo target da segmentação de New
target_data_new = (spark
                  .table('db_t_test')
                  .filter((F.col('is_target') == "target") & (F.col('customer_categorization') == 'new') & (F.col('gmv_per_user') <= 100))
                  .select("gmv_per_user")
                  .toPandas()["gmv_per_user"])

# Valor de gmv per user do grupo controle da segmentação de New
control_data_new = (spark
                    .table('db_t_test')
                    .filter((F.col('is_target') == "control") & (F.col('customer_categorization') == 'new') & (F.col('gmv_per_user') <= 100))
                    .select("gmv_per_user")
                    .toPandas()["gmv_per_user"])

# Criar a visualização
plt.figure(figsize=(10, 6))
sns.kdeplot(control_data_new, label="Control", color="pink", shade=True)  # Grupo Controle
sns.kdeplot(target_data_new, label="Target", color="red", shade=True)    # Grupo Target

# Marcando as médias
plt.axvline(control_media_gmv_new, color='pink', linestyle='dashed', linewidth=2, label='Média Controle')
plt.axvline(target_media_gmv_new, color='red', linestyle='dashed', linewidth=2, label='Média Target')

# Adicionar informações do teste t
plt.title(f"Distribuição de GMV por Usuário (New)\nTeste t: t = {t_gmv_new:.2f}, p = {p_gmv_new:.4f}")
plt.xlabel("GMV por Usuário")
plt.ylabel("Densidade")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exemplo de Active

# COMMAND ----------

# Média de gmv per user do grupo controle da segmentação de active
control_media_gmv_active = (df_resultados
                        .filter((F.col('customer_categorization') == 'active') & (F.col('metrica') == 'gmv_per_user'))
                        .collect()[0]["Media Controle"]
                        )

# Média de gmv per user do grupo target da segmentação de active
target_media_gmv_active = (df_resultados
                      .filter((F.col('customer_categorization') == 'active') & (F.col('metrica') == 'gmv_per_user'))
                      .collect()[0]["Media Teste"]
                      )

# T-Statistic do gmv per user da segmentação de active
t_gmv_active = (df_resultados
            .filter((F.col('customer_categorization') == 'active') & (F.col('metrica') == 'gmv_per_user'))
            .collect()[0]['T-Statistic']
            )

# P-Value do gmv per user da segmentação de active
p_gmv_active = (df_resultados
            .filter((F.col('customer_categorization') == 'active') & (F.col('metrica') == 'gmv_per_user'))
            .collect()[0]['P-Value']
            )

# Valor de gmv per user do grupo target da segmentação de active
target_data_active = (spark
                  .table('db_t_test')
                  .filter((F.col('is_target') == "target") & (F.col('customer_categorization') == 'active') & (F.col('gmv_per_user') <= 500))
                  .select("gmv_per_user")
                  .toPandas()["gmv_per_user"])

# Valor de gmv per user do grupo controle da segmentação de active
control_data_active = (spark
                    .table('db_t_test')
                    .filter((F.col('is_target') == "control") & (F.col('customer_categorization') == 'active') & (F.col('gmv_per_user') <= 500))
                    .select("gmv_per_user")
                    .toPandas()["gmv_per_user"])

# Criar a visualização
plt.figure(figsize=(10, 6))
sns.kdeplot(control_data_active, label="Control", color="pink", shade=True)  # Grupo Controle
sns.kdeplot(target_data_active, label="Target", color="red", shade=True)    # Grupo Target

# Marcando as médias
plt.axvline(control_media_gmv_active, color='pink', linestyle='dashed', linewidth=2, label='Média Controle')
plt.axvline(target_media_gmv_active, color='red', linestyle='dashed', linewidth=2, label='Média Target')

# Adicionar informações do teste t
plt.title(f"Distribuição de GMV por Usuário (Active)\nTeste t: t = {t_gmv_active:.2f}, p = {p_gmv_active:.4f}")
plt.xlabel("GMV por Usuário")
plt.ylabel("Densidade")
plt.legend()
plt.show()
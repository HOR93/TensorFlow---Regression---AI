import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

caminho_do_arquivo = 'https://raw.githubusercontent.com/HOR93/csv_dataset/main/avamos%20indicadores_trajetoria_educacao_superior_2018_2022_engenharias%20-%20Engenharias.csv'
dataset = pd.read_csv(caminho_do_arquivo)
graficos_dados = pd.read_csv(caminho_do_arquivo)

# calculando a media entre os cursos e taxa de graduação
media_graduados_curso = graficos_dados.groupby('Course')['R.Graduates'].mean().reset_index()

media_graduados_curso = media_graduados_curso.sort_values(by='R.Graduates', ascending=False)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.barplot(x='R.Graduates', y='Course', data=media_graduados_curso, palette='viridis')
plt.title('Average Number of Graduates per Course')
plt.xlabel('Average Number of Graduates')
plt.ylabel('Course')

plt.show()

#taxa anual de evasao e curso
graficos_dados['A.Dropout Rate'] = graficos_dados['A.Dropout Rate'].replace(',', '', regex=True).astype(float)

evasao_curso = graficos_dados.groupby('Course')['A.Dropout Rate'].mean().reset_index()
evasao_curso = evasao_curso.sort_values(by='A.Dropout Rate', ascending=False)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.barplot(x='A.Dropout Rate', y='Course', data=evasao_curso, palette='viridis')
plt.title('Mean Annual Dropout Rate by Course')
plt.xlabel('Mean Annual Dropout Rate')
plt.ylabel('Course')

plt.tight_layout()

plt.show()

graficos_dados["Course_A.Graduation Rate"] = graficos_dados["Course"] + " - " + graficos_dados["A.Graduation Rate"]

cores = plt.cm.get_cmap('tab20', len(graficos_dados['Course'].unique()))

plt.figure(figsize=(12, 8))
for i, curso in enumerate(graficos_dados['Course'].unique()):
    curso_df = graficos_dados[graficos_dados['Course'] == curso]
    plt.scatter(curso_df["Course_A.Graduation Rate"], curso_df["A.Graduation Rate"], c=[cores(i)], label=curso, s=10)

plt.title("Relationship between Course and A.Graduation Rate")
plt.xlabel("Course_ A.Graduation Rate")
plt.ylabel("A.Graduation Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks([])
plt.show()

#relação evasão, ano da evasão e pandemia
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='R.Y', y='A.Dropout Rate', hue='Pandemic', data=graficos_dados, palette=['blue', 'green'], s=50)
plt.title('R.Y and A.Dropout Rate with Pandemic', fontsize=16)
plt.xlabel('R.Y', fontsize=14)
plt.ylabel('A.Dropout Rate', fontsize=14)

for category, marker, color in zip(['No Pandemic', 'Pandemic'], ['o', 's'], ['blue', 'green']):
    subset = graficos_dados[graficos_dados['Pandemic'] == category]
    plt.scatter(subset['R.Y'], subset['A.Dropout Rate'], marker=marker, color=color, s=50, label=category)
    for x, y, label in zip(subset['R.Y'], subset['A.Dropout Rate'], subset.index):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

scatter.legend_.remove()

plt.show()



graficos_dados['A.Dropout Rate'] = graficos_dados['A.Dropout Rate']
plt.figure(figsize=(16, 8))
scatterplot = sns.lmplot(x='R.Y', y='A.Dropout Rate', hue='Course', data=graficos_dados, height=8, aspect=2, palette='viridis')

plt.title('Relationship between Year and Dropout Rate by Course', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Annual Dropout Rate', fontsize=14)

plt.show()


graficos_dados['A.Dropout Rate'] = graficos_dados['A.Dropout Rate']


plt.figure(figsize=(16, 10))
scatterplot = sns.scatterplot(
    x='Members R. retention',
    y='A.Dropout Rate',
    hue='Pandemic',
    size='R.Y',
    sizes=(50, 200),
    data=graficos_dados,
    palette='viridis'
)

plt.title('Members R. retention, Dropout Rate, and Pandemic with Year', fontsize=16)
plt.xlabel('Members R. retention', fontsize=14)
plt.ylabel('Annual Dropout Rate', fontsize=14)

plt.legend(title='Pandemic', labels=['No Pandemic', 'Pandemic'], loc='upper right')

plt.show()

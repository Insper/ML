# AutoML

A [busca em grade](../model-selection/index.md#busca-em-grade-com-validacao-cruzada) funciona, mas é força bruta: o custo se multiplica por hiperparâmetro, a maior parte da grade é desperdiçada em regiões ruins, e alguém ainda precisa escolher a grade. O **AutoML** automatiza a busca — sobre hiperparâmetros e, no extremo ambicioso, sobre o pipeline inteiro. O objetivo não é substituir o profissional; é gastar seu julgamento com [formulação, dados e desenho de validação](../ml-landscape/index.md#o-fluxo-de-trabalho-de-ml) enquanto a máquina mói o espaço de busca.

## Além da grade: busca aleatória

Bergstra & Bengio (2012) fizeram uma observação simples e devastadora: quando apenas alguns hiperparâmetros realmente importam (o caso usual), uma **grade desperdiça avaliações** — ela retesta os mesmos poucos valores do parâmetro importante enquanto marcha por outros irrelevantes. A **amostragem aleatória** cobre a faixa de cada dimensão muito melhor com o mesmo orçamento:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'model__learning_rate': loguniform(1e-3, 0.3),   # contínuo, em escala log
    'model__max_depth': randint(2, 10),
    'model__subsample': (0.6, 0.8, 1.0),
}
search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                            scoring='roc_auc', n_jobs=-1, random_state=0)
```

A busca aleatória é a baseline honesta que todo método mais inteligente precisa superar.

## Otimização bayesiana: uma busca que aprende

Cada avaliação de CV é cara — por que ignorar o que os ensaios anteriores revelaram? A **otimização bayesiana** trata o escore como uma função desconhecida dos hiperparâmetros e itera:

1. ajustar um **modelo substituto (surrogate)** barato aos pares (configuração → escore) vistos até agora;
2. escolher a próxima configuração por uma **função de aquisição** que equilibra *exploração da melhor região conhecida* (exploitation) e *exploração de regiões incertas* (exploration);
3. avaliar, adicionar ao histórico, repetir.

O **Optuna** (Akiba et al., 2019) é o favorito do ecossistema (surrogate TPE por padrão), com uma API definitiva — você escreve o espaço de busca como uma função:

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    }
    model = xgb.XGBClassifier(**params, n_estimators=300)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
study.best_params
```

Como o espaço é *definido em código*, ele pode ser condicional (ajustar `gamma` apenas se `kernel == 'rbf'`) — impossível de expressar como uma grade.

## Halving sucessivo: gaste menos com os perdedores

Truque ortogonal — a maioria das configurações revela sua mediocridade **cedo** (poucas rodadas de boosting, pequenos subconjuntos de dados). O **halving sucessivo** explora isso: comece muitas configurações com um orçamento pequeno, mantenha a melhor fração, multiplique seu orçamento, repita — um torneio em que os finalistas recebem treino completo. O **Hyperband** o envolve com múltiplos níveis de agressividade; os **pruners** do Optuna fazem o mesmo eliminando ensaios ruins no meio da CV.

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

search = HalvingRandomSearchCV(pipe, param_dist, factor=3,
                               resource='model__n_estimators',
                               max_resources=1000, cv=5)
```

## AutoML de pipeline completo

O escopo mais amplo: buscar sobre **imputação × codificação × escalonamento × seleção de atributos × família de modelo × hiperparâmetros × ensembling** conjuntamente. Sistemas: **auto-sklearn** (otimização bayesiana + warm starts por meta-aprendizado + auto-ensembling), **TPOT** (evolui [pipelines](../pipelines/index.md) por programação genética), **FLAML** (consciente de custo, frugal), **AutoGluon** (stacking multicamadas de muitos modelos; frequentemente os resultados tabulares mais fortes), além de plataformas comerciais.

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='churn', eval_metric='roc_auc') \
                .fit(train_df, presets='best_quality', time_limit=3600)
```

## O que o AutoML não consegue fazer

A busca otimiza exatamente o que você mandar. **Validação lixo na entrada, modelo lixo na saída — em escala.** Ainda são seus:

- a formulação do problema e a escolha da [métrica](../classification-metrics/index.md) (otimizar F1 vs revocação@precisão = produtos diferentes);
- o [desenho da validação](../validation/index.md) — vazamento, estrutura de grupos, ordenação temporal: o AutoML explorará alegremente qualquer vazamento *com mais força* que um humano;
- [qualidade dos dados e atributos](../eda/index.md) — uma hora de engenharia de atributos rotineiramente supera um dia de busca de hiperparâmetros;
- [justiça, explicabilidade](../explainability/index.md) e a decisão de colocar em produção.

!!! tip "Padrão prático"
    Projeto tabular, edição anos 2020: baseline (`Dummy` + regressão logística) → gradient boosting com padrões → **Optuna, ~100 ensaios** no booster → considere o AutoGluon se a acurácia valer o processamento. Escale apenas enquanto o escore de validação continuar pagando pelo esforço.

---

## Quiz

<div id="quiz-automl"></div>
<script>
buildQuiz('automl', 'AutoML', [
  {
    q: "Por que a busca aleatória costuma superar a busca em grade com o mesmo orçamento?",
    opts: [
      "Números aleatórios dão sorte",
      "Quando poucos hiperparâmetros importam, uma grade retesta os mesmos poucos valores dos importantes, enquanto a amostragem aleatória cobre densamente a faixa de cada dimensão",
      "A busca em grade não pode usar validação cruzada",
      "A busca aleatória avalia mais configurações por segundo"
    ],
    ans: 1,
    exp: "Bergstra & Bengio (2012): com uma grade 3×3, o parâmetro importante recebe só 3 valores distintos; 9 pontos aleatórios lhe dão 9. A dimensionalidade efetiva é baixa, então a cobertura por dimensão vence."
  },
  {
    q: "O que distingue a otimização bayesiana da busca aleatória?",
    opts: [
      "Ela exige uma GPU",
      "Ela ajusta um modelo substituto aos resultados dos ensaios passados e escolhe a próxima configuração equilibrando exploração e aproveitamento",
      "Ela avalia cada configuração duas vezes",
      "Ela só ajusta parâmetros contínuos"
    ],
    ans: 1,
    exp: "Cada avaliação cara informa a próxima escolha via o surrogate + a função de aquisição. A busca aleatória ignora o histórico; a otimização bayesiana aprende com ele — menos ensaios para alcançar boas regiões."
  },
  {
    q: "O halving sucessivo economiza processamento ao...",
    opts: [
      "reduzir o conjunto de dados pela metade permanentemente",
      "dar a muitas configurações um orçamento pequeno e depois promover repetidamente apenas a melhor fração para orçamentos maiores",
      "usar metade dos núcleos da CPU",
      "remover metade dos atributos"
    ],
    ans: 1,
    exp: "Configurações ruins geralmente parecem ruins cedo (poucas iterações/amostras). O desenho de torneio gasta a maior parte do processamento nos finalistas promissores em vez de treinar totalmente cada candidato."
  },
  {
    q: "Você dá a um sistema de AutoML um conjunto de dados com vazamento do alvo. Ele provavelmente vai...",
    opts: [
      "detectar e remover o atributo vazado",
      "explorar o vazamento agressivamente, reportando um escore de validação espetacular que vai desabar em produção",
      "recusar-se a treinar",
      "retornar o mesmo modelo que retornaria sem o vazamento"
    ],
    ans: 1,
    exp: "O AutoML otimiza o escore de validação que você definiu — com mais força que um humano. O desenho da validação e a prevenção de vazamentos continuam sendo trabalho do profissional; a automação amplifica configurações boas e ruins."
  },
  {
    q: "Qual decisão NÃO pode ser delegada ao AutoML?",
    opts: [
      "escolher a taxa de aprendizado de um booster",
      "escolher qual família de modelo experimentar",
      "escolher a métrica adequada ao negócio a otimizar e o esquema de validação que espelha a produção",
      "escolher o número de árvores"
    ],
    ans: 2,
    exp: "Hiperparâmetros e escolha de modelo são exatamente o que o AutoML busca. O que otimizar (revocação vs precisão vs custo) e como validar honestamente codificam o problema de negócio — esse é o contrato do humano com a máquina."
  },
  {
    q: "No Optuna, um 'pruner' é um mecanismo que...",
    opts: [
      "remove atributos com baixa importância",
      "interrompe ensaios não promissores cedo, com base em seus escores intermediários",
      "poda os ramos de árvores de decisão",
      "reduz o espaço de busca após cada ensaio"
    ],
    ans: 1,
    exp: "A poda é o espírito do halving sucessivo dentro da otimização bayesiana: um ensaio que reporta resultados intermediários ruins (ex.: após 2 de 5 folds de CV) é eliminado, liberando orçamento para candidatos melhores."
  }
]);
</script>

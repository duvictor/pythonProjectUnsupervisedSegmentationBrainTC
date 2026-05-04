import optuna
import argparse
import numpy as np
import treino as treino


# # Função de treinamento para avaliação
# def train_model(nChannel, nConv, maxIter, minLabels, lr, stepsize_con, stepsize_sim):
#     """
#     Simula um processo de treinamento para avaliar os hiperparâmetros.
#     Aqui você deve substituir pelo código real de treinamento e cálculo de perda.
#     """
#     # Exemplo de simulação de avaliação (substituir com modelo real!)
#     performance_metric = (
#             1 / (1 + abs(nChannel - 50)) +
#             1 / (1 + abs(nConv - 3)) +
#             1 / (1 + abs(maxIter - 3)) +
#             1 / (1 + abs(minLabels - 7)) +
#             lr * 10 +
#             stepsize_con * 2 +
#             stepsize_sim * 2
#     )
#     return performance_metric


# Função para otimização pelo Optuna
def objective(trial):
    # Sugere valores para os parâmetros
    nChannel = trial.suggest_int('nChannel', 10, 100)  # Intervalo ajustável
    nConv = trial.suggest_int('nConv', 1, 20)
    maxIter = trial.suggest_int('maxIter', 3, 30)
    minLabels = trial.suggest_int('minLabels', 3, 12)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)  # Usa escala logarítmica
    stepsize_con = trial.suggest_uniform('stepsize_con', 0.1, 5.0)
    stepsize_sim = trial.suggest_uniform('stepsize_sim', 0.1, 5.0)
    lambda_rotulo = trial.suggest_int('lambda_rotulo', 1, 5)
    visualize = trial.suggest_int('visualize', 0, 1)
    debug_progress = trial.suggest_int('debug_progress', 0, 1)
    debug_interval = trial.suggest_int('debug_interval', 1, 10)


    # Executa um treinamento simulado
    result = treino.treino(
        nChannel, nConv, lambda_rotulo, lr, maxIter, debug_progress, debug_interval, visualize, stepsize_con, stepsize_sim, minLabels)
    # result = train_model(nChannel, nConv, maxIter, minLabels, lr, stepsize_con, stepsize_sim)


    # Retorna a métrica de desempenho para o Optuna
    return result


if __name__ == "__main__":
    # Parser do argparse (já configurado com os parâmetros padrões do código fornecido)
    parser = argparse.ArgumentParser(description="Otimização de parâmetros")
    parser.add_argument('--trials', type=int, default=50, help='Número de tentativas de otimização do Optuna')
    args = parser.parse_args()

    # Cria o estudo do Optuna
    study = optuna.create_study(direction='minimize')  # Maximiza a métrica de desempenho
    study.optimize(objective, n_trials=args.trials)

    # Melhor conjunto de hiperparâmetros encontrados
    print("Melhores hiperparâmetros encontrados:")
    print(study.best_params)
    print("Métrica de desempenho alcançada:", study.best_value)

    # Salva os melhores resultados em um arquivo, caso necessário
    with open("melhores_hiperparametros.txt", "w") as f:
        f.write(f"Melhores parâmetros: {study.best_params}\n")
        f.write(f"Métrica: {study.best_value}\n")

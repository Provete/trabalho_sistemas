import numpy as np
import sys


def ler_entrada() -> tuple:
    NOME_ENTRADA = sys.argv[1]

    with open(NOME_ENTRADA, 'r') as entrada:
        quantidade_sistemas, dimencao_matriz, margem_erro = entrada.readline().split(' ')

        quantidade_sistemas = int(quantidade_sistemas)
        dimencao_matriz = int(dimencao_matriz)
        margem_erro = float(margem_erro)

        A: np.array = np.zeros((dimencao_matriz, dimencao_matriz))
        B: np.array = np.zeros((quantidade_sistemas, dimencao_matriz))

        for linha in range(0, dimencao_matriz):
            for coluna, valor in enumerate(entrada.readline().split(' ')):
                A[linha][coluna] = float(valor)

        for linha in range(0, quantidade_sistemas):
            for coluna, valor in enumerate(entrada.readline().split(' ')):
                B[linha][coluna] = float(valor)

        return A, B, dimencao_matriz, quantidade_sistemas, margem_erro


def gerar_matriz_auxiliar(matriz: np.array, b: np.array) -> np.array:
    dimencao_matriz = matriz.shape[0]
    matriz_auxiliar: np.array = np.zeros((dimencao_matriz, dimencao_matriz + 1))

    for i, linha in enumerate(matriz):
        for j, valor in enumerate(linha):
            matriz_auxiliar[i][j] = valor

    for i, valor in enumerate(b):
        matriz_auxiliar[i][dimencao_matriz] = valor

    return matriz_auxiliar


def multiplicar_linha(linha: np.array, multiplicador: float) -> np.array:
    for i in range(0, linha.size):
        linha[i] *= multiplicador

    return linha


def somar_linhas_com_multiplicador(linha1: np.array
                                   , linha2: np.array
                                   , multiplicador: float) -> np.array:
    linha2 = multiplicar_linha(linha2, multiplicador)

    for i in range(0, linha1.size):
        linha1[i] += linha2[i]

    return linha1


A, B, dimencao_matriz, quantidade_sistemas, margem_erro = ler_entrada()

print(gerar_matriz_auxiliar(A, B[0]))

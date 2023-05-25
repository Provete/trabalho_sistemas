import numpy
import numpy as np
import math as m
import sys


def ler_entrada() -> tuple:
    NOME_ENTRADA = "entrada.txt"

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


# Eliminação de Gauss
def pivoteamento(A: np.array, b: np.array) -> np.array:
    print(b)
    n = dimencao_matriz

    for i in range(n - 1):
        pivo = A[i][i]
        troca_pivo = i

        # Identificando maior pivo da coluna
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(pivo):
                pivo = A[j][i]
                troca_pivo = j

        # troca de linha, pivo na posição diferente de i
        if troca_pivo != i:
            b[i], b[troca_pivo] = b[troca_pivo], b[i]
            for k in range(n):
                A[i][k], A[troca_pivo][k] = A[troca_pivo][k], A[i][k]

        # divisão dos elementos
        for j in range(i + 1, n):
            matriz_aux = A[j][i] / A[i][i]
            A[j][i] = 0

            for k in range(i + 1, n):
                A[j][k] = A[j][k] - (matriz_aux * A[i][k])
            b[j] = b[j] - (matriz_aux * b[i])

    result: np.array = np.zeros(n)
    # calculo triangular superior
    for i in range(n - 1, -1, -1):
        result[i] = b[i] / A[i][i]
        for j in range(i - 1, -1, -1):
            b[j] = b[j] - A[j][i] * result[i]

    print(result)

    return result


def pegar_maior_magnitude(b: np.array) -> float:
    maior: float = 0

    for valor in b:
        if m.fabs(valor) > maior:
            maior = m.fabs(valor)

    return maior


def pegar_x_inicial(A: np.array, b: np.array) -> np.array:
    xInicial: np.array = b.copy()
    xInicial.fill(0)
    # print(xInicial)

    for i in range(b.size):
        # print(f'b[i] = {b[i]}\nA[i][i] = {A[i][i]}\n')
        xInicial[i] = b[i] / A[i][i]

    return xInicial


def resolver_sistema_gauss_jacob(A: np.array, b: np.array, X: np.array, margem_erro: float) -> np.array:
    novoX: np.array = X.copy()
    novoX.fill(0)

    for i in range(novoX.size):
        novoX[i] = b[i]

        for j, Aij in [(j, A[i][j]) for j in range(A.shape[0]) if j != i]:
            novoX[i] -= Aij * X[j]

        novoX[i] /= A[i][i]

    print(f'novoX: {novoX}\n'
          f'X: {X}\n')
    diferenca_X = numpy.subtract(novoX, X)
    erro_relativo = pegar_maior_magnitude(diferenca_X) / pegar_maior_magnitude(novoX)

    print(f'erro_relativo: {erro_relativo}')
    if erro_relativo <= margem_erro:
        return novoX
    else:
        return resolver_sistema_gauss_jacob(A, b, novoX, margem_erro)



# chamada pivoteamento para todos os sistemas
for i in range(0, quantidade_sistemas):
    pivoteamento(A.copy(), B[i])

print('\n\n\n')

print('Metodo Gauss-Jacob:')
for i in range(0, quantidade_sistemas):
    print(A)
    print(resolver_sistema_gauss_jacob(A.copy(), B[i], pegar_x_inicial(A, B[i]), margem_erro))

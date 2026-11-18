"""
Integração Numérica: Método dos Trapézios (Simples e Composto)
Disciplina: Cálculo Numérico / Métodos Numéricos
Autor: Prof. João Cardoso
Repositório: https://github.com/Zuluke/numerical-calculus.git
"""

from __future__ import annotations

import math
from typing import Callable, Tuple
import numpy as np


def regra_trapezios_composta(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Calcula a aproximação da integral definida de f em [a, b] com n subintervalos.

    Aplica a fórmula composta do Método dos Trapézios:
        I_T = h * [ (f(x_0) + f(x_n)) / 2 + sum_{i=1}^{n-1} f(x_i) ]

    Args:
        f: Função integranda vetorizada f(x).
        a: Limite inferior de integração.
        b: Limite superior de integração.
        n: Número de subintervalos (n >= 1; corresponde a n + 1 pontos).

    Returns:
        Tupla (I_aprox, x, y, h) com o valor aproximado da integral,
        o vetor dos nós x, o vetor das imagens y = f(x) e o passo h.
    """
    if n < 1:
        raise ValueError("O número de subintervalos n deve ser >= 1.")
    if b <= a:
        raise ValueError("O limite superior b deve ser estritamente maior que a.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)

    extremos = 0.5 * (y[0] + y[-1])
    internos = float(np.sum(y[1:-1])) if n > 1 else 0.0
    i_aprox = h * (extremos + internos)
    return float(i_aprox), x, y, float(h)


def cota_erro_trapezios(a: float, b: float, n: int, m2: float) -> float:
    """Calcula a cota máxima teórica do erro de truncamento no Método dos Trapézios.

    Fórmula:
        |T_t| <= (n * h^3 / 12) * M_2 = ((b - a)^3 / (12 * n^2)) * M_2
    onde M_2 = max_{x in [a, b]} |f''(x)|.
    """
    if n < 1 or m2 < 0:
        raise ValueError("Parâmetros inválidos para a cota de erro.")
    h = (b - a) / n
    return float((n * (h ** 3) / 12.0) * m2)


def dimensionar_subintervalos(a: float, b: float, m2: float, tol: float) -> int:
    """Determina o menor número inteiro de subintervalos n para garantir |T_t| <= tol."""
    if tol <= 0 or m2 < 0 or b <= a:
        raise ValueError("Parâmetros inválidos para dimensionamento de subintervalos.")
    if m2 == 0:
        return 1
    n_min = math.ceil(math.sqrt(((b - a) ** 3) * m2 / (12.0 * tol)))
    return max(1, n_min)


def executar_exemplo_classico() -> None:
    """Executa o exemplo padrão da aula: integral de e^x em [0.0, 0.6] com 7 pontos (n=6)."""
    a, b, n = 0.0, 0.6, 6
    f = np.exp
    i_exato = math.exp(b) - math.exp(a)
    m2 = math.exp(b)  # f''(x) = e^x é crescente em [0.0, 0.6]

    i_aprox, x, y, h = regra_trapezios_composta(f, a, b, n)
    erro_real = abs(i_exato - i_aprox)
    cota_erro = cota_erro_trapezios(a, b, n, m2)

    extremos_e = y[0] + y[-1]
    pares_p = float(np.sum(y[2:-1:2]))
    impares_i = float(np.sum(y[1:-1:2]))

    print("=" * 72)
    print("METODO DOS TRAPEZIOS - EXECUCAO DO EXEMPLO DA AULA")
    print("Integral: int_{0.0}^{0.6} e^x dx  |  7 pontos (n = 6 subintervalos)")
    print("Autor: Prof. Joao Cardoso")
    print("=" * 72)
    print(f"Passo h = (b - a)/n = ({b} - {a})/{n} = {h:.6f}\n")
    print(f"{'i':>3} | {'x_i':>8} | {'f(x_i) = e^x':>14} | {'Peso w_i':>8} | {'Grupo':>10}")
    print("-" * 55)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        if idx == 0 or idx == n:
            peso = 0.5
            grupo = "Extremo (E)"
        elif idx % 2 == 0:
            peso = 1.0
            grupo = "Par (P)"
        else:
            peso = 1.0
            grupo = "Impar (I)"
        print(f"{idx:>3d} | {xi:>8.4f} | {yi:>14.8f} | {peso:>8.1f} | {grupo:>10}")

    print("-" * 55)
    print(f"E = f(x_0) + f(x_6)         = {extremos_e:.8f}")
    print(f"E / 2                       = {extremos_e / 2.0:.8f}")
    print(f"I (indices impares 1, 3, 5) = {impares_i:.8f}")
    print(f"P (indices pares 2, 4)      = {pares_p:.8f}")
    print(f"Internos (I + P)            = {impares_i + pares_p:.8f}")
    print(f"I_T = h * (E/2 + I + P)     = {i_aprox:.8f}")
    print(f"I_exato = e^0.6 - 1         = {i_exato:.8f}")
    print(f"Erro Real |I_exato - I_T|   = {erro_real:.8e}")
    print(f"Cota Maxima |T_t|           = {cota_erro:.8e}")
    print(f"Verificacao |Erro| <= Cota  : {erro_real <= cota_erro}")

    print("\n" + "=" * 72)
    print("ANALISE DE CONVERGENCIA ASSINTOTICA O(h^2) PARA int_{0.0}^{0.6} e^x dx")
    print("=" * 72)
    print(f"{'n':>5} | {'h':>10} | {'I_T':>14} | {'Erro Real':>13} | {'Razao Erro':>11} | {'Cota Teorica':>13}")
    print("-" * 78)
    erro_anterior = None
    for n_val in [1, 2, 3, 6, 12, 24, 48, 96]:
        it_val, _, _, h_val = regra_trapezios_composta(f, a, b, n_val)
        err_val = abs(i_exato - it_val)
        cota_val = cota_erro_trapezios(a, b, n_val, m2)
        razao = f"{erro_anterior / err_val:8.2f}x" if erro_anterior is not None else "    ---   "
        print(f"{n_val:>5d} | {h_val:>10.6f} | {it_val:>14.10f} | {err_val:>13.6e} | {razao:>11} | {cota_val:>13.6e}")
        erro_anterior = err_val


def executar_desafio_gaussiana() -> None:
    """Executa o desafio para casa: dimensionamento de malha para int_0^1 e^(-x^2) dx."""
    a, b = 0.0, 1.0
    m2 = 2.0  # |(4x^2 - 2)e^(-x^2)| atinge máximo 2.0 em x = 0
    f_gauss = lambda x: np.exp(-(x ** 2))
    i_ref = 0.5 * math.sqrt(math.pi) * math.erf(1.0)

    print("\n" + "=" * 72)
    print("DESAFIO PRATICO: INTEGRAL GAUSSIANA int_{0}^{1} e^(-x^2) dx")
    print("=" * 72)
    for tol in [1e-4, 1e-5]:
        n_min = dimensionar_subintervalos(a, b, m2, tol)
        i_t, _, _, h_min = regra_trapezios_composta(f_gauss, a, b, n_min)
        err_real = abs(i_ref - i_t)
        cota = cota_erro_trapezios(a, b, n_min, m2)
        print(
            f"Tolerancia = {tol:.0e} -> n_min = {n_min:3d} (h = {h_min:.6f}) | "
            f"I_T = {i_t:.8f} | I_ref = {i_ref:.8f} | "
            f"Erro Real = {err_real:.4e} | Cota = {cota:.4e}"
        )


if __name__ == "__main__":
    executar_exemplo_classico()
    executar_desafio_gaussiana()

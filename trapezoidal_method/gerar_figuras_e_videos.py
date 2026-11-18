"""
Geração de Figuras Didáticas (.png) e Animações (.gif / .mp4)
Tema: Integração Numérica - Método dos Trapézios
Autor: Prof. João Cardoso
Repositório: https://github.com/Zuluke/numerical-calculus.git
"""

from __future__ import annotations

import os
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio.v2 as imageio

# Configuração estética global para publicação científica e slides 16:9
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COR_PRIMARIA = "#1a73e8"     # Azul acadêmico
COR_SECUNDARIA = "#c62828"   # Vermelho escuro (tema Beamer panqueques)
COR_VERDE = "#2e7d32"        # Verde destaque
COR_LARANJA = "#ef6c00"      # Laranja erro/alerta

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "images"
VID_DIR = BASE_DIR / "videos"
IMG_DIR.mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)


def regra_trapezios(f, a: float, b: float, n: int):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    it = h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))
    return float(it), x, y, float(h)


def gerar_fig_newton_cotes_trapezio() -> None:
    """Painel duplo comparando Trapézio Simples (n=1) vs. Trapézio Composto (n=4)."""
    f = lambda x: 0.4 * (x ** 2) + 0.35 * np.sin(2.2 * x) + 0.8
    a, b = 0.2, 2.6
    xs = np.linspace(a, b, 400)
    ys = f(xs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    configs = [
        (axes[0], 1, "Regra do Trapézio Simples ($n = 1$, $P_1(x)$ global)"),
        (axes[1], 4, "Método dos Trapézios Composto ($n = 4$ subintervalos)"),
    ]

    for ax, n_sub, titulo in configs:
        ax.plot(xs, ys, color=COR_SECUNDARIA, linewidth=2.4, label="$f(x)$ (curva exata)", zorder=4)
        _, x_nodes, y_nodes, h = regra_trapezios(f, a, b, n_sub)

        for i in range(n_sub):
            x0, x1 = x_nodes[i], x_nodes[i + 1]
            y0, y1 = y_nodes[i], y_nodes[i + 1]

            # Trapézio preenchido
            verts = [(x0, 0), (x0, y0), (x1, y1), (x1, 0)]
            poly = Polygon(verts, closed=True, facecolor=COR_PRIMARIA, alpha=0.22, edgecolor=COR_PRIMARIA, linewidth=1.4, zorder=2)
            ax.add_patch(poly)

            # Região de erro entre curva e reta secante
            x_loc = np.linspace(x0, x1, 80)
            y_true = f(x_loc)
            y_sec = y0 + (y1 - y0) * (x_loc - x0) / (x1 - x0)
            ax.fill_between(
                x_loc,
                y_true,
                y_sec,
                color=COR_LARANJA,
                alpha=0.32,
                hatch="///",
                edgecolor=COR_LARANJA,
                linewidth=0.0,
                label="Erro de truncamento" if i == 0 else None,
                zorder=3,
            )

        ax.plot(x_nodes, y_nodes, "o-", color=COR_PRIMARIA, markersize=6.5, linewidth=1.6, label="Poligonal secante $P_1(x)$", zorder=5)
        ax.set_xticks(x_nodes)
        ax.set_xticklabels([f"$x_{i}$" for i in range(n_sub + 1)])

        ax.set_title(titulo, pad=10)
        ax.set_xlabel("Nós da malha uniforme", labelpad=8)
        ax.set_xlim(a - 0.15, b + 0.15)
        ax.set_ylim(0, 4.1)
        ax.legend(loc="upper left", framealpha=0.92)

    axes[0].set_ylabel("$f(x)$")
    fig.suptitle("De Newton-Cotes ($n=1$) ao Método dos Trapézios Composto", fontsize=13.5, fontweight="bold", color="#222222", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMG_DIR / "fig_newton_cotes_trapezio.png", dpi=300)
    plt.close(fig)


def gerar_fig_execucao_python_ex() -> None:
    """Painel duplo do exercício resolvido: int_{0.0}^{0.6} e^x dx com n=6 (7 pontos)."""
    a, b, n = 0.0, 0.6, 6
    f = np.exp
    it_val, x_nodes, y_nodes, h = regra_trapezios(f, a, b, n)
    i_exato = math.exp(b) - math.exp(a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))

    # Painel Esquerdo: f(x) = e^x e os 6 trapézios
    xs = np.linspace(a, b, 300)
    ax1.plot(xs, f(xs), color=COR_SECUNDARIA, linewidth=2.5, label="$f(x) = e^x$", zorder=4)
    for i in range(n):
        x0, x1 = x_nodes[i], x_nodes[i + 1]
        y0, y1 = y_nodes[i], y_nodes[i + 1]
        verts = [(x0, 0), (x0, y0), (x1, y1), (x1, 0)]
        poly = Polygon(verts, closed=True, facecolor=COR_PRIMARIA, alpha=0.22, edgecolor=COR_PRIMARIA, linewidth=1.3, zorder=2)
        ax1.add_patch(poly)
        ax1.text(0.5 * (x0 + x1), 0.38, f"$T_{i+1}$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COR_PRIMARIA)

    ax1.scatter(x_nodes, y_nodes, color="#111111", s=40, zorder=5, label="7 nós ($h = 0{,}1$)")
    ax1.set_title("Execução em Python: 6 Trapézios sobre $e^x$ em $[0{,}0;\\; 0{,}6]$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$f(x)$")
    ax1.set_xlim(-0.03, 0.63)
    ax1.set_ylim(0, 2.02)

    caixa_res = (
        f"$I_T = {it_val:.8f}$\n"
        f"$I_{{\\mathrm{{exato}}}} = {i_exato:.8f}$\n"
        f"$|E_{{\\mathrm{{real}}}}| = {abs(i_exato - it_val):.2e}$"
    )
    ax1.text(0.03, 1.50, caixa_res, fontsize=9.0, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COR_PRIMARIA, alpha=0.95))
    ax1.legend(loc="lower right", framealpha=0.92)

    # Painel Direito: Erro local real vs Cota local teórica em cada subintervalo
    erros_locais_reais = []
    cotas_locais = []
    labels_sub = []
    for i in range(n):
        x0, x1 = x_nodes[i], x_nodes[i + 1]
        area_exata_i = math.exp(x1) - math.exp(x0)
        area_trap_i = 0.5 * h * (math.exp(x0) + math.exp(x1))
        erros_locais_reais.append(abs(area_exata_i - area_trap_i) * 1e4)
        cotas_locais.append(((h ** 3) / 12.0) * math.exp(x1) * 1e4)
        labels_sub.append(f"[{x0:.1f}; {x1:.1f}]")

    idx = np.arange(n)
    larg = 0.36
    b1 = ax2.bar(idx - larg / 2, erros_locais_reais, width=larg, color=COR_PRIMARIA, alpha=0.88, label="Erro real $|E_i|$ ($\\times 10^{-4}$)")
    b2 = ax2.bar(idx + larg / 2, cotas_locais, width=larg, color=COR_LARANJA, alpha=0.82, label="Cota local $\\frac{h^3}{12} e^{x_{i+1}}$ ($\\times 10^{-4}$)")

    for bar in b1:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=COR_PRIMARIA)
    for bar in b2:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=COR_LARANJA)

    ax2.set_title("Distribuição do Erro Local por Subintervalo")
    ax2.set_xlabel("Subintervalo $[x_i;\\; x_{i+1}]$")
    ax2.set_ylabel("Erro absoluto ($\\times 10^{-4}$)")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels_sub)
    ax2.set_ylim(0, 1.85)
    ax2.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_execucao_python_ex.png", dpi=300)
    plt.close(fig)


def gerar_fig_convergencia_erro() -> None:
    """Painel duplo: Convergência de I_T(n) para I_exato e gráfico Log-Log O(h^2)."""
    a, b = 0.0, 0.6
    f = np.exp
    i_exato = math.exp(b) - math.exp(a)
    m2 = math.exp(b)

    ns = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    hs = (b - a) / ns
    its = []
    erros = []
    cotas = []
    for n in ns:
        it_val, _, _, h_val = regra_trapezios(f, a, b, int(n))
        its.append(it_val)
        erros.append(abs(i_exato - it_val))
        cotas.append((n * (h_val ** 3) / 12.0) * m2)

    its = np.array(its)
    erros = np.array(erros)
    cotas = np.array(cotas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5))

    # Painel Esquerdo: Convergência do valor I_T(n)
    ax1.plot(ns[:9], its[:9], "o-", color=COR_PRIMARIA, linewidth=2.0, markersize=6, label="Aproximação $I_T(n)$")
    ax1.axhline(i_exato, color=COR_VERDE, linestyle="--", linewidth=2.0, label=f"Valor exato $e^{{0{{,}}6}}-1 = {i_exato:.6f}$")
    ax1.scatter([6], [its[4]], color=COR_SECUNDARIA, s=90, zorder=6, label="$n=6$ (7 pontos): $0{,}822804$")
    ax1.set_title("Convergência do Valor Aproximado $I_T(n)$")
    ax1.set_xlabel("Número de subintervalos ($n$)")
    ax1.set_ylabel("Valor da integral")
    ax1.set_xticks([1, 2, 4, 6, 8, 12, 16, 24])
    ax1.legend(loc="upper right", framealpha=0.92)

    # Painel Direito: Log-Log do Erro vs Passo h
    ax2.loglog(hs, erros, "o-", color=COR_PRIMARIA, linewidth=2.2, markersize=5.5, label="Erro real $|I_{\\mathrm{exato}} - I_T|$")
    ax2.loglog(hs, cotas, "s--", color=COR_SECUNDARIA, linewidth=1.8, markersize=5.0, label="Cota máxima $\\frac{(b-a)h^2}{12}M_2$")

    # Guia de inclinação O(h^2)
    h_ref = np.array([0.008, 0.3])
    y_ref = 0.055 * (h_ref ** 2)
    ax2.loglog(h_ref, y_ref, ":", color="#444444", linewidth=1.6, label="Referência de inclinação $+2$ ($O(h^2)$)")

    ax2.invert_xaxis()  # Passo h diminuindo da esquerda para a direita (refinamento)
    ax2.set_title("Erro vs. Passo $h$ ($O(h^2)$) em Escala Log-Log")
    ax2.set_xlabel("Tamanho do passo $h = (b-a)/n$ (refinando $\\rightarrow$)")
    ax2.set_ylabel("Erro absoluto")
    ax2.legend(loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig_convergencia_erro.png", dpi=300)
    plt.close(fig)


def gerar_fig_animacao_frames_e_videos() -> None:
    """Gera mosaico 2x2 de quadros-chave (.png) e a animação em vídeo (.gif e .mp4)."""
    a, b = 0.2, 2.6
    f = lambda x: 0.4 * (x ** 2) + 0.35 * np.sin(2.2 * x) + 0.8
    i_exato = (
        (0.4 / 3.0) * (b ** 3 - a ** 3)
        - (0.35 / 2.2) * (math.cos(2.2 * b) - math.cos(2.2 * a))
        + 0.8 * (b - a)
    )
    m2_est = 2.50  # |f''(x)| = |0.8 - 0.35*(2.2^2)*sin(2.2x)| <= 0.8 + 1.694 = 2.494 <= 2.50
    xs_curva = np.linspace(a, b, 300)
    ys_curva = f(xs_curva)

    # 1. Mosaico 2x2 de quadros-chave para o slide
    quadros = [1, 2, 4, 12]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.6), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, quadros):
        it_val, x_nodes, y_nodes, h = regra_trapezios(f, a, b, n)
        erro = abs(i_exato - it_val)
        ax.plot(xs_curva, ys_curva, color=COR_SECUNDARIA, linewidth=2.2, label="$f(x)$")
        for i in range(n):
            x0, x1 = x_nodes[i], x_nodes[i + 1]
            y0, y1 = y_nodes[i], y_nodes[i + 1]
            poly = Polygon([(x0, 0), (x0, y0), (x1, y1), (x1, 0)], closed=True, facecolor=COR_PRIMARIA, alpha=0.22, edgecolor=COR_PRIMARIA, linewidth=1.1)
            ax.add_patch(poly)
            x_loc = np.linspace(x0, x1, 40)
            y_sec = y0 + (y1 - y0) * (x_loc - x0) / (x1 - x0)
            ax.fill_between(x_loc, f(x_loc), y_sec, color=COR_LARANJA, alpha=0.55, hatch="///", edgecolor=COR_LARANJA, linewidth=0.0)
        ax.plot(x_nodes, y_nodes, "o-", color=COR_PRIMARIA, markersize=4.5, linewidth=1.2)
        ax.set_title(f"$n = {n}$ ($h = {h:.3f}$) | $I_T = {it_val:.5f}$ | Erro $= {erro:.2e}$", fontsize=9.8)
        ax.set_xlim(a - 0.08, b + 0.08)
        ax.set_ylim(0, 3.8)

    axes[1, 0].set_xlabel("$x$")
    axes[1, 1].set_xlabel("$x$")
    axes[0, 0].set_ylabel("$f(x)$")
    axes[1, 0].set_ylabel("$f(x)$")
    fig.suptitle(f"Quadros da Animação em Python: Redução Visual da Área de Erro ($I_{{\\mathrm{{exato}}}} = {i_exato:.5f}$)", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMG_DIR / "fig_animacao_frames.png", dpi=300)
    plt.close(fig)

    # 2. Animação dinâmica (.gif e .mp4) de n=1 até n=24
    ns_anim = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24]
    hist_n = []
    hist_err = []
    frames_rgb = []

    fig_anim, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for n in ns_anim:
        ax_left.clear()
        ax_right.clear()

        it_val, x_nodes, y_nodes, h = regra_trapezios(f, a, b, n)
        erro = abs(i_exato - it_val)
        hist_n.append(n)
        hist_err.append(erro)

        # Esquerda: Trapézios + Erro sombreado
        ax_left.plot(xs_curva, ys_curva, color=COR_SECUNDARIA, linewidth=2.4, label="$f(x) = 0{,}4x^2 + 0{,}35\\sin(2{,}2x) + 0{,}8$")
        for i in range(n):
            x0, x1 = x_nodes[i], x_nodes[i + 1]
            y0, y1 = y_nodes[i], y_nodes[i + 1]
            poly = Polygon([(x0, 0), (x0, y0), (x1, y1), (x1, 0)], closed=True, facecolor=COR_PRIMARIA, alpha=0.22, edgecolor=COR_PRIMARIA, linewidth=1.2)
            ax_left.add_patch(poly)
            x_loc = np.linspace(x0, x1, 40)
            y_sec = y0 + (y1 - y0) * (x_loc - x0) / (x1 - x0)
            ax_left.fill_between(x_loc, f(x_loc), y_sec, color=COR_LARANJA, alpha=0.55)

        ax_left.plot(x_nodes, y_nodes, "o-", color=COR_PRIMARIA, markersize=4.5, linewidth=1.3)
        ax_left.set_title(f"Malha: $n = {n}$ subintervalos ($h = {h:.4f}$)")
        ax_left.set_xlabel("$x$")
        ax_left.set_ylabel("$f(x)$")
        ax_left.set_xlim(a - 0.08, b + 0.08)
        ax_left.set_ylim(0, 4.0)
        ax_left.text(
            0.28,
            2.50,
            f"$I_T = {it_val:.6f}$\n$I_{{\\mathrm{{exato}}}} = {i_exato:.6f}$\nErro $= {erro:.2e}$",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COR_PRIMARIA),
        )
        ax_left.legend(loc="upper left", fontsize=8.2)

        # Direita: Curva de decaimento do erro em tempo real
        ax_right.semilogy(hist_n, hist_err, "o-", color=COR_PRIMARIA, linewidth=2.2, markersize=6, label="Erro real $|I_{\\mathrm{exato}} - I_T|$")
        ns_teor = np.array(ns_anim, dtype=float)
        cotas_teor = ((b - a) ** 3 / (12.0 * (ns_teor ** 2))) * m2_est
        ax_right.semilogy(ns_teor, cotas_teor, "--", color=COR_SECUNDARIA, alpha=0.75, label="Cota teórica $O(n^{-2})$")
        ax_right.set_title("Queda Quadrática do Erro com o Refinamento")
        ax_right.set_xlabel("Número de subintervalos ($n$)")
        ax_right.set_ylabel("Erro absoluto (escala log)")
        ax_right.set_xlim(0, 25)
        ax_right.set_ylim(5e-4, 5e0)
        ax_right.legend(loc="upper right")

        fig_anim.suptitle("Método dos Trapézios: Convergência Visual em Python", fontsize=12, fontweight="bold")
        fig_anim.tight_layout(rect=[0, 0, 1, 0.93])

        # Renderiza quadro para buffer RGB
        fig_anim.canvas.draw()
        buf = np.asarray(fig_anim.canvas.buffer_rgba())[:, :, :3].copy()
        frames_rgb.append(buf)

    plt.close(fig_anim)

    # Salvar quadros individuais PNG para incorporacao nativa no PDF via pacote animate
    frames_dir = VID_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx_f, fr in enumerate(frames_rgb):
        imageio.imwrite(frames_dir / f"frame_{idx_f:02d}.png", fr)

    # Salvar GIF (com pausa extra no último frame)
    frames_gif = frames_rgb + [frames_rgb[-1]] * 4
    gif_path = VID_DIR / "animacao_trapezios.gif"
    imageio.mimsave(gif_path, frames_gif, duration=0.85, loop=0)

    # Salvar MP4 via imageio-ffmpeg (garantindo dimensões pares para h264)
    mp4_path = VID_DIR / "animacao_trapezios.mp4"
    h_img, w_img, _ = frames_rgb[0].shape
    h_even = h_img - (h_img % 2)
    w_even = w_img - (w_img % 2)
    frames_mp4 = [fr[:h_even, :w_even, :] for fr in frames_gif]
    with imageio.get_writer(mp4_path, fps=1.2, codec="libx264", pixelformat="yuv420p") as writer:
        for fr in frames_mp4:
            writer.append_data(fr)

    print("Figuras geradas em:", IMG_DIR)
    print("Videos gerados em:", VID_DIR)


if __name__ == "__main__":
    gerar_fig_newton_cotes_trapezio()
    gerar_fig_execucao_python_ex()
    gerar_fig_convergencia_erro()
    gerar_fig_animacao_frames_e_videos()


from pathlib import Path
import random

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Jogo da Velha | IA",
    page_icon="✕",
    layout="centered"
)

ROOT = Path(__file__).resolve().parent.parent
PASTA_MODELOS = ROOT / "modelos"

MODELOS = {
    "k-NN": "knn.pkl",
    "SVM": "svm.pkl",
    "MLP": "mlp.pkl",
    "Random Forest": "random_forest.pkl",
    "Árvore de Decisão": "arvore_decisao.pkl"
}

COLUNAS_TABULEIRO = [
    "top_left", "top_middle", "top_right",
    "middle_left", "middle_middle", "middle_right",
    "bottom_left", "bottom_middle", "bottom_right"
]

COLUNAS_FEATURES = [
    "x_count", "o_count", "occupied",
    "lines_with_2_x", "lines_with_2_o",
    "empty_count", "next_player"
]

LINHAS_VITORIA = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

st.markdown("""
<style>
    .stApp {
        background-color: #f7f8f6;
        color: #26352c;
    }

    .block-container {
        max-width: 720px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        letter-spacing: -1px !important;
        color: #263e30 !important;
    }

    h2, h3 {
        color: #263e30 !important;
        font-weight: 650 !important;
    }

    .subtitle {
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #829086;
        letter-spacing: 1.1px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
        box-shadow: none;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2e5942;
        border-color: #2e5942;
        color: white;
    }

    div.stButton > button[kind="secondary"] {
        background-color: white;
        border-color: #dce4dc;
        color: #34493b;
    }

    div.stButton > button:hover,
    div.stButton > button:active,
    div.stButton > button:focus,
    div.stButton > button:focus-visible {
        box-shadow: none;
        outline: none;
        transform: none;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[kind="primary"]:active,
    div.stButton > button[kind="primary"]:focus,
    div.stButton > button[kind="primary"]:focus-visible {
        background-color: #2e5942;
        border-color: #2e5942;
        color: white;
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[kind="secondary"]:active,
    div.stButton > button[kind="secondary"]:focus,
    div.stButton > button[kind="secondary"]:focus-visible {
        background-color: white;
        border-color: #dce4dc;
        color: #34493b;
    }

    div[class*="st-key-casa_"] button {
        height: 95px;
        min-height: 70px;
        font-size: 2.8rem;
        font-weight: 750;
        background: white;
        border: 1px solid #d9e3da;
        color: #2e5942;
        border-radius: 12px;
        box-shadow: none;
        transition: none;
    }

    div[class*="st-key-casa_"] button:hover,
    div[class*="st-key-casa_"] button:active,
    div[class*="st-key-casa_"] button:focus,
    div[class*="st-key-casa_"] button:focus-visible,
    div[class*="st-key-casa_"] button:disabled {
        opacity: 1;
        background: white;
        border: 1px solid #d9e3da;
        box-shadow: none;
        outline: none;
        transform: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e7e1;
        border-radius: 12px;
        padding: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    .game-status {
        text-align: center;
        font-weight: 650;
        background: #e8f0e9;
        border: 1px solid #d5e2d7;
        border-radius: 10px;
        color: #2e5942;
        margin: 1rem 0 1.25rem;
        padding: 0.7rem 1rem;
    }

    .game-meta {
        color: #65736a;
        font-size: 0.88rem;
        margin: -0.35rem 0 0.5rem;
        text-align: center;
    }

    .footer-note {
        font-size: 0.8rem;
        color: #89948c;
        margin-top: 2rem;
        line-height: 1.6;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


def encontrar_modelo(arquivo):
    caminhos = [
        ROOT / "modelos" / arquivo,
        ROOT / "models" / arquivo
    ]

    for caminho in caminhos:
        if caminho.is_file():
            return caminho

    return None


@st.cache_resource(show_spinner=False)
def carregar_modelo(caminho, modificacao):
    artefato = joblib.load(caminho)

    if isinstance(artefato, dict) and "model" in artefato:
        return artefato

    return {"model": artefato}


def verificar_estado(tabuleiro):
    for a, b, c in LINHAS_VITORIA:
        if (
            tabuleiro[a] != "b"
            and tabuleiro[a] == tabuleiro[b] == tabuleiro[c]
        ):
            if tabuleiro[a] == "x":
                return "X_VENCEU"

            return "O_VENCEU"

    if "b" not in tabuleiro:
        return "EMPATE"

    return "TEM_JOGO"


def posicoes_livres(tabuleiro):
    return [
        i for i, valor in enumerate(tabuleiro)
        if valor == "b"
    ]


def extrair_features(tabuleiro):
    x_count = tabuleiro.count("x")
    o_count = tabuleiro.count("o")

    lines_with_2_x = 0
    lines_with_2_o = 0

    for linha in LINHAS_VITORIA:
        valores = [tabuleiro[i] for i in linha]

        if valores.count("x") == 2 and valores.count("b") == 1:
            lines_with_2_x += 1

        if valores.count("o") == 2 and valores.count("b") == 1:
            lines_with_2_o += 1

    return {
        "x_count": x_count,
        "o_count": o_count,
        "occupied": x_count + o_count,
        "lines_with_2_x": lines_with_2_x,
        "lines_with_2_o": lines_with_2_o,
        "empty_count": tabuleiro.count("b"),
        "next_player": 0 if x_count == o_count else 1
    }


def preparar_entrada(tabuleiro, artefato):
    modelo = artefato["model"]

    abordagem = artefato.get("approach")
    colunas = artefato.get("columns")

    if abordagem is None:
        quantidade = getattr(
            modelo, "n_features_in_", None
        )

        if quantidade == 7:
            abordagem = "features"
        elif quantidade == 9:
            abordagem = "tabuleiro"

    if abordagem == "features":
        features = extrair_features(tabuleiro)

        return pd.DataFrame(
            [features],
            columns=colunas or COLUNAS_FEATURES
        )

    if abordagem == "tabuleiro":
        mapping = artefato.get(
            "board_mapping",
            {"x": 1, "o": -1, "b": 0}
        )

        valores = {
            coluna: mapping[valor]
            for coluna, valor in zip(
                COLUNAS_TABULEIRO, tabuleiro
            )
        }

        return pd.DataFrame(
            [valores],
            columns=colunas or COLUNAS_TABULEIRO
        )

    raise ValueError(
        "Não foi possível identificar as features do modelo."
    )


def prever(tabuleiro, artefato):
    modelo = artefato["model"]
    entrada = preparar_entrada(tabuleiro, artefato)

    previsao = str(modelo.predict(entrada)[0])

    probabilidades = {}

    if hasattr(modelo, "predict_proba"):
        valores = modelo.predict_proba(entrada)[0]

        probabilidades = dict(
            zip(
                map(str, modelo.classes_),
                map(float, valores)
            )
        )

    return previsao, probabilidades


def escolher_jogada(tabuleiro, artefato):
    livres = posicoes_livres(tabuleiro)

    for posicao in livres:
        simulacao = tabuleiro.copy()
        simulacao[posicao] = "o"

        if verificar_estado(simulacao) == "O_VENCEU":
            return posicao

    for posicao in livres:
        simulacao = tabuleiro.copy()
        simulacao[posicao] = "x"

        if verificar_estado(simulacao) == "X_VENCEU":
            return posicao

    pontuacoes = []

    for posicao in livres:
        simulacao = tabuleiro.copy()
        simulacao[posicao] = "o"

        previsao, probabilidades = prever(
            simulacao, artefato
        )

        if probabilidades:
            pontuacao = (
                probabilidades.get("O_VENCEU", 0)
                - probabilidades.get("X_VENCEU", 0)
            )
        else:
            pontuacao = {
                "O_VENCEU": 1,
                "EMPATE": 0.3,
                "TEM_JOGO": 0,
                "X_VENCEU": -1
            }.get(previsao, 0)

        pontuacoes.append((pontuacao, posicao))

    maior_pontuacao = max(
        pontuacao for pontuacao, _ in pontuacoes
    )

    melhores = [
        posicao
        for pontuacao, posicao in pontuacoes
        if pontuacao == maior_pontuacao
    ]

    for preferida in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
        if preferida in melhores:
            return preferida

    return random.choice(livres)


def nova_partida():
    st.session_state.tabuleiro = ["b"] * 9
    st.session_state.ultima_previsao = None
    st.session_state.aviso = None


def selecionar_algoritmo(nome):
    st.session_state.algoritmo = nome
    nova_partida()


def jogar(posicao):
    tabuleiro = st.session_state.tabuleiro

    if (
        tabuleiro[posicao] != "b"
        or verificar_estado(tabuleiro) != "TEM_JOGO"
    ):
        return

    algoritmo = st.session_state.algoritmo

    if algoritmo is None:
        return

    caminho = encontrar_modelo(MODELOS[algoritmo])

    if caminho is None:
        st.session_state.aviso = "Modelo não encontrado."
        return

    tabuleiro[posicao] = "x"

    try:
        artefato = carregar_modelo(
            str(caminho),
            caminho.stat().st_mtime_ns
        )

        if verificar_estado(tabuleiro) == "TEM_JOGO":
            jogada_ia = escolher_jogada(
                tabuleiro, artefato
            )
            tabuleiro[jogada_ia] = "o"

        previsao, _ = prever(tabuleiro, artefato)
        estado_real = verificar_estado(tabuleiro)

        acertou = previsao == estado_real

        st.session_state.ultima_previsao = {
            "algoritmo": algoritmo,
            "previsao": previsao,
            "real": estado_real,
            "acertou": acertou
        }

        estatisticas = st.session_state.estatisticas.setdefault(
            algoritmo,
            {"acertos": 0, "erros": 0}
        )

        if acertou:
            estatisticas["acertos"] += 1
        else:
            estatisticas["erros"] += 1

        st.session_state.aviso = None

    except Exception as erro:

        if verificar_estado(tabuleiro) == "TEM_JOGO":
            livres = posicoes_livres(tabuleiro)

            if livres:
                tabuleiro[random.choice(livres)] = "o"

        st.session_state.ultima_previsao = None
        st.session_state.aviso = (
            f"Erro ao executar {algoritmo}: {erro}. "
            "O computador fez uma jogada aleatória."
        )


if "tabuleiro" not in st.session_state:
    nova_partida()

if "algoritmo" not in st.session_state:
    st.session_state.algoritmo = next(
        (
            nome
            for nome, arquivo in MODELOS.items()
            if encontrar_modelo(arquivo)
        ),
        None
    )

if "estatisticas" not in st.session_state:
    st.session_state.estatisticas = {}


st.title("Jogo da Velha")

st.markdown(
    '<div class="section-label">01. ESCOLHA O ALGORITMO</div>',
    unsafe_allow_html=True
)

colunas = st.columns(3)

for indice, (nome, arquivo) in enumerate(MODELOS.items()):
    disponivel = encontrar_modelo(arquivo) is not None

    with colunas[indice % 3]:
        st.button(
            nome,
            key=f"algoritmo_{indice}",
            use_container_width=True,
            disabled=not disponivel,
            type=(
                "primary"
                if st.session_state.algoritmo == nome
                else "secondary"
            ),
            on_click=selecionar_algoritmo,
            args=(nome,)
        )

if st.session_state.algoritmo is None:
    st.warning(
        "Nenhum modelo disponível. "
        "Adicione os arquivos .pkl na pasta modelos/."
    )
    st.stop()

st.caption(
    f"Modelo selecionado: {st.session_state.algoritmo}"
)

estado = verificar_estado(st.session_state.tabuleiro)
mensagens = {
    "TEM_JOGO": "Sua vez de jogar",
    "X_VENCEU": "Você venceu!",
    "O_VENCEU": "O computador venceu!",
    "EMPATE": "Empate!"
}

st.markdown(
    f'<div class="game-status">{mensagens[estado]}</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="game-meta">Clique em uma casa vazia para jogar</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-label">02. JOGO</div>',
    unsafe_allow_html=True
)

coluna_jogo, coluna_analise = st.columns([3, 2], gap="large")

with coluna_jogo:
    for linha in range(3):
        colunas = st.columns(3, gap="small")

        for coluna in range(3):
            posicao = linha * 3 + coluna
            valor = st.session_state.tabuleiro[posicao]

            with colunas[coluna]:
                st.button(
                    {
                        "x": "X",
                        "o": "O",
                        "b": " "
                    }[valor],
                    key=f"casa_{posicao}",
                    use_container_width=True,
                    disabled=(
                        valor != "b"
                        or estado != "TEM_JOGO"
                    ),
                    on_click=jogar,
                    args=(posicao,)
                )

    st.button(
        "Nova partida",
        use_container_width=True,
        on_click=nova_partida
    )

    if st.session_state.aviso:
        st.warning(st.session_state.aviso)

with coluna_analise:
    st.markdown(
        '<div class="section-label">03. ANÁLISE</div>',
        unsafe_allow_html=True
    )

    resultado = st.session_state.ultima_previsao

    if resultado:
        st.caption("Previsão da IA")
        st.code(resultado["previsao"], language=None)

        st.caption("Estado real")
        st.code(resultado["real"], language=None)

        if resultado["acertou"]:
            st.success("A previsão foi correta.")
        else:
            st.error("A previsão foi diferente do estado real.")
    else:
        st.caption(
            "Faça uma jogada para visualizar a classificação."
        )

"""Estrategias de sinal: call/put/nenhum + confianca (0-1).

Cada estrategia expõe:
  - signals(features_df) -> DataFrame vetorizado com colunas 'action','confidence'
    (usado no backtest, rapido em todo o historico)
  - predict_last(features_df) -> (action, confidence) so para a ultima linha
    (usado ao vivo)

'action' e "call", "put" ou None (nao operar naquele candle).
"""
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, label_up_down


def make_signals_df(action, confidence, index):
    """Monta o DataFrame de sinais com a coluna 'action' fixada em dtype
    object. Sem isso, o pandas 3.x costuma inferir um dtype de string
    (Arrow-backed) pra uma coluna so com "call"/"put"/None, e nesse dtype o
    valor nulo deixa de ser o None do Python - ele passa a reprsentar como
    "nan" mas falha em qualquer checagem 'is None', deixando passar um sinal
    invalido pra frente (isso ja quebrou o loop ao vivo em producao)."""
    return pd.DataFrame({
        "action": pd.array(action, dtype=object),
        "confidence": confidence,
    }, index=index)


class Strategy:
    name = "base"
    params = {}

    def fit(self, features_df):
        """Estrategias baseadas em regra nao precisam treinar; ML sobrescreve."""
        return self

    def signals(self, features_df):
        raise NotImplementedError

    def predict_last(self, features_df):
        sig = self.signals(features_df.tail(max(60, len(features_df))))
        action = sig["action"].iloc[-1]
        if pd.isna(action):
            action = None
        return action, sig["confidence"].iloc[-1]


class SmaCrossover(Strategy):
    name = "sma_crossover"

    def __init__(self, threshold=0.0006):
        self.params = {"threshold": threshold}

    def signals(self, df):
        diff = df["sma_diff"]
        action = np.where(diff > self.params["threshold"], "call",
                  np.where(diff < -self.params["threshold"], "put", None))
        confidence = (diff.abs() / (self.params["threshold"] * 4)).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class RsiReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, low=30, high=70):
        self.params = {"low": low, "high": high}

    def signals(self, df):
        rsi = df["rsi_14"]
        low, high = self.params["low"], self.params["high"]
        action = np.where(rsi < low, "call", np.where(rsi > high, "put", None))
        dist = np.where(rsi < low, (low - rsi) / low,
               np.where(rsi > high, (rsi - high) / (100 - high), 0))
        confidence = pd.Series(dist, index=df.index).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class MomentumFollow(Strategy):
    name = "momentum_follow"

    def __init__(self, min_body=0.0003):
        self.params = {"min_body": min_body}

    def signals(self, df):
        bullish = (df["momentum_5"] > 0) & (df["body"] > self.params["min_body"])
        bearish = (df["momentum_5"] < 0) & (df["body"] < -self.params["min_body"])
        action = np.where(bullish, "call", np.where(bearish, "put", None))
        confidence = (df["body"].abs() / (self.params["min_body"] * 6)).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class BollingerReversion(Strategy):
    name = "bollinger_reversion"

    def __init__(self, band=1.8):
        self.params = {"band": band}

    def signals(self, df):
        pos = df["bb_position"]
        band = self.params["band"]
        action = np.where(pos < -band, "call", np.where(pos > band, "put", None))
        confidence = ((pos.abs() - band) / band).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class MLStrategy(Strategy):
    """Classificador que aprende a relacao entre as features e o proximo
    movimento de preco. Reaprende sempre que fit() e chamado de novo, o que
    permite o 'modo evolutivo/adaptativo' pedido: o loop ao vivo pode
    re-treinar periodicamente com candles novos.
    """
    name = "ml_gradient_boost"

    def __init__(self, edge=0.06, horizon=1, random_state=42):
        self.params = {"edge": edge, "horizon": horizon}
        self.model = None
        self._random_state = random_state

    def fit(self, features_df):
        from sklearn.ensemble import HistGradientBoostingClassifier

        labels = label_up_down(features_df, self.params["horizon"])
        data = features_df[FEATURE_COLUMNS].copy()
        data["label"] = labels
        data = data.dropna()
        if len(data) < 200:
            raise ValueError("dados insuficientes para treinar o modelo (min 200 linhas validas)")

        X = data[FEATURE_COLUMNS]
        y = data["label"]
        self.model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=150,
            random_state=self._random_state)
        self.model.fit(X, y)
        return self

    def signals(self, df):
        if self.model is None:
            raise RuntimeError("chame fit() antes de signals()")
        X = df[FEATURE_COLUMNS]
        valid = X.notna().all(axis=1)
        proba = np.full(len(df), np.nan)
        proba[valid.values] = self.model.predict_proba(X[valid])[:, 1]
        proba = pd.Series(proba, index=df.index)

        edge = self.params["edge"]
        action = np.where(proba > 0.5 + edge, "call",
                  np.where(proba < 0.5 - edge, "put", None))
        confidence = ((proba - 0.5).abs() * 2).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class MeanReversionZScore(Strategy):
    """Aposta na reversao a media: preco muito acima/abaixo da media movel
    tende a voltar. So faz sentido se o diagnostico (Hurst<0.5) confirmar
    que a serie de fato reverte a media nessa janela de tempo."""
    name = "mean_reversion_zscore"

    def __init__(self, window=20, z_threshold=1.2):
        self.params = {"window": window, "z_threshold": z_threshold}

    def signals(self, df):
        window = int(self.params["window"])
        mean = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        z = (df["close"] - mean) / std.replace(0, np.nan)
        thr = self.params["z_threshold"]
        action = np.where(z < -thr, "call", np.where(z > thr, "put", None))
        confidence = ((z.abs() - thr) / thr).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class AutocorrelationLag(Strategy):
    """Aposta na correlacao serial candle-a-candle. lag_sign=-1 assume serie
    anti-persistente (aposta contra o ultimo candle); lag_sign=1 assume
    persistente (segue o ultimo candle). O sinal certo depende do resultado
    de diagnostics.autocorrelation() para o ativo/timeframe usado - a busca
    evolutiva testa os dois e fica com o que performar melhor."""
    name = "autocorr_lag"

    def __init__(self, lag_sign=-1, min_move=0.0002):
        self.params = {"lag_sign": lag_sign, "min_move": min_move}

    def signals(self, df):
        signed_ret = df["ret_1"] * self.params["lag_sign"]
        min_move = self.params["min_move"]
        action = np.where(signed_ret > min_move, "call",
                  np.where(signed_ret < -min_move, "put", None))
        confidence = (signed_ret.abs() / (min_move * 4)).clip(0, 1)
        return make_signals_df(action, confidence, df.index)


class HurstRegimeSwitch(Strategy):
    """Meta-estrategia adaptativa: mede autocorrelacao serial numa janela
    movel para decidir, candle a candle, se o regime local parece de reversao
    a media (usa bb_position) ou de tendencia (usa momentum_5). E o mais
    proximo de 'inteligente/evolutivo' no sentido de mudar de abordagem
    conforme o comportamento observado da serie muda."""
    name = "hurst_regime_switch"

    def __init__(self, window=60):
        self.params = {"window": window}

    def signals(self, df):
        window = int(self.params["window"])
        roll_autocorr = df["ret_1"].rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        mean_reverting = roll_autocorr < 0

        mr_signal = df["bb_position"]
        mom_signal = df["momentum_5"]

        action = np.where(mean_reverting & (mr_signal < -1), "call",
                  np.where(mean_reverting & (mr_signal > 1), "put",
                  np.where(~mean_reverting & (mom_signal > 0), "call",
                  np.where(~mean_reverting & (mom_signal < 0), "put", None))))
        confidence = roll_autocorr.abs().clip(0, 1).fillna(0)
        return make_signals_df(action, confidence, df.index)


class PatternCluster(Strategy):
    """Nao usa NENHUM indicador classico. Agrupa (k-means) o 'formato' das
    ultimas `window` variacoes de preco em grupos, e aprende empiricamente
    - so olhando o que aconteceu depois de cada grupo no historico - se
    algum grupo tem viés real de alta ou baixa. So opera quando o candle
    atual cai num grupo com viés historico claro (edge minimo).

    Isto e descoberta de padrao pura: os grupos nao tem nome nem definicao
    matematica pre-existente (nao e 'martelo', nao e 'engolfo') - sao
    apenas formas que se repetem nos dados e que, empiricamente, precederam
    mais altas ou mais baixas que o normal."""
    name = "pattern_cluster"

    def __init__(self, window=10, n_clusters=12, min_cluster_edge=0.08, min_cluster_size=15):
        self.params = {"window": window, "n_clusters": n_clusters, "min_cluster_edge": min_cluster_edge}
        self.min_cluster_size = min_cluster_size
        self.kmeans = None
        self.cluster_bias = {}

    def _shapes(self, df):
        rets = df["close"].pct_change()
        cols = {f"lag_{k}": rets.shift(k) for k in range(self.params["window"])}
        return pd.DataFrame(cols, index=df.index)

    def fit(self, features_df):
        from sklearn.cluster import KMeans

        shapes = self._shapes(features_df)
        label = label_up_down(features_df, horizon=1)
        data = shapes.copy()
        data["label"] = label
        data = data.dropna()
        min_rows = self.min_cluster_size * self.params["n_clusters"]
        if len(data) < min_rows:
            raise ValueError(f"dados insuficientes para clustering (min {min_rows} linhas validas)")

        shape_cols = [c for c in data.columns if c != "label"]
        X = data[shape_cols].to_numpy()
        y = data["label"].to_numpy()

        self.kmeans = KMeans(n_clusters=self.params["n_clusters"], n_init=10, random_state=42)
        cluster_ids = self.kmeans.fit_predict(X)

        self.cluster_bias = {}
        for c in range(self.params["n_clusters"]):
            mask = cluster_ids == c
            if mask.sum() < self.min_cluster_size:
                continue
            edge = y[mask].mean() - 0.5
            if edge > self.params["min_cluster_edge"]:
                self.cluster_bias[c] = ("call", edge)
            elif edge < -self.params["min_cluster_edge"]:
                self.cluster_bias[c] = ("put", -edge)
        return self

    def signals(self, df):
        if self.kmeans is None:
            raise RuntimeError("chame fit() antes de signals()")
        shapes = self._shapes(df)
        valid = shapes.notna().all(axis=1)
        action = np.full(len(df), None, dtype=object)
        confidence = np.zeros(len(df))

        if valid.any():
            X = shapes.loc[valid].to_numpy()
            cluster_ids = self.kmeans.predict(X)
            positions = np.where(valid.to_numpy())[0]
            for pos, cluster_id in zip(positions, cluster_ids):
                bias = self.cluster_bias.get(cluster_id)
                if bias:
                    action[pos], edge = bias
                    confidence[pos] = min(1.0, edge * 4)

        return make_signals_df(action, confidence, df.index)


DEFAULT_STRATEGY_FACTORIES = [
    lambda: SmaCrossover(),
    lambda: SmaCrossover(threshold=0.001),
    lambda: RsiReversion(),
    lambda: RsiReversion(low=25, high=75),
    lambda: MomentumFollow(),
    lambda: BollingerReversion(),
    lambda: MeanReversionZScore(),
    lambda: MeanReversionZScore(window=40, z_threshold=1.5),
    lambda: AutocorrelationLag(lag_sign=-1),
    lambda: AutocorrelationLag(lag_sign=1),
    lambda: HurstRegimeSwitch(),
    lambda: PatternCluster(),
    lambda: PatternCluster(window=15, n_clusters=16),
    lambda: MLStrategy(),
    lambda: MLStrategy(edge=0.1),
]

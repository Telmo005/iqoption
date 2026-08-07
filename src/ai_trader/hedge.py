"""Hedge simples para opcoes digitais de expiracao curta.

Nao existe 'delta continuo' pra hedgear como em opcoes vanilla - aqui e uma
opcao binaria de tudo-ou-nada. O hedge possivel e por reducao de variancia:
quando o sinal principal tem confianca baixa (perto do limiar de decisao),
abrir tambem uma posicao menor no lado oposto reduz o tamanho da perda no
pior caso, ao custo de reduzir o ganho esperado no melhor caso. Isso NAO
elimina risco, so troca variancia por expectativa - use com essa expectativa
realista.
"""
from . import config


def decide_hedge(action, confidence, primary_amount):
    """Retorna (hedge_action, hedge_amount) ou None se nao for o caso de hedgear."""
    if not config.HEDGE_ENABLED or action is None:
        return None
    if confidence >= config.HEDGE_CONFIDENCE_THRESHOLD:
        return None

    hedge_action = "put" if action == "call" else "call"
    hedge_amount = round(primary_amount * config.HEDGE_SIZE_RATIO, 2)
    if hedge_amount < config.MIN_TRADE_AMOUNT:
        # a IQ Option rejeita operacoes abaixo do valor minimo permitido;
        # nao faz sentido forcar o hedge pro minimo (mudaria a proporcao
        # pretendida), entao so pula o hedge nesse caso.
        return None
    return hedge_action, hedge_amount

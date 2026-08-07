import os
import sys

from iqoptionapi.stable_api import IQ_Option

EMAIL = os.getenv("IQ_EMAIL")
PASSWORD = os.getenv("IQ_PASSWORD")

if not EMAIL or not PASSWORD:
    sys.exit("Defina as variaveis de ambiente IQ_EMAIL e IQ_PASSWORD antes de rodar.")

Iq = IQ_Option(EMAIL, PASSWORD)
check, reason = Iq.connect()
if not check:
    sys.exit(f"Falha ao conectar: {reason}")

Iq.change_balance("PRACTICE")  # troque para "REAL" quando quiser operar com dinheiro real

ACTIVE = "EURUSD-OTC"  # pares nao-OTC estao com o mapeamento de ativos quebrado na lib hoje
DURATION = 1  # minutos: 1 ou 5
AMOUNT = 1
ACTION = "call"  # ou "put"

status, order_id = Iq.buy_digital_spot_v2(ACTIVE, AMOUNT, ACTION, DURATION)
if not status:
    sys.exit(f"Falha ao abrir operacao: {order_id}")

print(f"Operacao aberta com sucesso. ID: {order_id}")

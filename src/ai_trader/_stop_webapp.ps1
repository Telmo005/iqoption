# Fecha qualquer instancia anterior do PAINEL WEB (ai_trader.webapp) antes de
# abrir uma nova - chamado pelo "Iniciar AI Trader.bat" do Desktop. So mexe
# em processos cuja linha de comando contem "ai_trader.webapp" - nunca toca
# Sincronizador/Gerador/Executor, que tem os proprios botoes Parar/Iniciar
# na tela e nao devem ser afetados por isto.
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like '*ai_trader.webapp*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

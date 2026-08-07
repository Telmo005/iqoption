@echo off
setlocal enabledelayedexpansion
rem Launcher portatil: usa a propria localizacao do arquivo (%~dp0), entao
rem funciona em qualquer maquina/pasta, sem caminho fixo. Liga o painel web
rem escondido (pythonw, sem janela/log na tela) e, por baixo dos panos,
rem pede pra ele mesmo ligar Sincronizador/Gerador/Executor/Aprendiz via API
rem - pensado pra maquinas onde ninguem vai clicar nos botoes da tela toda
rem vez (ex: uma VPS reiniciando sozinha). Seguro rodar de novo com tudo ja
rem ligado: cada "iniciar" so cria processo novo se o anterior nao estiver
rem mais rodando de verdade.

rem Fecha uma pagina web anterior (se houver) antes de abrir uma nova.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_stop_webapp.ps1"

cd /d "%~dp0.."
start "" pythonw -m ai_trader.webapp
ping -n 5 127.0.0.1 >nul

rem Le a senha do painel direto do .env (IQ_DASHBOARD_PASSWORD) - configura
rem uma vez so, la, sem precisar editar este arquivo tambem.
set DASHBOARD_PASSWORD=
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if "%%A"=="IQ_DASHBOARD_PASSWORD" set DASHBOARD_PASSWORD=%%B
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_start_all.ps1" -Password "!DASHBOARD_PASSWORD!"

start "" "http://127.0.0.1:5000"

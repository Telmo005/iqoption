# Liga Sincronizador, Gerador, Executor e Aprendiz atraves da MESMA API que
# os botoes "Iniciar" da tela usam (/api/start/<componente>) - ja tem
# protecao contra duplicar processo embutida (confere o PID de verdade, nao
# so um heartbeat), entao rodar isto de novo com tudo ja ligado nao cria
# copias extras. Chamado pelo start_all.bat, DEPOIS do painel web subir.
#
# So existe pra maquinas "sem ninguem clicando nos botoes" (ex: uma VPS
# reiniciando sozinha) - no PC de uso manual, os botoes da tela continuam
# sendo o jeito normal de ligar cada componente.
param(
    [string]$Password = ""
)

$headers = @{}
if ($Password) {
    $pair = "admin:$Password"
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
    $b64 = [System.Convert]::ToBase64String($bytes)
    $headers["Authorization"] = "Basic $b64"
}

foreach ($comp in @("sync", "generator", "executor", "learner")) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/start/$comp" `
            -Method POST -Headers $headers -UseBasicParsing -TimeoutSec 15
        Write-Output "[ok] $comp -> $($resp.StatusCode)"
    } catch {
        # 409 = "ja esta rodando", nao e falha de verdade (e o proprio
        # mecanismo anti-duplicata funcionando) - so erros diferentes disso
        # sao problema de verdade.
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 409) {
            Write-Output "[ja rodando] $comp"
        } else {
            Write-Output "[erro ao iniciar $comp] $($_.Exception.Message)"
        }
    }
}

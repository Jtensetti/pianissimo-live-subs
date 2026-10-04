# Pianissimo Live Subs

En grön textyta för OBS. Starta från cmd, prata i mikrofonen och få löpande svenska undertexter med Klang Pianissimo. All taligenkänning sker lokalt.

## Starta på Windows

Installera **Python 3.11 med Python Launcher** en gång. Öppna cmd i projektmappen och kör:

```bat
start.cmd
```

Första starten installerar Python-paketen i `.venv` och hämtar modellen (cirka 660 MB). Senare starter återanvänder filerna. När modellen är klar öppnas den gröna textsidan i din vanliga webbläsare. Programmet använder standardmikrofonen. **Ctrl+C stoppar.** Tryck en gång till för direkt avbrott om en modellkörning eller nedladdning fastnat. Ingen Node.js, kontrollpanel eller separat webbbyggnad behövs.

## OBS

1. Lägg till en **Webbläsare / Browser** som källa.
2. URL: **http://localhost:8000**. Bredd **1920**, höjd **1080**.
3. Öppna källans **Filter**, lägg till **Chroma Key** och välj **grön** nyckelfärg.

Det är en webbläsarkälla med ett OBS-filter, inte ett separat OBS-plugin. Sidan visar bara en grön bakgrund och text med svart kontur längst ned. OBS-sidan behöver inte mikrofonåtkomst; cmd-programmet läser mikrofonen. Låt cmd-fönstret vara öppet under sändningen.

Källan måste läggas till i OBS en gång; att sidan öppnas i webbläsaren gör inte att den automatiskt visas i OBS. När källan är på plats kan du starta med `start.cmd --no-browser` för att bara visa texten i OBS.

## Välj en annan mikrofon

```bat
start.cmd --list-devices
start.cmd --device 1
```

Byt `1` mot enhetsnumret i listan. För dator- eller mötesljud väljer du en ingång från ljudkortets loopback eller en virtuell ljudkabel. OBS-ljud skickas inte automatiskt till programmet.

Om svagt tal inte kommer med: prova `start.cmd --sensitivity 5`. Standard är `4`; ett lägre tal filtrerar bort mer svagt ljud. Det är en ljudnivågrind, så hög känslighet kan också släppa in bakgrundsljud. När en fras väl börjat sänks gränsen för att behålla svagare ordslut.

`start.cmd --diagnostics` visar mikrofonens ljudnivå var femte sekund och modellens kö-/beräkningstid vid färdig text. Diagnostiken visas bara i cmd; inget ljud eller text sparas och inget av detta visas i OBS.

## Modell och kvalitet

Pianissimo har cirka 600 miljoner parametrar. Följande är olika precisionsformat av samma modell, inte separata small/medium/large-modeller:

| Startalternativ | Modellfiler | Användning |
| --- | --- | --- |
| `--precision int8` (standard) | cirka 660 MB | Snabbast på CPU enligt Klang, liten kvalitetsförlust jämfört med originalet. |
| `--precision fp16` | cirka 1,28 GB | Högre precision; något bättre resultat i Klangs tester men långsammare på CPU. |
| `--precision fp32` | cirka 2,56 GB | Full precision; ingen tydlig kvalitetsvinst över fp16 i Klangs publicerade ONNX-tester. |

Exempel: `start.cmd --precision fp16`. Första starten med ett nytt format hämtar dess modellfiler. Inga ytterligare Python-paket krävs. [Klangs modelljämförelse](https://huggingface.co/KlangAI/pianissimo-sv) gäller deras testmaterial och datorer; den garanterar inte bättre resultat för en viss mikrofon eller dialekt.

## Ingen lång kö av gammalt ljud

- Första preliminära texten planeras efter cirka 1,2 sekunders ljud och uppdateras sedan ungefär var 0,6 sekund. Den kan rättas när mer sammanhang kommer in. OBS visar de senaste två raderna direkt, med föregående fras som kort kontext.
- Mikrofonkön är begränsad. Överflöd eller ljud som blivit mer än en sekund gammalt stoppar textningen med ett fel.
- Ett modelljobb får högst **3 sekunder för kötid och beräkning tillsammans**. Om gränsen passeras töms OBS-texten, textningen stoppas och sena resultat kastas. Gränsen räknas från färdigt ljudfönster, inte från mikrofonens första sampel.
- En pågående ONNX-beräkning kan inte avbrytas säkert mitt i en Python-tråd. Den får avslutas utan att visa sitt resultat. Vid en fastnad beräkning: stäng programmet och starta igen.

Modellen använder CPU, fyra trådar som standard. Grafikkortet kan användas av OBS. I ett lokalt test på i9-11950H tog ett fyrasekunders ljudfönster cirka **0,34 sekunder** att beräkna. Detta är inte hela fördröjningen till OBS; mikrofon och samtidig OBS-sändning behöver provas i din miljö.

Långa talpass behandlas i ljudfönster på högst åtta sekunder med 1,2 sekunders överlapp. Vid skarvar jämförs ord och modellens tidsstämplar; nästa fönster kan rätta ett tidigare avhugget ord. Samma textfras fortsätter över skarven. Om ingen säker match finns behålls orden, vilket kan ge en upprepning i stället för att ord raderas. Modellens tidsstämplar är ungefärliga och fel kan fortfarande uppstå.

Hela den mottagna frassvansen skickas till modellen, även svaga slutljud. Kort fortsättning efter en fönstergräns behålls också. Talgrinden kan fortfarande missa en helt svag fras, och musik eller brus kan trigga den. Bedöm resultatet med din egen mikrofon och riktiga talpass.

## Offline

Modellfilerna sparas i `models/`. Efter första lyckade modelladdningen:

```bat
set HF_HUB_OFFLINE=1
start.cmd
```

Ljud skickas inte till någon molntjänst och sparas inte till fil. Servern lyssnar endast på den egna datorn.

## För utvecklare

`run.py` startar mikrofon och lokal server i en process. `backend/overlay.html` är en vanlig HTML-fil. ONNX Runtime/onnx-asr kör modellen; sounddevice läser mikrofonen; soxr gör löpande samplingsomvandling. FastAPI/Uvicorn/WebSocket sköter den lokala textutgången. Beroendena finns i `backend/requirements.txt`.

Tester: installera `backend/requirements-dev.txt` i `.venv` och kör `python -m pytest backend/tests -q`. Visningens separata regressionstester kan köras med `node --test backend/tests/test_overlay.mjs`; Node behövs inte för att använda appen. `scripts/benchmark.py --help` visar hur modellens beräkningstid mäts på en WAV-fil, även med `--precision fp16`. `run.py --no-microphone` används för integrationstest utan mikrofoninspelning och öppnar ingen webbläsare.

Modell: [Klang Pianissimo ONNX int8](https://huggingface.co/KlangAI/pianissimo-sv-onnx), Klang AI AB, CC BY 4.0. Bygger vidare på [mickekring/Live-Local-Subtitles](https://github.com/mickekring/Live-Local-Subtitles). Se [NOTICE.md](NOTICE.md) för ursprung och modelllicens.

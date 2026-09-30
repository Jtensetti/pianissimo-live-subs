# Ursprung och modell

Pianissimo Live Subs bygger vidare på [Live-Local-Subtitles av mickekring](https://github.com/mickekring/Live-Local-Subtitles), utgångscommit `42e9e39`.

Denna version byter transkriptionsmotor till Klang Pianissimo och anpassar ljudbuffring, kontrollpanel och OBS-utgång. Originalets README innehåller ingen fastställd kodlicens; detta dokument tilldelar därför ingen ny licens till originalkoden.

Talmodellen [Klang Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) är utvecklad av Klang AI AB och publicerad under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Programmet använder Klangs officiella [ONNX int8-export](https://huggingface.co/KlangAI/pianissimo-sv-onnx) via [onnx-asr](https://github.com/istupakov/onnx-asr). Modellvikterna har inte ändrats i detta projekt och ingår inte i källkodspaketet. De hämtas separat vid första modelladdningen.

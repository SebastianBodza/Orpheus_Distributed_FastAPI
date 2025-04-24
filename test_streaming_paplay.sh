

curl -X POST "http://localhost:8001/generate-audio-stream/" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Martin: Dies ist ein längerer Text, um die Streaming-Fähigkeiten des Systems zu testen. Wir möchten sehen, wie die Audioausgabe generiert wird, während der Text noch verarbeitet wird. Die Idee ist, dass der Benutzer nicht warten muss, bis die gesamte Audiodatei erstellt ist, sondern fast sofort mit dem Hören beginnen kann. Das ist besonders nützlich für längere Abschnitte oder bei interaktiven Anwendungen. Mal sehen, ob die Latenz gering genug ist und die Wiedergabe flüssig bleibt, auch wenn der Server im Hintergrund noch rechnet und weitere Audiosegmente an den Client sendet. Hoffentlich funktioniert die Verkettung der einzelnen Chunks ohne hörbare Unterbrechungen oder Störungen. Das wäre ein gutes Zeichen für die Robustheit der Implementierung. Wir sind gespannt auf die Ergebnisse und hoffen, dass alles reibungslos abläuft. Das wäre ein großer Fortschritt in der Audioverarbeitung und -wiedergabe.",
           "voice": "in_prompt"
         }' \
     --no-buffer \
     | paplay --raw --format=s16le --rate=24000 --channels=1

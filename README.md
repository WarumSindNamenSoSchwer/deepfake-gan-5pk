# GANs & Deepfakes — 5. Prüfungskomponente (Abitur)

Ein selbst trainiertes Generative Adversarial Network (GAN), entstanden als meine
**5. Prüfungskomponente** im Abitur — fächerverbindend zwischen **Informatik**
(die Technik hinter GANs) und **Politik** (Deepfakes als Gefahr für demokratische
Prozesse).

> **Leitfrage:** Wie ermöglichen neuronale Netzwerke, insbesondere GANs, die
> Erstellung von Deepfakes — und welche Bedrohung stellen sie für demokratische
> Prozesse dar?

## Kurzfassung

Ich habe ein DCGAN von Grund auf in TensorFlow/Keras gebaut und auf einer kleinen
Menge Gesichtsbilder trainiert, um die Technik hinter Deepfakes praktisch zu
verstehen. Der schriftliche/mündliche Teil analysiert die politische Dimension:
Desinformation, Wahlmanipulation und schwindendes Vertrauen in Institutionen.

## Methode (Informatik-Teil)

- **Architektur:** DCGAN — `Conv2DTranspose`-Generator (512 → 256 → 128 → 64),
  `BatchNormalization` + `LeakyReLU`, faltender Diskriminator.
- **Eingabe:** 128×128 Bilder, Graustufen (einige Experimente in Farbe).
- **Latentraum:** Rauschvektor der Dimension 100 (eine Variante 120).
- **Framework:** TensorFlow / Keras, lokal trainiert auf macOS.
- **Training:** bis zu 1000 Epochen; mehrere Script-Varianten für Checkpoints,
  Fixed-Noise-Tracking, höhere Auflösung und Effizienz (siehe `src/`).

### Ehrliches Ergebnis — und warum es zählt

Der Datensatz war winzig und die Batch-Größe entsprach dem ganzen Datensatz
(`BATCH_SIZE = len(images)`). Über viele Epochen trainiert, hat das Modell seine
Eingaben **auswendig gelernt / overfitted**, statt neue Gesichter zu erzeugen —
die Outputs sind im Grunde degradierte Kopien der Trainingsbilder, keine
überzeugenden Fälschungen.

Dieses negative Ergebnis ist für das Thema ein ehrlicher, nützlicher Befund: ein
*überzeugender* Deepfake ist schwer, daten- und rechenhungrig. Die Gefahr für die
Demokratie geht weniger von einem Hobby-GAN aus als von gut ausgestatteten
Akteuren — genau das entwickelt der Politik-Teil.

## Politik-Teil — Deepfakes & Demokratie

- Deepfakes können glaubhafte Aufnahmen öffentlicher Personen fälschen → direkte
  Gefahr für die Glaubwürdigkeit politischer Akteure und die Integrität von Wahlen.
- Fallbeispiele: Biden/Trump (US 2020), Nancy Pelosi (2018), Rumin Farhana
  (Bangladesch 2023) sowie ein hypothetisches Szenario ausländischer
  Einflussnahme gegen Deutschland.
- Kernprobleme: Erkennung hinkt der Erzeugung hinterher; Rechtsrahmen unreif.
- Gegenmaßnahmen: Erkennungsforschung, Medienkompetenz, Regulierung
  (EU AI Act, DSGVO Art. 22, Persönlichkeitsrecht).

## Aufbau des Repos

```
src/                GAN-Trainingsscripts und Experimente (TensorFlow/Keras)
Bilder/             Trainings-Eingaben
Renders/            Trainingsfortschritt (50 → 1000 Epochen)
Generierte_Bilder/  generierte Outputs pro Lauf
```

Die Präsentation (`GANs und Deepfakes.pptx`) und das Exposé liegen bei.

## Ethik-Statement

Dieses Repo dokumentiert eine akademische Auseinandersetzung mit GANs und ihren
politischen Folgen, geschrieben für mein Abitur-Capstone. Das Modell erzeugt
**keine** überzeugenden synthetischen Medien — es overfittet und reproduziert
seine Eingaben. Der Code dient ausschließlich Bildungszwecken. Wer diese Arbeit
nachbaut, sollte DSGVO/GDPR, das deutsche Persönlichkeitsrecht und die
EU-AI-Act-Regeln zu synthetischen Medien beachten und damit keine Bilder realer
Personen ohne deren Einwilligung erzeugen.

## Lizenz

Code: MIT (siehe `LICENSE`). Die schriftliche Arbeit und Bilder Dritter bleiben
unter ihren jeweiligen Rechten.

---

# English

A self-trained Generative Adversarial Network (GAN) built as my German Abitur
capstone (**5. Prüfungskomponente**), combining **Informatik** (the technical
side of GANs) and **Politik** (deepfakes as a threat to democratic processes).

> **Research question:** How do neural networks, in particular GANs, enable the
> creation of deepfakes — and what threat do they pose to democratic processes?

## TL;DR

I implemented a DCGAN from scratch in TensorFlow/Keras and trained it on a small
set of face images to understand, hands-on, how the technology behind deepfakes
actually works. The written/oral part analyses the political dimension:
disinformation, election manipulation, and erosion of trust in institutions.

## Method (Informatik part)

- **Architecture:** DCGAN — `Conv2DTranspose` generator (512 → 256 → 128 → 64),
  `BatchNormalization` + `LeakyReLU`, convolutional discriminator.
- **Input:** 128×128 images, grayscale (some experiments in color).
- **Latent space:** noise vector of dimension 100 (one variant 120).
- **Framework:** TensorFlow / Keras, trained locally on macOS.
- **Training:** up to 1000 epochs; several script variants explore checkpoints,
  fixed-noise tracking, higher resolution, and efficiency tweaks (see `src/`).

### Honest result — and why it matters

The dataset was tiny and the batch size equalled the full dataset
(`BATCH_SIZE = len(images)`). Trained for many epochs, the model **overfit /
memorised its inputs** instead of synthesising novel faces — outputs are
essentially degraded copies of the training images, not convincing fakes.

That negative result is, for the topic, an honest and useful finding: building a
*convincing* deepfake is hard, data-hungry, and compute-hungry. The threat to
democracy comes less from a hobbyist GAN and more from well-resourced actors —
which is exactly the argument the Politik part develops.

## Politik part — Deepfakes & Democracy

- Deepfakes can fabricate believable footage of public figures → direct threat to
  the credibility of political actors and the integrity of elections.
- Case studies: Biden/Trump (US 2020), Nancy Pelosi (2018), Rumin Farhana
  (Bangladesh 2023), plus a hypothetical foreign-influence scenario against Germany.
- Core problems: detection lags behind generation; legal frameworks are immature.
- Countermeasures: detection research, media literacy, regulation
  (EU AI Act, GDPR Art. 22, German personality rights).

## Ethics statement

This repository documents an academic exploration of GANs and their political
implications, written for my Abitur capstone. The model does **not** produce
convincing synthetic media — it overfits and reproduces its inputs. The code is
provided for educational reference only. Anyone reproducing this work should
respect GDPR, German Persönlichkeitsrecht, and the EU AI Act provisions on
synthetic media, and should not use it to fabricate imagery of real people
without consent.

## License

Code: MIT (see `LICENSE`). The written work and any third-party images remain
under their respective rights.

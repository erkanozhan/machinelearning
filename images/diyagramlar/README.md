# Diyagramlar

Ders notundaki akış diyagramları burada SVG görsel olarak durur. Mermaid diyagramları GitHub'da ayrı bir sunucuda çizildiği için bazen "Unable to render rich display" hatası veriyordu; görseller her yerde (GitHub, GitHub Pages, mobil uygulama) aynı görünür.

Her `.svg` dosyasının kaynağı, aynı adlı `.mmd` (Mermaid) dosyasıdır. Bir diyagramı değiştirmek için:

1. `.mmd` dosyasını düzenleyin.
2. [mermaid.live](https://mermaid.live) adresine yapıştırın. **Config** sekmesine şunu yazın (yazılar HTML yerine SVG metni olsun; GitHub `<img>` içindeki HTML etiketleri göstermez):
   ```json
   { "htmlLabels": false, "flowchart": { "htmlLabels": false }, "fontFamily": "Arial, Helvetica, sans-serif" }
   ```
3. **Actions → SVG** ile indirip aynı adla bu klasöre kaydedin.

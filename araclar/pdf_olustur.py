"""Ders notunun tüm bölümlerini tek bir PDF dosyasında birleştirir.

Kullanım (deponun kök klasöründen):
    python araclar/pdf_olustur.py                 # Ders_Notu.pdf üretir
    python araclar/pdf_olustur.py cikti.pdf       # farklı bir dosya adıyla

Gerekenler: Python 3 ve bilgisayarda Google Chrome ya da Microsoft Edge.
Formüller ve kod renklendirme internetten (CDN) yüklendiği için internet bağlantısı gerekir.

Nasıl çalışır?
  1. Depo klasörü geçici bir yerel web sunucusuyla yayınlanır.
  2. yazdir.html sayfası, _sidebar.md'deki sırayla tüm bölümleri tek sayfada birleştirir.
  3. Tarayıcı görünmez (headless) kipte açılır ve sayfayı PDF olarak kaydeder.
"""
import functools
import http.server
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading

DEPO = pathlib.Path(__file__).resolve().parent.parent
BEKLEME_MS = 90000  # sayfanın yüklenip formüllerin çizilmesi için tanınan süre


def tarayici_bul():
    adaylar = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for ad in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"):
        yol = shutil.which(ad)
        if yol:
            adaylar.insert(0, yol)
    for yol in adaylar:
        if pathlib.Path(yol).exists():
            return yol
    sys.exit("Chrome veya Edge bulunamadı. Alternatif: web sitesinde yazdir.html sayfasını açıp Ctrl+P ile 'PDF olarak kaydet' seçilebilir.")


class SessizIstek(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    # Türkçe karakterleri desteklemeyen konsollarda yazdırma hatası olmasın
    sys.stdout.reconfigure(errors="replace")
    cikti = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEPO / "Ders_Notu.pdf").resolve()
    tarayici = tarayici_bul()

    isleyici = functools.partial(SessizIstek, directory=str(DEPO))
    sunucu = http.server.ThreadingHTTPServer(("127.0.0.1", 0), isleyici)
    port = sunucu.server_address[1]
    threading.Thread(target=sunucu.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory() as profil:
        komut = [
            tarayici,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profil}",
            f"--virtual-time-budget={BEKLEME_MS}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={cikti}",
            f"http://127.0.0.1:{port}/yazdir.html",
        ]
        print("PDF hazırlanıyor (bir-iki dakika sürebilir)...")
        subprocess.run(komut, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    sunucu.shutdown()

    if not cikti.exists() or cikti.stat().st_size < 100_000:
        sys.exit("PDF üretilemedi ya da eksik görünüyor. İnternet bağlantısını kontrol edip tekrar deneyin.")
    print(f"Hazır: {cikti} ({cikti.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()

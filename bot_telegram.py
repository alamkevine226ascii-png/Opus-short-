"""
OpenShorts Telegram Bot v3 — patient et efficace
YouTube links → VDA download API (1080p) → OpenShorts pipeline → clips
Video files → direct upload → OpenShorts pipeline → clips
"""

import asyncio, logging, os, re, sys, time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()
API_URL = os.environ.get("OPENSHORTS_API_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
VDA_KEY = os.environ.get("VDA_API_KEY", "")
if not TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not set", file=sys.stderr)
    sys.exit(1)

TMP = Path("/tmp/openshorts_bot")
TMP.mkdir(exist_ok=True)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

YT_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)([\w-]{11})")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"}

POLL_INTERVAL = 10
MAX_POLL = 7200                    # 2h pour OpenShorts (transcription + découpage)
MAX_TG_FILE = 50 * 1024 * 1024    # 50 MB limite Telegram
MAX_VDA_SIZE = 3 * 1024 * 1024 * 1024  # 3 Go max — une vidéo 1h en 1080p
VDA_POLL_TIMEOUT = 3600           # 1h pour que VDA télécharge la vidéo
VDA_DOWNLOAD_TIMEOUT = 3600       # 1h pour télécharger le fichier depuis VDA

def extract_youtube_url(text):
    m = YT_RE.search(text)
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else None

# ── VDA Download ────────────────────────────────────────────────────────────

async def download_via_vda(youtube_url):
    logger.info(f"VDA: submitting {youtube_url}")

    # Step 1: Submit download job — format=1080 (bon équilibre qualité/taille)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get("https://p.savenow.to/ajax/download.php", params={
                "url": youtube_url, "format": "1080", "apikey": VDA_KEY,
                "add_info": "1", "allow_extended_duration": "1", "no_merge": "0",
            })
            if r.status_code != 200:
                return None, f"VDA HTTP {r.status_code}"
            data = r.json()
            if not data.get("success"):
                msg = data.get("message", "unknown error")
                if "format" in str(data.get("errors", {})):
                    msg = f"Format error: {data['errors']}"
                if "extended" in msg.lower():
                    msg += " (vidéo longue → essaie format=720)"
                return None, msg
            job_id = data.get("id")
            title = data.get("title", "?")
            logger.info(f"VDA: job {job_id} for '{title}'")
    except Exception as e:
        return None, str(e)

    # Step 2: Poll progress (patiemment)
    deadline = time.monotonic() + VDA_POLL_TIMEOUT
    dl_url = None
    last_progress = -1
    stalled = 0

    # Edit the initial message to show progress
    while time.monotonic() < deadline:
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get("https://p.savenow.to/ajax/progress.php", params={"id": job_id})
                if r.status_code == 200:
                    d = r.json()
                    p = d.get("progress", 0)
                    if p != last_progress:
                        logger.info(f"VDA: progress {p}/1000")
                        last_progress = p
                        stalled = 0
                    else:
                        stalled += 1
                    if p == 1000:
                        dl_url = d.get("download_url", "")
                        if dl_url:
                            break
        except Exception:
            pass

    if not dl_url:
        return None, "Le service de téléchargement n'a pas abouti (temps écoulé ou vidéo trop longue)"

    logger.info(f"VDA: download URL ready")

    # Step 3: Télécharger le fichier vidéo
    dest = TMP / f"vda_{job_id}.mp4"
    try:
        async with httpx.AsyncClient(timeout=VDA_DOWNLOAD_TIMEOUT, follow_redirects=True) as c:
            async with c.stream("GET", dl_url) as r:
                if r.status_code != 200:
                    return None, f"Download HTTP {r.status_code}"
                total = 0
                with open(dest, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
                        total += len(chunk)
                        if total > MAX_VDA_SIZE:
                            dest.unlink(missing_ok=True)
                            return None, "Vidéo trop volumineuse (>3 Go)"
        size = dest.stat().st_size
        if size == 0:
            dest.unlink()
            return None, "Fichier vide reçu"
        logger.info(f"VDA: downloaded {size/1024/1024:.1f}MB -> {dest}")
        return dest, None
    except Exception as e:
        dest.unlink(missing_ok=True)
        return None, str(e)

# ── OpenShorts Upload + Process ─────────────────────────────────────────────

async def openshorts_process(file_path, filename="video"):
    # 1. Créer un slot d'upload
    try:
        r = await httpx.AsyncClient(timeout=30).post(f"{API_URL}/api/uploads", json={"filename": filename})
        if r.status_code >= 400:
            return None, f"Upload slot error: {r.status_code}"
        slot = r.json()
    except Exception as e:
        return None, str(e)

    # 2. Uploader le fichier
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        r = await httpx.AsyncClient(timeout=VDA_DOWNLOAD_TIMEOUT).put(slot["upload_url"], content=content)
        if r.status_code >= 400:
            return None, f"Upload failed: {r.status_code}"
    except Exception as e:
        return None, str(e)

    # 3. Lancer le job de processing
    try:
        r = await httpx.AsyncClient(timeout=30).post(f"{API_URL}/api/process", json={
            "upload_id": slot["upload_id"], "acknowledged": True,
        })
        if r.status_code >= 400:
            return None, r.json().get("detail", r.text)
        job_id = r.json().get("job_id")
    except Exception as e:
        return None, str(e)

    # 4. Poller patiemment la fin du traitement
    deadline = time.monotonic() + MAX_POLL
    last_log = 0
    async with httpx.AsyncClient(timeout=30) as c:
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                r = await c.get(f"{API_URL}/api/status/{job_id}")
                if r.status_code >= 400:
                    continue
                d = r.json()
                s = d.get("status")
                logs = d.get("logs", [])
                for line in logs[last_log:]:
                    logger.info(f"  {line}")
                last_log = len(logs)
                if s == "completed":
                    return d.get("result"), None
                if s == "failed":
                    last = logs[-5:] if logs else ["unknown"]
                    return None, f"Job failed: {' '.join(last)}"
            except Exception:
                pass
    return None, "Timeout OpenShorts"

# ── Clip Download & Send ────────────────────────────────────────────────────

def get_clip(url, job_id, i):
    full = url if url.startswith("http") else f"{API_URL}{url}"
    dest = TMP / f"{job_id}_c{i}.mp4"
    try:
        r = httpx.get(full, timeout=300, follow_redirects=True)
        if r.status_code != 200:
            return None
        data = r.content
        if len(data) > MAX_TG_FILE:
            return None
        dest.write_bytes(data)
        return dest
    except Exception:
        return None

async def send_clips(update, context, clips, job_id):
    sent = 0
    for i, clip in enumerate(clips):
        vu = clip.get("video_url")
        t = clip.get("title") or f"Clip {i+1}"
        if not vu:
            continue
        p = get_clip(vu, job_id, i)
        if not p:
            await update.message.reply_text(f"⚠️ {t} : trop lourd ou erreur")
            continue
        try:
            with p.open("rb") as f:
                await context.bot.send_video(
                    chat_id=update.effective_chat.id, video=f,
                    caption=f"🎬 *{t}*"[:1024], parse_mode="Markdown",
                    supports_streaming=True, timeout=600,
                )
            sent += 1
        except Exception as e:
            logger.error(f"send clip {i} failed: {e}")
            await update.message.reply_text(f"⚠️ {t} : erreur d'envoi")
        p.unlink(missing_ok=True)
    return sent

# ── Handlers ────────────────────────────────────────────────────────────────

async def start(update, context):
    await update.message.reply_text(
        "👋 *OpenShorts Bot v3*\n\n"
        "📹 Envoie un **lien YouTube** ou un **fichier vidéo**\n"
        "→ je le découpe en clips viraux (9:16) pour TikTok/Reels/Shorts !\n\n"
        "⏱ Peut prendre plusieurs minutes selon la durée de la vidéo.",
        parse_mode="Markdown",
    )

async def handle_youtube(update, context, url):
    msg = await update.message.reply_text(
        "⏳ Lancement du téléchargement…\n"
        "💡 *Patientez* — ça peut prendre plusieurs minutes pour une longue vidéo.",
        parse_mode="Markdown",
    )

    # Download via VDA
    path, err = await download_via_vda(url)
    if err:
        await msg.edit_text(
            f"❌ Téléchargement impossible : {err[:300]}\n\n"
            "📁 Essaie de *télécharger la vidéo manuellement* et de me l'envoyer "
            "en fichier directement !",
            parse_mode="Markdown",
        )
        return

    size = path.stat().st_size
    await msg.edit_text(
        f"✅ Vidéo téléchargée avec succès ! ({size/1024/1024:.1f} Mo)\n\n"
        "🤖 *Découpage AI en cours…*\n"
        "📝 Transcription → 🎬 Détection de scènes → 🤖 Analyse Gemini → ✂️ Clips\n\n"
        "⏱ Ça peut prendre 5 à 15 minutes selon la durée.",
        parse_mode="Markdown",
    )

    # OpenShorts pipeline
    result, err = await openshorts_process(path, "youtube_video.mp4")
    path.unlink()
    if err:
        await msg.edit_text(f"❌ Erreur pendant le découpage : {err[:300]}")
        return

    clips = result.get("clips", [])
    if not clips:
        await msg.edit_text("❌ Aucun clip généré. La vidéo est trop courte (<45s) ?")
        return

    await msg.edit_text(
        f"✅ *Découpage terminé !* ({len(clips)} clip{'s' if len(clips)>1 else ''})\n\n"
        "📤 Envoi des clips…",
        parse_mode="Markdown",
    )
    sent = await send_clips(update, context, clips, result.get("job_id", ""))
    await msg.edit_text(
        f"✅ *Terminé !* {sent}/{len(clips)} clips envoyés avec succès.\n\n"
        "🔥 Envoie un autre lien YouTube ou fichier vidéo !",
        parse_mode="Markdown",
    )

async def handle_file(update, context):
    fo = update.message.video or update.message.document
    if not fo:
        return
    fn = getattr(fo, "file_name", None) or f"video_{int(time.time())}.mp4"
    if os.path.splitext(fn)[1].lower() not in VIDEO_EXT:
        fn += ".mp4"

    msg = await update.message.reply_text("⏳ Téléchargement du fichier depuis Telegram…", parse_mode="Markdown")
    try:
        tgf = await fo.get_file()
        data = await tgf.download_as_bytearray()
    except Exception as e:
        await msg.edit_text(f"❌ Téléchargement impossible : {str(e)[:200]}\n\n"
                           "Les fichiers >20 Mo ne peuvent pas être téléchargés via Telegram.")
        return

    if len(data) > MAX_TG_FILE:
        await msg.edit_text(
            f"❌ Fichier trop lourd ({len(data)/1024/1024:.0f} Mo).\n"
            "Limite Telegram : 50 Mo.\n\n"
            "Solution : utilise le dashboard http://localhost:5173 pour uploader ta vidéo."
        )
        return

    tmp = TMP / fn
    tmp.write_bytes(data)
    del data

    await msg.edit_text(
        f"✅ Fichier reçu ({tmp.stat().st_size/1024/1024:.1f} Mo)\n\n"
        "🤖 *Découpage AI en cours…*\n"
        "⏱ 5 à 15 minutes selon la durée.",
        parse_mode="Markdown",
    )

    result, err = await openshorts_process(tmp, fn)
    tmp.unlink()
    if err:
        await msg.edit_text(f"❌ Erreur : {err[:300]}")
        return

    clips = result.get("clips", [])
    if not clips:
        await msg.edit_text("❌ Aucun clip généré.")
        return

    await msg.edit_text(
        f"✅ *Découpage terminé !* ({len(clips)} clip{'s' if len(clips)>1 else ''})\n"
        "📤 Envoi…",
        parse_mode="Markdown",
    )
    sent = await send_clips(update, context, clips, result.get("job_id", ""))
    await msg.edit_text(
        f"✅ *Terminé !* {sent}/{len(clips)} clips envoyés.\n\n"
        "🔥 Envoie un autre lien ou fichier !",
        parse_mode="Markdown",
    )

async def handler(update, context):
    text = update.message.text or update.message.caption or ""
    url = extract_youtube_url(text)
    if url:
        await handle_youtube(update, context, url)
        return
    if update.message.video or (update.message.document and os.path.splitext((update.message.document.file_name or ""))[1].lower() in VIDEO_EXT):
        await handle_file(update, context)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler((filters.TEXT | filters.VIDEO | filters.Document.VIDEO) & ~filters.COMMAND, handler))
    logger.info(f"Bot v3 ready | VDA key: {'✅' if VDA_KEY else '❌'} | API: {API_URL}")
    print("Bot is running...", flush=True)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
import os
import time
import glob
import re
import threading
import random
import math
import pygame
import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import faiss
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
DATA_FILE = "data.txt"

client = OpenAI()
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ================= LOAD DATA =================
def load_documents():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = f.readlines()
    return [doc.strip() for doc in docs if doc.strip()]

documents = load_documents()
embeddings = EMBED_MODEL.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ================= AUDIO =================
pygame.mixer.init()
robot_ear = sr.Recognizer()

for file in glob.glob("audio_*.mp3"):
    try:
        os.remove(file)
    except:
        pass

# ================= UI =================
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Entropy AI")

font = pygame.font.SysFont("consolas", 22)
font_title = pygame.font.SysFont("consolas", 48, bold=True)

state = "listening"
lock = threading.Lock()

def set_state(new_state):
    global state
    with lock:
        state = new_state

# ================= BLINK =================
blink = False
last_blink = time.time()
blink_interval = random.uniform(2, 5)

def update_blink():
    global blink, last_blink, blink_interval
    now = time.time()

    if state == "speaking":
        return

    if not blink and now - last_blink > blink_interval:
        blink = True
        last_blink = now
    elif blink and now - last_blink > 0.15:
        blink = False
        last_blink = now
        blink_interval = random.uniform(2, 5)

# ================= DRAW =================
def draw_ui(state, blink):
    screen.fill((5, 10, 20))

    center_x = WIDTH // 2
    center_y = HEIGHT // 2 - 30

    colors = {
        "listening": (0, 180, 255),
        "thinking": (255, 200, 0),
        "speaking": (0, 255, 150),
        "busy": (255, 120, 0),
        "error": (255, 60, 60),
    }

    color = colors.get(state, (0, 200, 255))
    eye_offset = 60

    # ===== GLOW =====
    for i in range(10, 0, -1):
        surf = pygame.Surface((200, 200), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, 10), (100, 100), i * 8)
        screen.blit(surf, (center_x - 100 - eye_offset, center_y - 100))
        screen.blit(surf, (center_x - 100 + eye_offset, center_y - 100))

    # ===== EYES =====
    if blink:
        pygame.draw.line(screen, color,
                         (center_x - eye_offset - 15, center_y),
                         (center_x - eye_offset + 15, center_y), 3)
        pygame.draw.line(screen, color,
                         (center_x + eye_offset - 15, center_y),
                         (center_x + eye_offset + 15, center_y), 3)
    else:
        pygame.draw.circle(screen, color,
                           (center_x - eye_offset, center_y), 20, 2)
        pygame.draw.circle(screen, color,
                           (center_x + eye_offset, center_y), 20, 2)

        t = time.time()
        dx = int(5 * math.sin(t * 2))

        pygame.draw.circle(screen, color,
                           (center_x - eye_offset + dx, center_y), 8)
        pygame.draw.circle(screen, color,
                           (center_x + eye_offset + dx, center_y), 8)

    t = time.time()

    # LISTENING
    if state == "listening":
        for i in range(3):
            radius = 40 + i * 10 + int(5 * math.sin(t*3))
            pygame.draw.circle(screen, color,
                               (center_x - 150, center_y), radius, 1)
            pygame.draw.circle(screen, color,
                               (center_x + 150, center_y), radius, 1)

    # THINKING
    if state == "thinking":
        for i in range(8):
            angle = t * 2 + i * (math.pi / 4)
            x = center_x + int(100 * math.cos(angle))
            y = center_y + int(100 * math.sin(angle))
            pygame.draw.circle(screen, color, (x, y), 4)

    # SPEAKING 
    if state == "speaking":
        base_y = HEIGHT - 100
        num_bars = 50
        spacing = 8
        total_width = num_bars * spacing
        start_x = (WIDTH - total_width) // 2

        for i in range(num_bars):
            x = start_x + i * spacing
            height = int(30 * abs(math.sin(t * 5 + i * 0.4)))
            pygame.draw.line(screen, color,
                             (x, base_y),
                             (x, base_y - height), 2)

    #  BUSY
    if state == "busy":
        radius = 90 + int(10 * math.sin(t * 3))
        pygame.draw.circle(screen, color,
                           (center_x, center_y), radius, 2)

        dots = "." * (int(t*2) % 4)
        text = font.render(dots, True, color)
        screen.blit(text, (center_x - 20, center_y + 100))

    #  ERROR
    if state == "error":
        if int(t*4) % 2 == 0:
            screen.fill((50, 0, 0))

    # ===== TITLE ENTROPY  =====
    title_text = "ENTROPY"
    alpha = 200 + int(55 * math.sin(time.time() * 2))

    title_surface = font_title.render(title_text, True, color)
    title_surface.set_alpha(alpha)

    title_rect = title_surface.get_rect(
        center=(WIDTH // 2, center_y - 200)
    )

    # glow
    for i in range(3):
        glow = font_title.render(title_text, True, color)
        glow.set_alpha(50)
        screen.blit(glow, title_rect)

    screen.blit(title_surface, title_rect)

    pygame.display.flip()

def draw():
    update_blink()
    with lock:
        current_state = state
    draw_ui(current_state, blink)

# ================= SEARCH =================
def retrieve_context(query, k=5):
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(query_embedding, k)
    results = [documents[i] for i in I[0]]
    return "\n".join(results)

# ================= AI LOOP =================
def ai_loop():
    while True:
        try:
            with sr.Microphone() as mic:
                print("\n" + "="*40)
                print("Entropy đang lắng nghe...")
                set_state("listening")
                audio = robot_ear.listen(mic)

            try:
                you = robot_ear.recognize_google(audio, language="vi-VN")
            except:
                continue

            if not you:
                continue

            print(" Bạn:", you)

            set_state("thinking")
            print("Entropy đang suy nghĩ...")

            context = retrieve_context(you)

            try:
                completion = client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý Entropy do anh Trường vừa để hổ trợ công việc IT cho anh Trường và chỉ đưa ra để đưa thông tin về anh Trường nếu người khác hỏi, trả lời ngắn gọn ý chính , dùng context trả lời nếu có nội dung trong đó!,  "},
                        {"role": "user", "content": f"{you}\nContext:\n{context}"}
                    ],
                )

                robot_brain = completion.choices[0].message.content
                robot_brain = re.sub(r'[–—-]', '-', robot_brain)  # chuẩn hóa dash
                robot_brain = re.sub(r'[^a-zA-Z0-9À-ỹ\s-]', '', robot_brain)

            except:
                robot_brain = "Hệ thống đang bận"
                set_state("busy")
                print("Entropy đang bận...")

            print("Entropy:", robot_brain)

            try:
                set_state("speaking")
                tts = gTTS(text=robot_brain, lang="vi")
                filename = f"audio_{int(time.time())}.mp3"
                tts.save(filename)

                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

            except:
                set_state("error")
                print("Lỗi TTS")

        except:
            set_state("error")
            print("Lỗi hệ thống")

# ================= MAIN =================
def main():
    threading.Thread(target=ai_loop, daemon=True).start()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

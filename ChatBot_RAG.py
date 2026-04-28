import os
import time
import glob
import re
import pygame
import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import faiss
from sentence_transformers import SentenceTransformer
#by Truong Viet Tran
client = OpenAI()
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DATA_FILE = "data.txt"
def load_documents():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = f.readlines()
    return [doc.strip() for doc in docs if doc.strip()]

documents = load_documents()
embeddings = EMBED_MODEL.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

pygame.mixer.init()
robot_ear = sr.Recognizer()

for file in glob.glob("audio_*.mp3"):
    try:
        os.remove(file)
    except:
        pass

def retrieve_context(query, k=5):
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(query_embedding, k)
    results = [documents[i] for i in I[0]]
    return "\n".join(results)

while True:
    with sr.Microphone() as mic:
        print("Entropy đang nghe...")
        audio = robot_ear.listen(mic)

    try:
        you = robot_ear.recognize_google(audio, language="vi-VN")
    except:
        you = ""

    if not you:
        continue

    print("Bạn:", you)

    context = retrieve_context(you)

    try:
        completion = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý Entropy do anh Trường vừa để hổ trợ công việc vừa đưa ra để đưa thông tin về anh Trường cho người khác hỏi, trả lời ngắn gọn đầy đủ ý chính , dùng context trả lời đầy đủ nếu có !  "
                },
                {
                    "role": "user",
                    "content": f"Câu hỏi: {you}\n\nContext:\n{context}"
                }
            ],
        )

        robot_brain = completion.choices[0].message.content
        robot_brain = re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', robot_brain)

    except Exception as e:
        print("Lỗi GPT:", e)
        robot_brain = "Hệ thống đang bận vui lòng thử lại sau !"

    print(" Entropy:", robot_brain)

    try:
        tts = gTTS(text=robot_brain, lang="vi")
        filename = f"audio_{int(time.time())}.mp3"
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        print(" Lỗi TTS:", e)
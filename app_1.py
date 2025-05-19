import streamlit as st
import sounddevice as sd
from pydub import AudioSegment
import numpy as np
import time
import wave
import whisper
import torch
import io
import tempfile
from openai import OpenAI
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

st.title("Cine Shazam")
st.subheader("Based on the Scene, We’ll Find Your Movie 🎥")

# Load OpenAI key once
with open('keys/openai_key.txt') as f:
    OPENAI_API_KEY = f.read().strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# Embedding setup (only once)
class SentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

embedding_function = SentenceTransformerEmbedding('all-MiniLM-L6-v2')
db = Chroma(
    collection_name="vector_database",
    embedding_function=embedding_function,
    persist_directory="./chroma_db_"
)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
PROMPT_TEMPLATE = """Answer the question based on the following context:
{context}

Based on the given subtitle/dialogues below:
{question}

Provide the following details strictly based on the context and 
show in below format

1. The title of the movie or web series, or TV show.
DONT SHOW THE BELOW LINE IF YOU DIDNT FIND IN THE CONTEXT VERY STRICTLY
2. The year of release. 
SHOW THE 3RD POINT IF ITS A WEBSERIES OR ELSE DONT PRINT THE BELOW LINE VERY STRICTLY
3. the season and episode number. (show this only if its webseries)

Do not include any ID (VERY STRICTLY)
Do not add any extra information.
Do not justify your answers.
DO NOT MENTION Not SPECIFIED
DO NOT MENTION NOT APPLICABLE
DO NOT MENTION NOT FOUND
DO NOT MENTION N/A

Avoid phrases like "according to the context/question" or "mentioned in the context/question."
Simply provide the requested details.
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt_template | chat_model | parser

def record_audio(duration=40, samplerate=44100, channels=1):
    st.success("Recording started...")
    progress_bar = st.progress(0)
    audio_data = []
    def callback(indata, frames, t, status):
        if status:
            st.error(status)
        audio_data.append(indata.copy())
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', callback=callback):
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress(int((i+1)/duration * 100))
    audio = np.concatenate(audio_data, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf

if st.button("🔴 Record"):
    total_start = time.perf_counter()

    # 1) Record
    t0 = time.perf_counter()
    wav_buf = record_audio()
    t1 = time.perf_counter()
    st.success(f"Recording complete! ({t1-t0:.2f}s)")

    # 2) Transcribe
    t2 = time.perf_counter()
    # write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_buf.getvalue())
        tmp_path = tmp.name
    transcription = client.audio.transcriptions.create(model="whisper-1",
                                                      file=open(tmp_path, "rb"))
    text = transcription.text
    t3 = time.perf_counter()
    st.subheader("Speech Recognition:")
    st.write(text)
    st.info(f"Transcription took {t3-t2:.2f}s")

    # 3) RAG search + generation
    t4 = time.perf_counter()
    result = rag_chain.invoke(text)
    t5 = time.perf_counter()
    st.subheader("Result:")
    st.write(result)
    st.info(f"RAG pipeline took {t5-t4:.2f}s")

    total_end = time.perf_counter()
    st.balloons()
    st.success(f"Total end‑to‑end time: {total_end-total_start:.2f}s")

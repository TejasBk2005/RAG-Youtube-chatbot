from youtube_transcript_api import YouTubeTranscriptApi
import re
import faiss
import numpy
from openai import OpenAI
from config import OPENROUTER_API_KEY, EMBEDDING_MODEL, LLM_MODEL

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def extract_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})"
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns :
        match = re.search(pattern , url)
        if match :
            return match.group(1)
    return None

def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        try:
            transcript_data = ytt_api.fetch(video_id, languages=["en"])
        except Exception:
            transcript_list = ytt_api.list(video_id)   
            transcript_data = next(iter(transcript_list)).fetch() 

        full_text = " ".join(item.text for item in transcript_data)
        return re.sub(r"\s+", " ",full_text)
    except Exception as e:
        print("Transcript Error:" ,e)
        return None

def splite_text(text, chunk_size=150):
    words=text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0 , len(words), chunk_size)
    ]

def create_embeddings(text_list):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text_list
    )
    embeddings = [
        item.embeddings for item in response.data
    ]
    return numpy.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return  index

def retreive_chunks(index, query_embedding, k=3):
    distances, indices = index.search(
        numpy.array([query_embedding]).astype(" float32"),k
    )
    return indices[0]

def ask_llm(context, question):
    if not context.strip():
        return "Sorry , I couldn't find relavant information in the video transcript"
    context = context[:6000]

    prompt = f""" Yor are an AI assistant answering question about a Youtube video.
    The transcript may be in any language (Kannada, Hindi , English ,etc).
    Always answer in english using the provided context.
    
    Transcript Context: {context}
    Question: {question}

    Answer clearly in English: """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system", "content":"You answer question about youtube videos."},
            {"role":"user" , "content": prompt}
        ]
    )
    return response.choices[0].message.content
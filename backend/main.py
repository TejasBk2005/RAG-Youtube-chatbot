from fastapi import FastAPI
from rag_utils import *
from rag_utils import extract_video_id
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)

chunks=[]
index= None

@app.post("/process_video")
def process_video(url : str):
    global chunks, index

    video_id = extract_video_id(url)
    if not video_id:
        return {"Error ": "Invalid Youtube URL"}
    text = get_transcript(video_id)
    if text is None:
        return {"Error":"Transcript not avilable for this video."}

    chunks = splite_text(text)
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    return {"message": "Video processed sucessfully"}

@app.post("/ask")
def ask(question: str):
    if index is None :
        return{"error":"Please process a video first"}
    try:
        query_embedding = create_embeddings([question])[0]
        top_chunks = retreive_chunks(index, query_embedding)
        if len(top_chunks) == 0 :
            return{"answer":"No relavant content in this video."}

        context_chunks = []
        for i in top_chunks:
            if i < len(chunks):
                context_chunks.append(chunks[1])
        context = " ".join(context_chunks)
        print("Retrieved chunks: ",top_chunks)
        print("Context preview:", context[:300])  

        answer = ask_llm(context , question)
        return{"answer" : answer}   
    except Exception as e :
        print("Ask error", e)
        return{"error": str(e)}

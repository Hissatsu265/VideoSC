import re
import os
import torch
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from langchain_community.llms import HuggingFacePipeline

VECTOR_STORE_PATH = "vector_store"

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    return None

def get_youtube_subtitles(video_url, language="en"):
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("Không thể trích xuất ID video từ URL.")

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    subtitles_text = " ".join([item['text'] for item in transcript])
    return subtitles_text

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = [chunk.strip() for chunk in text_splitter.split_text(text) if chunk.strip()]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'batch_size': 32}  
    )
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store

def generate_summary(text,choice="normal"):
    # model_name = "philschmid/bart-large-cnn-samsum"
    # tokenizer = BartTokenizer.from_pretrained(model_name)
    # model = BartForConditionalGeneration.from_pretrained(model_name)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)

    # inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    # summary_ids = model.generate(inputs["input_ids"], max_length=512, min_length=50, num_beams=4, early_stopping=True)
    
    # return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    model_name = "philschmid/bart-large-cnn-samsum" 
    
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    chunk_size = 1024
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    intermediate_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=512, 
            min_length=50, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        intermediate_summaries.append(summary)
    
    combined_text = " ".join(intermediate_summaries)
    if choice!="normal":
        inputs = tokenizer(combined_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], max_length=1024, min_length=50, num_beams=4, early_stopping=True)
        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return final_summary
    else:
        return combined_text

def setup_llm():
    model_name = "microsoft/phi-2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def process_video(url,choice):
    subtitles = get_youtube_subtitles(url)
    processed_text = preprocess_text(subtitles)
    
    vector_store = create_vector_store(processed_text)
    summary = generate_summary(processed_text,choice)
    
    return {"summary": summary, "message": "Video processed successfully"}
def create_chat_chain(vector_store, llm):
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=None,
        return_source_documents=False
    )
    return chain
def ask_question(question):
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'batch_size': 32}  
        ), 
        allow_dangerous_deserialization=True 
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    llm = setup_llm()
    chain = create_chat_chain(vector_store, llm)

    formatted_query = f"""
        Based on the YouTube video subtitles, please answer the following question:
        
        Question: {question}
        
        Only respond using the information found in the video subtitles. If the information is not mentioned, state that it is not available in the video.
    """
    
    response = chain({"question": formatted_query, "chat_history": []})
    return {"answer": response["answer"]}

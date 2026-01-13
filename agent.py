import base64
import os
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field

# --- LANGCHAIN IMPORTS ---
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- CONFIGURATION ---
TEXT_MODEL = "mistral"
VISION_MODEL = "llava" 
EMBEDDING_MODEL = "nomic-embed-text"

# --- HELPER: IMAGE ENCODER ---
def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return None

# 1. THE STATE
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    vision_analysis: str      
    generation: str          
    web_search_needed: bool
    sources: List[str]        # We ensure this persists
    retry_count: int          

# --- LLM SETUP ---
llm_text = ChatOllama(model=TEXT_MODEL, temperature=0)
llm_vision = ChatOllama(model=VISION_MODEL, temperature=0)

# 2. THE NODES

def retrieve(state: AgentState):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="local-os-rag", 
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
    
    try:
        # Boosted k to 6 to capture more potential images
        retriever = vectorstore.as_retriever(search_kwargs={'k': 6})
        docs = retriever.invoke(question)
    except:
        docs = []

    return {"documents": docs, "question": question, "retry_count": 0}

def grade_documents(state: AgentState):
    print("---GRADING RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    class Grade(BaseModel):
        score: str = Field(description="'yes' or 'no'")
    
    # We use a relaxed grader to let images pass through to Vision model
    grader = llm_text.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template(
        "Is this file likely to contain info about '{question}'? If it's an image file or relevant text, say yes. Doc: {doc}"
    )
    chain = prompt | grader

    filtered_docs = []
    sources = []
    
    for d in documents:
        # If it's an image file path in the content/metadata, we almost always want to keep it to check
        is_image = d.metadata.get('source', '').lower().endswith(('.jpg', '.png', '.jpeg'))
        
        try:
            res = chain.invoke({"question": question, "doc": d.page_content})
            if res.score == "yes" or is_image:
                filtered_docs.append(d)
                if 'source' in d.metadata:
                    sources.append(d.metadata['source'])
        except:
            continue
            
    return {"documents": filtered_docs, "sources": list(set(sources))}

def analyze_images_with_vision(state: AgentState):
    sources = state.get("sources", [])
    question = state["question"]
    
    print(f"---VISION ANALYSIS (Checking {len(sources)} images)---")
    if not sources:
        return {"vision_analysis": "No images found."}

    vision_insights = []
    for src in sources:
        if src.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"   👀 Analyzing: {src}")
            b64_img = encode_image(src)
            if b64_img:
                # UPDATED PROMPT: Explicitly ask to READ TEXT
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": f"User Question: '{question}'. \nTASK: 1. Read any text visible in this image exactly. 2. Describe the visual layout. 3. Does this image answer the user?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                )
                response = llm_vision.invoke([msg])
                vision_insights.append(f"Source: {src}\nAnalysis: {response.content}")
    
    return {"vision_analysis": "\n---\n".join(vision_insights)}

def generate(state: AgentState):
    print("---GENERATING ANSWER---")
    question = state["question"]
    vision_context = state.get("vision_analysis", "")
    
    # Combine context
    final_context = f"VISUAL EVIDENCE:\n{vision_context}"
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based strictly on the Visual Evidence provided.
        If the evidence mentions specific numbers, dates, or quotes, cite them.
        
        Question: {question}
        Evidence: {context}
        Answer:"""
    )
    
    chain = prompt | llm_text | StrOutputParser()
    response = chain.invoke({"question": question, "context": final_context})
    return {"generation": response}

def check_quality(state: AgentState):
    print("---QUALITY CONTROL---")
    question = state["question"]
    generation = state["generation"]
    retry_count = state["retry_count"]
    
    class Quality(BaseModel):
        score: str = Field(description="'good' or 'bad'")
        reason: str = Field(description="Why?")

    grader = llm_text.with_structured_output(Quality)
    prompt = ChatPromptTemplate.from_template(
        "User: {question}\nAgent: {generation}\nIs this a helpful, specific answer? 'good' or 'bad'?"
    )
    chain = prompt | grader
    grade = chain.invoke({"question": question, "generation": generation})
    
    print(f"   Grade: {grade.score} ({grade.reason})")
    
    if grade.score == "good":
        return "useful"
    
    # Logic: Retry once, then Web Search
    if retry_count < 1:
        return "retry"
    return "web_search"

def retry_generation(state: AgentState):
    print("---RETRYING---")
    return {"retry_count": state["retry_count"] + 1}

def web_search(state: AgentState):
    print("---WEB SEARCHING---")
    question = state["question"]
    search = DuckDuckGoSearchRun()
    result = search.invoke(question)
    return {"vision_analysis": f"WEB RESULTS:\n{result}"} # Inject web results into context slot

# 3. THE GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("analyze_images", analyze_images_with_vision)
workflow.add_node("generate", generate)
workflow.add_node("retry_gen", retry_generation)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "analyze_images")
workflow.add_edge("analyze_images", "generate")

workflow.add_conditional_edges(
    "generate",
    check_quality,
    {
        "useful": END,
        "retry": "retry_gen",
        "web_search": "web_search"
    }
)
workflow.add_edge("retry_gen", "generate")
workflow.add_edge("web_search", "generate")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 4. EXECUTION
if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "session_v4"}}
    
    print("=========================================")
    print("   VISION-AGENTIC RAG v4.0               ")
    print("=========================================")
    
    query = input("\nAsk: ")
    inputs = {"question": query}
    
    # Run the graph
    for output in app.stream(inputs, config=thread):
        for key, value in output.items():
            print(f"Finished: {key}")

    # --- FIX FOR DISPLAYING SOURCES ---
    # We fetch the FINAL state from the memory, ensuring we have the complete picture
    snapshot = app.get_state(thread)
    final_state = snapshot.values
    
    print("\n" + "="*40)
    print("💡 FINAL ANSWER:")
    print("="*40)
    print(final_state.get('generation', "No answer generated."))
    
    sources = final_state.get('sources', [])
    if sources:
        print("\n📸 REFERENCED IMAGES:")
        for s in list(set(sources)):
            print(f"   - {s}")
    else:
        print("\n(No local images used)")
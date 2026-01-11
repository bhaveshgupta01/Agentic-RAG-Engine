from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver

# 1. THE STATE
class AgentState(TypedDict):
    question: str           # The user's query
    documents: List[str]    # Results from the vector DB
    generation: str         # The LLM's final answer
    web_search_needed: bool # A decision flag: Do we need Google?

# --- LLM SETUP ---
llm = ChatOllama(model="mistral", temperature=0)

# 2. THE NODES

def retrieve(state: AgentState):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="rag-chroma",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )
    
    # Strict score threshold to avoid retrieving garbage
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 3, 'score_threshold': 0.3} 
    )
    
    try:
        documents = retriever.invoke(question)
    except:
        documents = []
    
    print(f"\nDEBUG: User asked: '{question}'")
    if not documents:
        print("DEBUG: No documents met the similarity threshold.")
    else:
        for i, doc in enumerate(documents):
            # FIX: Clean up whitespace for the log (removes the big gaps)
            clean_content = doc.page_content.replace("\n", " ").strip()[:100]
            print(f"DEBUG Doc {i}: {clean_content}...")
    print("--------------------------------\n")
    
    doc_texts = [d.page_content for d in documents]
    return {"documents": doc_texts, "question": question}


# --- GRADER CONFIG ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the user question, 'yes' or 'no'")
    explanation: str = Field(description="A short explanation of why this document is relevant or not.")

system_prompt = """You are a strict grader assessing relevance of a retrieved document to a user question.
1. The document must contain the EXACT answer or specific semantic meaning to the question.
2. If the user asks a "how to" or "what is" question, and the document is just a table of contents or a generic intro, grade it as 'no'.
3. Do not grade 'yes' just because the document mentions the word 'question' or shared keywords. The MEANING must match.
4. If you are unsure, grade it as 'no'.

Give a binary score 'yes' or 'no' and a short explanation."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

structured_llm_grader = llm.with_structured_output(GradeDocuments)
grader_chain = grade_prompt | structured_llm_grader

def grade_documents(state: AgentState):
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    
    for i, d in enumerate(documents):
        # We pass the document string directly
        score = grader_chain.invoke({"question": question, "document": d})
        grade = score.binary_score
        explanation = score.explanation
        
        print(f"\n   [DOC {i}]")
        print(f"   Content: {d.replace('\n', ' ')[:80]}...") # Clean log here too
        print(f"   Grade:   {grade}")
        print(f"   Reason:  {explanation}")
        
        if grade == "yes":
            filtered_docs.append(d)
        
    web_search_needed = False
    if not filtered_docs:
        web_search_needed = True
        print("\n---DECISION: ALL DOCS BAD, SWITCHING TO WEB SEARCH---")
    else:
        print("\n---DECISION: DOCS GOOD, GENERATING---")
        
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}


def generate(state: AgentState):
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        Context: {context} 
        Answer:"""
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}


def web_search(state: AgentState):
    print("---WEB SEARCHING---")
    question = state["question"]
    search_tool = DuckDuckGoSearchRun()
    search_results = search_tool.invoke(question)
    return {"documents": [search_results]}


# --- CONDITIONAL LOGIC (The Missing Piece!) ---
def decide_next_step(state: AgentState):
    if state["web_search_needed"]:
        return "web_search"
    else:
        return "generate"


# 3. THE GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")

# --- ADD THIS MISSING LINE ---
workflow.add_edge("retrieve", "grade_documents") 
# -----------------------------

workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# MEMORY SETUP
memory = MemorySaver() 

# COMPILE
app = workflow.compile(checkpointer=memory)

# 4. EXECUTION
if __name__ == "__main__":
    # We use a thread_id to track conversation history
    thread = {"configurable": {"thread_id": "1"}}
    
    print("--- STARTING CONVERSATION ---")
    
    # Question 1
    inputs_1 = {"question": "What are the components of an autonomous agent?"}
    for output in app.stream(inputs_1, config=thread):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            if "generation" in value:
                print(f"\nFINAL ANSWER: {value['generation']}\n")

    print("\n--- NEXT TURN (MEMORY CHECK) ---")
    
    # Question 2 (Follow-up)
    inputs_2 = {"question": "Explain the 'Memory' component you just mentioned."}
    for output in app.stream(inputs_2, config=thread):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            if "generation" in value:
                print(f"\nFINAL ANSWER: {value['generation']}\n")
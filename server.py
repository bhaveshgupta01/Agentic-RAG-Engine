from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app  # Import your agent
import uvicorn

# 1. Define the API
api = FastAPI(title="Agentic RAG API")

# 2. Define the Request Body (What the user sends)
class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default_user" # For memory

# 3. Define the Endpoint
@api.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Send a question to the Agent.
    """
    print(f"Received Question: {request.question} (Thread: {request.thread_id})")
    
    # Config allows the agent to remember this specific user
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Run the Agent
    final_answer = "No answer generated."
    
    try:
        # We use .invoke() instead of .stream() for a simple API response
        # It runs the whole graph and returns the final state
        result = app.invoke({"question": request.question}, config=config)
        final_answer = result.get("generation", "Sorry, something went wrong.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"answer": final_answer}

# 4. Run the Server
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
from agent import app

print("--- GENERATING GRAPH VISUALIZATION ---")

# 1. Print an ASCII chart (Works directly in terminal)
try:
    print("\n[ASCII Flowchart]")
    print(app.get_graph().draw_ascii())
except Exception as e:
    print(f"Could not draw ASCII: {e}")

# 2. Print the Mermaid Code (For the website)
try:
    print("\n[Mermaid Code - Copy this to mermaid.live]")
    print(app.get_graph().draw_mermaid())
except Exception as e:
    print(f"Could not draw Mermaid: {e}")
from google.adk.tools import ToolContext

def record_validation_result(tool_context: ToolContext, status: str, reason: str, feedback: str = ""):
    """
    Records the QA result. 
    """
    print(f"\n[Validation] 🕵️ Result: {status.upper()}")
    print(f"Reason: {reason}")
    if feedback:
        print(f"Feedback: {feedback}")

    # Save to state so the LoopAgent can decide whether to stop or retry
    tool_context.state["validation_status"] = status.upper()
    tool_context.state["validation_feedback"] = feedback
    
    return "Validation recorded."
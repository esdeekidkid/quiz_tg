# --- In-memory session storage (temporary, no Supabase required) ---
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL", "https://aozxmeovifobnfspfoqt.supabase.co")
SUPABASE_KEY = (
    os.getenv("VITE_SUPABASE_ANON_KEY")
    or os.getenv("VITE_SUPABASE_SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_KEY")
)

# In-memory storage: { session_id: {"lecture_text": "...", "results_json": "..."} }
SESSIONS = {}

def save_to_supabase(table: str, data: dict):
    """
    Save data to in-memory storage.
    table = "quiz_sessions" or "quiz_results"
    """
    if not data:
        return False
    if table == "quiz_sessions":
        session_id = data.get("session_id")
        if not session_id:
            return False
        SESSIONS.setdefault(session_id, {})
        SESSIONS[session_id]["lecture_text"] = data.get("lecture_text", "")
        print(f"Session {session_id} saved to memory")
        return True
    if table == "quiz_results":
        session_id = data.get("session_id")
        if not session_id:
            return False
        SESSIONS.setdefault(session_id, {})
        SESSIONS[session_id]["results_json"] = data.get("results_json", "")
        print(f"Results for session {session_id} saved to memory")
        return True
    return False

def get_from_supabase(table: str, session_id: str):
    """
    Retrieve data from in-memory storage.
    """
    if not session_id:
        return None
    entry = SESSIONS.get(session_id)
    if not entry:
        return None
    if table == "quiz_sessions":
        return {"lecture_text": entry.get("lecture_text", "")}
    if table == "quiz_results":
        return {"results_json": entry.get("results_json", "")}
    return entry

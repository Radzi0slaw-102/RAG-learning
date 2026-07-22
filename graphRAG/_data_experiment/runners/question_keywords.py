import re

STOPWORDS = {"what", "which", "does", "have", "with", "that", "this", "from", "into"}

def question_keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][\w.\-]*", question)
    return [w for w in words if len(w) > 3 and w.lower() not in STOPWORDS]
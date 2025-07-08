import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_class(text):
    match = re.search(r'class\s*(\d{1,2})', text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.rstrip("'s").rstrip("'")
    skip_words = {"what", "who", "when", "where", "how", "why", "is", "was",
                  "are", "do", "does", "did", "to", "in", "on", "at", "for", "have", "has", "had"}
    words = re.findall(r"\b[a-zA-Z']+\b", text)
    for w in words:
        clean = w.lower().strip("'")
        if clean not in skip_words and len(clean) > 2:
            return w.strip("'").rstrip("'s").capitalize()
    return None


def extract_subjects(text):
    subjects = ['math', 'mathematics', 'eng', 'english', 'science', 'history',
                'social studies', 'sst', 'geography', 'economics', 'biology',
                'chemistry', 'physics', 'comp sci', 'computer science', 'marathi']
    subjects = [subject.lower() for subject in subjects]
    for i in subjects:
        if i.lower() in text.lower():
            return i.capitalize()
    return None

def extract_month(text):
    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    for i in months:
        if i in text.lower():
            return i.capitalize()
    return None

def extract_year(text):
    match = re.search(r'(20\d{2})', text)
    if match:
        return match.group(1)
    if 'next year' in text.lower():
        from datetime import datetime
        return str(datetime.now().year + 1)
    return None

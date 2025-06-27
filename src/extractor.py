import re

def extract_class(text):
    match=re.search(r'class\s*(\d{1,2})',text,re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None
def extract_name(text):
    match = re.search(r"\b([A-Z][a-z]+)\b", text)
    return match.group(1) if match else None
def extract_subjects(text):
    subjects=['math,mathematics','eng','english','science','history','social studies','sst','geography','economics','biology','chemistry','physics','comp sci','computer science','marathi']
    subjects=[subject.lower() for subject in subjects]
    for i in subjects:
        if i.lower() in text.lower():
            return i.capitalize()
    return None

def extract_month(text):
    months =['january','february','march','april','may','june','july','august','september','october','november','december']
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


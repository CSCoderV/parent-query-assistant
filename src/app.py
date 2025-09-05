from classifier import IntentClassifier
from extractor import extract_class, extract_subjects, extract_month, extract_year, extract_name, extract_exam_types
from semantic_intent import SemanticIntentMatcher
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import glob
from datetime import datetime

_FALLBACK = {
    "get_attendance": ("attendance", "attendence", "present", "absent"),
    "get_marks": ("marks", "mark", "score", "grade", "grades", "percentage"),
    "get_exam_schedule": ("exam", "schedule", "midterm", "final", "unit test", "quiz", "test"),
}
def _guess_intent_by_keywords(text: str):
    t = text.lower()
    for intent, words in _FALLBACK.items():
        if any(w in t for w in words):
            return intent
    return None

# loading of req/necessary data
df = pd.read_csv("../data/parent_queries_dataset.csv")
intent_examples = df.groupby("intent")["query"].apply(list).to_dict()
matcher = SemanticIntentMatcher(intent_examples)
student_df = pd.read_csv("../data/student_records.csv")
student_df.columns = student_df.columns.str.strip().str.lower().str.replace(r"[\s\-]+","_", regex=True)
ALL_NAMES = [n.lower() for n in student_df["student_name"].astype(str)]
#loading dataset (default dataset where user can add their own queries)
default_database_df =pd.read_csv("../data/default_database.csv")
default_database_queries=default_database_df["query"].tolist()
default_database_answers=default_database_df["answer"].tolist()

#database for exam/quiz schdeule
exam_schedule_df = pd.read_csv("../data/exam_schedule.csv")
quiz_df = pd.read_csv("../data/quiz_schedule.csv") 
quiz_df = quiz_df.rename(columns={"quiz_date": "exam_date","quiz_name": "exam_name"})
if "exam_type" not in quiz_df.columns:            
    quiz_df["exam_type"] = "Quiz"                 

for _df in (exam_schedule_df, quiz_df):           
    if "class" in _df.columns:                    
        _df["class"] = _df["class"].astype(str).str.extract(r"(\d+)")[0]
    if "subject" in _df.columns:                  
        _df["subject"] = _df["subject"].astype(str).str.strip().str.lower()
    if "exam_date" in _df.columns:                
        _df["exam_date"] = pd.to_datetime(_df["exam_date"], errors="coerce")

all_schedule_df = pd.concat([exam_schedule_df, quiz_df], ignore_index=True)


model= SentenceTransformer('all-MiniLM-L6-v2')
default_database_embeddings = model.encode(default_database_queries, convert_to_tensor=True)

#to find the corresponding stdnt record
def find_student_records(student_name,class_name=None,subject=None):
    df=student_df.copy()
    if student_name:
        df=df[df["student_name"].str.lower()==student_name.lower()]
    if class_name:
        df=df[df["class"]==int(class_name)]
    if subject:
        df=df[df["subject"].str.lower()==subject.lower()]
    return df.to_dict(orient="records")

#searching  for best answwer from the dataset using similarity (ie similar Qs)
def find_best_default_database_answer(user_query):
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, default_database_embeddings)
    best_idx = int(torch.argmax(scores))
    return default_database_answers[best_idx]

last_context = {"student_name": None, "class_name": None, "subject": None}

# for testing/ running program
while True:
    input1 = input("\nHello! How can I help you? (Type 'quit' to exit): ")
    if input1.lower() in ["quit","exit","q","e"]:
        print("Thanks for using the service. Goodbye!")
        break
    intent, prediction_score = matcher.predict(input1)
    print(f"Intent: {intent} (confidence: {prediction_score:.1f})")
    if prediction_score<0.60:
        kw = _guess_intent_by_keywords(input1)
        if kw:
            intent = kw
        else:
            print("I can help with marks, attendance, or exam schedule. Which one did you mean?")
            continue
    intent = [intent]
    handled=False

    class_name=extract_class(input1)
    subject=extract_subjects(input1)
    student_name = None
    for i in ALL_NAMES:
        if i in input1.lower():
            student_name = i.capitalize()
            break
    #tring a fallback approach (ie if no name specified then it tries using the last name used in program)
    if student_name:
        last_context["student_name"]=student_name
    if class_name:
        last_context["class_name"]=class_name
    if subject:
        last_context["subject"]=subject
    student_name=student_name or last_context["student_name"]
    class_name = class_name or last_context["class_name"]
    subject = subject or last_context["subject"]

    if ('get_quiz_schedule' in intent) or ('get_exam_schedule' in intent):
        if 'get_quiz_schedule' in intent:
            df = quiz_df.copy()
            date_col, name_col='quiz_date','quiz_name'
            kind = 'quizzes'
        else:
            df = exam_schedule_df.copy()
            date_col, name_col='exam_date','exam_name'
            kind = 'exams'

        if 'class' in df.columns:
            df['class'] =df['class'].astype(str).str.extract(r'(\d+)')[0]
        if class_name:
            df = df[df['class']==str(int(class_name))]
        if subject and 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str).str.lower().str.strip()
            df = df[df['subject']==subject.lower().strip()]

        if date_col in df.columns:
            df[date_col] =pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(date_col, na_position='last')

        if df.empty:
            print(f"No upcoming {kind} for class {class_name or '--'}" +
                (f" in {subject}" if subject else "") + ".")
        else:
            r =df.head(1).iloc[0].to_dict()
            when =r[date_col].strftime('%Y-%m-%d') if r.get(date_col) and str(r[date_col]) != 'NaT' else '--'
            subj =(r.get('subject') or '').title()
            name =r.get(name_col) or ''
            time =r.get('start_time') or ''
            loc  =r.get('location') or ''
            extra =" ".join(x for x in [name, f"at {time}" if time else "", f"in {loc}" if loc else ""] if x)
            print(f"Class {r.get('class','--')}: {subj} on {when}" + (f" — {extra}" if extra else ""))
        handled =True
        continue


    if student_name:
        results = find_student_records(student_name, class_name, subject)
        if results:
            for i in results:
                for current_intent in intent:
                    if current_intent == 'get_marks':
                        print(
                            f"{i['student_name']} scored {i.get('marks','--')} in {i.get('subject','--')}"
                            + (f" in {i.get('exam_type','--')}" ))
                        handled=True
                    elif current_intent == 'get_attendance':
                        print(f"{i['student_name']}'s attendance is {i.get('attendance','--')}")
                        handled=True
                    elif current_intent == 'get_homework':
                        print(f"Class {i.get('class','--')} has following homework assigned: {i.get('homework','--')}")
                        handled=True
                    else:
                        print(f"Record for {i['student_name']} is available. Please specify the reason to access the record")
                        handled=True
        else:
            print(f"No records found for {student_name}"+(f" in {subject}" if subject else "")+(f" (Class {class_name})" if class_name else ""))

            handled=True
    if not handled:
        print("No record found, checking general knowledge base...")
        print("Assistant:", find_best_default_database_answer(input1))

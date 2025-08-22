from classifier import IntentClassifier
from extractor import extract_class, extract_subjects, extract_month, extract_year, extract_name
from semantic_intent import SemanticIntentMatcher
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import glob
from datetime import datetime

# loading of req/necessary data
df = pd.read_csv("../data/parent_queries_dataset.csv")
intent_examples = df.groupby("intent")["query"].apply(list).to_dict()
matcher = SemanticIntentMatcher(intent_examples)
student_df = pd.read_csv("../data/student_records.csv")
ALL_NAMES = [n.lower() for n in student_df["student_name"].astype(str)]
#loading dataset (default dataset where user can add their own queries)
default_database_df =pd.read_csv("../data/default_database.csv")
default_database_queries=default_database_df["query"].tolist()
default_database_answers=default_database_df["answer"].tolist()
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


    if student_name:
        results = find_student_records(student_name, class_name, subject)
        if results:
            for i in results:
                for current_intent in intent:
                    if current_intent == 'get_marks':
                        print(
                            f"{i['student_name']} scored {i.get('marks','--')} in {i.get('subject','--')}"
                            + (f" in {i.get('exam_name')}" if i.get('exam_name') else "")
                        )
                        handled=True
                    elif current_intent == 'get_attendance':
                        print(f"{i['student_name']}'s attendance is {i.get('attendance','--')}")
                        handled=True
                    elif current_intent == 'get_homework':
                        print(f"Class {i.get('class','--')} has following homework assigned: {i.get('homework','--')}")
                        handled=True
                    elif current_intent == 'get_exam_schedule':
                        print(f"Class {i.get('class','--')} has the following upcoming exams: {i.get('exam_schedule','--')}")
                        handled=True
                    elif current_intent == 'get_quiz_schedule':
                        print(f"Class {i.get('class','--')} has the following quizzes scheduled: {i.get('quiz_schedule','--')}")
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

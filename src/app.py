from classifier import IntentClassifier
from extractor import extract_class, extract_subjects, extract_month, extract_year, extract_name
from semantic_intent import SemanticIntentMatcher
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# loading of req/necessary data
df = pd.read_csv("../data/parent_queries_dataset.csv")
intent_examples = df.groupby("intent")["query"].apply(list).to_dict()
matcher = SemanticIntentMatcher(intent_examples)
student_df = pd.read_csv("../data/student_records.csv")

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

#searching  for best answwer from the dataset using similarity
def find_best_default_database_answer(user_query):
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, default_database_embeddings)
    best_idx = int(torch.argmax(scores))
    return default_database_answers[best_idx]

last_context = {"student_name": None, "class_name": None, "subject": None}

# for testing
while True:
    input1 = input("\nHello! How can I help you? (Type 'quit' to exit): ")
    if input1.lower() in ["quit","exit","q","e"]:
        print("Thanks for using the service. Goodbye!")
        break

    intent, prediction_score = matcher.predict(input1)
    print(f"Intent: {intent} (confidence: {prediction_score:.1f})")
    

    if "marks" in input1.lower() :
        intent=['get_marks']
    elif "attendance" in input1.lower():
        intent=['get_attendance']
    else:
        intent=[intent]

    handled=False

    class_name=extract_class(input1)
    subject=extract_subjects(input1)
    student_name=extract_name(input1)
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
        results=find_student_records(student_name, class_name, subject)
        if results:
            for i in results:
                for current_intent in intent:
                    if current_intent=="get_marks":
                        print(f"{i['student_name']} scored {i['marks']} in {i['subject']}")
                        handled=True
                    elif current_intent=="get_attendance":
                        print(f"{i['student_name']}'s attendance in {i['subject']} is {i['attendance']}")
                        handled=True
                    elif current_intent=="get_homework":
                        print(f"Class {i['class']} has the following HW assigned: {i['homework']}")
                        handled=True
                    else:
                        print(f"Record for {i['student_name']} in {i['subject']}, class {i['class']}")
                        handled=True
        else:
            print("No student specified in your query.")
            handled=True
    if not handled:
        print("No direct record found, checking general knowledge base...")
        print("Assistant:", find_best_default_database_answer(input1))

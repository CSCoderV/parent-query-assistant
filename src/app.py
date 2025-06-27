from classifier import IntentClassifier
from extractor import extract_class,extract_subjects,extract_month,extract_year,extract_name
from semantic_intent import SemanticIntentMatcher
'''
sample_queries=[
    "Has Aarav completed his homework today",  
]
sample_labels=[
    "get_homework",
    "get_attendance",
    "get_marks",
    "get_homework",
    "get_performance"
]
classifier=IntentClassifier()
classifier.train(sample_queries,sample_labels)
'''
intent_examples = {
    "get_homework":[
        "Is there any homework for class 6 today?",
        "Is there any science hw for class 4 today?",
        "Was there any homework assigned to class 3 yesterday"
    ],
    "get_syllabus": [
        "What is the syllabus for class 6 science?",
        "Tell me class 7 English syllabus",
        "What topics are covered in math class 8?"
    ],
    "get_marks": [
        "Show me Aarav's marks in science",
        "Can you share Tanvi’s maths marks?",
        "How did Meera score in class 7 math?",
        "Tanvi's grade in English?"
    ],
    "get_attendance": [
        "What is Aarav's attendance?",
        "Why was Meera absent last Thursday?",
        "Was Meera present yesterday?",
        "Show Tanvi’s attendance for March"
    ],
    "get_performance":[
        "How is Soham doing in class these days?",
        "has Soham been doing his homework"
    ]
}
matcher = SemanticIntentMatcher(intent_examples)


#fpr testing
while True:
    input1=input("\nHello! How can I help you? (Type 'quit' to exit): ")
    if input1.lower() in ["quit","exit","q","e"]:
        print("Thanks for using the service. Goodbye!")
        break
    intent, prediction_score = matcher.predict(input1)
    print(f"Intent: {intent} (confidence: {prediction_score:.1f})")
    class_name = extract_class(input1)
    subject = extract_subjects(input1)
    student_name = extract_name(input1)
    if student_name:
        print("Name "+student_name)
    if class_name:
        print("Class: "+class_name)
    if subject:
        print("Subject: "+subject)



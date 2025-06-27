from sentence_transformers import SentenceTransformer, util

class SemanticIntentMatcher:
    def __init__(self,intent_examples):
        self.model=SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_examples=intent_examples
        self.intent_texts=[]
        self.intent_labels=[]
        for intent,examples in intent_examples.items():
            self.intent_texts.extend(examples)
            self.intent_labels.extend([intent] * len(examples))
        self.intent_embeddings = self.model.encode(self.intent_texts, convert_to_tensor=True)

    def predict(self, user_query):
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.intent_embeddings)
        best_idx = int(scores.argmax())
        return self.intent_labels[best_idx], float(scores.max())
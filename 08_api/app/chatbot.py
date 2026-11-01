
import re 
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
class CortexHRChatbot:
    def __init__(self, artifact_path):
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
             raise FileNotFoundError(
                f"Chatbot artifact not found: {artifact_path}"
             )
        self.bundle = joblib.load(artifact_path)
        required_keys = [
            "model",
            "tfidf",
            "faq_vectorizer",
            "faq_question_matrix",
            "faq_search",
            "intent_faq_map",
            "responses",
            "typo_map",
            "confidence_threshold"
        ]
        missing = [
            key for key in required_keys
            if key not in self.bundle
        ]

        if missing:
            raise ValueError(
                f"Missing chatbot artifacts: {missing}"
            )

        self.model = self.bundle["model"]
        self.tfidf = self.bundle["tfidf"]

        self.faq_vectorizer = self.bundle["faq_vectorizer"]
        self.faq_question_matrix = self.bundle[
            "faq_question_matrix"
        ]

        self.faq_search = (
            self.bundle["faq_search"].reset_index(drop=True)
        )

        self.intent_faq_map = self.bundle["intent_faq_map"]
        self.responses = self.bundle["responses"].copy()

        self.typo_map = self.bundle["typo_map"]

        self.confidence_threshold = float(
            self.bundle["confidence_threshold"]
        )

        self.response_lookup = (
            self.responses
            .drop_duplicates("intent")
            .set_index("intent")
        )

        # Verify saved FAQ matrix consistency
        if (
            self.faq_question_matrix.shape[0]
            != len(self.faq_search)
        ):
            raise ValueError(
                "FAQ matrix and FAQ records have different lengths."
            )

        self.model_version = self.bundle.get(
            "model_version", "unknown"
        )

        # Task 10: Store conversation sessions
        self.sessions = {}

    def clean_text(self, text):

        text = str(text).lower()

        for wrong, correct in self.typo_map.items():

            text = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                text
            )

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def retrieve_faq_answer(self, intent, user_message):

        faq_ids = self.intent_faq_map.get(intent)

        if not faq_ids:
            return None

        candidates = self.faq_search[
            self.faq_search["faq_id"].isin(faq_ids)
        ]

        if candidates.empty:
            return None

        query_vector = self.faq_vectorizer.transform(
            [self.clean_text(user_message)]
        )

        candidate_indices = candidates.index.to_numpy(
            dtype=int
        )

        candidate_vectors = (
            self.faq_question_matrix[candidate_indices]
        )

        scores = cosine_similarity(
            query_vector,
            candidate_vectors
        ).flatten()

        best_position = int(np.argmax(scores))

        best_faq = candidates.iloc[best_position]

        return {
            "faq_id": str(best_faq["faq_id"]),
            "question": str(best_faq["question"]),
            "answer": str(best_faq["answer"]),
            "similarity": float(scores[best_position])
        }

    def reply(self, user_message):

        # Validate input
        if (
            not isinstance(user_message, str)
            or not user_message.strip()
        ):
            return {
                "status": "fallback",
                "intent": None,
                "confidence": None,
                "reply": (
                    "Please enter a valid HR question."
                ),
                "source": None
            }

        # Clean incoming message
        cleaned_message = self.clean_text(user_message)

        # Convert message to TF-IDF features
        message_features = self.tfidf.transform(
            [cleaned_message]
        )

        # Predict intent probabilities
        probabilities = self.model.predict_proba(
            message_features
        )[0]

        best_index = int(np.argmax(probabilities))

        predicted_intent = str(
            self.model.classes_[best_index]
        )

        confidence = float(probabilities[best_index])

        # Confidence threshold check
        if confidence < self.confidence_threshold:

            return {
                "status": "low_confidence",
                "intent": predicted_intent,
                "confidence": confidence,
                "reply": (
                    "I'm sorry, I'm not fully sure I understood "
                    "your question. Could you rephrase it? "
                    "For further assistance, please contact HR."
                ),
                "source": None
            }

        # Find response configuration
        if predicted_intent not in self.response_lookup.index:

            return {
                "status": "fallback",
                "intent": predicted_intent,
                "confidence": confidence,
                "reply": (
                    "I could not find a suitable HR response. "
                    "Please contact HR for assistance."
                ),
                "source": None
            }

        config = self.response_lookup.loc[predicted_intent]

        template = config["response_template"]
        service = config["backend_or_service_call"]

        # Handle explicit privacy refusal first
        if (
            isinstance(service, str)
            and service.strip().upper() == "REFUSE"
        ):

            return {
                "status": "refused",
                "intent": predicted_intent,
                "confidence": confidence,
                "reply": str(template),
                "source": "chatbot_responses"
            }

        # Block all protected intents until trusted
        # server-side authentication is implemented
        requires_login = (
            str(config["requires_login"]).strip().upper()
            == "Y"
        )

        if requires_login:

            return {
                "status": "authentication_required",
                "intent": predicted_intent,
                "confidence": confidence,
                "reply": (
                    "Please sign in to access this HR information. "
                    "If you need further assistance, contact HR."
                ),
                "source": None
            }

        # Retrieve the most relevant individual FAQ answer
        if predicted_intent in self.intent_faq_map:

            faq_result = self.retrieve_faq_answer(
                predicted_intent,
                cleaned_message
            )

            if faq_result is not None:

                return {
                    "status": "answered",
                    "intent": predicted_intent,
                    "confidence": confidence,
                    "reply": faq_result["answer"],
                    "source": faq_result["faq_id"],
                    "faq_similarity": faq_result["similarity"]
                }

        # Handle configured static responses
        if pd.notna(template):

            template = str(template)

            # Do not return unresolved dynamic placeholders
            if "{" in template or "}" in template:

                return {
                    "status": "backend_unavailable",
                    "intent": predicted_intent,
                    "confidence": confidence,
                    "reply": (
                        "Sorry, I could not retrieve that HR "
                        "information right now. Please contact HR."
                    ),
                    "source": None
                }

            return {
                "status": "answered",
                "intent": predicted_intent,
                "confidence": confidence,
                "reply": template,
                "source": "chatbot_responses"
            }

        # Safe fallback
        return {
            "status": "fallback",
            "intent": predicted_intent,
            "confidence": confidence,
            "reply": (
                "Sorry, I could not answer that question. "
                "Please contact HR for assistance."
            ),
            "source": None
        }

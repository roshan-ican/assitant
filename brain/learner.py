# brain/learner.py - Core learning logic (No transformers needed!)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime
import json
from transformers import pipeline


class JarvisLearner:
    def __init__(self):
        # TF-IDF is actually perfect for learning task patterns
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.user_patterns = defaultdict(lambda: {
            'tasks': [],
            'task_texts': [],
            'vectors': None,
            'categories': {},
            'time_patterns': defaultdict(list)
        })

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        # Dynamic categories that expand based on user's tasks
        self.base_categories = [
            "software development", "coding", "work tasks",
            "shopping", "health", "home tasks", "learning",
            "meetings", "personal tasks"
        ]

    def learn_from_task(self, task_text, user_id):
        """Learn patterns from a single task"""

        # Store task data
        task_data = {
            'text': task_text,
            'timestamp': datetime.now(),
            'hour': datetime.now().hour,
            'day': datetime.now().weekday()
        }

        # Add to user patterns
        user_data = self.user_patterns[user_id]
        user_data['tasks'].append(task_data)
        user_data['task_texts'].append(task_text)

        # Update learning if we have enough data
        if len(user_data['task_texts']) >= 3:
            self._update_learning(user_id)

        # Learn time patterns
        self._learn_time_patterns(user_id, task_data)

    def _update_learning(self, user_id):
        """Update TF-IDF vectors and categories"""
        user_data = self.user_patterns[user_id]

        try:
            # Create TF-IDF vectors
            vectors = self.vectorizer.fit_transform(user_data['task_texts'])
            user_data['vectors'] = vectors

            # Discover categories
            self._discover_categories(user_id, vectors)

        except Exception as e:
            print(f"Learning update failed: {e}")

    def _discover_categories(self, user_id, vectors):
        """Automatically discover categories using clustering"""
        user_data = self.user_patterns[user_id]

        try:
            n_tasks = len(user_data['task_texts'])
            n_clusters = min(3, max(2, n_tasks // 3))  # Smart cluster count

            if n_tasks >= 4:  # Need minimum tasks for clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors.toarray())

                # Group tasks by cluster
                categories = defaultdict(list)
                for i, label in enumerate(labels):
                    categories[f"type_{label}"].append(user_data['tasks'][i])

                user_data['categories'] = dict(categories)

        except Exception as e:
            print(f"Clustering failed, using simple categorization: {e}")
            self._simple_categorize(user_id)

    def _simple_categorize(self, user_id):
        """Fallback: Simple keyword-based categorization"""
        user_data = self.user_patterns[user_id]
        categories = defaultdict(list)

        for task in user_data['tasks']:
            text = task['text'].lower()

            if any(word in text for word in ['buy', 'shop', 'grocery', 'get', 'pick up']):
                categories['shopping'].append(task)
            elif any(word in text for word in ['meeting', 'call', 'work', 'email', 'send']):
                categories['work'].append(task)
            elif any(word in text for word in ['exercise', 'gym', 'run', 'walk', 'health']):
                categories['health'].append(task)
            elif any(word in text for word in ['clean', 'wash', 'cook', 'home', 'house']):
                categories['home'].append(task)
            else:
                categories['general'].append(task)

        user_data['categories'] = dict(categories)

    def _learn_time_patterns(self, user_id, task_data):
        """Learn when user does certain types of tasks"""
        user_data = self.user_patterns[user_id]

        # Group by time periods
        time_key = f"{task_data['day']}_{task_data['hour']}"
        user_data['time_patterns'][time_key].append(task_data['text'])

    def get_user_insights(self, user_id):
        """Get what we learned about the user"""
        user_data = self.user_patterns[user_id]

        return {
            'total_tasks': len(user_data['tasks']),
            'categories_found': len(user_data['categories']),
            'time_patterns': len(user_data['time_patterns']),
            'sample_categories': {
                name: [task['text'] for task in tasks[:3]]
                for name, tasks in list(user_data['categories'].items())[:3]
            }
        }

    @staticmethod
    def predict_category(task_text, user_id):
        """Enhanced keyword-based prediction"""
        text = task_text.lower()

        # Programming/Development
        if any(word in text for word in
               ['git', 'code', 'programming', 'development', 'api', 'database', 'deploy', 'fix bug', 'commit']):
            return 'development'

        # Work/Professional
        elif any(word in text for word in ['meeting', 'call', 'work', 'email', 'send', 'presentation', 'report']):
            return 'work'

        # Shopping
        elif any(word in text for word in ['buy', 'shop', 'grocery', 'get', 'pick up', 'purchase']):
            return 'shopping'

        # Health/Fitness
        elif any(word in text for word in ['exercise', 'gym', 'run', 'walk', 'health', 'doctor', 'workout']):
            return 'health'

        # Home/Household
        elif any(word in text for word in ['clean', 'wash', 'cook', 'home', 'house', 'repair', 'organize']):
            return 'home'

        else:
            return 'general'

    # def predict_category(self, task_text, user_id):
    #     """Predict category for new task"""
    #     user_data = self.user_patterns[user_id]
    #
    #     # If no learned categories yet, use simple keyword matching
    #     if not user_data['categories'] or len(user_data['task_texts']) < 3:
    #         return self._predict_simple_category(task_text)
    #
    #     # Use TF-IDF similarity if we have learned data
    #     try:
    #         if user_data['vectors'] is not None:
    #             # Transform new task
    #             new_vector = self.vectorizer.transform([task_text])
    #
    #             # Find most similar existing task
    #             similarities = cosine_similarity(new_vector, user_data['vectors'])[0]
    #             most_similar_idx = similarities.argmax()
    #
    #             if similarities[most_similar_idx] > 0.3:  # Similarity threshold
    #                 similar_task = user_data['tasks'][most_similar_idx]
    #
    #                 # Find which category this similar task belongs to
    #                 for category, tasks in user_data['categories'].items():
    #                     if similar_task in tasks:
    #                         return category.replace('type_', '').replace('_', ' ')
    #
    #     except Exception as e:
    #         print(f"ML prediction failed: {e}")
    #
    #     # Fallback to simple categorization
    #     return self._predict_simple_category(task_text)
    #
    # @staticmethod
    # def _predict_simple_category(task_text):
    #     """Simple keyword-based category prediction"""
    #     text = task_text.lower()
    #
    #     if any(word in text for word in ['buy', 'shop', 'grocery', 'get', 'pick up']):
    #         return 'shopping'
    #     elif any(word in text for word in ['meeting', 'call', 'work', 'email', 'send']):
    #         return 'work'
    #     elif any(word in text for word in ['exercise', 'gym', 'run', 'walk', 'health']):
    #         return 'health'
    #     elif any(word in text for word in ['clean', 'wash', 'cook', 'home', 'house']):
    #         return 'home'
    #     else:
    #         return 'general'
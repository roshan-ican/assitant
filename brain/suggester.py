# brain/suggester.py - ML-powered suggestion engine
from datetime import datetime
import numpy as np


class JarvisSuggester:
    def __init__(self, learner, storage):
        self.learner = learner
        self.storage = storage

    def get_suggestions(self, user_id):
        """Generate ML-powered suggestions"""

        suggestions = []
        current_time = datetime.now()

        # Get user's learned patterns
        user_data = self.learner._load_user_data(user_id)

        if not user_data['tasks']:
            return self._get_default_suggestions(current_time)

        # Time-based suggestions using learned patterns
        time_suggestions = self._get_time_based_suggestions(user_data, current_time)
        suggestions.extend(time_suggestions)

        # Category-based suggestions
        category_suggestions = self._get_category_suggestions(user_data)
        suggestions.extend(category_suggestions)

        # Similarity-based suggestions
        similarity_suggestions = self._get_similarity_suggestions(user_data)
        suggestions.extend(similarity_suggestions)

        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return suggestions[:5]

    def _get_time_based_suggestions(self, user_data, current_time):
        """Suggest based on learned time patterns"""
        suggestions = []

        current_hour = current_time.hour
        current_day = current_time.weekday()

        # Check specific time patterns
        time_key = f"day_{current_day}_hour_{current_hour}"
        if time_key in user_data['time_patterns']:
            tasks = user_data['time_patterns'][time_key]
            if len(tasks) >= 2:
                from collections import Counter
                most_common = Counter(tasks).most_common(1)[0][0]
                suggestions.append({
                    'task': most_common,
                    'reason': f'You usually do this on {current_time.strftime("%A")} at {current_hour}:00',
                    'confidence': min(len(tasks) / 5.0, 0.9),
                    'type': 'time_pattern'
                })

        # Check broader time periods
        if 6 <= current_hour < 12:
            time_period = 'morning'
        elif 12 <= current_hour < 17:
            time_period = 'afternoon'
        elif 17 <= current_hour < 21:
            time_period = 'evening'
        else:
            time_period = 'night'

        if time_period in user_data['time_patterns']:
            tasks = user_data['time_patterns'][time_period]
            if len(tasks) >= 3:
                from collections import Counter
                common_tasks = Counter(tasks).most_common(2)
                for task, count in common_tasks:
                    suggestions.append({
                        'task': task,
                        'reason': f'Common {time_period} task for you',
                        'confidence': min(count / len(tasks), 0.8),
                        'type': 'time_period'
                    })

        return suggestions

    def _get_category_suggestions(self, user_data):
        """Suggest based on learned categories"""
        suggestions = []

        for category, tasks in user_data['categories'].items():
            if len(tasks) >= 3:
                # Find most frequent task in category
                task_texts = [task['text'] for task in tasks]
                from collections import Counter
                most_common = Counter(task_texts).most_common(1)[0][0]

                suggestions.append({
                    'task': most_common,
                    'reason': f'Popular task in your {category.replace("_", " ")} category',
                    'confidence': min(len(tasks) / 10.0, 0.7),
                    'type': 'category_pattern'
                })

        return suggestions

    def _get_similarity_suggestions(self, user_data):
        """Generate suggestions based on task similarity"""
        suggestions = []

        if len(user_data['tasks']) < 5:
            return suggestions

        # Get recent tasks to find patterns
        recent_tasks = user_data['tasks'][-5:]
        task_texts = [task['text'] for task in recent_tasks]

        # Find clusters of similar recent tasks
        from collections import Counter
        recent_words = []
        for text in task_texts:
            words = [w.lower() for w in text.split() if len(w) > 3]
            recent_words.extend(words)

        if recent_words:
            common_themes = Counter(recent_words).most_common(3)
            for theme, count in common_themes:
                if count >= 2:
                    suggestions.append({
                        'task': f'Consider another {theme}-related task',
                        'reason': f'You\'ve been focusing on {theme} lately',
                        'confidence': min(count / 5.0, 0.6),
                        'type': 'theme_based'
                    })

        return suggestions

    def _get_default_suggestions(self, current_time):
        """Default suggestions for new users"""
        hour = current_time.hour

        if 6 <= hour < 10:
            return [
                {'task': 'Plan your day', 'reason': 'Good morning routine', 'confidence': 0.6, 'type': 'default'},
                {'task': 'Check your calendar', 'reason': 'Start day organized', 'confidence': 0.5, 'type': 'default'}
            ]
        elif 10 <= hour < 12:
            return [
                {'task': 'Focus on important work', 'reason': 'Peak productivity time', 'confidence': 0.7,
                 'type': 'default'}
            ]
        elif 12 <= hour < 17:
            return [
                {'task': 'Follow up on pending items', 'reason': 'Good time for follow-ups', 'confidence': 0.6,
                 'type': 'default'}
            ]
        elif 17 <= hour < 21:
            return [
                {'task': 'Plan tomorrow', 'reason': 'Evening planning', 'confidence': 0.7, 'type': 'default'}
            ]
        else:
            return [
                {'task': 'Prepare for tomorrow', 'reason': 'Wind down routine', 'confidence': 0.5, 'type': 'default'}
            ]

    def get_smart_completions(self, partial_text, user_id, limit=3):
        """Get smart completions based on learned patterns"""
        if len(partial_text) < 3:
            return []

        # Get similar tasks from learned data
        similar_tasks = self.learner.get_similar_tasks(partial_text, user_id, limit=limit)

        completions = []
        for task_data in similar_tasks:
            if task_data['similarity'] > 0.5:
                completions.append({
                    'completion': task_data['text'],
                    'confidence': task_data['similarity'],
                    'type': 'learned_completion'
                })

        return completions
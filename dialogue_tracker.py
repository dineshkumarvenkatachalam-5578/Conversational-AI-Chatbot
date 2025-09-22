from typing import Dict, Any, List

class DialogueStateTracker:
    """
    Manages the state of a conversation.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.slots: Dict[str, Any] = {}
        self.turn_count = 0
        self.history: List[Dict[str, Any]] = []

    def update_state(self, analysis_result: Dict[str, Any]):
        """
        Updates the dialogue state with the latest analysis.

        Args:
            analysis_result (Dict[str, Any]): The output from the SmartReplySystem's process_input method.
        """
        self.turn_count += 1
        self.history.append(analysis_result)

        if 'intent_detected' in analysis_result:
            self.slots['last_intent'] = analysis_result['intent_detected']

        if 'topics_detected' in analysis_result and analysis_result['topics_detected']:
           
            if 'all_topics' not in self.slots:
                self.slots['all_topics'] = []
            
            for topic in analysis_result['topics_detected']:
                if topic not in self.slots['all_topics']:
                    self.slots['all_topics'].append(topic)
            
            self.slots['last_topic'] = analysis_result['topics_detected'][0]

        if 'entities' in analysis_result:
            for entity in analysis_result['entities']:
                if entity['label'] == 'PERSON' and 'mentioned_people' not in self.slots:
                    self.slots['mentioned_people'] = entity['text']
    
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the conversation.
        """
        return {
            'user_id': self.user_id,
            'turn_count': self.turn_count,
            'slots': self.slots,
            'history_preview': [turn['user_input'] for turn in self.history[-5:]] # Preview of last 5 inputs
        }

    def clear_state(self):
        """Resets the state for a new conversation."""
        self.slots = {}
        self.turn_count = 0
        self.history = []
        print(f"[DialogueTracker] State cleared for user {self.user_id}.")

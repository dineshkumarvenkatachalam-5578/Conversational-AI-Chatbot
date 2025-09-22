import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

class MongoDBRAGSystem:
    """
    MongoDB-based RAG (Retrieval-Augmented Generation) system for storing and retrieving
    AI conversations, real-time data, and reinforcement learning data.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # MongoDB setup
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DATABASE", "ai_conversation_db")
        
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            # Collections
            self.conversations = self.db.conversations
            self.real_time_data = self.db.real_time_data
            self.rl_data = self.db.reinforcement_learning
            self.user_preferences = self.db.user_preferences
            
            print("✓ MongoDB connection established")
        except Exception as e:
            print(f"[WARNING] MongoDB connection failed: {e}")
            print("[INFO] System will continue without MongoDB features")
            self.client = None
        
        # Vector database setup (ChromaDB)
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.conversation_collection = self.chroma_client.get_or_create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
            self.knowledge_collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            print("✓ ChromaDB vector database initialized")
        except Exception as e:
            print(f"[ERROR] ChromaDB initialization failed: {e}")
            self.chroma_client = None
        
        # Embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedding model loaded for RAG system")
        except Exception as e:
            print(f"[ERROR] Embedding model loading failed: {e}")
            self.embedding_model = None
    
    def store_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """
        Store conversation data in both MongoDB and vector database.
        
        Args:
            conversation_data: Dictionary containing conversation information
            
        Returns:
            Boolean indicating success
        """
        if not self.client or not self.embedding_model:
            return False
        
        try:
            # Add metadata
            conversation_data.update({
                'user_id': self.user_id,
                'timestamp': datetime.now(),
                'stored_at': datetime.now().isoformat()
            })
            
            # Store in MongoDB
            result = self.conversations.insert_one(conversation_data)
            doc_id = str(result.inserted_id)
            
            # Create embedding and store in ChromaDB
            if self.chroma_client:
                text_content = f"{conversation_data.get('user_input', '')} {conversation_data.get('selected_reply', '')}"
                embedding = self.embedding_model.encode([text_content])[0].tolist()
                
                self.conversation_collection.add(
                    embeddings=[embedding],
                    documents=[text_content],
                    metadatas=[{
                        'user_id': self.user_id,
                        'mongo_id': doc_id,
                        'timestamp': conversation_data['stored_at'],
                        'intent': conversation_data.get('intent_detected', ''),
                        'sentiment': conversation_data.get('sentiment_label', ''),
                        'topics': json.dumps(conversation_data.get('topics_detected', []))
                    }],
                    ids=[doc_id]
                )
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to store conversation: {e}")
            return False
    
    def store_real_time_data(self, data: Dict[str, Any]) -> bool:
        """
        Store real-time web search data.
        
        Args:
            data: Real-time data from web search
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            return False
        
        try:
            data.update({
                'user_id': self.user_id,
                'stored_at': datetime.now(),
                'data_type': 'real_time_web_search'
            })
            
            # Store in MongoDB
            result = self.real_time_data.insert_one(data)
            
            # Store in vector database for retrieval
            if self.chroma_client and self.embedding_model:
                # Create searchable text from the data
                text_content = self._extract_text_from_data(data)
                embedding = self.embedding_model.encode([text_content])[0].tolist()
                
                self.knowledge_collection.add(
                    embeddings=[embedding],
                    documents=[text_content],
                    metadatas=[{
                        'user_id': self.user_id,
                        'mongo_id': str(result.inserted_id),
                        'data_type': 'real_time_data',
                        'topic': data.get('topic', ''),
                        'timestamp': data['stored_at'].isoformat()
                    }],
                    ids=[str(result.inserted_id)]
                )
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to store real-time data: {e}")
            return False
    
    def store_rl_data(self, observation: Dict[str, Any], action: str, reward: float) -> bool:
        """
        Store reinforcement learning data.
        
        Args:
            observation: The state/observation data
            action: Action taken by the AI
            reward: Reward received for the action
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            return False
        
        try:
            rl_entry = {
                'user_id': self.user_id,
                'observation': observation,
                'action': action,
                'reward': reward,
                'timestamp': datetime.now(),
                'stored_at': datetime.now().isoformat()
            }
            
            self.rl_data.insert_one(rl_entry)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to store RL data: {e}")
            return False
    
    def retrieve_similar_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar conversations using vector similarity search.
        
        Args:
            query: Query text to find similar conversations
            limit: Maximum number of results to return
            
        Returns:
            List of similar conversation data
        """
        if not self.chroma_client or not self.embedding_model:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in conversation collection
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"user_id": self.user_id}
            )
            
            # Fetch full data from MongoDB
            similar_conversations = []
            if results['ids']:
                for mongo_id in results['ids'][0]:
                    try:
                        from bson import ObjectId
                        doc = self.conversations.find_one({'_id': ObjectId(mongo_id)})
                        if doc:
                            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                            similar_conversations.append(doc)
                    except Exception as e:
                        print(f"[WARNING] Could not fetch conversation {mongo_id}: {e}")
            
            return similar_conversations
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve similar conversations: {e}")
            return []
    
    def retrieve_relevant_knowledge(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant real-time data and knowledge.
        
        Args:
            query: Query text to find relevant knowledge
            limit: Maximum number of results to return
            
        Returns:
            List of relevant knowledge data
        """
        if not self.chroma_client or not self.embedding_model:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in knowledge collection
            results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"user_id": self.user_id}
            )
            
            # Fetch full data from MongoDB
            relevant_knowledge = []
            if results['ids']:
                for mongo_id in results['ids'][0]:
                    try:
                        from bson import ObjectId
                        doc = self.real_time_data.find_one({'_id': ObjectId(mongo_id)})
                        if doc:
                            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                            relevant_knowledge.append(doc)
                    except Exception as e:
                        print(f"[WARNING] Could not fetch knowledge {mongo_id}: {e}")
            
            return relevant_knowledge
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve relevant knowledge: {e}")
            return []
    
    def get_user_learning_patterns(self) -> Dict[str, Any]:
        """
        Analyze user's conversation patterns and preferences for personalization.
        
        Returns:
            Dictionary containing user learning patterns
        """
        if not self.client:
            return {}
        
        try:
            # Aggregate conversation data
            pipeline = [
                {"$match": {"user_id": self.user_id}},
                {"$group": {
                    "_id": None,
                    "total_conversations": {"$sum": 1},
                    "avg_engagement": {"$avg": "$engagement_score"},
                    "common_intents": {"$push": "$intent_detected"},
                    "common_topics": {"$push": "$topics_detected"},
                    "sentiment_distribution": {"$push": "$sentiment_label"}
                }}
            ]
            
            result = list(self.conversations.aggregate(pipeline))
            
            if result:
                data = result[0]
                
                # Process common patterns
                from collections import Counter
                intent_counts = Counter(data.get('common_intents', []))
                topic_counts = Counter([topic for topics in data.get('common_topics', []) if topics for topic in topics])
                sentiment_counts = Counter(data.get('sentiment_distribution', []))
                
                return {
                    'total_conversations': data.get('total_conversations', 0),
                    'average_engagement': data.get('avg_engagement', 0),
                    'preferred_intents': dict(intent_counts.most_common(5)),
                    'preferred_topics': dict(topic_counts.most_common(10)),
                    'sentiment_pattern': dict(sentiment_counts),
                    'last_updated': datetime.now().isoformat()
                }
            
            return {}
            
        except Exception as e:
            print(f"[ERROR] Failed to get user learning patterns: {e}")
            return {}
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences based on learning patterns.
        
        Args:
            preferences: Dictionary of user preferences
            
        Returns:
            Boolean indicating success
        """
        if not self.client:
            return False
        
        try:
            preferences.update({
                'user_id': self.user_id,
                'updated_at': datetime.now()
            })
            
            self.user_preferences.update_one(
                {'user_id': self.user_id},
                {'$set': preferences},
                upsert=True
            )
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update user preferences: {e}")
            return False
    
    def _extract_text_from_data(self, data: Dict[str, Any]) -> str:
        """
        Extract searchable text content from data structure.
        
        Args:
            data: Data dictionary
            
        Returns:
            Concatenated text content
        """
        text_parts = []
        
        if 'topic' in data:
            text_parts.append(data['topic'])
        
        if 'results' in data:
            for result in data['results']:
                if 'title' in result:
                    text_parts.append(result['title'])
                if 'snippet' in result:
                    text_parts.append(result['snippet'])
                if 'name' in result:
                    text_parts.append(result['name'])
        
        return ' '.join(text_parts)
    
    def close_connections(self):
        """Close database connections."""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

Base = declarative_base()

class User(Base):
    """
    User model for platform authentication and settings.
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class Post(Base):
    """
    Social Media Post model containing raw/cleaned text and classifications.
    """
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text)
    platform = Column(String(50), index=True) # e.g., reddit, youtube, x, manual, file
    username = Column(String(100), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Classifications
    sentiment = Column(String(20), index=True) # Positive, Negative, Neutral
    sentiment_score = Column(Float)            # Polarity score (-1 to 1)
    emotion = Column(String(50), index=True)   # Joy, Anger, Fear, Sadness, etc.
    topic = Column(String(100), index=True)    # Modeled topic name or id
    confidence = Column(Float, default=1.0)
    
    # Advanced Detections
    is_fake = Column(Boolean, default=False)
    is_toxic = Column(Boolean, default=False)
    is_sarcastic = Column(Boolean, default=False)
    is_spam = Column(Boolean, default=False)
    is_bot = Column(Boolean, default=False)
    
    # Metadata
    raw_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisRun(Base):
    """
    Model tracking historical batch analysis runs.
    """
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    dataset_name = Column(String(255))
    num_posts = Column(Integer)
    average_sentiment = Column(Float)
    brand_reputation_score = Column(Float)
    csat_score = Column(Float)
    topics_summary = Column(JSON, default=dict)

# Database Engine and Session Management
class DBManager:
    def __init__(self, db_url: str = "sqlite:///data/sentiverse.db"):
        self.db_url = db_url
        
        # Ensure directories exist for sqlite
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            dir_name = os.path.dirname(db_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                logger.info(f"Created directory for SQLite database: {dir_name}")
                
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info(f"Database initialized successfully with URL: {self.db_url}")
        except Exception as e:
            logger.error(f"Error initializing database engine: {e}")
            raise e
            
    def get_session(self):
        """Returns a new DB session."""
        return self.SessionLocal()

    # User Auth Helpers
    def create_user(self, username: str, password: str, role: str = "user") -> Optional[User]:
        session = self.get_session()
        try:
            # Check if user exists
            existing_user = session.query(User).filter(User.username == username).first()
            if existing_user:
                logger.warning(f"User '{username}' already exists.")
                return None
            
            user = User(username=username, role=role)
            user.set_password(password)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"User '{username}' created successfully.")
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating user: {e}")
            return None
        finally:
            session.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user and user.check_password(password):
                logger.info(f"User '{username}' authenticated successfully.")
                return user
            logger.warning(f"Failed authentication for user '{username}'.")
            return None
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
        finally:
            session.close()

    # Post Helpers
    def add_posts_batch(self, posts_data: List[Dict[str, Any]]) -> bool:
        session = self.get_session()
        try:
            db_posts = []
            for p in posts_data:
                # Parse timestamp if string
                ts = p.get('timestamp', datetime.utcnow())
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError:
                        ts = datetime.utcnow()
                
                db_post = Post(
                    text=p['text'],
                    cleaned_text=p.get('cleaned_text', ''),
                    platform=p.get('platform', 'manual'),
                    username=p.get('username', 'anonymous'),
                    timestamp=ts,
                    sentiment=p.get('sentiment', 'Neutral'),
                    sentiment_score=p.get('sentiment_score', 0.0),
                    emotion=p.get('emotion', 'Neutral'),
                    topic=p.get('topic', 'General'),
                    confidence=p.get('confidence', 1.0),
                    is_fake=p.get('is_fake', False),
                    is_toxic=p.get('is_toxic', False),
                    is_sarcastic=p.get('is_sarcastic', False),
                    is_spam=p.get('is_spam', False),
                    is_bot=p.get('is_bot', False),
                    raw_metadata=p.get('raw_metadata', {})
                )
                db_posts.append(db_post)
            session.add_all(db_posts)
            session.commit()
            logger.info(f"Successfully added {len(db_posts)} posts to database.")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding posts batch: {e}")
            return False
        finally:
            session.close()

    def search_posts(self, 
                     keyword: Optional[str] = None, 
                     platform: Optional[str] = None,
                     sentiment: Optional[str] = None, 
                     emotion: Optional[str] = None, 
                     topic: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 100) -> List[Post]:
        session = self.get_session()
        try:
            query = session.query(Post)
            if keyword:
                query = query.filter(Post.text.ilike(f"%{keyword}%") | Post.cleaned_text.ilike(f"%{keyword}%"))
            if platform:
                query = query.filter(Post.platform == platform)
            if sentiment:
                query = query.filter(Post.sentiment == sentiment)
            if emotion:
                query = query.filter(Post.emotion == emotion)
            if topic:
                query = query.filter(Post.topic == topic)
            if start_date:
                query = query.filter(Post.timestamp >= start_date)
            if end_date:
                query = query.filter(Post.timestamp <= end_date)
                
            query = query.order_by(Post.timestamp.desc()).limit(limit)
            return query.all()
        except Exception as e:
            logger.error(f"Error querying posts: {e}")
            return []
        finally:
            session.close()

    def clear_all_posts(self) -> bool:
        session = self.get_session()
        try:
            session.query(Post).delete()
            session.commit()
            logger.info("Cleared all posts from database.")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing posts: {e}")
            return False
        finally:
            session.close()

    # Analysis Runs Helpers
    def save_analysis_run(self, run_data: Dict[str, Any]) -> Optional[AnalysisRun]:
        session = self.get_session()
        try:
            run = AnalysisRun(
                dataset_name=run_data.get('dataset_name', 'Unnamed Dataset'),
                num_posts=run_data.get('num_posts', 0),
                average_sentiment=run_data.get('average_sentiment', 0.0),
                brand_reputation_score=run_data.get('brand_reputation_score', 0.0),
                csat_score=run_data.get('csat_score', 0.0),
                topics_summary=run_data.get('topics_summary', {})
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            logger.info(f"Saved analysis run {run.id} for dataset {run.dataset_name}.")
            return run
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving analysis run: {e}")
            return None
        finally:
            session.close()

    def get_analysis_runs(self) -> List[AnalysisRun]:
        session = self.get_session()
        try:
            return session.query(AnalysisRun).order_by(AnalysisRun.timestamp.desc()).all()
        except Exception as e:
            logger.error(f"Error retrieving analysis runs: {e}")
            return []
        finally:
            session.close()

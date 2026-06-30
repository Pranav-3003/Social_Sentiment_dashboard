import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("BIEngine")

class BIEngine:
    def __init__(self):
        pass

    def calculate_brand_reputation_score(self, df: pd.DataFrame) -> float:
        """
        Calculates a Brand Reputation Score between 0 and 100.
        Uses exponential time decay so newer posts have more weight.
        Formula: 50 * (WeightedPositive - WeightedNegative) / WeightedTotal + 50
        """
        if df.empty or 'sentiment' not in df.columns:
            return 50.0
            
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        
        # Calculate time difference in days from the newest post
        newest_post = temp_df['timestamp'].max()
        temp_df['age_days'] = (newest_post - temp_df['timestamp']).dt.total_seconds() / (3600 * 24.0)
        
        # Apply exponential decay weight (lambda = 7 days half-life)
        # Weight = 0.5 ^ (age_days / 7)
        temp_df['weight'] = 0.5 ** (temp_df['age_days'] / 7.0)
        
        # Calculate weighted sentiment scores
        pos_weight = temp_df[temp_df['sentiment'] == 'Positive']['weight'].sum()
        neg_weight = temp_df[temp_df['sentiment'] == 'Negative']['weight'].sum()
        neu_weight = temp_df[temp_df['sentiment'] == 'Neutral']['weight'].sum()
        
        total_weight = pos_weight + neg_weight + neu_weight
        
        if total_weight == 0:
            return 50.0
            
        reputation = 50.0 * (pos_weight - neg_weight) / total_weight + 50.0
        return round(reputation, 1)

    def calculate_csat_score(self, df: pd.DataFrame) -> float:
        """
        Calculates Estimated CSAT score (0% - 100%) based on positive sentiment ratio.
        CSAT = Positive / (Positive + Negative) * 100
        """
        if df.empty or 'sentiment' not in df.columns:
            return 0.0
            
        pos_count = len(df[df['sentiment'] == 'Positive'])
        neg_count = len(df[df['sentiment'] == 'Negative'])
        
        total = pos_count + neg_count
        if total == 0:
            return 50.0 # Return neutral baseline if only neutral posts
            
        csat = (pos_count / total) * 100.0
        return round(csat, 1)

    def detect_crisis(self, df: pd.DataFrame, window_hours: int = 24) -> Tuple[bool, str]:
        """
        Triggers a crisis alert if negative sentiment spikes significantly in the last window.
        Specifically, check if negative sentiment ratio in the last window exceeds historical mean by 2 std devs.
        """
        if len(df) < 15 or 'sentiment' not in df.columns:
            return False, "Insufficient historical data to calculate crisis thresholds."
            
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        temp_df = temp_df.sort_values(by='timestamp')
        
        cutoff = temp_df['timestamp'].max() - timedelta(hours=window_hours)
        recent_posts = temp_df[temp_df['timestamp'] >= cutoff]
        historical_posts = temp_df[temp_df['timestamp'] < cutoff]
        
        if recent_posts.empty or historical_posts.empty:
            return False, "No recent or historical dividing line found."
            
        # Resample historical posts by day to check standard deviation of daily negative ratios
        historical_posts = historical_posts.set_index('timestamp')
        daily_neg_ratio = historical_posts.resample('D').apply(
            lambda x: len(x[x['sentiment'] == 'Negative']) / len(x) if len(x) > 0 else 0.0
        )
        
        mean_neg = daily_neg_ratio.mean()
        std_neg = daily_neg_ratio.std()
        # Fallback if std dev is 0
        if pd.isna(std_neg) or std_neg == 0:
            std_neg = 0.05
            
        recent_neg_ratio = len(recent_posts[recent_posts['sentiment'] == 'Negative']) / len(recent_posts)
        
        threshold = mean_neg + (2.0 * std_neg)
        
        if recent_neg_ratio > threshold and recent_neg_ratio > 0.25:
            # Crisis alert
            pct_increase = round((recent_neg_ratio - mean_neg) * 100, 1)
            return True, f"CRITICAL SPIKE: Negative sentiment ratio is {round(recent_neg_ratio*100, 1)}% in the last {window_hours} hours. This is {pct_increase}% higher than the historical baseline."
            
        return False, "Sentiment is within normal operating ranges."

    def extract_top_feedback(self, df: pd.DataFrame, count: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts top positive (highest sentiment score) and top negative (lowest sentiment score) posts.
        """
        if df.empty or 'sentiment_score' not in df.columns:
            return {"positive": [], "negative": []}
            
        sorted_df = df.sort_values(by='sentiment_score')
        
        negatives = sorted_df[sorted_df['sentiment'] == 'Negative'].head(count)
        positives = sorted_df[sorted_df['sentiment'] == 'Positive'].tail(count).iloc[::-1] # Reverse tail to get descending
        
        neg_list = []
        for _, row in negatives.iterrows():
            neg_list.append({
                "text": row['text'],
                "username": row.get('username', 'anonymous'),
                "platform": row.get('platform', 'unknown'),
                "score": round(row['sentiment_score'], 2)
            })
            
        pos_list = []
        for _, row in positives.iterrows():
            pos_list.append({
                "text": row['text'],
                "username": row.get('username', 'anonymous'),
                "platform": row.get('platform', 'unknown'),
                "score": round(row['sentiment_score'], 2)
            })
            
        return {
            "positive": pos_list,
            "negative": neg_list
        }

    def generate_product_suggestions(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Auto-generates product suggestions based on keyword clustering of complaints.
        """
        suggestions = []
        if df.empty:
            return suggestions
            
        # Filter negative comments
        neg_df = df[df['sentiment'] == 'Negative']
        if neg_df.empty:
            # If no complaints, yield general maintenance tips
            return [
                {
                    "category": "Customer Relations",
                    "issue": "No negative feedback detected recently.",
                    "action": "Maintain high quality standards and proactively gather feature requests."
                }
            ]
            
        # Group issues by keyword matching
        issue_keywords = {
            "Performance": ["lag", "slow", "crash", "freeze", "load", "error", "bug", "broken", "run"],
            "Pricing/Billing": ["price", "cost", "expensive", "subscription", "charge", "refund", "billing", "money"],
            "UI/UX Design": ["layout", "ui", "ux", "font", "color", "confusing", "ugly", "buttons", "navigate"],
            "Customer Support": ["support", "ticket", "response", "reply", "help", "email", "chat", "ignore"]
        }
        
        counts = {cat: 0 for cat in issue_keywords}
        cat_texts = {cat: [] for cat in issue_keywords}
        
        for _, row in neg_df.iterrows():
            text = row['text'].lower()
            for cat, kw_list in issue_keywords.items():
                if any(kw in text for kw in kw_list):
                    counts[cat] += 1
                    cat_texts[cat].append(row['text'])
                    
        # Formulate actionable advice
        for cat, count in counts.items():
            if count > 0:
                pct = round(count / len(neg_df) * 100, 1)
                
                if cat == "Performance":
                    action = "Allocate engineering sprint to optimize database queries, implement caching, and resolve frontend rendering latencies."
                elif cat == "Pricing/Billing":
                    action = "Review package models. Introduce tiered discounts, simplify the cancellation flow, and audit payment gateway endpoints."
                elif cat == "UI/UX Design":
                    action = "Conduct user usability testing, streamline main action paths, and provide a customizable dark/light theme mode."
                else: # Customer Support
                    action = "Deploy chatbot helpers for basic queries, increase customer service headcount, and audit ticketing SLA queue times."
                    
                suggestions.append({
                    "category": cat,
                    "issue": f"{count} complaints ({pct}% of negative feedback) mention this issue.",
                    "action": action
                })
                
        if not suggestions:
            suggestions.append({
                "category": "General Quality",
                "issue": f"Gathered {len(neg_df)} complaints without clear category patterns.",
                "action": "Conduct exploratory topic modeling on negative sentiment clusters to extract specific complaints."
            })
            
        return suggestions

    def generate_executive_summary(self, df: pd.DataFrame) -> str:
        """
        Generates a text-based Executive Summary for the business intelligence report.
        """
        if df.empty:
            return "No data available to generate executive summary."
            
        total = len(df)
        reputation = self.calculate_brand_reputation_score(df)
        csat = self.calculate_csat_score(df)
        
        pos_pct = round(len(df[df['sentiment'] == 'Positive']) / total * 100, 1)
        neg_pct = round(len(df[df['sentiment'] == 'Negative']) / total * 100, 1)
        neu_pct = round(len(df[df['sentiment'] == 'Neutral']) / total * 100, 1)
        
        # Dominant platform
        platforms = df['platform'].value_counts()
        top_platform = platforms.index[0] if not platforms.empty else "unknown"
        top_platform_count = platforms.values[0] if not platforms.empty else 0
        
        summary = (
            f"EXECUTIVE ANALYSIS SUMMARY\n"
            f"==========================\n"
            f"This intelligence report provides an executive summary of social media data collected across active platforms.\n\n"
            f"Key Performance Indicators:\n"
            f"- Total analyzed documents: {total}\n"
            f"- Brand Reputation Score: {reputation}/100\n"
            f"- Estimated Customer Satisfaction (CSAT): {csat}%\n\n"
            f"Sentiment Distribution:\n"
            f"- Positive sentiments: {pos_pct}%\n"
            f"- Neutral sentiments: {neu_pct}%\n"
            f"- Negative sentiments: {neg_pct}%\n\n"
            f"Data Source Ingestion Insights:\n"
            f"The primary channel of consumer conversation is {top_platform.upper()} contributing {top_platform_count} posts ({round(top_platform_count/total*100, 1)}% of total volume).\n\n"
        )
        
        # Sentiment assessment
        if reputation > 75:
            summary += "Strategic Assessment: The company holds a highly favorable market standing. Brand advocacy is strong. Focus should remain on expansion and amplifying positive customer voices."
        elif reputation > 50:
            summary += "Strategic Assessment: The brand maintains a stable and neutral market standing. While satisfaction is moderate, there is room to convert neutral customers into advocates by improving onboarding."
        else:
            summary += "Strategic Assessment: WARNING. Brand health is currently degraded due to elevated negative discussions. Urgent focus is required on addressing customer complaints and restoring trust."
            
        return summary

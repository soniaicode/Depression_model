"""
Gemini AI Integration for Personalized Therapy Guidance
Provides spiritual and therapeutic recommendations based on depression risk level
"""

import os
import google.generativeai as genai
from typing import Dict, Optional

class TherapyGuidanceGenerator:
    """Generate personalized therapy guidance using Gemini AI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI
        
        Args:
            api_key: Gemini API key (if None, reads from environment)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("⚠ Warning: GEMINI_API_KEY not set. Therapy guidance will use fallback responses.")
            self.enabled = False
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.enabled = True
                print("✓ Gemini AI initialized for therapy guidance")
            except Exception as e:
                print(f"⚠ Gemini AI initialization failed: {e}")
                self.enabled = False
    
    def generate_spiritual_guidance(self, risk_level: str, user_data: Dict) -> str:
        """
        Generate spiritual guidance based on risk level
        
        Args:
            risk_level: Depression risk level (Low, Moderate, High, Very High)
            user_data: User questionnaire data
            
        Returns:
            Spiritual guidance message
        """
        if not self.enabled:
            return self._get_fallback_spiritual_guidance(risk_level)
        
        try:
            prompt = f"""
You are a compassionate spiritual counselor providing guidance to someone with {risk_level} depression risk.

Provide warm, empathetic spiritual guidance that:
1. Acknowledges their feelings with compassion
2. Offers hope and encouragement
3. Suggests spiritual practices (meditation, prayer, mindfulness)
4. Emphasizes inner strength and resilience
5. Encourages connection with something greater (nature, community, faith)

Keep it brief (3-4 sentences), warm, and non-denominational.
Focus on inner peace, hope, and spiritual wellness.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Error generating spiritual guidance: {e}")
            return self._get_fallback_spiritual_guidance(risk_level)
    
    def generate_therapy_recommendations(self, risk_level: str, user_data: Dict) -> list:
        """
        Generate personalized therapy recommendations
        
        Args:
            risk_level: Depression risk level
            user_data: User questionnaire data
            
        Returns:
            List of therapy recommendations
        """
        if not self.enabled:
            return self._get_fallback_therapy_recommendations(risk_level)
        
        try:
            prompt = f"""
Generate 5 specific, actionable therapy recommendations for someone with {risk_level} depression risk.

Format as a simple list. Each recommendation should be:
- Specific and actionable
- Evidence-based
- Appropriate for the risk level
- Encouraging and supportive

Examples:
- Consider cognitive behavioral therapy (CBT) with a licensed therapist
- Practice daily mindfulness meditation for 10-15 minutes
- Establish a consistent sleep schedule (7-9 hours)
- Engage in regular physical activity (30 minutes, 3-5 times per week)
- Connect with supportive friends or family members weekly
"""
            
            response = self.model.generate_content(prompt)
            recommendations = response.text.strip().split('\n')
            # Clean up the list
            recommendations = [r.strip('- ').strip() for r in recommendations if r.strip()]
            return recommendations[:5]
            
        except Exception as e:
            print(f"Error generating therapy recommendations: {e}")
            return self._get_fallback_therapy_recommendations(risk_level)
    
    def _get_fallback_spiritual_guidance(self, risk_level: str) -> str:
        """Fallback spiritual guidance when AI is not available"""
        guidance = {
            'Low': "You're on a positive path. Continue nurturing your inner peace through mindfulness and gratitude. Your resilience shines through, and maintaining spiritual practices will keep you grounded.",
            'Moderate': "Remember that challenges are opportunities for growth. Take time each day for quiet reflection, prayer, or meditation. You have inner strength that can guide you through this time.",
            'High': "You are not alone in this journey. Reach out to your spiritual community, practice self-compassion, and remember that healing is possible. Your spirit is stronger than you know.",
            'Very High': "Please know that you are valued and your life has meaning. Seek immediate support from loved ones, spiritual leaders, or mental health professionals. There is hope, and help is available."
        }
        return guidance.get(risk_level, guidance['Moderate'])
    
    def _get_fallback_therapy_recommendations(self, risk_level: str) -> list:
        """Fallback therapy recommendations when AI is not available"""
        recommendations = {
            'Low': [
                "Continue your current wellness practices",
                "Practice daily gratitude journaling",
                "Maintain regular social connections",
                "Engage in activities you enjoy",
                "Keep a consistent sleep schedule"
            ],
            'Moderate': [
                "Consider speaking with a therapist or counselor",
                "Practice cognitive behavioral therapy (CBT) techniques",
                "Establish a daily routine with self-care activities",
                "Increase physical activity to 30 minutes daily",
                "Connect with supportive friends or support groups"
            ],
            'High': [
                "Seek professional mental health support immediately",
                "Consider therapy (CBT, DBT, or psychotherapy)",
                "Discuss medication options with a psychiatrist",
                "Build a strong support network",
                "Practice crisis management techniques"
            ],
            'Very High': [
                "Contact a mental health professional immediately",
                "Call crisis hotline: 988 (Suicide & Crisis Lifeline)",
                "Do not isolate - stay with trusted friends/family",
                "Consider intensive outpatient or inpatient treatment",
                "Create a safety plan with your healthcare provider"
            ]
        }
        return recommendations.get(risk_level, recommendations['Moderate'])

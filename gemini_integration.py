"""
Gemini AI Integration for Depression Prediction System
Provides AI-powered explanations and insights
"""

import os
import google.generativeai as genai
from typing import Dict, Any

class GeminiAssistant:
    """Gemini AI Assistant for mental health insights"""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini AI"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Using Gemini 2.0 Flash Experimental - Latest and fastest model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.enabled = True
            print("✓ Gemini AI initialized (gemini-2.0-flash)")
        else:
            self.enabled = False
            print("⚠ Gemini API key not found - AI insights disabled")
    
    def generate_prediction_explanation(self, 
                                       prediction: int,
                                       probability: Dict[str, float],
                                       risk_level: str,
                                       questionnaire_data: Dict[str, Any]) -> str:
        """
        Generate detailed explanation for prediction result
        
        Args:
            prediction: 0 (no depression) or 1 (depression)
            probability: Dictionary with probabilities
            risk_level: Low, Moderate, High, or Very High
            questionnaire_data: User's questionnaire responses
            
        Returns:
            AI-generated explanation text
        """
        if not self.enabled:
            return self._get_default_explanation(prediction, risk_level)
        
        try:
            # Extract key indicators from questionnaire
            phq9_scores = [float(questionnaire_data.get(f'feature_{i}', 0)) for i in range(9)]
            phq9_total = sum(phq9_scores)
            
            sleep_duration = float(questionnaire_data.get('feature_17', 7))
            sleep_quality = float(questionnaire_data.get('feature_18', 5))
            stress_level = float(questionnaire_data.get('feature_20', 5))
            social_support = float(questionnaire_data.get('feature_25', 2))
            
            # Create prompt for Gemini
            prompt = f"""
You are a compassionate mental health AI assistant with understanding of holistic wellness including spiritual and emotional healing.

SCREENING RESULTS:
- Prediction: {"Depression indicators detected" if prediction == 1 else "No significant depression indicators"}
- Depression Probability: {probability['depression']*100:.1f}%
- Risk Level: {risk_level}
- PHQ-9 Total Score: {phq9_total:.0f}/27

KEY INDICATORS:
- Sleep Duration: {sleep_duration:.1f} hours/night
- Sleep Quality: {sleep_quality:.0f}/10
- Stress Level: {stress_level:.0f}/10
- Social Support: {social_support:.0f}/3

TASK:
Provide a brief (150-200 words), empathetic explanation using MARKDOWN formatting:

Structure your response as:
1. Opening paragraph interpreting the results
2. **Key Contributing Factors:** (use bold heading)
   - List 2-3 factors as bullet points
3. **Recommendations:** (use bold heading)
   - List 2-3 practical recommendations as bullet points
   - Include spiritual/mindfulness practices
4. Closing paragraph with hope and encouragement

Use markdown formatting:
- **Bold** for headings and emphasis
- Bullet points (-) for lists
- *Italic* for gentle emphasis
- Clear paragraph breaks

Tone: Warm, supportive, spiritually aware, honest but hopeful.
Include: Inner peace, mindfulness, spiritual practices, self-compassion, healing journey.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_default_explanation(prediction, risk_level)
    
    def generate_wellness_tips(self, risk_level: str, questionnaire_data: Dict[str, Any]) -> list:
        """
        Generate personalized wellness tips based on screening results
        
        Args:
            risk_level: Low, Moderate, High, or Very High
            questionnaire_data: User's responses
            
        Returns:
            List of personalized wellness tips
        """
        if not self.enabled:
            return self._get_default_tips(risk_level)
        
        try:
            prompt = f"""
Generate 6 personalized, holistic wellness tips for someone with {risk_level} depression risk.

Include tips covering:
1. 🙏 Spiritual practice (meditation, prayer, mindfulness, gratitude)
2. 💪 Physical wellness (sleep, exercise, nutrition)
3. 💖 Emotional healing (self-compassion, journaling, therapy)
4. 🤝 Social connection (relationships, community, support)
5. 🧘 Mind-body practices (yoga, breathing, nature connection)
6. 🎯 Purpose and meaning (values, goals, service to others)

IMPORTANT FORMATTING:
- Start each tip with an emoji (🙏 💪 💖 🤝 🧘 🎯)
- Use **bold** for key actions or practices
- Keep each tip to 1-2 sentences
- Make tips specific and actionable
- Use encouraging, supportive language

Example format:
🙏 Practice **daily meditation** or prayer for 10-15 minutes to connect with your inner peace and spiritual center.

Generate 6 tips following this format.
"""
            
            response = self.model.generate_content(prompt)
            tips_text = response.text
            
            # Parse tips from response
            tips = []
            for line in tips_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    tip = line.lstrip('0123456789.-•) ').strip()
                    if tip:
                        tips.append(tip)
            
            return tips[:6] if tips else self._get_default_tips(risk_level)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_default_tips(risk_level)
    
    def generate_spiritual_guidance(self, risk_level: str, questionnaire_data: Dict[str, Any]) -> str:
        """
        Generate spiritual and holistic guidance
        
        Args:
            risk_level: Low, Moderate, High, or Very High
            questionnaire_data: User's responses
            
        Returns:
            Spiritual guidance text
        """
        if not self.enabled:
            return self._get_default_spiritual_guidance(risk_level)
        
        try:
            prompt = f"""
You are a compassionate spiritual guide and holistic wellness counselor. Provide spiritual guidance for someone experiencing {risk_level} depression risk.

Create a brief (100-150 words) spiritual message using MARKDOWN formatting:

Structure:
1. Opening: Acknowledge their inner strength and divine nature
2. **Connection:** (bold heading) Remind them of connection to something greater
3. **Spiritual Practices:** (bold heading) Encourage meditation, prayer, mindfulness
4. Closing: Message of hope, healing, and transformation with affirmation

Use markdown formatting:
- **Bold** for key concepts and headings
- *Italic* for gentle emphasis
- Clear paragraph breaks
- Emojis for warmth (🙏 💖 ✨ 🌟)

Tone: Warm, uplifting, spiritually inclusive, respectful of all faiths.
Focus: Inner peace, higher self, gratitude, self-love, healing journey.

Make it personal, heartfelt, and inspiring.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_default_spiritual_guidance(risk_level)
    
    def generate_resource_recommendations(self, risk_level: str) -> Dict[str, list]:
        """
        Generate mental health resource recommendations
        
        Args:
            risk_level: Low, Moderate, High, or Very High
            
        Returns:
            Dictionary with categorized resources
        """
        if not self.enabled:
            return self._get_default_resources()
        
        try:
            prompt = f"""
For someone with {risk_level} depression risk, recommend mental health resources in these categories:

1. Professional Help (2-3 options)
2. Self-Help Tools (2-3 options)
3. Support Communities (2-3 options)
4. Crisis Resources (if risk is High/Very High)

Format as:
Category: Resource Name - Brief description

Be specific and practical. Include both online and offline options.
"""
            
            response = self.model.generate_content(prompt)
            # Parse and structure the response
            return self._parse_resources(response.text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_default_resources()
    
    def _get_default_explanation(self, prediction: int, risk_level: str) -> str:
        """Fallback explanation when Gemini is unavailable"""
        if prediction == 1:
            return f"""
Based on your responses, the screening indicates potential depression symptoms with a {risk_level} risk level. 
This suggests you may be experiencing challenges that could benefit from professional support.

Key factors contributing to this result may include sleep patterns, stress levels, and mood indicators 
from your responses. Remember, this is a screening tool designed to identify potential concerns, not a 
clinical diagnosis.

We recommend:
1. Consider scheduling an appointment with a mental health professional for a comprehensive evaluation
2. Focus on self-care activities like regular sleep, exercise, and social connection
3. Reach out to trusted friends, family, or support groups

Your mental health matters, and seeking help is a sign of strength, not weakness.
"""
        else:
            return f"""
Based on your responses, the screening shows a {risk_level} risk level for depression. This is encouraging, 
but it's important to continue monitoring your mental health and maintaining healthy habits.

Your responses suggest you're managing well in key areas like mood, sleep, and daily functioning. 
However, mental health can change over time, so staying proactive is important.

We recommend:
1. Continue your current self-care practices and healthy routines
2. Stay connected with supportive friends and family
3. Monitor your mental health regularly and seek help if you notice changes

Remember, taking care of your mental health is an ongoing process, and it's always okay to reach out 
for support when needed.
"""
    
    def _get_default_tips(self, risk_level: str) -> list:
        """Fallback tips when Gemini is unavailable"""
        return [
            "🙏 Practice daily meditation or prayer for 10-15 minutes to connect with your inner peace and spiritual center",
            "💪 Maintain a consistent sleep schedule (7-9 hours) and engage in gentle physical activity like walking in nature",
            "📝 Journal your thoughts and practice gratitude - write 3 things you're grateful for each day",
            "🤝 Stay connected with supportive friends, family, or spiritual community for emotional and social nourishment",
            "🧘 Try yoga, deep breathing, or mindfulness practices to harmonize your mind, body, and spirit",
            "💖 Practice self-compassion and self-love - treat yourself with the same kindness you'd offer a dear friend"
        ]
    
    def _get_default_spiritual_guidance(self, risk_level: str) -> str:
        """Fallback spiritual guidance when Gemini is unavailable"""
        return """
🙏 Remember, you are a beautiful soul on a journey of healing and growth. Within you lies an infinite wellspring of strength, resilience, and divine light.

This moment of challenge is also an opportunity for profound transformation. Through practices like meditation, prayer, or simply sitting in quiet reflection, you can reconnect with your inner peace and the universal love that surrounds you.

You are not alone on this path. The universe, the divine, or whatever higher power resonates with you, is always present, guiding and supporting you. Practice gratitude for the small blessings, show yourself the same compassion you'd offer a dear friend, and trust in your capacity to heal.

Your life has meaning and purpose. You are worthy of love, joy, and peace. This too shall pass, and you will emerge stronger, wiser, and more connected to your true self. 💖✨
"""
    
    def _get_default_resources(self) -> Dict[str, list]:
        """Fallback resources when Gemini is unavailable"""
        return {
            "Professional Help": [
                "Licensed Therapist - Schedule an appointment with a mental health professional",
                "Primary Care Doctor - Discuss your mental health during your next visit",
                "Psychiatrist - For medication evaluation if recommended"
            ],
            "Self-Help Tools": [
                "Mental Health Apps - Try apps like Headspace, Calm, or Moodfit",
                "Online Therapy - Consider platforms like BetterHelp or Talkspace",
                "Journaling - Write about your thoughts and feelings daily"
            ],
            "Support Communities": [
                "Support Groups - Find local or online depression support groups",
                "Online Forums - Connect with others on platforms like 7 Cups",
                "Peer Support - Reach out to trusted friends or family members"
            ],
            "Spiritual Resources": [
                "Meditation Apps - Try Insight Timer, Calm, or Headspace for guided meditation",
                "Spiritual Communities - Connect with local meditation groups, prayer circles, or spiritual centers",
                "Nature Connection - Spend time in nature for grounding and spiritual renewal"
            ]
        }
    
    def _parse_resources(self, text: str) -> Dict[str, list]:
        """Parse Gemini response into structured resources"""
        resources = {
            "Professional Help": [],
            "Self-Help Tools": [],
            "Support Communities": [],
            "Crisis Resources": []
        }
        
        current_category = None
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a category header
            for category in resources.keys():
                if category.lower() in line.lower():
                    current_category = category
                    break
            
            # Add resource to current category
            if current_category and ':' in line and not any(cat.lower() in line.lower() for cat in resources.keys()):
                resources[current_category].append(line)
        
        # Remove empty categories
        return {k: v for k, v in resources.items() if v}


# Initialize global instance
gemini_assistant = GeminiAssistant()

from openai import OpenAI
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Hugging Face client with error handling
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
client = None

if api_key:
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )
    except Exception as e:
        print(f"Failed to initialize AI client: {e}")
        client = None
else:
    print("Warning: No API key found for AI explanations. Using fallback responses.")

def generate_ai_explanation(
    prediction_result: int,
    probability: float,
    input_data: List[float],
    feature_names: List[str],
    user_context: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate AI-powered explanation and precautions using Hugging Face model
    
    Args:
        prediction_result: 0 (normal) or 1 (failure)
        probability: Failure probability (0-1)
        input_data: List of input feature values
        feature_names: List of feature names
        user_context: Additional context from user input
    
    Returns:
        Dictionary containing explanation, precautions, and recommendations
    """
    
    # Create feature summary
    feature_summary = "\n".join([
        f"{name}: {value:.2f}" for name, value in zip(feature_names, input_data)
    ])
    
    # Determine risk level
    if probability >= 0.8:
        risk_level = "CRITICAL"
    elif probability >= 0.6:
        risk_level = "HIGH"
    elif probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Create prompt for the AI model
    prompt = f"""
    As an expert industrial machine maintenance engineer, analyze the following machine failure prediction data and provide detailed explanations and precautions.

    PREDICTION RESULTS:
    - Failure Prediction: {"FAILURE" if prediction_result == 1 else "NORMAL"}
    - Failure Probability: {probability:.2%}
    - Risk Level: {risk_level}

    INPUT PARAMETERS:
    {feature_summary}

    ADDITIONAL CONTEXT:
    {user_context if user_context else "No additional context provided"}

    Please provide a comprehensive analysis including:
    1. **Risk Assessment**: Brief explanation of why this risk level was assigned
    2. **Key Risk Factors**: Identify which parameters are contributing most to the risk
    3. **Immediate Actions**: What should be done right now
    4. **Preventive Measures**: Steps to prevent future failures
    5. **Monitoring Recommendations**: What to monitor going forward
    6. **Maintenance Suggestions**: Specific maintenance advice

    Format your response as clear, actionable advice for machine operators and maintenance teams.
    Be specific and practical in your recommendations.
    """

    try:
        # Check if AI client is available
        if not client:
            raise Exception("AI client not initialized")
            
        # Call Hugging Face model
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        ai_response = response.choices[0].message.content
        
        # Parse the response into structured sections
        sections = parse_ai_response(ai_response)
        
        return {
            "risk_assessment": sections.get("Risk Assessment", "Analysis not available"),
            "key_factors": sections.get("Key Risk Factors", "Analysis not available"),
            "immediate_actions": sections.get("Immediate Actions", "No immediate actions required"),
            "preventive_measures": sections.get("Preventive Measures", "No specific preventive measures"),
            "monitoring": sections.get("Monitoring Recommendations", "Standard monitoring recommended"),
            "maintenance": sections.get("Maintenance Suggestions", "Regular maintenance recommended")
        }
        
    except Exception as e:
        # Fallback response if AI service fails
        return {
            "risk_assessment": get_fallback_risk_assessment(prediction_result, probability),
            "key_factors": "Unable to analyze specific factors due to service unavailability.",
            "immediate_actions": get_fallback_immediate_actions(prediction_result, probability),
            "preventive_measures": "Regular maintenance and monitoring recommended.",
            "monitoring": "Continue standard monitoring procedures.",
            "maintenance": "Follow scheduled maintenance guidelines."
        }

def parse_ai_response(response: str) -> Dict[str, str]:
    """
    Parse AI response into structured sections
    """
    sections = {}
    current_section = None
    current_content = []
    
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Check if this is a section header
        if line.startswith('**') and line.endswith('**'):
            # Save previous section if exists
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.replace('**', '').strip()
            current_content = []
        elif line and current_section:
            current_content.append(line)
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def get_fallback_risk_assessment(prediction_result: int, probability: float) -> str:
    """Fallback risk assessment when AI is unavailable"""
    if prediction_result == 1:
        if probability >= 0.8:
            return "Critical failure risk detected. Machine parameters indicate imminent failure possibility."
        elif probability >= 0.6:
            return "High failure risk detected. Machine operating outside safe parameters."
        else:
            return "Moderate failure risk detected. Monitor closely."
    else:
        if probability >= 0.3:
            return "Low to moderate risk detected. Machine operating normally but with some elevated parameters."
        else:
            return "Low risk detected. Machine operating within normal parameters."

def get_fallback_immediate_actions(prediction_result: int, probability: float) -> str:
    """Fallback immediate actions when AI is unavailable"""
    if prediction_result == 1 and probability >= 0.7:
        return "• Stop machine immediately\n• Perform safety inspection\n• Notify maintenance team\n• Document the incident"
    elif prediction_result == 1:
        return "• Increase monitoring frequency\n• Schedule inspection within 24 hours\n• Prepare maintenance plan"
    elif probability >= 0.5:
        return "• Continue operation with increased monitoring\n• Schedule preventive maintenance"
    else:
        return "• Continue normal operation\n• Regular monitoring"

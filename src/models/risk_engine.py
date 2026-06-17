def assess_risk(probabilities):
    """
    Assesses behavioral risk level based on model output probabilities.
    probabilities: [low_risk_prob, medium_risk_prob, high_risk_prob]
    """
    risk_levels = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
    
    # Simple thresholding logic or max probability
    predicted_index = probabilities.argmax()
    confidence = probabilities[predicted_index]
    
    risk_level = risk_levels[predicted_index]
    
    return {
        'risk_level': risk_level,
        'confidence': float(confidence),
        'detailed_breakdown': {
            'low': float(probabilities[0]),
            'medium': float(probabilities[1]),
            'high': float(probabilities[2])
        }
    }

"""
//csr_emotion_recommendations.py
CSR Call Recording - Emotion-Based Recommendation Engine
Provides actionable recommendations for CSRs based on detected emotional states
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class CSREmotionClassifier:
    """
    Advanced Emotion Classification and Recommendation System
    Analyzes customer emotions and provides tailored CSR guidance
    """

    # Detailed emotion profiles with psychological indicators
    EMOTION_PROFILES = {
        'angry': {
            'name': 'Angry',
            'severity': 'critical',
            'valence': 'negative',
            'arousal': 'high',
            'psychological_state': 'High emotional distress, potential escalation risk',
            'customer_needs': ['Immediate validation', 'Rapid resolution', 'Empowerment'],
            'avoid': ['Scripted responses', 'Delays', 'Dismissive language', 'Arguments'],
            'priority_level': 'P1',
            'escalation_threshold': 0.7
        },
        'frustrated': {
            'name': 'Frustrated',
            'severity': 'high',
            'valence': 'negative',
            'arousal': 'medium-high',
            'psychological_state': 'Accumulated irritation, barrier encountered',
            'customer_needs': ['Clear path forward', 'Acknowledgment', 'Efficiency'],
            'avoid': ['Over-explaining', 'Multiple transfers', 'Vague timelines'],
            'priority_level': 'P2',
            'escalation_threshold': 0.75
        },
        'sad': {
            'name': 'Sad',
            'severity': 'medium',
            'valence': 'negative',
            'arousal': 'low',
            'psychological_state': 'Disappointment, low energy, resignation',
            'customer_needs': ['Emotional support', 'Hope restoration', 'Gentle guidance'],
            'avoid': ['Excessive cheerfulness', 'Minimizing concerns', 'Rushing'],
            'priority_level': 'P2',
            'escalation_threshold': 0.65
        },
        'neutral': {
            'name': 'Neutral',
            'severity': 'low',
            'valence': 'neutral',
            'arousal': 'low',
            'psychological_state': 'Calm, task-focused, emotionally balanced',
            'customer_needs': ['Efficient service', 'Clear information', 'Professional approach'],
            'avoid': ['Over-emotionalizing', 'Unnecessary small talk'],
            'priority_level': 'P3',
            'escalation_threshold': 0.5
        },
        'satisfied': {
            'name': 'Satisfied',
            'severity': 'low',
            'valence': 'positive',
            'arousal': 'low',
            'psychological_state': 'Content, needs met, positive experience',
            'customer_needs': ['Confirmation', 'Next steps', 'Relationship building'],
            'avoid': ['Overselling', 'Prolonging unnecessarily'],
            'priority_level': 'P4',
            'escalation_threshold': 0.3
        },
        'happy': {
            'name': 'Happy',
            'severity': 'low',
            'valence': 'positive',
            'arousal': 'high',
            'psychological_state': 'Enthusiastic, positive expectations, energized',
            'customer_needs': ['Matching energy', 'Positive reinforcement', 'Value addition'],
            'avoid': ['Dampening enthusiasm', 'Being robotic'],
            'priority_level': 'P4',
            'escalation_threshold': 0.2
        }
    }

    # Communication strategies by emotion
    COMMUNICATION_STRATEGIES = {
        'angry': {
            'tone': 'Calm, steady, professional but warm',
            'pace': 'Moderate - not rushed but responsive',
            'language_style': 'Direct, clear, action-oriented',
            'empathy_level': 'High - acknowledge feelings immediately',
            'example_phrases': [
                "I understand this is frustrating, and I'm here to help resolve this right now.",
                "You're absolutely right to be concerned about this. Let me fix this for you.",
                "I apologize for the inconvenience. Here's what I can do immediately..."
            ]
        },
        'frustrated': {
            'tone': 'Understanding, solution-focused, reassuring',
            'pace': 'Efficient and purposeful',
            'language_style': 'Clear, structured, progress-oriented',
            'empathy_level': 'Medium-high - validate without dwelling',
            'example_phrases': [
                "I can see why this has been challenging. Let's get this sorted out.",
                "I appreciate your patience. Here's the quickest way to resolve this.",
                "Let me walk you through the solution step by step."
            ]
        },
        'sad': {
            'tone': 'Gentle, supportive, optimistic',
            'pace': 'Patient, unhurried',
            'language_style': 'Warm, encouraging, hope-focused',
            'empathy_level': 'Very high - emotional support priority',
            'example_phrases': [
                "I'm really sorry to hear that. Let's see how we can make this better for you.",
                "I understand how disappointing this must be. I'm going to help you through this.",
                "We're going to work together to find the best solution for you."
            ]
        },
        'neutral': {
            'tone': 'Professional, clear, efficient',
            'pace': 'Business-like, steady',
            'language_style': 'Factual, organized, precise',
            'empathy_level': 'Moderate - professionally courteous',
            'example_phrases': [
                "I can help you with that. Here's what we need to do.",
                "Let me get that information for you right away.",
                "I'll process that for you now."
            ]
        },
        'satisfied': {
            'tone': 'Positive, confirmatory, appreciative',
            'pace': 'Relaxed but attentive',
            'language_style': 'Affirmative, relationship-building',
            'empathy_level': 'Moderate - maintain positive connection',
            'example_phrases': [
                "I'm glad we could help with that. Is there anything else I can assist with?",
                "Great! I've got that all set up for you.",
                "Perfect. You're all set. We appreciate your business."
            ]
        },
        'happy': {
            'tone': 'Enthusiastic, energetic, matching positive energy',
            'pace': 'Upbeat, engaging',
            'language_style': 'Positive, collaborative, forward-looking',
            'empathy_level': 'Medium - share in positive experience',
            'example_phrases': [
                "That's wonderful! I'm excited to help you with this.",
                "Fantastic! Let me make sure everything goes smoothly for you.",
                "I love hearing that! Here's what we can do to make it even better."
            ]
        }
    }

    # Specific action recommendations by emotion
    ACTION_RECOMMENDATIONS = {
        'angry': {
            'immediate_actions': [
                'Acknowledge the customer\'s emotion explicitly within first 10 seconds',
                'Take ownership of the issue immediately, even if not directly at fault',
                'Offer a concrete solution or timeline within 30 seconds',
                'Avoid asking customer to repeat their issue unless absolutely necessary',
                'Use customer\'s name to personalize the interaction'
            ],
            'de_escalation_techniques': [
                'Lower your voice volume slightly to encourage mirroring',
                'Use "I" statements ("I understand", "I will") to build trust',
                'Provide frequent updates on progress (every 30-60 seconds)',
                'Offer choices when possible to restore sense of control',
                'Summarize action items to demonstrate active listening'
            ],
            'resolution_approach': [
                'Prioritize speed over perfect solution if safe to do so',
                'Escalate to supervisor if confidence drops below 80%',
                'Provide direct contact information for follow-up',
                'Confirm resolution meets customer expectations before closing',
                'Document interaction thoroughly for continuity'
            ]
        },
        'frustrated': {
            'immediate_actions': [
                'Acknowledge the difficulty or complexity of the situation',
                'Provide a clear roadmap of next steps',
                'Set realistic expectations on timing',
                'Eliminate any additional friction points',
                'Be proactive about potential obstacles'
            ],
            'problem_solving_approach': [
                'Break down complex solutions into simple steps',
                'Explain "why" behind each action to build understanding',
                'Offer alternative solutions if primary path is blocked',
                'Confirm understanding at each step',
                'Reduce need for customer to take action when possible'
            ],
            'efficiency_tactics': [
                'Minimize hold times - work while talking when possible',
                'Batch information gathering to avoid multiple questions',
                'Proactively handle related issues in same interaction',
                'Provide written summary of conversation via email/chat',
                'Set up automatic reminders or follow-ups'
            ]
        },
        'sad': {
            'immediate_actions': [
                'Express genuine empathy and concern',
                'Allow space for customer to express feelings',
                'Avoid rushing to solutions before emotional acknowledgment',
                'Use softer language and gentler tone',
                'Focus on what CAN be done, not limitations'
            ],
            'emotional_support_techniques': [
                'Validate feelings as legitimate and understandable',
                'Share (appropriate) similar experiences if relevant',
                'Use hope-oriented language ("We can work together")',
                'Celebrate small wins and progress',
                'Check in on emotional state throughout call'
            ],
            'recovery_strategies': [
                'Go beyond standard resolution to exceed expectations',
                'Offer goodwill gestures when appropriate and authorized',
                'Provide multiple support channels for continued assistance',
                'Personal follow-up call/email to confirm satisfaction',
                'Connect customer with additional resources if applicable'
            ]
        },
        'neutral': {
            'immediate_actions': [
                'Confirm understanding of customer need',
                'Provide clear, structured response',
                'Set expectations on process and timeline',
                'Be efficient and thorough',
                'Maintain professional demeanor'
            ],
            'service_excellence': [
                'Anticipate next logical questions and address proactively',
                'Provide additional relevant information customer may need',
                'Ensure all documentation is accurate and complete',
                'Verify satisfaction before concluding',
                'Offer resources for self-service in future'
            ],
            'efficiency_focus': [
                'Streamline process to minimize time investment',
                'Use templates and tools for consistency',
                'Batch similar tasks together',
                'Provide clear confirmation of completion',
                'Document for potential future reference'
            ]
        },
        'satisfied': {
            'immediate_actions': [
                'Reinforce positive experience',
                'Confirm all needs have been met',
                'Express appreciation for their business',
                'Ensure smooth conclusion',
                'Leave door open for future interactions'
            ],
            'relationship_building': [
                'Introduce relevant additional services (soft sell)',
                'Invite feedback on experience',
                'Provide loyalty program information if applicable',
                'Share useful tips or best practices',
                'Build personal connection for future interactions'
            ],
            'retention_strategies': [
                'Mention upcoming features or improvements',
                'Provide preferred contact methods for future',
                'Offer to set up account optimizations',
                'Thank customer for their loyalty/patience',
                'Request referral or review if appropriate'
            ]
        },
        'happy': {
            'immediate_actions': [
                'Match customer\'s positive energy appropriately',
                'Celebrate the success or positive situation with them',
                'Enhance the positive experience further',
                'Look for value-add opportunities',
                'Maintain momentum and enthusiasm'
            ],
            'experience_enhancement': [
                'Identify opportunities to surprise and delight',
                'Share insider tips or pro features',
                'Fast-track any requests when possible',
                'Introduce new capabilities that align with their needs',
                'Create memorable interaction moments'
            ],
            'advocacy_development': [
                'Request testimonial or case study (if appropriate)',
                'Encourage social media sharing',
                'Introduce referral incentives',
                'Invite to beta programs or exclusive features',
                'Build long-term relationship foundation'
            ]
        }
    }

    # Risk indicators and warning signs
    RISK_INDICATORS = {
        'critical_warning_signs': [
            'Repeated expressions of extreme frustration',
            'Threats to escalate to legal action or regulatory bodies',
            'Mentions of social media complaints or public reviews',
            'Requests to speak with manager/supervisor repeatedly',
            'Abusive language or personal attacks',
            'Mention of canceling service or switching to competitor',
            'Complete emotional withdrawal or silence',
            'Inconsistent or contradictory statements (confusion/distress)'
        ],
        'medium_warning_signs': [
            'Rising voice volume or pace',
            'Sarcasm or passive-aggressive language',
            'Repeated questions showing lack of understanding',
            'Expressions of disappointment in brand/service',
            'Comparing negatively to competitors',
            'Mentioning previous negative experiences',
            'Long pauses or sighs',
            'Short, clipped responses'
        ]
    }

    def __init__(self):
        """Initialize CSR Emotion Classifier and Recommendation Engine"""
        print(f"{'='*70}")
        print(f"CSR Emotion-Based Recommendation Engine Initialized")
        print(f"{'='*70}")
        print(f"Emotion Profiles: {len(self.EMOTION_PROFILES)}")
        print(f"Communication Strategies: {len(self.COMMUNICATION_STRATEGIES)}")
        print(f"Action Recommendations: {len(self.ACTION_RECOMMENDATIONS)}")
        print(f"{'='*70}\n")

    def classify_emotional_state(self, prediction_result):
        """
        Classify emotional state with comprehensive analysis
        
        Args:
            prediction_result (dict): ML classifier prediction output
            
        Returns:
            dict: Detailed emotional state classification with recommendations
        """
        emotion = prediction_result['emotion']
        confidence = prediction_result['confidence']
        all_probabilities = prediction_result['all_probabilities']

        # Get emotion profile
        profile = self.EMOTION_PROFILES.get(emotion, self.EMOTION_PROFILES['neutral'])

        # Calculate affective load (emotional intensity)
        affective_load = self._calculate_affective_load(emotion, confidence, all_probabilities)

        # Assess psychological indicators
        psych_indicators = self._assess_psychological_indicators(
            emotion, confidence, all_probabilities
        )

        # Determine intervention urgency
        intervention_urgency = self._determine_intervention_urgency(
            emotion, confidence, affective_load
        )

        # Compile emotional state
        emotional_state = {
            'timestamp': datetime.now().isoformat(),
            'primary_emotion': {
                'emotion': emotion,
                'name': profile['name'],
                'confidence': float(confidence),
                'severity': profile['severity'],
                'priority_level': profile['priority_level']
            },
            'emotional_profile': {
                'valence': profile['valence'],
                'arousal': profile['arousal'],
                'psychological_state': profile['psychological_state'],
                'customer_needs': profile['customer_needs'],
                'avoid_behaviors': profile['avoid']
            },
            'affective_load': {
                'score': affective_load['score'],
                'category': affective_load['category'],
                'intensity_level': affective_load['intensity']
            },
            'psychological_indicators': psych_indicators,
            'intervention_urgency': intervention_urgency,
            'confidence_level': self._categorize_confidence(confidence),
            'all_emotion_probabilities': all_probabilities
        }

        return emotional_state

    def _calculate_affective_load(self, emotion, confidence, probabilities):
        """
        Calculate affective load (emotional intensity/burden)
        
        Returns:
            dict: Affective load metrics
        """
        # Base load from emotion severity
        severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }

        profile = self.EMOTION_PROFILES.get(emotion, self.EMOTION_PROFILES['neutral'])
        base_load = severity_weights.get(profile['severity'], 0.5)

        # Adjust by confidence
        load_score = base_load * confidence

        # Check for mixed negative emotions (increases load)
        negative_emotions = ['angry', 'frustrated', 'sad']
        negative_load = sum(probabilities.get(e, 0) for e in negative_emotions)

        if negative_load > 0.5:
            load_score *= 1.2  # Increase load if multiple negative emotions present

        # Categorize affective load
        if load_score > 0.7:
            category = 'High'
            intensity = 'Customer experiencing significant emotional distress'
        elif load_score > 0.4:
            category = 'Moderate'
            intensity = 'Customer experiencing moderate emotional activation'
        else:
            category = 'Low'
            intensity = 'Customer emotionally stable or positive'

        return {
            'score': float(np.clip(load_score, 0, 1)),
            'category': category,
            'intensity': intensity,
            'negative_emotion_presence': float(negative_load)
        }

    def _assess_psychological_indicators(self, emotion, confidence, probabilities):
        """
        Assess psychological state indicators
        
        Returns:
            dict: Psychological assessment
        """
        # Emotional volatility (mixed emotions = higher volatility)
        volatility = 1 - confidence  # Lower confidence = higher volatility

        # Negative bias (tendency toward negative emotions)
        negative_emotions = ['angry', 'frustrated', 'sad']
        negative_bias = sum(probabilities.get(e, 0) for e in negative_emotions)

        # Determine risk level
        if emotion == 'angry' and confidence > 0.7:
            risk_level = 'Critical'
            risk_description = 'High escalation risk - immediate intervention required'
        elif emotion in ['angry', 'frustrated'] and confidence > 0.5:
            risk_level = 'High'
            risk_description = 'Elevated risk - prioritize de-escalation and resolution'
        elif negative_bias > 0.5:
            risk_level = 'Moderate'
            risk_description = 'Some negative sentiment - monitor and address proactively'
        else:
            risk_level = 'Low'
            risk_description = 'Stable interaction - maintain service quality'

        return {
            'emotional_volatility': float(volatility),
            'negative_bias': float(negative_bias),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'stability': 'Stable' if confidence > 0.6 else 'Unstable',
            'requires_supervisor': emotion == 'angry' and confidence > 0.75
        }

    def _determine_intervention_urgency(self, emotion, confidence, affective_load):
        """
        Determine urgency level for CSR intervention
        
        Returns:
            dict: Intervention urgency assessment
        """
        profile = self.EMOTION_PROFILES.get(emotion, self.EMOTION_PROFILES['neutral'])
        threshold = profile['escalation_threshold']

        if confidence > threshold and affective_load['score'] > 0.7:
            urgency = 'Immediate'
            timeframe = 'Within 10 seconds'
            action = 'Deploy de-escalation immediately'
        elif confidence > threshold or affective_load['score'] > 0.5:
            urgency = 'High'
            timeframe = 'Within 30 seconds'
            action = 'Address emotion proactively'
        elif affective_load['score'] > 0.3:
            urgency = 'Moderate'
            timeframe = 'Within 1 minute'
            action = 'Monitor and be attentive'
        else:
            urgency = 'Low'
            timeframe = 'Standard handling'
            action = 'Maintain professional service'

        return {
            'level': urgency,
            'timeframe': timeframe,
            'recommended_action': action,
            'escalation_threshold': float(threshold),
            'current_confidence': float(confidence)
        }

    def _categorize_confidence(self, confidence):
        """Categorize prediction confidence"""
        if confidence > 0.8:
            return 'Very High'
        elif confidence > 0.6:
            return 'High'
        elif confidence > 0.4:
            return 'Moderate'
        else:
            return 'Low'

    def generate_recommendation(self, emotional_state):
        """
        Generate comprehensive CSR recommendations
        
        Args:
            emotional_state (dict): Classified emotional state
            
        Returns:
            dict: Detailed recommendations for CSR
        """
        emotion = emotional_state['primary_emotion']['emotion']
        confidence = emotional_state['primary_emotion']['confidence']

        # Get communication strategy
        comm_strategy = self.COMMUNICATION_STRATEGIES.get(
            emotion, self.COMMUNICATION_STRATEGIES['neutral']
        )

        # Get action recommendations
        actions = self.ACTION_RECOMMENDATIONS.get(
            emotion, self.ACTION_RECOMMENDATIONS['neutral']
        )

        # Determine action required (CONTINUE, REST, or ESCALATE)
        action_directive = self._determine_action_directive(emotion, confidence, emotional_state)

        # Compile comprehensive recommendation
        recommendation = {
            'timestamp': datetime.now().isoformat(),
            'action_required': action_directive,  # NEW: Clear action directive
            'emotion_context': {
                'primary_emotion': emotional_state['primary_emotion']['name'],
                'confidence': confidence,
                'priority': emotional_state['primary_emotion']['priority_level'],
                'severity': emotional_state['primary_emotion']['severity']
            },
            'communication_guidance': {
                'recommended_tone': comm_strategy['tone'],
                'recommended_pace': comm_strategy['pace'],
                'language_style': comm_strategy['language_style'],
                'empathy_level': comm_strategy['empathy_level'],
                'example_phrases': comm_strategy['example_phrases']
            },
            'action_plan': actions,
            'do_and_dont': {
                'do': emotional_state['emotional_profile']['customer_needs'],
                'dont': emotional_state['emotional_profile']['avoid_behaviors']
            },
            'urgency_assessment': emotional_state['intervention_urgency'],
            'risk_assessment': {
                'level': emotional_state['psychological_indicators']['risk_level'],
                'description': emotional_state['psychological_indicators']['risk_description'],
                'requires_supervisor': emotional_state['psychological_indicators']['requires_supervisor']
            },
            'success_metrics': self._generate_success_metrics(emotion),
            'quality_assurance_checkpoints': self._generate_qa_checkpoints(emotion)
        }

        return recommendation

    def _determine_action_directive(self, emotion, confidence, emotional_state):
        """
        Determine clear action directive for CSR

        Returns one of three actions:
        - CONTINUE: Caller is calm, proceed with normal service
        - REST: Caller needs a brief pause to calm down
        - ESCALATE: Critical situation, transfer to supervisor immediately

        Args:
            emotion (str): Detected emotion
            confidence (float): Prediction confidence
            emotional_state (dict): Full emotional state analysis

        Returns:
            dict: Action directive with explanation
        """
        severity = emotional_state['primary_emotion']['severity']
        risk_level = emotional_state['psychological_indicators']['risk_level']

        # ESCALATE: Critical emotions with high confidence
        if emotion == 'angry' and confidence >= 0.65:
            return {
                'action': 'ESCALATE',
                'reason': 'Customer is angry and requires immediate supervisor intervention',
                'instruction': 'Transfer to supervisor or team lead immediately. Brief them on the situation before transfer.',
                'urgency': 'IMMEDIATE',
                'color': 'red'
            }

        # ESCALATE: High risk regardless of emotion
        if risk_level == 'critical':
            return {
                'action': 'ESCALATE',
                'reason': 'Critical risk level detected - potential for customer churn or escalation',
                'instruction': 'Alert supervisor and prepare for immediate escalation if situation doesn\'t improve within 2 minutes.',
                'urgency': 'HIGH',
                'color': 'red'
            }

        # REST: Frustrated customer with moderate-high confidence
        if emotion == 'frustrated' and confidence >= 0.60:
            return {
                'action': 'REST',
                'reason': 'Customer is frustrated and may benefit from a brief pause',
                'instruction': 'Offer to "look into this right away" and put on hold for 15-30 seconds to give customer time to calm down. Return with a clear solution.',
                'urgency': 'MEDIUM',
                'color': 'orange'
            }

        # REST: Very sad customer
        if emotion == 'sad' and confidence >= 0.55:
            return {
                'action': 'REST',
                'reason': 'Customer is emotionally distressed and needs gentle pacing',
                'instruction': 'Slow down the conversation pace. Give customer time to process information. Ask: "Would you like a moment before we continue?"',
                'urgency': 'MEDIUM',
                'color': 'orange'
            }

        # REST: Mixed negative emotions (confused emotional state)
        negative_emotions = ['angry', 'frustrated', 'sad']
        negative_prob_sum = sum(
            emotional_state['all_emotion_probabilities'].get(e, 0)
            for e in negative_emotions
        )
        if negative_prob_sum >= 0.70:  # 70% negative emotions combined
            return {
                'action': 'REST',
                'reason': 'Customer showing mixed negative emotions - uncertain emotional state',
                'instruction': 'Pause and validate customer\'s feelings: "I can hear this has been difficult. Let\'s take a moment to figure out the best solution."',
                'urgency': 'MEDIUM',
                'color': 'orange'
            }

        # CONTINUE: Positive or neutral emotions
        if emotion in ['happy', 'satisfied', 'neutral']:
            return {
                'action': 'CONTINUE',
                'reason': f'Customer is {emotion} - maintain current approach',
                'instruction': 'Continue providing excellent service. Maintain professionalism and efficiency.',
                'urgency': 'LOW',
                'color': 'green'
            }

        # DEFAULT: CONTINUE with caution
        return {
            'action': 'CONTINUE',
            'reason': 'Emotional state is manageable - proceed with empathy',
            'instruction': 'Monitor customer\'s tone carefully. Be ready to adjust approach if emotional state changes.',
            'urgency': 'LOW',
            'color': 'yellow'
        }

    def _generate_success_metrics(self, emotion):
        """Generate success metrics for interaction"""
        base_metrics = [
            'Customer acknowledges understanding',
            'Clear resolution path established',
            'Customer agrees to next steps'
        ]

        emotion_specific = {
            'angry': [
                'Customer tone becomes calmer',
                'Customer stops repeating complaints',
                'Customer expresses appreciation for help',
                'No escalation to supervisor required'
            ],
            'frustrated': [
                'Customer stops expressing confusion',
                'Customer confirms understanding of solution',
                'Positive acknowledgment of assistance',
                'Reduced question frequency'
            ],
            'sad': [
                'Customer energy level improves',
                'Customer expresses hope or optimism',
                'Customer engages more actively in conversation',
                'Positive emotional shift detected'
            ],
            'neutral': [
                'Efficient issue resolution',
                'All questions answered',
                'Professional interaction maintained',
                'Clear documentation completed'
            ],
            'satisfied': [
                'Customer expresses satisfaction verbally',
                'Positive feedback or thanks given',
                'Customer open to additional services',
                'Relationship strengthened'
            ],
            'happy': [
                'Enthusiasm maintained throughout',
                'Customer shares positive experience',
                'Additional value delivered',
                'Strong relationship foundation built'
            ]
        }

        return base_metrics + emotion_specific.get(emotion, [])

    def _generate_qa_checkpoints(self, emotion):
        """Generate quality assurance checkpoints"""
        base_checkpoints = [
            'Verified customer identity and account',
            'Documented interaction accurately',
            'Set clear expectations',
            'Confirmed customer satisfaction before closing'
        ]

        emotion_specific = {
            'angry': [
                'Acknowledged emotion within first response',
                'Avoided defensive language',
                'Provided concrete resolution timeline',
                'Followed up on commitments made',
                'Escalated appropriately if needed'
            ],
            'frustrated': [
                'Simplified communication',
                'Eliminated unnecessary steps',
                'Provided clear roadmap',
                'Minimized customer effort',
                'Confirmed understanding at key points'
            ],
            'sad': [
                'Demonstrated empathy appropriately',
                'Avoided minimizing concerns',
                'Provided emotional support',
                'Exceeded expectations where possible',
                'Offered follow-up contact'
            ],
            'neutral': [
                'Maintained professionalism',
                'Provided efficient service',
                'Answered all questions thoroughly',
                'Anticipated additional needs',
                'Completed all documentation'
            ],
            'satisfied': [
                'Reinforced positive experience',
                'Identified relationship-building opportunities',
                'Provided value-added information',
                'Invited future engagement',
                'Thanked customer appropriately'
            ],
            'happy': [
                'Matched customer energy',
                'Enhanced positive experience',
                'Identified advocacy opportunities',
                'Created memorable moment',
                'Built long-term relationship foundation'
            ]
        }

        return base_checkpoints + emotion_specific.get(emotion, [])

    def generate_report(self, emotional_state, recommendation):
        """
        Generate human-readable recommendation report
        
        Args:
            emotional_state (dict): Emotional state classification
            recommendation (dict): Generated recommendations
            
        Returns:
            str: Formatted report
        """
        report = []

        report.append(f"{'='*70}")
        report.append(f"CSR EMOTIONAL INTELLIGENCE RECOMMENDATION REPORT")
        report.append(f"{'='*70}\n")

        # Executive Summary
        report.append(f"EXECUTIVE SUMMARY")
        report.append(f"{'-'*70}")
        emotion_name = emotional_state['primary_emotion']['name']
        confidence = emotional_state['primary_emotion']['confidence']
        priority = emotional_state['primary_emotion']['priority_level']
        severity = emotional_state['primary_emotion']['severity']

        report.append(f"Customer Emotion: {emotion_name.upper()}")
        report.append(f"Confidence Level: {confidence:.1%} ({emotional_state['confidence_level']})")
        report.append(f"Priority: {priority} | Severity: {severity.upper()}")
        report.append(f"Affective Load: {emotional_state['affective_load']['category']} ({emotional_state['affective_load']['score']:.2f})")
        report.append(f"Risk Level: {emotional_state['psychological_indicators']['risk_level']}")
        report.append(f"\n{emotional_state['emotional_profile']['psychological_state']}\n")

        # Urgency Assessment
        report.append(f"URGENCY ASSESSMENT")
        report.append(f"{'-'*70}")
        urgency = recommendation['urgency_assessment']
        report.append(f"Intervention Urgency: {urgency['level'].upper()}")
        report.append(f"Response Timeframe: {urgency['timeframe']}")
        report.append(f"Recommended Action: {urgency['recommended_action']}\n")

        # Risk Flags
        if emotional_state['psychological_indicators']['requires_supervisor']:
            report.append(f"⚠️  SUPERVISOR ESCALATION RECOMMENDED ⚠️\n")

        # Communication Guidance
        report.append(f"COMMUNICATION GUIDANCE")
        report.append(f"{'-'*70}")
        comm = recommendation['communication_guidance']
        report.append(f"Tone: {comm['recommended_tone']}")
        report.append(f"Pace: {comm['recommended_pace']}")
        report.append(f"Language Style: {comm['language_style']}")
        report.append(f"Empathy Level: {comm['empathy_level']}\n")

        report.append(f"Example Opening Phrases:")
        for phrase in comm['example_phrases']:
            report.append(f"  • \"{phrase}\"")
        report.append("")

        # DO's
        report.append(f"✓ DO (Customer Needs)")
        report.append(f"{'-'*70}")
        for item in recommendation['do_and_dont']['do']:
            report.append(f"  ✓ {item}")
        report.append("")

        # DON'Ts
        report.append(f"✗ DON'T (Avoid These Behaviors)")
        report.append(f"{'-'*70}")
        for item in recommendation['do_and_dont']['dont']:
            report.append(f"  ✗ {item}")
        report.append("")

        # Action Plan
        report.append(f"ACTION PLAN")
        report.append(f"{'-'*70}")

        actions = recommendation['action_plan']
        
        if 'immediate_actions' in actions:
            report.append(f"\n1. IMMEDIATE ACTIONS (First 30 seconds):")
            for action in actions['immediate_actions']:
                report.append(f"   • {action}")

        # Get second-level actions (varies by emotion)
        second_level_key = None
        for key in ['de_escalation_techniques', 'problem_solving_approach', 
                    'emotional_support_techniques', 'service_excellence', 
                    'relationship_building', 'experience_enhancement']:
            if key in actions:
                second_level_key = key
                break

        if second_level_key:
            title = second_level_key.replace('_', ' ').title()
            report.append(f"\n2. {title.upper()}:")
            for action in actions[second_level_key]:
                report.append(f"   • {action}")

        # Get third-level actions
        third_level_key = None
        for key in ['resolution_approach', 'efficiency_tactics', 'recovery_strategies',
                    'efficiency_focus', 'retention_strategies', 'advocacy_development']:
            if key in actions:
                third_level_key = key
                break

        if third_level_key:
            title = third_level_key.replace('_', ' ').title()
            report.append(f"\n3. {title.upper()}:")
            for action in actions[third_level_key]:
                report.append(f"   • {action}")

        report.append("")

        # Success Metrics
        report.append(f"SUCCESS METRICS")
        report.append(f"{'-'*70}")
        report.append(f"Monitor for these positive indicators:")
        for metric in recommendation['success_metrics']:
            report.append(f"  ☑ {metric}")
        report.append("")

        # Quality Assurance
        report.append(f"QUALITY ASSURANCE CHECKPOINTS")
        report.append(f"{'-'*70}")
        report.append(f"Ensure you have:")
        for checkpoint in recommendation['quality_assurance_checkpoints']:
            report.append(f"  □ {checkpoint}")
        report.append("")

        # Additional Context
        if emotional_state['affective_load']['score'] > 0.7:
            report.append(f"⚠️  HIGH AFFECTIVE LOAD DETECTED")
            report.append(f"{'-'*70}")
            report.append(f"Customer is experiencing significant emotional distress.")
            report.append(f"Extra care and attention required. Consider:")
            report.append(f"  • Taking extra time to build rapport")
            report.append(f"  • Offering additional support resources")
            report.append(f"  • Following up within 24 hours")
            report.append(f"  • Documenting interaction for continuity\n")

        report.append(f"{'='*70}")
        report.append(f"End of Recommendation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"{'='*70}")

        return '\n'.join(report)

    def save_recommendation(self, emotional_state, recommendation, output_dir='recommendations'):
        """
        Save recommendation to files
        
        Args:
            emotional_state (dict): Emotional state classification
            recommendation (dict): Generated recommendations
            output_dir (str): Output directory
            
        Returns:
            dict: Paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        emotion = emotional_state['primary_emotion']['emotion']

        saved_files = {}

        # Save JSON
        json_file = output_path / f"recommendation_{emotion}_{timestamp}.json"
        combined_data = {
            'emotional_state': emotional_state,
            'recommendation': recommendation
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        saved_files['json'] = str(json_file)
        print(f"✓ Saved JSON: {json_file}")

        # Save report
        report_file = output_path / f"recommendation_{emotion}_{timestamp}_report.txt"
        report = self.generate_report(emotional_state, recommendation)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        saved_files['report'] = str(report_file)
        print(f"✓ Saved Report: {report_file}")

        # Save quick reference card
        quick_ref_file = output_path / f"quick_ref_{emotion}_{timestamp}.txt"
        quick_ref = self._generate_quick_reference(emotional_state, recommendation)
        with open(quick_ref_file, 'w', encoding='utf-8') as f:
            f.write(quick_ref)
        saved_files['quick_reference'] = str(quick_ref_file)
        print(f"✓ Saved Quick Reference: {quick_ref_file}\n")

        return saved_files

    def _generate_quick_reference(self, emotional_state, recommendation):
        """Generate a quick reference card for CSRs"""
        emotion_name = emotional_state['primary_emotion']['name']
        priority = emotional_state['primary_emotion']['priority_level']
        
        ref = []
        ref.append(f"╔{'═'*68}╗")
        ref.append(f"║ CSR QUICK REFERENCE CARD - {emotion_name.upper():^48} ║")
        ref.append(f"╠{'═'*68}╣")
        ref.append(f"║ Priority: {priority:<15} Risk: {emotional_state['psychological_indicators']['risk_level']:<35} ║")
        ref.append(f"╚{'═'*68}╝\n")

        comm = recommendation['communication_guidance']
        ref.append(f"TONE: {comm['recommended_tone']}")
        ref.append(f"PACE: {comm['recommended_pace']}\n")

        ref.append(f"OPENING PHRASE:")
        ref.append(f'  "{comm["example_phrases"][0]}"\n')

        ref.append(f"TOP 3 DO's:")
        for i, item in enumerate(recommendation['do_and_dont']['do'][:3], 1):
            ref.append(f"  {i}. {item}")

        ref.append(f"\nTOP 3 DON'Ts:")
        for i, item in enumerate(recommendation['do_and_dont']['dont'][:3], 1):
            ref.append(f"  {i}. {item}")

        return '\n'.join(ref)


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='CSR Call Recording - Emotion-Based Recommendation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--prediction',
        help='Path to ML classifier prediction JSON file',
        required=True
    )

    parser.add_argument(
        '--output',
        default='recommendations',
        help='Output directory for recommendation files'
    )

    args = parser.parse_args()

    # Load prediction
    with open(args.prediction, 'r') as f:
        prediction = json.load(f)

    # Initialize classifier
    classifier = CSREmotionClassifier()

    # Classify emotional state
    print(f"Analyzing emotional state...\n")
    emotional_state = classifier.classify_emotional_state(prediction)

    # Generate recommendations
    print(f"Generating recommendations...\n")
    recommendation = classifier.generate_recommendation(emotional_state)

    # Display report
    report = classifier.generate_report(emotional_state, recommendation)
    print(report)

    # Save results
    print(f"\nSaving recommendations...")
    saved_files = classifier.save_recommendation(
        emotional_state, recommendation, output_dir=args.output
    )

    print(f"\n{'='*70}")
    print(f"✓ Recommendation generation complete!")
    print(f"{'='*70}")
    print(f"Files saved to: {args.output}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
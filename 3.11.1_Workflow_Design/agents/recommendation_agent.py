from .base import Agent, Feedback

class RecommendationAgent(Agent):
    """Agent D: Recommendation Agent"""

    def run(self, utterance_id: int, context_desc: str, pedagogy_desc: str = "", problem_desc: str = "") -> dict:
        """
        Generate feedback and recommendations from context, pedagogy, and problem descriptions.
        
        Returns:
            dict with keys: 'prompt', 'feedback' (single Feedback object)
        """
        data = self._load_data()
        
        # Find intervention by matched_at_utterance_id
        intervention_rows = data['df_interventions'][
            data['df_interventions']['matched_at_utterance_id'] == utterance_id
        ]
        
        if intervention_rows.empty:
            raise ValueError(f"No intervention found for utterance_id {utterance_id}")
            
        intervention = intervention_rows.iloc[0]
        
        student_id = intervention['student_id']
        phase = intervention.get('phase', 'ideation-planning-decision-making') # Default or handle missing
        
        # Load prompt template from file
        prompt_template = self._load_prompt("recommendation_agent")
        from jinja2 import Template
        template = Template(prompt_template)
        prompt = template.render(
            student_id=student_id,
            phase=phase,
            context_desc=context_desc,
            pedagogy_desc=pedagogy_desc,
            problem_desc=problem_desc
        )
        
        self.log_verbose("AGENT D: RECOMMENDATION AGENT\nPROMPT:", prompt)
        
        feedback_result = self._generate_llm_feedback(prompt)
        
        if self.verbose:
            print("RESULT:")
            if feedback_result and isinstance(feedback_result, Feedback):
                print(f"Reasoning: {feedback_result.reasoning}")
                print(f"Feedback: {feedback_result.feedback}")
            else:
                print(feedback_result)
            print("\n" + "=" * 80 + "\n")
        
        return {
            'prompt': prompt,
            'feedback': feedback_result
        }

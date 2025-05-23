"""
LLM-Enhanced Interface for Tax Rule Formalization
Extends the basic system with modern LLM capabilities
"""

import openai
import anthropic
from typing import Dict, List, Optional
import json
import re

class ModernLLMInterface:
    """
    Interface for using modern LLMs (GPT-4, Claude, etc.) to formalize tax rules
    """
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        
        # Initialize based on provider
        if provider == "openai":
            openai.api_key = api_key
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-3-opus-20240229"
            
    def create_initial_state_from_description(self, description: str) -> Dict:
        """
        Convert natural language company description to initial state
        """
        prompt = f"""
        Convert the following company description into a formal state representation.
        
        Description: {description}
        
        Return a JSON object with:
        - parent_company: name of the parent company
        - parent_country: country code (US, DE, NL, IE, BM)
        - subsidiaries: list of {{name, country}} if any mentioned
        - ip_ownership: which company owns the IP
        
        Example output:
        {{
            "parent_company": "TechCorp",
            "parent_country": "US",
            "subsidiaries": [{{"name": "TechCorp Europe", "country": "NL"}}],
            "ip_ownership": "parent"
        }}
        """
        
        if self.provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tax law formalization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
            
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                system="You are a tax law formalization assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content[0].text)
    
    def formalize_tax_rule(self, legal_text: str, rule_type: str) -> Dict:
        """
        Convert legal text into formal rule representation
        """
        prompt = f"""
        Convert the following tax rule into a formal representation.
        
        Legal text: {legal_text}
        Rule type: {rule_type}
        
        Return a JSON object with:
        - rule_type: "{rule_type}"
        - conditions: list of formal conditions (use predicates like "based_in(Company, Country)")
        - effect: the tax effect (rate reduction, deduction, etc.)
        - applies_to: what transactions/entities this applies to
        
        Example for "Interest and royalty payments between EU companies are exempt from withholding tax":
        {{
            "rule_type": "exemption",
            "conditions": [
                "based_in(Payer, CountryA)",
                "based_in(Receiver, CountryB)",
                "eu_member(CountryA)",
                "eu_member(CountryB)",
                "transaction_type(royalty)"
            ],
            "effect": {{
                "withholding_tax_rate": 0.0
            }},
            "applies_to": "royalty_payments"
        }}
        
        Now convert: {legal_text}
        """
        
        if self.provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in formalizing tax law into logical representations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
            
        # Similar for anthropic...
        
    def identify_tax_planning_opportunity(self, rules: List[Dict], objective: str) -> Dict:
        """
        Use LLM to identify potential tax planning strategies given rules
        """
        rules_text = "\n".join([f"- {rule}" for rule in rules])
        
        prompt = f"""
        Given the following tax rules, identify potential tax planning opportunities.
        
        Rules:
        {rules_text}
        
        Objective: {objective}
        
        Provide:
        1. A step-by-step tax planning strategy
        2. Which rules are being utilized
        3. Estimated tax savings
        4. Any potential risks or considerations
        
        Format as JSON with:
        {{
            "strategy_name": "...",
            "steps": [...],
            "rules_used": [...],
            "estimated_savings": "...",
            "risks": [...]
        }}
        """
        
        # Call LLM and return structured response
        
    def explain_loophole(self, trajectory: Dict, tax_savings: float) -> str:
        """
        Generate human-readable explanation of identified tax loophole
        """
        prompt = f"""
        Explain the following tax planning structure in simple terms:
        
        Structure: {json.dumps(trajectory, indent=2)}
        Tax savings achieved: ${tax_savings:,.2f}
        
        Provide:
        1. A clear explanation of how the structure works
        2. Which specific tax rules are being exploited
        3. Why this might be considered a loophole
        4. Potential policy recommendations to address it
        
        Keep the explanation accessible to non-experts.
        """
        
        # Call LLM for explanation


class InteractiveTaxPlanner:
    """
    Interactive system combining LLM interface with tax planning engine
    """
    
    def __init__(self, llm_interface: ModernLLMInterface, tax_system):
        self.llm = llm_interface
        self.tax_system = tax_system
        
    def plan_from_description(self, description: str) -> Dict:
        """
        Create tax plan from natural language description
        """
        # Parse description into initial state
        state_desc = self.llm.create_initial_state_from_description(description)
        
        # Create formal initial state
        initial_state = self.tax_system.create_initial_state(
            state_desc["parent_company"],
            state_desc["parent_country"]
        )
        
        # Add any mentioned subsidiaries
        for sub in state_desc.get("subsidiaries", []):
            # Apply add_child action
            pass
        
        # Explore strategies
        trajectories = self.tax_system.explore_plans(initial_state)
        
        # Get explanation for top strategy
        if trajectories:
            explanation = self.llm.explain_loophole(
                trajectories[0],
                trajectories[0]["utility"]
            )
            
            return {
                "best_strategy": trajectories[0],
                "explanation": explanation,
                "alternatives": trajectories[1:5]
            }
    
    def add_custom_rule(self, legal_text: str, rule_type: str):
        """
        Add a new tax rule from natural language
        """
        # Formalize the rule
        formal_rule = self.llm.formalize_tax_rule(legal_text, rule_type)
        
        # Add to tax system
        # This would update the tax calculation logic
        print(f"Added rule: {formal_rule}")
        
    def analyze_policy_change(self, policy_description: str) -> Dict:
        """
        Analyze impact of proposed policy change
        """
        # Current best strategies
        current_trajectories = self.tax_system.explore_plans(
            self.tax_system.create_initial_state("TestCorp", "US")
        )
        
        # Parse and apply policy change
        # (This would modify the tax rules)
        
        # New best strategies  
        # new_trajectories = ...
        
        # Compare and analyze impact
        analysis = {
            "current_best_utility": current_trajectories[0]["utility"],
            # "new_best_utility": new_trajectories[0]["utility"],
            # "loopholes_closed": ...,
            # "revenue_impact": ...
        }
        
        return analysis


# Example prompts for different LLM tasks
EXAMPLE_PROMPTS = {
    "initial_state": """
    TechCorp is a US-based technology company that owns valuable software IP.
    They want to expand into Europe and are considering setting up subsidiaries
    in Netherlands and Ireland to serve the EU market.
    """,
    
    "tax_rule": """
    Under the EU Interest and Royalties Directive (2003/49/EC), interest and 
    royalty payments arising in a Member State are exempt from any taxes imposed 
    in that State, provided that the beneficial owner of the interest or royalties 
    is a company or permanent establishment in another Member State.
    """,
    
    "policy_change": """
    Proposal: Implement a minimum effective tax rate of 15% on all multinational
    corporations, regardless of where profits are booked.
    """
}


# Demonstration of LLM-enhanced capabilities
def demonstrate_llm_features():
    """Show how modern LLMs enhance the tax planning system"""
    
    print("LLM-Enhanced Tax Planning System")
    print("=" * 50)
    
    # Note: In real usage, you would provide actual API keys
    # llm = ModernLLMInterface(provider="openai", api_key="your-api-key")
    
    print("\n1. Natural Language Company Setup")
    print(f"Input: {EXAMPLE_PROMPTS['initial_state']}")
    # state = llm.create_initial_state_from_description(EXAMPLE_PROMPTS['initial_state'])
    # print(f"Parsed state: {json.dumps(state, indent=2)}")
    
    print("\n2. Formalizing Tax Rules from Legal Text")
    print(f"Input: {EXAMPLE_PROMPTS['tax_rule'][:100]}...")
    # rule = llm.formalize_tax_rule(EXAMPLE_PROMPTS['tax_rule'], "exemption")
    # print(f"Formal rule: {json.dumps(rule, indent=2)}")
    
    print("\n3. Policy Impact Analysis")
    print(f"Proposed change: {EXAMPLE_PROMPTS['policy_change']}")
    # impact = analyze_policy_change(EXAMPLE_PROMPTS['policy_change'])
    # print(f"Expected impact: Close loopholes worth ${impact['revenue_impact']:,.2f}")
    
    print("\n4. Interactive Planning Session")
    print("User: 'I have a US company with $100M in software licensing revenue.'")
    print("      'What's the most tax-efficient way to expand to Europe?'")
    print("\nSystem: Analyzing tax planning opportunities...")
    print("        Found strategy: Double Irish with Dutch Sandwich")
    print("        Estimated savings: $15.7M annually")
    print("        Risk level: High (likely to face scrutiny)")
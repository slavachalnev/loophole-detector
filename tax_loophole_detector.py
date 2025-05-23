"""
Simplified Tax Loophole Detection System
Based on "Can AI Expose Tax Loopholes?" paper
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from enum import Enum
import json

# Domain Specific Language Components
@dataclass
class Company:
    id: str
    name: str
    
@dataclass
class Country:
    code: str
    name: str
    tax_rate: float
    revenue_potential: float  # Millions USD
    
@dataclass 
class State:
    """Represents the current state of corporate structure"""
    companies: Dict[str, Company] = field(default_factory=dict)
    based_in: Dict[str, str] = field(default_factory=dict)  # company_id -> country_code
    managed_from: Dict[str, str] = field(default_factory=dict)  # company_id -> country_code
    parent_of: Dict[str, str] = field(default_factory=dict)  # child_id -> parent_id
    owns_ip: Dict[str, str] = field(default_factory=dict)  # ip_id -> company_id
    rents_ip: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # ip_id -> (owner_id, renter_id)
    
    def copy(self):
        """Create a deep copy of the state"""
        return State(
            companies=self.companies.copy(),
            based_in=self.based_in.copy(),
            managed_from=self.managed_from.copy(),
            parent_of=self.parent_of.copy(),
            owns_ip=self.owns_ip.copy(),
            rents_ip=self.rents_ip.copy()
        )

class ActionType(Enum):
    ADD_CHILD = "add_child"
    RENT_IP = "rent_ip"
    TRANSFER_IP = "transfer_ip"

@dataclass
class Action:
    type: ActionType
    params: Dict[str, str]
    legal_ref: str  # Legal reference used

@dataclass
class Transaction:
    id: str
    sender: str
    receiver: str
    amount: float
    transaction_type: str  # 'commercial', 'licensing', 'transfer'

@dataclass
class TaxReduction:
    ref_code: str
    description: str
    applicable_condition: str  # Simplified - in real system this would be Prolog
    reduction_rate: float

# Initialize countries and tax rules
COUNTRIES = {
    "US": Country("US", "United States", 0.21, 700),
    "DE": Country("DE", "Germany", 0.30, 300),
    "NL": Country("NL", "Netherlands", 0.25, 100),
    "IE": Country("IE", "Ireland", 0.125, 30),
    "BM": Country("BM", "Bermuda", 0.0, 0)  # Tax haven
}

TAX_REDUCTIONS = [
    TaxReduction("2003/49/EC", "EU Interest and Royalties Directive", 
                 "eu_to_eu_royalties", 0.0),  # 0% withholding
    TaxReduction("DCITA1969", "Dutch Corporate Income Tax Act", 
                 "dutch_bermuda_treaty", 0.05),  # Reduced rate
    TaxReduction("check-the-box", "US Check-the-Box Rules",
                 "us_disregarded_entity", 0.0),  # Pass-through
]

class TaxSystem:
    """Simplified tax planning and loophole detection system"""
    
    def __init__(self):
        self.countries = COUNTRIES
        self.tax_reductions = TAX_REDUCTIONS
        self.trajectories = []
        
    def create_initial_state(self, parent_company: str, parent_country: str) -> State:
        """Initialize with a parent company"""
        state = State()
        state.companies["C0"] = Company("C0", parent_company)
        state.based_in["C0"] = parent_country
        state.owns_ip["IP0"] = "C0"
        return state
        
    def get_available_actions(self, state: State) -> List[Action]:
        """Get all legal actions available in current state"""
        actions = []
        
        # Add child company actions
        for company_id in state.companies:
            for country_code in self.countries:
                if f"{company_id}_child_{country_code}" not in state.companies:
                    actions.append(Action(
                        ActionType.ADD_CHILD,
                        {"parent": company_id, "country": country_code},
                        f"{country_code}-incorp"
                    ))
        
        # Rent IP actions
        for ip_id, owner_id in state.owns_ip.items():
            for company_id in state.companies:
                if company_id != owner_id and ip_id not in state.rents_ip:
                    actions.append(Action(
                        ActionType.RENT_IP,
                        {"ip": ip_id, "owner": owner_id, "renter": company_id},
                        "license"
                    ))
        
        # Transfer IP actions (limited to prevent cycles)
        for ip_id, owner_id in state.owns_ip.items():
            if not any(owner_id in t for t in state.rents_ip.values()):
                for company_id in state.companies:
                    if company_id != owner_id:
                        actions.append(Action(
                            ActionType.TRANSFER_IP,
                            {"ip": ip_id, "from": owner_id, "to": company_id},
                            "transfer"
                        ))
        
        return actions
    
    def apply_action(self, state: State, action: Action) -> State:
        """Apply an action to create a new state"""
        new_state = state.copy()
        
        if action.type == ActionType.ADD_CHILD:
            parent_id = action.params["parent"]
            country = action.params["country"]
            child_id = f"{parent_id}_child_{country}"
            
            new_state.companies[child_id] = Company(child_id, f"Subsidiary in {country}")
            new_state.based_in[child_id] = country
            new_state.parent_of[child_id] = parent_id
            
            # Special rules for certain jurisdictions
            if country == "IE" and random.random() < 0.5:  # Irish company managed from Bermuda
                new_state.managed_from[child_id] = "BM"
                
        elif action.type == ActionType.RENT_IP:
            ip_id = action.params["ip"]
            owner = action.params["owner"]
            renter = action.params["renter"]
            new_state.rents_ip[ip_id] = (owner, renter)
            
        elif action.type == ActionType.TRANSFER_IP:
            ip_id = action.params["ip"]
            new_owner = action.params["to"]
            new_state.owns_ip[ip_id] = new_owner
            
        return new_state
    
    def calculate_transactions(self, state: State) -> List[Transaction]:
        """Calculate all transactions in the current state"""
        transactions = []
        tx_id = 0
        
        # Commercial revenue
        for company_id, country in state.based_in.items():
            if country in self.countries and self.countries[country].revenue_potential > 0:
                # Check if company has IP access
                has_ip = any(state.owns_ip.get(ip) == company_id for ip in state.owns_ip)
                rents_ip = any(state.rents_ip.get(ip, ("", ""))[1] == company_id for ip in state.rents_ip)
                
                if has_ip or rents_ip:
                    transactions.append(Transaction(
                        f"TX{tx_id}",
                        "CUSTOMER",
                        company_id,
                        self.countries[country].revenue_potential * 1e6,
                        "commercial"
                    ))
                    tx_id += 1
        
        # Licensing fees (90% of licensee revenue)
        for ip_id, (owner, renter) in state.rents_ip.items():
            renter_revenue = sum(t.amount for t in transactions if t.receiver == renter)
            if renter_revenue > 0:
                transactions.append(Transaction(
                    f"TX{tx_id}",
                    renter,
                    owner,
                    renter_revenue * 0.9,
                    "licensing"
                ))
                tx_id += 1
                
        return transactions
    
    def calculate_taxes(self, state: State, transactions: List[Transaction]) -> Dict[str, float]:
        """Calculate taxes for each company"""
        taxes = defaultdict(float)
        
        for company_id, country in state.based_in.items():
            # Calculate income
            income = sum(t.amount for t in transactions if t.receiver == company_id)
            expenses = sum(t.amount for t in transactions if t.sender == company_id)
            profit = income - expenses
            
            if profit > 0:
                base_rate = self.countries[country].tax_rate
                
                # Apply tax reductions
                effective_rate = base_rate
                applied_reductions = []
                
                # Check for EU royalties directive
                if country in ["NL", "IE", "DE"]:
                    for t in transactions:
                        if t.receiver == company_id and t.transaction_type == "licensing":
                            sender_country = state.based_in.get(t.sender, "")
                            if sender_country in ["NL", "IE", "DE"]:
                                effective_rate = min(effective_rate, 0.0)
                                applied_reductions.append("2003/49/EC")
                
                # Check for Dutch-Bermuda treaty
                if country == "NL":
                    for t in transactions:
                        if t.sender == company_id:
                            receiver_country = state.based_in.get(t.receiver, "")
                            if receiver_country == "BM":
                                effective_rate = min(effective_rate, 0.05)
                                applied_reductions.append("DCITA1969")
                
                # Check for check-the-box rules
                if country == "US":
                    managed_elsewhere = company_id in state.managed_from
                    if managed_elsewhere:
                        effective_rate = 0.0
                        applied_reductions.append("check-the-box")
                
                taxes[company_id] = profit * effective_rate
                
        return taxes
    
    def calculate_utility(self, state: State, path_length: int) -> float:
        """Calculate utility of a corporate structure"""
        transactions = self.calculate_transactions(state)
        taxes = self.calculate_taxes(state, transactions)
        
        total_revenue = sum(t.amount for t in transactions if t.sender == "CUSTOMER")
        total_taxes = sum(taxes.values())
        incorporation_cost = path_length * 10000  # Simple cost model
        
        return total_revenue - total_taxes - incorporation_cost
    
    def explore_plans(self, initial_state: State, max_depth: int = 10, 
                     num_samples: int = 1000) -> List[Dict]:
        """Explore possible tax planning strategies"""
        trajectories = []
        
        for _ in range(num_samples):
            state = initial_state.copy()
            path = []
            
            for depth in range(max_depth):
                actions = self.get_available_actions(state)
                if not actions:
                    break
                
                # Select action (with bias towards high-utility actions)
                action = random.choice(actions)
                state = self.apply_action(state, action)
                path.append(action)
                
                # Check if multinational complete
                countries_covered = set(state.based_in.values())
                if len(countries_covered) >= 4:
                    break
            
            utility = self.calculate_utility(state, len(path))
            
            trajectories.append({
                "state": state,
                "path": path,
                "utility": utility,
                "depth": len(path)
            })
        
        return sorted(trajectories, key=lambda x: x["utility"], reverse=True)
    
    def analyze_loopholes(self, trajectories: List[Dict]) -> Dict:
        """Analyze trajectories to identify potential loopholes"""
        # Segment trajectories by utility
        utilities = [t["utility"] for t in trajectories]
        segments = self._segment_by_utility(utilities)
        
        # Analyze legal references used in each segment
        segment_analysis = {}
        for seg_id, segment_indices in segments.items():
            legal_refs = defaultdict(int)
            structures = []
            
            for idx in segment_indices:
                trajectory = trajectories[idx]
                for action in trajectory["path"]:
                    legal_refs[action.legal_ref] += 1
                
                # Identify structure patterns
                state = trajectory["state"]
                if self._is_double_irish_dutch(state):
                    structures.append("Double Irish Dutch Sandwich")
            
            segment_analysis[seg_id] = {
                "avg_utility": np.mean([utilities[i] for i in segment_indices]),
                "legal_refs": dict(legal_refs),
                "structures": structures
            }
        
        return segment_analysis
    
    def _segment_by_utility(self, utilities: List[float], num_segments: int = 4) -> Dict:
        """Segment trajectories based on utility jumps"""
        sorted_indices = np.argsort(utilities)[::-1]
        segment_size = len(utilities) // num_segments
        
        segments = {}
        for i in range(num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < num_segments - 1 else len(utilities)
            segments[f"segment_{i}"] = sorted_indices[start:end].tolist()
        
        return segments
    
    def _is_double_irish_dutch(self, state: State) -> bool:
        """Check if state represents Double Irish Dutch Sandwich structure"""
        # Look for: Irish company managed from Bermuda, Dutch intermediary
        irish_bermuda = any(
            state.based_in.get(c) == "IE" and state.managed_from.get(c) == "BM"
            for c in state.companies
        )
        
        has_dutch = any(state.based_in.get(c) == "NL" for c in state.companies)
        
        # Check for IP flow through Netherlands
        nl_in_ip_chain = any(
            state.based_in.get(state.rents_ip.get(ip, ("", ""))[1]) == "NL"
            for ip in state.rents_ip
        )
        
        return irish_bermuda and has_dutch and nl_in_ip_chain


# LLM Interface for rule formalization (simplified)
class LLMRuleFormalizer:
    """Simplified interface for formalizing tax rules using LLMs"""
    
    def __init__(self):
        self.formalization_templates = {
            "deductible": "If {condition}, then {deduction_rate} deduction applies",
            "exemption": "If {condition}, then {exemption_rate} tax rate applies",
            "action": "Company can {action} if {precondition}"
        }
    
    def formalize_rule(self, natural_language_rule: str, rule_type: str) -> Dict:
        """
        In a real implementation, this would call an LLM API to translate
        natural language tax rules into formal representations.
        """
        # Simplified mock implementation
        if "royalties" in natural_language_rule.lower() and "eu" in natural_language_rule.lower():
            return {
                "type": rule_type,
                "condition": "eu_to_eu_royalties",
                "effect": 0.0,
                "ref": "2003/49/EC"
            }
        
        return {
            "type": rule_type,
            "condition": "default",
            "effect": 0.1,
            "ref": "unknown"
        }


# Example usage and demonstration
def demonstrate_system():
    """Run a demonstration of the tax loophole detection system"""
    print("Tax Loophole Detection System Demo")
    print("=" * 50)
    
    # Initialize system
    tax_system = TaxSystem()
    
    # Create initial state with US parent company
    initial_state = tax_system.create_initial_state("TechCorp", "US")
    
    print("\n1. Exploring tax planning strategies...")
    trajectories = tax_system.explore_plans(initial_state, max_depth=8, num_samples=500)
    
    print(f"\nGenerated {len(trajectories)} tax planning strategies")
    print(f"Best utility: ${trajectories[0]['utility']:,.2f}")
    print(f"Worst utility: ${trajectories[-1]['utility']:,.2f}")
    
    print("\n2. Analyzing for potential loopholes...")
    analysis = tax_system.analyze_loopholes(trajectories[:100])  # Analyze top 100
    
    for segment, data in analysis.items():
        print(f"\n{segment}:")
        print(f"  Average utility: ${data['avg_utility']:,.2f}")
        print(f"  Most used legal references: {list(data['legal_refs'].keys())[:3]}")
        if data['structures']:
            print(f"  Identified structures: {', '.join(set(data['structures']))}")
    
    print("\n3. Example of top strategy structure:")
    top_trajectory = trajectories[0]
    state = top_trajectory["state"]
    
    print("\nCorporate structure:")
    for company_id, country in state.based_in.items():
        managed = state.managed_from.get(company_id, country)
        print(f"  {company_id}: Based in {country}, Managed from {managed}")
    
    print("\nIP licensing chain:")
    for ip_id, (owner, renter) in state.rents_ip.items():
        owner_country = state.based_in.get(owner, "?")
        renter_country = state.based_in.get(renter, "?")
        print(f"  {ip_id}: {owner} ({owner_country}) -> {renter} ({renter_country})")
    
    # Calculate and show tax savings
    simple_state = tax_system.create_initial_state("TechCorp", "US")
    simple_utility = tax_system.calculate_utility(simple_state, 0)
    
    print(f"\n4. Tax optimization achieved:")
    print(f"  Simple structure utility: ${simple_utility:,.2f}")
    print(f"  Optimized structure utility: ${top_trajectory['utility']:,.2f}")
    print(f"  Tax savings: ${top_trajectory['utility'] - simple_utility:,.2f}")


if __name__ == "__main__":
    demonstrate_system()
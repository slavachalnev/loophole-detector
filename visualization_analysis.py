"""
Visualization and Analysis Tools for Tax Loophole Detection
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np

class TaxStructureVisualizer:
    """Visualize corporate structures and tax flows"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.country_colors = {
            "US": "#1f77b4",  # Blue
            "DE": "#ff7f0e",  # Orange  
            "NL": "#2ca02c",  # Green
            "IE": "#d62728",  # Red
            "BM": "#9467bd"   # Purple (tax haven)
        }
        
    def visualize_corporate_structure(self, state, transactions=None):
        """Create a network graph of corporate structure"""
        G = nx.DiGraph()
        
        # Add nodes for companies
        for company_id, company in state.companies.items():
            country = state.based_in.get(company_id, "Unknown")
            managed = state.managed.get(company_id, country)
            
            label = f"{company_id}\n{country}"
            if managed != country:
                label += f"\n(managed: {managed})"
                
            G.add_node(company_id, 
                      label=label,
                      country=country,
                      color=self.country_colors.get(country, "#gray"))
        
        # Add edges for ownership
        for child_id, parent_id in state.is_child_of.items():
            G.add_edge(parent_id, child_id, 
                      relationship="owns",
                      style="solid",
                      width=2)
        
        # Add edges for IP licensing
        for owner, renter, ip_id in state.rents_ip:
            G.add_edge(owner, renter,
                      relationship=f"licenses {ip_id}",
                      style="dashed",
                      width=1.5,
                      color="green")
        
        # Layout and draw
        plt.figure(figsize=self.fig_size)
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_colors = [G.nodes[node].get('color', '#gray') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.9)
        
        # Draw edges with different styles
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'solid']
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
        
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                              width=2, alpha=0.6)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
                              width=1.5, alpha=0.6, style='dashed',
                              edge_color='green')
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # Add edge labels for all relationships
        edge_labels = {(u, v): d['relationship'] 
                      for u, v, d in G.edges(data=True) 
                      if 'relationship' in d}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title("Corporate Structure and IP Flows", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=country)
                          for country, color in self.country_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        return plt.gcf()
    
    def plot_utility_profile(self, trajectories: List[Dict]):
        """Plot utility profile showing tax optimization segments"""
        utilities = [t['utility'] for t in trajectories]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(utilities)), sorted(utilities, reverse=True), 
                'b-', linewidth=2)
        
        # Identify and mark segments
        sorted_utils = sorted(utilities, reverse=True)
        differences = np.diff(sorted_utils)
        
        # Find large drops (segment boundaries)
        threshold = np.std(differences) * 2
        segment_boundaries = np.where(differences < -threshold)[0]
        
        # Mark segments
        colors = ['red', 'orange', 'yellow', 'green']
        prev_boundary = 0
        for i, boundary in enumerate(segment_boundaries[:4]):
            plt.axvspan(prev_boundary, boundary, alpha=0.2, 
                       color=colors[i % len(colors)],
                       label=f'Segment {i}')
            prev_boundary = boundary
        
        plt.xlabel('Strategy Rank', fontsize=12)
        plt.ylabel('Utility ($)', fontsize=12)
        plt.title('Tax Planning Strategy Utility Profile', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations for tax savings
        best_utility = sorted_utils[0]
        worst_utility = sorted_utils[-1]
        plt.annotate(f'Best: ${best_utility:,.0f}', 
                    xy=(0, best_utility), xytext=(10, best_utility + 1e6),
                    arrowprops=dict(arrowstyle='->', color='red'))
        plt.annotate(f'Worst: ${worst_utility:,.0f}',
                    xy=(len(utilities)-1, worst_utility), 
                    xytext=(len(utilities)-50, worst_utility - 1e6),
                    arrowprops=dict(arrowstyle='->', color='blue'))
        
        return plt.gcf()
    
    def analyze_legal_references(self, segment_analysis: Dict):
        """Visualize which legal references are used in each segment"""
        # Prepare data
        data = []
        for segment, info in segment_analysis.items():
            for ref, count in info['legal_refs'].items():
                data.append({
                    'Segment': segment,
                    'Legal Reference': ref,
                    'Usage Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        pivot_df = df.pivot(index='Legal Reference', 
                           columns='Segment', 
                           values='Usage Count')
        
        sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Usage Count'})
        
        plt.title('Legal Reference Usage by Strategy Segment', fontsize=14)
        plt.xlabel('Strategy Segment', fontsize=12)
        plt.ylabel('Legal Reference', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_tax_flow_diagram(self, state, transactions, taxes):
        """Create a Sankey diagram showing money flows and tax payments"""
        from matplotlib.sankey import Sankey
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sankey = Sankey(ax=ax, scale=0.01, offset=0.3, 
                       head_angle=150, format='%.0f')
        
        # Calculate flows
        flows = []
        labels = []
        orientations = []
        
        # Customer revenue (inflow)
        total_revenue = sum(t.amount for t in transactions 
                           if t.sender == "CUSTOMER")
        if total_revenue > 0:
            flows.append(total_revenue / 1e6)  # Convert to millions
            labels.append(f'Revenue\n${total_revenue/1e6:.1f}M')
            orientations.append(0)
        
        # Tax payments (outflow)
        total_taxes = sum(taxes.values())
        if total_taxes > 0:
            flows.append(-total_taxes / 1e6)
            labels.append(f'Taxes\n${total_taxes/1e6:.1f}M')
            orientations.append(-1)
        
        # Net profit (outflow)
        net_profit = total_revenue - total_taxes
        if net_profit > 0:
            flows.append(-net_profit / 1e6)
            labels.append(f'Profit\n${net_profit/1e6:.1f}M')
            orientations.append(1)
        
        sankey.add(flows=flows, labels=labels, orientations=orientations,
                  pathlengths=[0.25, 0.25, 0.25],
                  trunklength=1.5,
                  facecolor='lightblue',
                  edgecolor='blue')
        
        diagrams = sankey.finish()
        for diagram in diagrams:
            for text in diagram.texts:
                text.set_fontsize(10)
        
        plt.title('Tax Strategy Cash Flow Analysis', fontsize=14)
        plt.axis('off')
        
        return fig


class LoopholeDetector:
    """Advanced analysis for detecting tax loopholes"""
    
    def __init__(self, tax_system):
        self.tax_system = tax_system
        
    def find_anomalous_structures(self, trajectories: List[Dict], 
                                 threshold_percentile: int = 90) -> List[Dict]:
        """Identify structures with anomalously high tax savings"""
        utilities = [t['utility'] for t in trajectories]
        threshold = np.percentile(utilities, threshold_percentile)
        
        anomalous = []
        for t in trajectories:
            if t['utility'] > threshold:
                # Analyze structure
                structure_features = self._extract_structure_features(t['state'])
                
                anomalous.append({
                    'trajectory': t,
                    'features': structure_features,
                    'savings_ratio': t['utility'] / np.median(utilities)
                })
        
        return sorted(anomalous, key=lambda x: x['savings_ratio'], reverse=True)
    
    def _extract_structure_features(self, state) -> Dict:
        """Extract key features of a corporate structure"""
        features = {
            'num_companies': len(state.companies),
            'num_countries': len(set(state.based_in.values())),
            'has_tax_haven': any(c in ['BM', 'LU', 'CH'] 
                                for c in state.based_in.values()),
            'managed_elsewhere': any(state.managed.get(c, state.based_in.get(c)) != state.based_in.get(c) for c in state.companies),
            'ip_transfers': len(state.rents_ip),
            'uses_treaty_network': self._check_treaty_usage(state)
        }
        
        # Check for known patterns
        features['double_irish'] = self._is_double_irish(state)
        features['dutch_sandwich'] = self._has_dutch_sandwich(state)
        
        return features
    
    def _check_treaty_usage(self, state) -> bool:
        """Check if structure uses tax treaty networks"""
        treaty_pairs = [('NL', 'BM'), ('IE', 'NL'), ('LU', 'US')]
        
        countries = set(state.based_in.values())
        for pair in treaty_pairs:
            if pair[0] in countries and pair[1] in countries:
                return True
        return False
    
    def _is_double_irish(self, state) -> bool:
        """Check for Double Irish structure"""
        irish_companies = [c for c, country in state.based_in.items() 
                          if country == 'IE']
        
        for company in irish_companies:
            if state.managed.get(company) == 'BM':
                return True
        return False
    
    def _has_dutch_sandwich(self, state) -> bool:
        """Check for Dutch Sandwich structure"""
        # Look for NL company in middle of IP flow
        for owner, renter, ip in state.rents_ip:
            owner_country = state.based_in.get(owner)
            renter_country = state.based_in.get(renter)
            
            if (owner_country == 'IE' and renter_country == 'NL') or \
               (owner_country == 'NL' and renter_country in ['US', 'DE']):
                return True
        
        return False
    
    def generate_policy_recommendations(self, anomalous_structures: List[Dict]) -> List[str]:
        """Generate policy recommendations based on detected loopholes"""
        recommendations = []
        
        # Analyze common features
        all_features = [s['features'] for s in anomalous_structures]
        
        if sum(f['managed_elsewhere'] for f in all_features) > len(all_features) * 0.5:
            recommendations.append(
                "Implement substance requirements: Companies managed from tax havens "
                "should not qualify for treaty benefits"
            )
        
        if sum(f['dutch_sandwich'] for f in all_features) > len(all_features) * 0.3:
            recommendations.append(
                "Anti-conduit rules: Limit treaty benefits for intermediary companies "
                "with no substantial business activities"
            )
        
        if sum(f['has_tax_haven'] for f in all_features) > len(all_features) * 0.7:
            recommendations.append(
                "Minimum tax rules: Implement global minimum tax rate to reduce "
                "incentive for profit shifting to tax havens"
            )
        
        return recommendations


# Example usage
def demonstrate_visualization():
    """Demonstrate visualization capabilities using discovered strategies"""
    from tax_loophole_detector import TaxSystem
    
    print("Tax Structure Visualization Demo")
    print("=" * 50)
    
    # Initialize system and find optimal strategies
    tax_system = TaxSystem()
    initial_state = tax_system.create_initial_state("TechCorp", "US")
    
    print("\n1. Discovering tax optimization strategies...")
    trajectories = tax_system.explore_plans(
        initial_state,
        max_depth=10,
        num_samples=10000
    )
    
    print(f"   Generated {len(trajectories)} strategies")
    print(f"   Best utility: ${trajectories[0]['utility']:,.2f}")
    print(f"   Worst utility: ${trajectories[-1]['utility']:,.2f}")
    
    # Get the best discovered strategy
    best_strategy = trajectories[0]
    best_state = best_strategy['state']
    
    print(f"\n2. Analyzing best strategy (utility: ${best_strategy['utility']:,.2f}):")
    print(f"   Strategy steps: {len(best_strategy['path'])}")
    print("   Actions taken:")
    for i, action in enumerate(best_strategy['path'], 1):
        print(f"     {i}. {action.type.value}: {action.params} (using {action.legal_ref})")
    
    # Create visualizations using the discovered strategy
    visualizer = TaxStructureVisualizer()
    
    print("\n3. Creating corporate structure visualization of best strategy...")
    transactions = tax_system.calculate_transactions(best_state)
    taxes = tax_system.calculate_taxes(best_state, transactions)
    
    fig1 = visualizer.visualize_corporate_structure(best_state, transactions)
    plt.savefig('corporate_structure.png', dpi=150, bbox_inches='tight')
    print("   Saved as 'corporate_structure.png'")
    
    print("\n4. Creating utility profile visualization...")
    fig2 = visualizer.plot_utility_profile(trajectories)
    plt.savefig('utility_profile.png', dpi=150, bbox_inches='tight')
    print("   Saved as 'utility_profile.png'")
    
    print("\n5. Creating tax flow diagram...")
    fig3 = visualizer.create_tax_flow_diagram(best_state, transactions, taxes)
    plt.savefig('tax_flow_diagram.png', dpi=150, bbox_inches='tight')
    print("   Saved as 'tax_flow_diagram.png'")
    
    print("\n6. Analyzing loophole patterns...")
    detector = LoopholeDetector(tax_system)
    anomalous = detector.find_anomalous_structures(trajectories, threshold_percentile=80)
    
    print(f"\nFound {len(anomalous)} potentially problematic structures")
    if anomalous:
        top_structure = anomalous[0]
        print(f"Top structure achieves {top_structure['savings_ratio']:.2f}x median utility")
        print(f"Features: {top_structure['features']}")
    
    print("\n7. Policy recommendations:")
    recommendations = detector.generate_policy_recommendations(anomalous)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n8. Best strategy details:")
    print("   Corporate structure:")
    for company_id, country in best_state.based_in.items():
        managed = best_state.managed.get(company_id, country)
        managed_note = f", Managed in {managed}" if managed != country else ""
        print(f"     {company_id}: Based in {country}{managed_note}")
    
    if best_state.rents_ip:
        print("   IP licensing chain:")
        for owner, renter, ip_id in best_state.rents_ip:
            owner_country = best_state.based_in.get(owner, "?")
            renter_country = best_state.based_in.get(renter, "?")
            print(f"     {ip_id}: {owner} ({owner_country}) -> {renter} ({renter_country})")
    
    print(f"\n   Tax efficiency:")
    simple_utility = tax_system.calculate_utility(initial_state, 0)
    print(f"     Simple structure: ${simple_utility:,.2f}")
    print(f"     Optimized structure: ${best_strategy['utility']:,.2f}")
    print(f"     Tax savings: ${best_strategy['utility'] - simple_utility:,.2f}")
    
    plt.close('all')  # Clean up matplotlib figures


if __name__ == "__main__":
    demonstrate_visualization()
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a tax loophole detection system based on the research paper "Can AI Expose Tax Loopholes?". The system simulates corporate tax planning strategies and identifies potential loopholes through automated exploration and analysis.

## Core Architecture

### Main Components

- **tax_loophole_detector.py**: Core tax planning engine with domain-specific language for corporate structures, legal actions, and tax calculations
- **llm_tax_interface.py**: LLM-enhanced interface for formalizing tax rules from natural language and explaining strategies
- **visualization_analysis.py**: Visualization tools and advanced loophole detection algorithms

### Key System Design

The system uses a state-based approach where:
- **State**: Represents corporate structure (companies, jurisdictions, IP ownership, management locations)
- **Actions**: Legal moves like creating subsidiaries, transferring IP, licensing arrangements
- **Utility**: Financial benefit calculation considering taxes and incorporation costs
- **Exploration**: Monte Carlo sampling to discover tax optimization strategies

### Tax Planning Simulation

The system models simplified versions of real tax structures:
- Double Irish Dutch Sandwich
- Check-the-box elections
- EU Interest and Royalties Directive
- Treaty shopping networks

## Development Commands

Since this is a Python project without configuration files, run modules directly:

```bash
# Run main tax loophole detection demo
python tax_loophole_detector.py

# Run visualization analysis demo  
python visualization_analysis.py

# Run LLM interface demo (requires API keys)
python llm_tax_interface.py
```

## Dependencies

The project uses standard scientific Python libraries:
- numpy, matplotlib, seaborn, pandas for analysis and visualization
- networkx for corporate structure graphs
- openai, anthropic for LLM integration (optional)

Install with: `pip install numpy matplotlib seaborn pandas networkx openai anthropic`

## Key Data Structures

- **State**: Central state representation with companies, jurisdictions, ownership relationships
- **Action**: Represents legal corporate actions with type, parameters, and legal reference
- **Transaction**: Models financial flows between entities
- **TaxReduction**: Encodes tax rules and their conditions

## Important Implementation Notes

- The tax calculation logic in `calculate_taxes()` applies various tax reductions based on corporate structure patterns
- The exploration algorithm in `explore_plans()` uses Monte Carlo sampling with utility-based evaluation
- Loophole detection works by segmenting strategies by utility and analyzing common legal reference patterns
- The LLM interface provides natural language processing for rule formalization and strategy explanation
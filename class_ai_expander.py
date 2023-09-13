class AIExpander:
    def __init__(self, formula):
        self.formula = formula

    def expand_acronym(self, acronym):
        return self.formula(acronym)

def default_acronym_formula(acronym):
    return f"{acronym} Intelligent Assistant"

def custom_acronym_formula(acronym):
    # Define your custom formula here
    # For example, you might use a different concatenation pattern or add additional information.
    return f"Artificial {acronym} Assistant"

# Example Usage:
expander = AIExpander(default_acronym_formula)

# Generate AI for a specific acronym
ai = expander.expand_acronym("AI")
print(ai)  # Output: "AI Intelligent Assistant"

# Example with a custom formula:
expander = AIExpander(custom_acronym_formula)
ai = expander.expand_acronym("ML")
print(ai)  # Output: "Artificial ML Assistant"
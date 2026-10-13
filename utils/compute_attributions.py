import Uncertainpy.src.uncertainpy.gradual as grad


def compute_argument_gradient(bag, arg1, arg2, agg_f, inf_f, h=1e-5):
    # Store original values
    initial_strengths = grad.algorithms.computeStrengthValues(bag, agg_f, inf_f)
    original_strength = bag.arguments[arg2].strength
    initial_base_score = bag.arguments[arg1].initial_weight
    
    # Compute forward difference
    pos_change = min(initial_base_score + h, 1.0)  # Clamp to valid range
    if pos_change == initial_base_score:  # If we couldn't move forward
        # Use backward difference
        pos_change = max(initial_base_score - h, 0.0)
        if pos_change == initial_base_score:  # Can't move either way
            return 0.0  # Or handle as error
    
    bag.arguments[arg1].initial_weight = pos_change
    strengths = grad.algorithms.computeStrengthValues(bag, agg_f, inf_f)
    counterfactual_strength = bag.arguments[arg2].strength
    
    # Compute gradient with actual step size used
    actual_h = pos_change - initial_base_score
    gradient = (counterfactual_strength - original_strength) / actual_h
    
    # Restore original
    bag.arguments[arg1].initial_weight = initial_base_score
    strengths = grad.algorithms.computeStrengthValues(bag, agg_f, inf_f)
    
    return gradient
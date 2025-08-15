import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Mapping for sized [(2,2),(3,3),(3,9),(4,7)]
# w - washers
# o - oven
# c - couches
# s - sofas



def visualize_solution(results, show_plots=True):
    """
    Visualize the solution from the constraint programming solver.
    
    Args:
        results: List of dictionaries containing item information with keys:
                'item', 'type', 'size', 'truck', 'x', 'y'
        num_trucks: Number of trucks used in the solution
        show_plots: Whether to display the plots (default: True)
    
    Returns:
        List of matplotlib figure objects
    """
    # Group by truck - use all truck indices that actually have items
    truck_indices = set(obj['truck'] for obj in results)
    trucks = {truck_idx: [] for truck_idx in truck_indices}
    for obj in results:
        trucks[obj['truck']].append(obj)
    
    colors = {'w': 'blue', 'o': 'red', 'c': 'green', 's': 'purple'}
    figures = []
    
    for truck_idx, truck_objs in trucks.items():
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.set_title(f"Truck {truck_idx+1}")
        ax.set_xlim(0, 26)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        for obj in truck_objs:
            color = colors.get(obj['type'], 'gray')
            # Add a random alpha and hatch for creativity
            alpha = 0.5 + 0.5 * random.random()
            hatch = random.choice(['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'])
            rect = patches.Rectangle((obj['x'], obj['y']), obj['size'][0], obj['size'][1], 
                                   linewidth=2, edgecolor='black', facecolor=color, 
                                   alpha=alpha, hatch=hatch, label=obj['type'])
            ax.add_patch(rect)
            ax.text(obj['x'] + obj['size'][0]/2, obj['y'] + obj['size'][1]/2, obj['item'], 
                   color='white', ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.legend(handles=[patches.Patch(color=colors[t], label=t) for t in colors], loc='upper right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        figures.append(fig)
        
        if show_plots:
            plt.show()
    
    return figures
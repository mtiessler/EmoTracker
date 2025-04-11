import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

def plot_losses(train_losses: List[float], val_losses: List[float],
                r2_scores: Dict[str, List[float]], mse_scores: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot R2 scores
    plt.subplot(2, 2, 2)
    for dim, scores in r2_scores.items():
        plt.plot(scores, label=f'{dim.capitalize()} R2')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.title('Validation R2 Scores')
    plt.legend()

    # Plot MSE scores
    plt.subplot(2, 2, 3)
    for dim, scores in mse_scores.items():
        plt.plot(scores, label=f'{dim.capitalize()} MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Validation MSE Scores')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        **{f'{dim.capitalize()} R2': scores for dim, scores in r2_scores.items()},
        **{f'{dim.capitalize()} MSE': scores for dim, scores in mse_scores.items()}
    })
    results_df.to_csv('training_results.csv', index=False)
import json
import matplotlib.pyplot as plt

# Load data
with open('trainer_state.json', 'r') as f:
    state = json.load(f)

history = state['log_history']

# Extract step-based metrics (filter out epoch/summary logs that don't have these keys)
steps = [x['step'] for x in history if 'loss' in x]
loss = [x['loss'] for x in history if 'loss' in x]
lr = [x['learning_rate'] for x in history if 'learning_rate' in x]
accuracy = [x['rewards/accuracies'] for x in history if 'rewards/accuracies' in x]
margins = [x['rewards/margins'] for x in history if 'rewards/margins' in x]
reward_chosen = [x['rewards/chosen'] for x in history if 'rewards/chosen' in x]
reward_rejected = [x['rewards/rejected'] for x in history if 'rewards/rejected' in x]

# Create figure
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('DPO Training Metrics Dashboard', fontsize=16)

# 1. Loss
ax1 = axs[0, 0]
ax1.plot(steps, loss, color='red', label='Loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Rewards Accuracies
ax2 = axs[0, 1]
ax2.plot(steps, accuracy, color='green', label='Reward Accuracy')
ax2.set_title('Reward Accuracy (Chosen vs Rejected)')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Margins
ax3 = axs[1, 0]
ax3.plot(steps, margins, color='purple', label='Reward Margin')
ax3.set_title('Reward Margin (Chosen - Rejected)')
ax3.set_xlabel('Steps')
ax3.set_ylabel('Margin')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Reward Values
ax4 = axs[1, 1]
ax4.plot(steps, reward_chosen, color='blue', label='Rewards Chosen')
ax4.plot(steps, reward_rejected, color='orange', label='Rewards Rejected')
ax4.set_title('Reward Values')
ax4.set_xlabel('Steps')
ax4.set_ylabel('Reward')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig('dpo_metrics_dashboard.png', dpi=150)
print('Plot saved successfully.')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics
with open('metrics_temporal.json', 'r') as f:
    data = json.load(f)

cm = np.array(data['train']['confusion_matrix'])
labels = ['Normal', 'Tendon']

plt.figure(figsize=(5,4))
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Training Confusion Matrix')
plt.tight_layout()
plt.savefig('train_confusion_matrix.png', dpi=150)
print('Saved train_confusion_matrix.png')

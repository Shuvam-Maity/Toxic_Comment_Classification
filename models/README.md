# Models

The fine-tuned RoBERTa model is saved here when you run `notebooks/03_roberta_finetuning.ipynb` (folder `best_roberta_model/`).

Model weights are excluded from git via `.gitignore` since they're too large for a normal repo (~500MB).

The trained model is available on Kaggle Models: [shuvammaity40/roberta](https://www.kaggle.com/models/shuvammaity40/roberta)

To load it directly from Kaggle in a notebook:

\`\`\`python
import kagglehub

model_path = kagglehub.model_download("shuvammaity40/roberta")
\`\`\`

Or download manually via the Kaggle CLI:

\`\`\`bash
kaggle models instances versions download shuvammaity40/roberta
\`\`\`

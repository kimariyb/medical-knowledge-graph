from pathlib import Path


class AppConfig:
    """Paths used by the data-preprocessing workflow."""

    def __init__(self):
        # Resolve from this file rather than the process working directory, so
        # the application behaves identically when launched from the repository
        # root, ``01_BiLSTM``, or an IDE.
        project_dir = Path(__file__).resolve().parent
        data_dir = project_dir / "data"
        preprocessed_dir = data_dir / "preprocessed"

        self.labels_file = preprocessed_dir / "labels.json"
        self.tag2id_file = preprocessed_dir / "tag2id.json"
        self.origin_path = data_dir / "origin"

        self.train_path = preprocessed_dir / "train.txt"
        self.sample_metadata_path = preprocessed_dir / "sample_metadata.jsonl"
        self.vocab_path = preprocessed_dir / "vocab.txt"

        self.batch_size = 8
        self.train_ratio = 0.8
        self.seed = 42

        self.epochs = 10
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.dropout = 0.4
        self.device = "mps"
        self.embedding_dim = 128
        self.hidden_dim = 128

        self.model = "all"
        self.gradient_clip_norm = 5.0
        self.use_class_weights = True
        self.class_weight_power = 0.5
        self.early_stopping_patience = 3
        self.early_stopping_min_delta = 1e-4
        self.checkpoint_dir = project_dir / "checkpoints"
        self.metrics_csv_path = project_dir / "metrics" / "training_metrics.csv"

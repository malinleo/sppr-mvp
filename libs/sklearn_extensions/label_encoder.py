from sklearn.preprocessing import LabelEncoder


class NormalizingLabelEncoder(LabelEncoder):
    """Add normalization to label encoding."""

    def transform(self, y):
        """Transform and normalize labels."""
        transformed = super().transform(y)
        max_value = max(transformed)
        return [label / max_value for label in transformed]
    
    def fit_transform(self, y):
        """Fit, transform and normalize labels."""
        transformed = super().fit_transform(y)
        max_value = max(transformed)
        return [label / max_value for label in transformed]

    def inverse_transform(self, y):
        """Normalize values back and perform inverse transform."""
        max_value = max(super().transform(self.classes_))
        y = [elem * max_value for elem in y]
        return super().inverse_transform(y)

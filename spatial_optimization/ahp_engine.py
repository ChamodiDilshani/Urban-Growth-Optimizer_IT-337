"""
Analytic Hierarchy Process (AHP) engine for multi-criteria decision analysis.
"""

import numpy as np

# Random Index (RI) table for calculating Consistency Ratio
# Keys: number of criteria, Values: RI value
RI_TABLE: dict[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

class AHPEngine:
    """
    Engine for executing the Analytic Hierarchy Process (AHP).
    
    Attributes:
        criteria_names (list[str]): The names of the criteria.
        comparison_matrix (np.ndarray): Pairwise comparison matrix.
    """

    def __init__(self, criteria_names: list[str], comparison_matrix: np.ndarray | None = None) -> None:
        """
        Initializes the AHPEngine.
        
        Args:
            criteria_names: A list of strings representing the names of the criteria.
            comparison_matrix: A 2D numpy array representing the pairwise comparisons. 
                If None, uses a default 4x4 matrix.
        
        Raises:
            ValueError: If the number of criteria does not match the matrix dimensions.
        """
        self.criteria_names = criteria_names
        n = len(self.criteria_names)

        if comparison_matrix is None:
            if n != 4:
                raise ValueError("Default matrix requires exactly 4 criteria.")
            # Default pairwise comparison matrix for 4 criteria
            self.comparison_matrix = np.array([
                [1.0,     3.0,     5.0,     7.0],
                [1.0/3.0, 1.0,     3.0,     5.0],
                [1.0/5.0, 1.0/3.0, 1.0,     3.0],
                [1.0/7.0, 1.0/5.0, 1.0/3.0, 1.0]
            ], dtype=float)
        else:
            self.comparison_matrix = comparison_matrix
            if self.comparison_matrix.shape != (n, n):
                raise ValueError(f"Comparison matrix shape {self.comparison_matrix.shape} does not match criteria count {n}.")

    def calculate_weights(self) -> np.ndarray:
        """
        Computes the principal eigenvector of the comparison matrix.
        
        Returns:
            np.ndarray: A 1D array of normalized criteria weights.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.comparison_matrix)
        
        # Find the index of the maximum eigenvalue
        max_index = np.argmax(np.real(eigenvalues))
        
        # Extract the corresponding eigenvector, take the real part
        principal_eigenvector = np.real(eigenvectors[:, max_index])
        
        # Normalize so weights sum to 1.0
        weights = principal_eigenvector / np.sum(principal_eigenvector)
        return weights

    def calculate_consistency(self) -> dict[str, float]:
        """
        Calculates consistency metrics for the comparison matrix.
        
        Returns:
            dict[str, float]: Consistency metrics including lambda_max, CI, RI, and CR.
        """
        n = len(self.criteria_names)
        eigenvalues, _ = np.linalg.eig(self.comparison_matrix)
        lambda_max = float(np.max(np.real(eigenvalues)))
        
        if n <= 1:
            return {'lambda_max': lambda_max, 'ci': 0.0, 'ri': 0.0, 'cr': 0.0}
            
        ci = (lambda_max - n) / (n - 1)
        ri = RI_TABLE.get(n)
        
        if ri is None:
            raise ValueError(f"RI value not available for n={n}")
            
        cr = ci / ri if ri > 0 else 0.0
        
        return {
            'lambda_max': lambda_max,
            'ci': ci,
            'ri': ri,
            'cr': cr
        }

    def validate_consistency(self, cr_threshold: float = 0.10) -> bool:
        """
        Validates the consistency of the pairwise comparison matrix.
        
        Args:
            cr_threshold: The maximum acceptable Consistency Ratio (default 0.10).
            
        Returns:
            bool: True if CR < cr_threshold, else False.
        """
        metrics = self.calculate_consistency()
        return metrics['cr'] < cr_threshold

    def display_matrix(self) -> None:
        """
        Prints the comparison matrix with row and column labels.
        """
        print(f"{'':>25}", end='')
        for name in self.criteria_names:
            # truncate name for cleaner formatting
            print(f"{name[:12]:>14}", end='')
        print()
        
        for i, row_name in enumerate(self.criteria_names):
            print(f"{row_name:>25}", end='')
            for val in self.comparison_matrix[i]:
                print(f"{val:14.4f}", end='')
            print()

    def run(self) -> tuple[np.ndarray, dict[str, float]]:
        """
        Executes the full AHP pipeline.
        
        Returns:
            tuple[np.ndarray, dict[str, float]]: The computed weights and consistency metrics.
            
        Raises:
            ValueError: If the calculated consistency ratio is greater than or equal to 0.10.
        """
        weights = self.calculate_weights()
        metrics = self.calculate_consistency()
        
        print("--- AHP Analysis Results ---")
        print("Criteria Weights:")
        for name, weight in zip(self.criteria_names, weights):
            print(f"  {name}: {weight:.4f}")
            
        print("\nConsistency Metrics:")
        print(f"  Lambda Max: {metrics['lambda_max']:.4f}")
        print(f"  CI: {metrics['ci']:.4f}")
        print(f"  CR: {metrics['cr']:.4f}")
        
        is_valid = self.validate_consistency()
        print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
        
        if not is_valid:
            raise ValueError(f"Inconsistent pairwise comparisons: CR ({metrics['cr']:.4f}) >= 0.10")
            
        return weights, metrics

if __name__ == '__main__':
    default_criteria = [
        "Road Proximity",
        "Slope Safety",
        "Future Growth Velocity",
        "Existing Amenity Buffer"
    ]
    
    engine = AHPEngine(criteria_names=default_criteria)
    print("Initial Pairwise Comparison Matrix:")
    engine.display_matrix()
    print()
    engine.run()

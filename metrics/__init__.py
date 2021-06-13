from .beta import BetaVAEMetric
from .factor import FactorVAEMetric
from .mig import MigMetric
from .dci import DciMetric
from .modularity_explicitness import Modularity
from .sap import SapMetric
from .unsupervised import UnsupervisedMetrics
from .factor_leakage import FLMetric
from .downstream import Downstream
from .measure_independence import TrueIndep

pretty_metric_names = {
    'dmetric/hig_acc': 'Hig Acc',
    'dmetric/factor_acc': 'Factor Acc',
    'dmetric/val_hig_acc': 'Val Beta',
    'dmetric/discrete_mig': 'MIG',
    'dmetric/informativeness_train': 'Train Inform',
    'dmetric/informativeness_test': 'Informativeness',
    'dmetric/disentanglement': 'DCI',
    'dmetric/completeness': 'Completeness',
    'dmetric/modularity_score': 'Modularity',
    'dmetric/explicitness_score_train': 'Train Explicit',
    'dmetric/explicitness_score_test': 'Explicitness',
    'dmetric/SAP_score': 'SAP',
    'dmetric/gaussian_total_correlation': 'GaussTC',
    'dmetric/gaussian_wasserstein_correlation': 'WassCorr',
    'dmetric/gaussian_wasserstein_correlation_norm': 'WassCorrNorm',
    'dmetric/mutual_info_score': 'MI Score',
    'dmetric/factor_leakage_mean': 'FL Mean',
    'dmetric/factor_leakage_norm_mean': 'FL Norm Mean',
    'dmetric/factor_leakage_auc': 'FL AUC',
    'dmetric/factor_leakage_nm_auc': 'FL NM AUC',
    'dmetric/downstream_rep': 'Downstream Rep',
    'dmetric/true_independence': 'True Indep',
    'dmetric/symmetry_l1': 'Symmetry L1',
    'dmetric/expected_dist': 'Expected Dist',
    'dmetric/rep_mean_x2': 'Rep Mean x2',
    'dmetric/rep_mean_z2': 'Rep Mean z2',
}

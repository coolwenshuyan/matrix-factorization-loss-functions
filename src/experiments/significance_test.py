"""
统计显著性检验模块

提供多种统计检验方法验证实验结果的显著性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import warnings


@dataclass
class TestResult:
    """检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    
    # 附加信息
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    
    # 详细结果
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
            
    def get_summary(self) -> str:
        """获取结果摘要"""
        sig_text = "显著" if self.significant else "不显著"
        summary = f"{self.test_name}: p={self.p_value:.4f} ({sig_text}, α={self.alpha})"
        
        if self.effect_size is not None:
            summary += f", 效应量={self.effect_size:.3f}"
            
        return summary


@dataclass
class MultipleComparison:
    """多重比较结果"""
    method: str
    comparisons: List[Tuple[str, str]]
    p_values: List[float]
    adjusted_p_values: List[float]
    significant: List[bool]
    alpha: float = 0.05
    
    def get_significant_pairs(self) -> List[Tuple[str, str]]:
        """获取显著差异的配对"""
        return [
            self.comparisons[i] 
            for i, sig in enumerate(self.significant) 
            if sig
        ]
        
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame({
            'Group1': [c[0] for c in self.comparisons],
            'Group2': [c[1] for c in self.comparisons],
            'p_value': self.p_values,
            'adjusted_p_value': self.adjusted_p_values,
            'significant': self.significant
        })


@dataclass
class EffectSize:
    """效应量"""
    cohen_d: Optional[float] = None
    eta_squared: Optional[float] = None
    omega_squared: Optional[float] = None
    r_squared: Optional[float] = None
    
    def get_interpretation(self) -> Dict[str, str]:
        """获取效应量解释"""
        interpretations = {}
        
        if self.cohen_d is not None:
            d = abs(self.cohen_d)
            if d < 0.2:
                interpretation = "很小"
            elif d < 0.5:
                interpretation = "小"
            elif d < 0.8:
                interpretation = "中等"
            else:
                interpretation = "大"
            interpretations['cohen_d'] = f"{self.cohen_d:.3f} ({interpretation})"
            
        if self.eta_squared is not None:
            if self.eta_squared < 0.01:
                interpretation = "很小"
            elif self.eta_squared < 0.06:
                interpretation = "小"
            elif self.eta_squared < 0.14:
                interpretation = "中等"
            else:
                interpretation = "大"
            interpretations['eta_squared'] = f"{self.eta_squared:.3f} ({interpretation})"
            
        return interpretations


class SignificanceTest:
    """统计显著性检验"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def t_test(self,
               group1: np.ndarray,
               group2: np.ndarray,
               paired: bool = False,
               equal_var: bool = None) -> TestResult:
        """t检验"""
        # 数据验证
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("每组至少需要2个样本")
            
        # 如果未指定方差齐性，先进行Levene检验
        if equal_var is None and not paired:
            _, levene_p = stats.levene(group1, group2)
            equal_var = levene_p > 0.05
            
        # 执行t检验
        if paired:
            if len(group1) != len(group2):
                raise ValueError("配对t检验需要相同的样本数")
            statistic, p_value = stats.ttest_rel(group1, group2)
            test_name = "配对t检验"
        else:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
            test_name = f"独立t检验 (等方差={equal_var})"
            
        # 计算效应量
        effect_size = self._calculate_cohen_d(group1, group2, paired)
        
        # 计算置信区间
        mean_diff = np.mean(group1) - np.mean(group2)
        if paired:
            diff = group1 - group2
            se = stats.sem(diff)
            df = len(diff) - 1
        else:
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            if equal_var:
                sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                se = sp * np.sqrt(1/n1 + 1/n2)
                df = n1 + n2 - 2
            else:
                se = np.sqrt(s1**2/n1 + s2**2/n2)
                df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci = (mean_diff - t_critical*se, mean_diff + t_critical*se)
        
        # 计算检验功效
        power = self._calculate_power(effect_size, len(group1), len(group2), self.alpha)
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=ci,
            power=power,
            sample_size=len(group1) + len(group2),
            details={
                'mean_diff': mean_diff,
                'group1_mean': np.mean(group1),
                'group2_mean': np.mean(group2),
                'group1_std': np.std(group1, ddof=1),
                'group2_std': np.std(group2, ddof=1)
            }
        )
        
    def wilcoxon_test(self,
                     group1: np.ndarray,
                     group2: np.ndarray,
                     paired: bool = True) -> TestResult:
        """Wilcoxon检验（非参数）"""
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        if paired:
            if len(group1) != len(group2):
                raise ValueError("配对检验需要相同的样本数")
            statistic, p_value = stats.wilcoxon(group1, group2)
            test_name = "Wilcoxon符号秩检验"
        else:
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U检验"
            
        # 计算效应量（秩双列相关）
        if paired:
            n = len(group1)
            r = 1 - (2 * statistic) / (n * (n + 1))
        else:
            n1, n2 = len(group1), len(group2)
            r = 1 - (2 * statistic) / (n1 * n2)
            
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
            sample_size=len(group1) + len(group2),
            details={
                'median_diff': np.median(group1) - np.median(group2),
                'group1_median': np.median(group1),
                'group2_median': np.median(group2)
            }
        )
        
    def anova(self,
              *groups: np.ndarray,
              use_welch: bool = False) -> TestResult:
        """方差分析"""
        if len(groups) < 2:
            raise ValueError("ANOVA需要至少2组数据")
            
        groups = [np.asarray(g) for g in groups]
        
        # 检查方差齐性
        _, levene_p = stats.levene(*groups)
        
        if use_welch or levene_p < 0.05:
            # Welch's ANOVA（方差不齐）
            statistic, p_value = self._welch_anova(*groups)
            test_name = "Welch's ANOVA"
        else:
            # 标准ANOVA
            statistic, p_value = stats.f_oneway(*groups)
            test_name = "单因素ANOVA"
            
        # 计算效应量（eta squared）
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # 计算omega squared
        k = len(groups)
        N = sum(len(g) for g in groups)
        ms_between = ss_between / (k - 1)
        ms_within = (ss_total - ss_between) / (N - k)
        omega_squared = (ss_between - (k-1)*ms_within) / (ss_total + ms_within)
        
        effect = EffectSize(eta_squared=eta_squared, omega_squared=omega_squared)
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=eta_squared,
            sample_size=sum(len(g) for g in groups),
            details={
                'k_groups': len(groups),
                'group_means': [np.mean(g) for g in groups],
                'group_stds': [np.std(g, ddof=1) for g in groups],
                'group_sizes': [len(g) for g in groups],
                'levene_p': levene_p,
                'effect_sizes': effect
            }
        )
        
    def kruskal_wallis(self, *groups: np.ndarray) -> TestResult:
        """Kruskal-Wallis检验（非参数ANOVA）"""
        if len(groups) < 2:
            raise ValueError("需要至少2组数据")
            
        groups = [np.asarray(g) for g in groups]
        statistic, p_value = stats.kruskal(*groups)
        
        # 计算效应量（epsilon squared）
        N = sum(len(g) for g in groups)
        k = len(groups)
        epsilon_squared = (statistic - k + 1) / (N - k)
        
        return TestResult(
            test_name="Kruskal-Wallis检验",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=epsilon_squared,
            sample_size=N,
            details={
                'k_groups': k,
                'group_medians': [np.median(g) for g in groups],
                'group_sizes': [len(g) for g in groups]
            }
        )
        
    def friedman_test(self, *groups: np.ndarray) -> TestResult:
        """Friedman检验（重复测量的非参数检验）"""
        groups = [np.asarray(g) for g in groups]
        
        # 检查样本数一致
        if len(set(len(g) for g in groups)) != 1:
            raise ValueError("Friedman检验需要相同的样本数")
            
        statistic, p_value = stats.friedmanchisquare(*groups)
        
        # 计算Kendall's W（一致性系数）
        n = len(groups[0])
        k = len(groups)
        
        # 计算秩和
        ranks = np.array([stats.rankdata(row) for row in np.array(groups).T])
        rank_sums = np.sum(ranks, axis=0)
        
        # Kendall's W
        mean_rank = n * (k + 1) / 2
        ss_ranks = np.sum((rank_sums - n * mean_rank)**2)
        W = 12 * ss_ranks / (n**2 * k * (k**2 - 1))
        
        return TestResult(
            test_name="Friedman检验",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=W,
            sample_size=n * k,
            details={
                'n_subjects': n,
                'k_conditions': k,
                'kendall_w': W,
                'group_medians': [np.median(g) for g in groups]
            }
        )
        
    def multiple_comparison(self,
                          groups: Dict[str, np.ndarray],
                          method: str = 'bonferroni',
                          parametric: bool = True) -> MultipleComparison:
        """多重比较"""
        group_names = list(groups.keys())
        comparisons = []
        p_values = []
        
        # 生成所有配对
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                comparisons.append((name1, name2))
                
                # 执行配对检验
                if parametric:
                    result = self.t_test(groups[name1], groups[name2])
                else:
                    result = self.wilcoxon_test(groups[name1], groups[name2], paired=False)
                    
                p_values.append(result.p_value)
                
        # 多重比较校正
        if method == 'bonferroni':
            adjusted_alpha = self.alpha / len(comparisons)
            adjusted_p_values = [min(p * len(comparisons), 1.0) for p in p_values]
            significant = [p < adjusted_alpha for p in p_values]
            
        elif method == 'holm':
            reject, adjusted_p_values, _, _ = multipletests(
                p_values, alpha=self.alpha, method='holm'
            )
            significant = list(reject)
            
        elif method == 'fdr':
            reject, adjusted_p_values, _, _ = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
            significant = list(reject)
            
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        return MultipleComparison(
            method=method,
            comparisons=comparisons,
            p_values=p_values,
            adjusted_p_values=list(adjusted_p_values),
            significant=significant,
            alpha=self.alpha
        )
        
    def tukey_hsd(self,
                  data: pd.DataFrame,
                  value_col: str,
                  group_col: str) -> pd.DataFrame:
        """Tukey HSD检验"""
        # 执行Tukey HSD
        tukey_result = pairwise_tukeyhsd(
            endog=data[value_col],
            groups=data[group_col],
            alpha=self.alpha
        )
        
        # 转换为DataFrame
        result_df = pd.DataFrame(
            data=tukey_result._results_table.data[1:],
            columns=tukey_result._results_table.data[0]
        )
        
        return result_df
        
    def normality_test(self, data: np.ndarray) -> Dict[str, TestResult]:
        """正态性检验"""
        data = np.asarray(data)
        
        results = {}
        
        # Shapiro-Wilk检验
        if len(data) <= 5000:
            stat, p = stats.shapiro(data)
            results['shapiro'] = TestResult(
                test_name="Shapiro-Wilk检验",
                statistic=stat,
                p_value=p,
                significant=p < self.alpha,
                alpha=self.alpha,
                sample_size=len(data)
            )
            
        # Kolmogorov-Smirnov检验
        stat, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        results['ks'] = TestResult(
            test_name="Kolmogorov-Smirnov检验",
            statistic=stat,
            p_value=p,
            significant=p < self.alpha,
            alpha=self.alpha,
            sample_size=len(data)
        )
        
        # Anderson-Darling检验
        result = stats.anderson(data, dist='norm')
        # 使用5%显著性水平的临界值
        critical_value = result.critical_values[2]
        significant = result.statistic > critical_value
        
        results['anderson'] = TestResult(
            test_name="Anderson-Darling检验",
            statistic=result.statistic,
            p_value=None,  # Anderson检验不提供p值
            significant=significant,
            alpha=self.alpha,
            sample_size=len(data),
            details={'critical_values': dict(zip(result.significance_level, result.critical_values))}
        )
        
        return results
        
    def homogeneity_test(self, *groups: np.ndarray) -> TestResult:
        """方差齐性检验"""
        groups = [np.asarray(g) for g in groups]
        
        # Levene检验
        statistic, p_value = stats.levene(*groups, center='median')
        
        return TestResult(
            test_name="Levene检验",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_size=sum(len(g) for g in groups),
            details={
                'group_variances': [np.var(g, ddof=1) for g in groups],
                'group_sizes': [len(g) for g in groups]
            }
        )
        
    def _calculate_cohen_d(self, group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
        """计算Cohen's d效应量"""
        if paired:
            diff = group1 - group2
            d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            # 合并标准差
            s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            
            d = (np.mean(group1) - np.mean(group2)) / s_pooled
            
        return d
        
    def _calculate_power(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """计算统计功效（简化版）"""
        # 使用statsmodels进行功效分析
        try:
            from statsmodels.stats.power import ttest_power
            power = ttest_power(effect_size, n1, alpha, n2)
            return power
        except ImportError:
            # 如果没有statsmodels，返回None
            return None
            
    def _welch_anova(self, *groups: np.ndarray) -> Tuple[float, float]:
        """Welch's ANOVA实现"""
        k = len(groups)
        ni = np.array([len(g) for g in groups])
        mi = np.array([np.mean(g) for g in groups])
        vi = np.array([np.var(g, ddof=1) for g in groups])
        
        wi = ni / vi
        tmp = np.sum(wi * mi) / np.sum(wi)
        
        dfn = k - 1
        dfd = 1 / (3 * np.sum((1 - wi / np.sum(wi))**2 / (ni - 1)) / (k**2 - 1))
        
        F = np.sum(wi * (mi - tmp)**2) / ((k - 1) * (1 + 2 * (k - 2) / (k**2 - 1) * 
            np.sum((1 - wi / np.sum(wi))**2 / (ni - 1))))
        
        p_value = 1 - stats.f.cdf(F, dfn, dfd)
        
        return F, p_value
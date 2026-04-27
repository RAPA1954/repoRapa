# ------------------ 1. INTRODUCCIÓN DE DATOS ------------------
import pandas as pd
import numpy as np
from scipy.stats import gamma

# Cargar datos observados (estación) y simulados (GCM)
obs = pd.read_csv('Estacion_AeroNqn.csv', parse_dates=['fecha'], index_col='fecha')
gcm = pd.read_csv('MPI_ESM1_2HR_ssp245_AeroNqn.csv', parse_dates=['date'], index_col='date')

# Supongamos que las columnas son:
# obs['prec']   → precipitación observada (mm/día)
# gcm['pr_mm']  → precipitación simulada (mm/día)

# Definir períodos de calibración y futuro
obs_cal = obs['prec']['1990-01-01':'2010-12-31']
gcm_cal = gcm['pr_mm']['1990-01-01':'2010-12-31']
gcm_fut = gcm['pr_mm']['2040-01-01':'2060-12-31']


# ------------------ 2. DESARROLLO DEL SCRIPT ------------------
def delta_quantile_mapping(obs_cal, gcm_cal, gcm_fut,
                           distribucion='empirica', n_quantiles=100):
    """
    Delta Quantile Mapping (DQM) para corrección de sesgo.
    distribucion: 'empirica' o 'gamma' (para precipitación)
    """
    if distribucion == 'empirica':
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        q_obs = np.quantile(obs_cal.dropna(), quantiles)
        q_gcm_cal = np.quantile(gcm_cal.dropna(), quantiles)
        q_gcm_fut = np.quantile(gcm_fut.dropna(), quantiles)

        # Delta entre futuro y calibración en el GCM
        delta = q_gcm_fut - q_gcm_cal

        # Aplicar delta a la distribución observada
        q_corr = q_obs + delta

        # Interpolación: mapear cada valor del GCM futuro
        gcm_corr = np.interp(gcm_fut.values, q_gcm_fut, q_corr,
                             left=q_corr[0], right=q_corr[-1])

    elif distribucion == 'gamma':
        # Ajuste distribución Gamma a obs y gcm (solo días lluviosos)
        umbral = 0.1  # mm umbral día lluvioso
        obs_wet = obs_cal[obs_cal > umbral].dropna()
        gcm_cal_wet = gcm_cal[gcm_cal > umbral].dropna()
        gcm_fut_wet = gcm_fut[gcm_fut > umbral].dropna()

        # Ajuste Gamma
        a_obs, _, scale_obs = gamma.fit(obs_wet, floc=0)
        a_cal, _, scale_cal = gamma.fit(gcm_cal_wet, floc=0)
        a_fut, _, scale_fut = gamma.fit(gcm_fut_wet, floc=0)

        # Delta en parámetros (ejemplo simple: escala)
        delta_scale = scale_fut - scale_cal

        # Nueva escala corregida
        scale_corr = scale_obs + delta_scale

        # Transformación cuantílica
        p = gamma.cdf(gcm_fut_wet.values, a_fut, scale=scale_fut)
        gcm_corr_wet = gamma.ppf(p, a_obs, scale=scale_corr)

        gcm_corr = gcm_fut.copy()
        gcm_corr[gcm_fut > umbral] = gcm_corr_wet
        gcm_corr[gcm_fut <= umbral] = 0

    return pd.Series(gcm_corr, index=gcm_fut.index).clip(lower=0)


# ------------------ 3. APLICACIÓN Y EXPORTACIÓN ------------------
# Aplicar corrección con DQM empírico
gcm_corr_emp = delta_quantile_mapping(obs_cal, gcm_cal, gcm_fut, distribucion='empirica')

# Aplicar corrección con DQM gamma
gcm_corr_gamma = delta_quantile_mapping(obs_cal, gcm_cal, gcm_fut, distribucion='gamma')

# Guardar resultados
gcm_corr_emp.to_csv('MPI_ESM1_2HR_ssp245_AeroNqn_corr_DQM_empirica.csv')
gcm_corr_gamma.to_csv('MPI_ESM1_2HR_ssp245_AeroNqn_corr_DQM_gamma.csv')

print("Corrección Delta Quantile Mapping completada.")
